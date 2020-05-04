import os
import torch
import argparse
import torch.optim as optim
import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from multiprocessing import Pool

import rbp_probe_modules.rbp_dataloader
from rbp_probe_modules.rbp_dataloader import RBPFolder
from rbp_probe_modules.rbp_classifier import RBP_EMB_Classifier
import lib.plot_utils, lib.logger

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', required=True)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--dataset_name', type=str, default='data_RBPsmed.h5')

all_output_size = {
    'data_RBPslow.h5': 27,
    'data_RBPsmed.h5': 21,
    'data_RBPshigh.h5': 11
}

expr_investigate = '/home/zichao/lstm_baseline_output/20200429-222820-flow-prior-resumed-5e-4-1e-2'
epochs_to_load = []
for dirname in os.listdir(expr_investigate):
    if dirname.startswith('model'):
        epochs_to_load.append(int(dirname.split('-')[-1]))
epochs_to_load = list(np.sort(epochs_to_load))


def evaluate(loader):
    all_loss, all_preds, all_label = 0., [], []
    size = 0
    with torch.no_grad():
        for latent_vec, label in loader:
            # compute various metrics
            ret_dict = rbp_probe(latent_vec, label)
            loss = ret_dict['loss'] / ret_dict['nb_preds']
            all_loss += loss.item()
            preds = ret_dict['preds']
            all_preds.append(preds)
            all_label.append(label)

            size += 1

        all_preds = np.concatenate(all_preds, axis=0)
        all_label = np.concatenate(all_label, axis=0)

        cate_roc_auc, cate_ap = [], []

        for cate_idx in range(output_size):
            roc_score = roc_auc_score(all_label[:, cate_idx], all_preds[:, cate_idx])
            cate_roc_auc.append(roc_score)
            ap_score = average_precision_score(all_label[:, cate_idx], all_preds[:, cate_idx])
            cate_ap.append(ap_score)

        cate_roc_auc = np.array(cate_roc_auc)
        cate_ap = np.array(ap_score)
        all_loss /= size

        return all_loss, cate_roc_auc, cate_ap


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    input_size = 128  # latent dimension
    output_size = all_output_size[args.dataset_name]

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = '/'.join(args.save_dir.split('/')[:-1] + [cur_time + '-' + args.save_dir.split('/')[-1]])

    all_test_loss, all_test_roc, all_test_ap = [], [], []

    for enc_epoch_to_load in epochs_to_load:
        rbp_probe = RBP_EMB_Classifier(input_size, args.hidden_size, output_size, device=device).to(device)
        print(rbp_probe)
        optimizer = optim.Adam(rbp_probe.parameters(), lr=args.lr)

        enc_epoch_weight_path = os.path.join(expr_investigate, 'model.epoch-%d' % enc_epoch_to_load)
        enc_epoch_dir = os.path.join(save_dir, 'enc-epoch-%d' % (enc_epoch_to_load))

        if not os.path.exists(enc_epoch_dir):
            os.makedirs(enc_epoch_dir)

        lib.plot_utils.set_output_dir(enc_epoch_dir)
        lib.plot_utils.suppress_stdout()

        all_fields = ['epoch', 'train_loss', 'valid_loss'] + \
                     ['train_roc_score_%d' % (cate_idx) for cate_idx in range(output_size)] + \
                     ['valid_roc_score_%d' % (cate_idx) for cate_idx in range(output_size)] + \
                     ['train_ap_score_%d' % (cate_idx) for cate_idx in range(output_size)] + \
                     ['valid_ap_score_%d' % (cate_idx) for cate_idx in range(output_size)]

        logger = lib.logger.CSVLogger('run.csv', enc_epoch_dir, all_fields)

        best_valid_loss = np.inf
        best_valid_weight_path = None

        mp_pool = Pool(8)

        train_loader = RBPFolder(args.dataset_name, args.batch_size, num_workers=4,
                                 mode='train', weight_path=enc_epoch_weight_path, device=device,
                                 mp_pool=mp_pool)

        valid_loader = RBPFolder(args.dataset_name, args.batch_size, num_workers=4,
                                 shuffle=False, mode='valid', weight_path=enc_epoch_weight_path,
                                 device=device, mp_pool=mp_pool)

        test_loader = RBPFolder(args.dataset_name, args.batch_size, num_workers=4,
                                shuffle=False, mode='test', weight_path=enc_epoch_weight_path,
                                device=device, mp_pool=mp_pool)

        for epoch in range(1, args.epoch + 1):

            # training loop
            for latent_vec, label in train_loader:
                rbp_probe.zero_grad()

                ret_dict = rbp_probe(latent_vec, label)
                loss = ret_dict['loss'] / ret_dict['nb_preds']
                loss.backward()
                optimizer.step()

            # validation loop
            train_loss, train_roc_auc, train_ap_score = evaluate(train_loader)
            valid_loss, valid_roc_auc, valid_ap_score = evaluate(valid_loader)

            lib.plot_utils.plot('train_loss', train_loss)
            lib.plot_utils.plot('averaged_train_roc_score', train_roc_auc.mean())
            lib.plot_utils.plot('averaged_train_ap_score', train_ap_score.mean())

            lib.plot_utils.plot('valid_loss', valid_loss)
            lib.plot_utils.plot('averaged_valid_roc_score', valid_roc_auc.mean())
            lib.plot_utils.plot('averaged_valid_ap_score', valid_ap_score.mean())

            lib.plot_utils.set_xlabel_for_tick(index=0, label='epoch')
            lib.plot_utils.flush()
            lib.plot_utils.tick(index=0)

            save_dict = {'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss}
            for cate_idx in range(output_size):
                save_dict['train_roc_score_%d' % (cate_idx)] = train_roc_auc[cate_idx]
                save_dict['valid_roc_score_%d' % (cate_idx)] = valid_roc_auc[cate_idx]
                save_dict['train_ap_score_%d' % (cate_idx)] = train_ap_score[cate_idx]
                save_dict['valid_ap_score_%d' % (cate_idx)] = valid_ap_score[cate_idx]

            logger.update_with_dict(save_dict)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_weight_path = os.path.join(save_dir, "model.epoch-" + str(epoch))
                torch.save(
                    {'model_weights': rbp_probe.state_dict(),
                     'opt_weights': optimizer.state_dict()},
                    best_valid_weight_path)

        if best_valid_weight_path is not None:
            print('Loading best weights from: %s' % (best_valid_weight_path))
            rbp_probe.load_state_dict(torch.load(best_valid_weight_path)['model_weights'])

        test_loss, test_roc_auc, test_ap_score = evaluate(test_loader)

        all_test_loss.append(test_loss)
        all_test_roc.append(test_roc_auc)
        all_test_ap.append(test_ap_score)

        logger.close()

        lib.plot_utils.set_output_dir(save_dir)
        lib.plot_utils.plot('all_test_loss', test_loss, index=1)
        lib.plot_utils.plot('all_test_roc_score', test_roc_auc, index=1)
        lib.plot_utils.plot('all_test_ap', test_ap_score, index=1)

        lib.plot_utils.set_xlabel_for_tick(index=1, label='epoch')
        lib.plot_utils.flush()
        lib.plot_utils.tick(index=1)

        del rbp_probe_modules.rbp_dataloader.model
        rbp_probe_modules.rbp_dataloader.model = None

    np.savetxt(os.path.join(save_dir, 'all_test_loss'), all_test_loss)
    np.savetxt(os.path.join(save_dir, 'all_test_roc_scores'), all_test_roc)
    np.savetxt(os.path.join(save_dir, 'all_test_ap_scores'), all_test_ap)

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
