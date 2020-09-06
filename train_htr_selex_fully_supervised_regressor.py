import os
import sys
import torch
import argparse
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from supervised_encoder_models.task_dataloader import TaskFolder, read_htr_selex_fasta
from supervised_encoder_models.supervised_encoder_model import FULL_ENC_Model
import lib.plot_utils, lib.logger

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='mlp')
parser.add_argument('--hidden_size', type=eval, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--mode', type=str, default='lstm')
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--epoch', type=int, default=100)


def evaluate(loader):
    all_loss = 0.
    size = loader.size
    all_preds, all_label = [], []

    with torch.no_grad():
        for batch_input, batch_label in loader:
            # compute various metrics
            ret_dict = model(batch_input, batch_label)
            all_loss += ret_dict['loss'].item()

            all_preds.append(ret_dict['preds'])
            all_label.append(batch_label)

    all_preds = np.concatenate(all_preds, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    cate_roc_auc, cate_ap = [], []

    for cate_idx in range(output_size):
        roc_score = roc_auc_score(all_label[:, cate_idx], all_preds[:, cate_idx])
        cate_roc_auc.append(roc_score)
        ap_score = average_precision_score(all_label[:, cate_idx], all_preds[:, cate_idx])
        cate_ap.append(ap_score)

    cate_roc_auc = np.array(cate_roc_auc)
    cate_ap = np.array(cate_ap)
    all_loss /= size * output_size

    return all_loss, cate_roc_auc, cate_ap


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)
    # rbp_name = args.rbp_name
    # if rbp_name == 'all':
    #     pass
    #     all_rbps = rnacompete_s_all_rbps
    # else:
    #     assert rbp_name in rnacompete_s_all_rbps
    #     all_rbps = [rbp_name]

    preprocess_type = args.mode
    input_size = 128  # latent dimension
    output_size = 10
    train_val_split_ratio = 0.1
    loss_type = 'binary_ce'

    # for rbp_name in all_rbps:
    for rbp_name in ['10-RBPs']:
        save_dir = os.path.join('full-htr-selex-regressor', args.save_dir, rbp_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            continue

        outfile = open(os.path.join(save_dir, '%s.out' % (args.save_dir)), "w")
        sys.stdout = outfile
        sys.stderr = outfile

        datapath_train = 'data/htr_selex_10rbps_multilabel/train.fa'
        train_seq, train_targets = read_htr_selex_fasta(datapath_train, output_size)

        perm = np.random.permutation(len(train_seq))
        all_valid_idx = perm[:int(train_val_split_ratio * len(train_seq))]
        all_train_idx = perm[int(train_val_split_ratio * len(train_seq)):]
        # all_train_idx, all_valid_idx = [], []
        # for class_label in range(output_size):
        #     class_idx = np.where(np.array(train_targets) == class_label)[0]
        #     perm = np.random.permutation(len(class_idx))
        #     all_valid_idx.extend(class_idx[perm[:int(train_val_split_ratio * len(class_idx))]])
        #     all_train_idx.extend(class_idx[perm[int(train_val_split_ratio * len(class_idx)):]])

        valid_seq = np.array(train_seq)[all_valid_idx]
        valid_targets = np.array(train_targets)[all_valid_idx]
        valid_size = len(valid_seq)
        train_seq = np.array(train_seq)[all_train_idx]
        train_targets = np.array(train_targets)[all_train_idx]
        train_size = len(train_seq)

        datapath_test = 'data/htr_selex_10rbps_multilabel/test.fa'
        test_seq, test_targets = read_htr_selex_fasta(datapath_test, output_size)
        test_targets = np.array(test_targets)

        model = FULL_ENC_Model(input_size, args.hidden_size, output_size, device=device,
                               vae_type=preprocess_type, loss_type=loss_type).to(device)
        print(model)
        import torch.nn as nn

        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            elif param.dim() >= 2:
                nn.init.xavier_normal_(param)

        print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        from importlib import reload

        reload(lib.plot_utils)
        lib.plot_utils.set_output_dir(save_dir)
        lib.plot_utils.suppress_stdout()

        all_fields = ['epoch', 'train_loss', 'valid_loss'] + \
                     ['train_roc_score_%d' % (cate_idx) for cate_idx in range(output_size)] + \
                     ['valid_roc_score_%d' % (cate_idx) for cate_idx in range(output_size)] + \
                     ['train_ap_score_%d' % (cate_idx) for cate_idx in range(output_size)] + \
                     ['valid_ap_score_%d' % (cate_idx) for cate_idx in range(output_size)]

        logger = lib.logger.CSVLogger('run.csv', save_dir, all_fields)

        best_valid_loss = np.inf
        best_valid_weight_path = None

        train_loader = TaskFolder(train_seq, train_targets, args.batch_size, shuffle=True,
                                  preprocess_type=preprocess_type, num_workers=8)
        valid_loader = TaskFolder(valid_seq, valid_targets, args.batch_size, shuffle=False,
                                  preprocess_type=preprocess_type, num_workers=8)
        test_loader = TaskFolder(test_seq, test_targets, args.batch_size, shuffle=False,
                                 preprocess_type=preprocess_type, num_workers=8)
        last_improved = 0
        last_5_epochs = []
        for epoch in range(1, args.epoch + 1):
            if last_improved >= 10:
                print('Have\'t improved for %d epochs' % (last_improved))
                break

            # training loop
            model.train()
            for batch_input, batch_label in train_loader:
                model.zero_grad()
                ret_dict = model(batch_input, batch_label)
                loss = ret_dict['loss'] / ret_dict['nb_preds'] / output_size
                # print(sum(np.argmax(ret_dict['preds'], axis=-1) == np.array(batch_label)) / ret_dict['nb_preds'])
                loss.backward()
                optimizer.step()

            model.eval()
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

            print('Epoch %d, train_loss: %.2f, averaged_train_roc_score: %.2f, averaged_train_ap_score: %.2f, '
                  'valid_loss: %.2f, averaged_valid_roc_score: %.2f, averaged_valid_ap_score: %.2f' %
                  (epoch, train_loss, train_roc_auc.mean(), train_ap_score.mean(),
                   valid_loss, valid_roc_auc.mean(), valid_ap_score.mean()))

            save_dict = {'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss}
            for cate_idx in range(output_size):
                save_dict['train_roc_score_%d' % (cate_idx)] = train_roc_auc[cate_idx]
                save_dict['valid_roc_score_%d' % (cate_idx)] = valid_roc_auc[cate_idx]
                save_dict['train_ap_score_%d' % (cate_idx)] = train_ap_score[cate_idx]
                save_dict['valid_ap_score_%d' % (cate_idx)] = valid_ap_score[cate_idx]

            logger.update_with_dict(save_dict)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if len(last_5_epochs) >= 5:
                    to_remove_epoch = last_5_epochs.pop(0)
                    os.remove(os.path.join(save_dir, "model.epoch-" + str(to_remove_epoch)))
                last_5_epochs.append(epoch)
                best_valid_weight_path = os.path.join(save_dir, "model.epoch-" + str(epoch))
                torch.save(
                    {'model_weights': model.state_dict(),
                     'opt_weights': optimizer.state_dict()},
                    best_valid_weight_path)
                print('Validation loss improved, saving current weights to path:', best_valid_weight_path)
                last_improved = 0
            else:
                last_improved += 1

        if best_valid_weight_path is not None:
            print('Loading best weights from: %s' % (best_valid_weight_path))
            model.load_state_dict(torch.load(best_valid_weight_path)['model_weights'])

        model.eval()
        test_loss, test_roc_auc, test_ap_score = evaluate(test_loader)
        print('test_loss: %.2f, averaged_test_roc_score: %.2f, averaged_test_ap_score: %.2f' %
              (test_loss, test_roc_auc.mean(), test_ap_score.mean()))

        logger.close()
