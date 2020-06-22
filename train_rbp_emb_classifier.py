import os
import sys
import torch
import argparse
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from emb_models.emb_dataloader import convert_seq_to_embeddings, read_rbp_h5py, \
    rbp_datapath, rbp_dataset_options
from emb_models.base_emb_model import EMB_Classifier
from baseline_models.FlowLSTMVAE import LSTMVAE
from baseline_models.GraphLSTMVAE import GraphLSTMVAE
from jtvae_models.VAE import JunctionTreeVAE
import lib.plot_utils, lib.logger

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='mlp')
parser.add_argument('--expr_path', type=str, default='')
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--dataset_name', type=str, default='data_RBPsmed.h5')
parser.add_argument('--mode', type=str, default='lstm')

all_output_size = {
    'data_RBPslow.h5': 27,
    'data_RBPsmed.h5': 21,
    'data_RBPshigh.h5': 11
}


def evaluate(embedding, label):
    all_loss, nb_preds, all_preds, all_label = 0., 0., [], []

    with torch.no_grad():
        for idx in range(0, len(label), args.batch_size):
            latent_vec = embedding[idx: idx + args.batch_size]
            batch_label = label[idx: idx + args.batch_size]
            # compute various metrics
            ret_dict = rbp_probe(latent_vec, batch_label)
            all_loss += ret_dict['loss'].item()
            nb_preds += ret_dict['nb_preds']

            preds = ret_dict['preds']
            all_preds.append(preds)
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
        all_loss /= nb_preds

        return all_loss, cate_roc_auc, cate_ap


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    input_size = 128  # latent dimension
    output_size = all_output_size[args.dataset_name]
    preprocess_type = args.mode

    train_val_split_ratio = 0.1
    train_seq, train_label = read_rbp_h5py(rbp_datapath.format(args.dataset_name), 'train')
    test_seq, test_label = read_rbp_h5py(rbp_datapath.format(args.dataset_name), 'test')
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    valid_idx = np.random.choice(np.arange(len(train_label)), int(len(train_label) * train_val_split_ratio),
                                 replace=False)
    valid_idx = np.array(valid_idx)
    train_idx = np.setdiff1d(np.arange(len(train_label)), valid_idx)

    valid_seq = np.array(train_seq)[valid_idx]
    valid_label = np.array(train_label)[valid_idx]
    val_size = len(valid_seq)

    train_seq = np.array(train_seq)[train_idx]
    train_label = np.array(train_label)[train_idx]
    train_size = len(train_seq)

    expr_investigate = args.expr_path
    assert os.path.exists(expr_investigate), '%s does not exist' % (expr_investigate)
    epochs_to_load = []
    for dirname in os.listdir(expr_investigate):
        if dirname.startswith('model'):
            epochs_to_load.append(int(dirname.split('-')[-1]))
    epochs_to_load = list(np.sort(epochs_to_load))

    save_dir = os.path.join(args.expr_path, 'RBP-classification-%s-%s' % (args.dataset_name, args.save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    outfile = open(os.path.join(save_dir, '%s.out' % (args.save_dir)), "w")
    sys.stdout = outfile
    sys.stderr = outfile

    all_test_loss, all_test_roc, all_test_ap = [], [], []

    mp_pool = Pool(8)

    for enc_epoch_to_load in epochs_to_load:
        rbp_probe = EMB_Classifier(input_size, args.hidden_size, output_size, device=device, loss_type='binary_ce').to(device)
        print(rbp_probe)
        optimizer = optim.Adam(rbp_probe.parameters(), lr=args.lr, amsgrad=True)

        enc_epoch_weight_path = os.path.join(expr_investigate, 'model.epoch-%d' % enc_epoch_to_load)
        enc_epoch_dir = os.path.join(save_dir, 'enc-epoch-%d' % (enc_epoch_to_load))

        if preprocess_type == 'lstm':
            pretrain_model = LSTMVAE(
                512, 128, 2, device=device, use_attention=True,
                use_flow_prior=True, use_aux_regressor=False).to(device)
        elif preprocess_type == 'graph_lstm':
            pretrain_model = GraphLSTMVAE(
                512, 128, 10, device=device, use_attention=False,
                use_flow_prior=True, use_aux_regressor=False).to(device)
        elif preprocess_type == 'jtvae':
            pretrain_model = JunctionTreeVAE(
                512, 64, 5, 10, decode_nuc_with_lstm=True, tree_encoder_arch='baseline',
                use_flow_prior=True, device=device).to(device)
        elif preprocess_type == 'jtvae_branched':
            pretrain_model = JunctionTreeVAE(
                512, 64, 5, 10, decode_nuc_with_lstm=True, tree_encoder_arch='branched',
                decoder_version='v1', use_flow_prior=True, device=device).to(device)

        pretrain_model.load_state_dict(
            torch.load(enc_epoch_weight_path, map_location=device)['model_weights'])

        if not os.path.exists(enc_epoch_dir):
            os.makedirs(enc_epoch_dir)

        from importlib import reload

        reload(lib.plot_utils)
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

        print('converting embeddings')
        train_emb = convert_seq_to_embeddings(train_seq, pretrain_model, mp_pool, preprocess_type=preprocess_type)
        valid_emb = convert_seq_to_embeddings(valid_seq, pretrain_model, mp_pool, preprocess_type=preprocess_type)

        print('Probing', enc_epoch_weight_path)
        last_improved = 0
        for epoch in range(1, args.epoch + 1):
            if last_improved >= 20:
                print('Have\'t improved for %d epochs' % (last_improved))
                break

            shuffle_idx = np.random.permutation(train_emb.size(0))
            shuffled_train_emb = train_emb[shuffle_idx]
            shuffled_train_label = np.array(train_label)[shuffle_idx]

            rbp_probe.train()

            for idx in range(0, train_size, args.batch_size):
                rbp_probe.zero_grad()

                ret_dict = rbp_probe(shuffled_train_emb[idx: idx + args.batch_size],
                                     shuffled_train_label[idx: idx + args.batch_size])
                loss = ret_dict['loss'] / ret_dict['nb_preds']
                loss.backward()
                optimizer.step()

            # validation loop
            rbp_probe.eval()
            train_loss, train_roc_auc, train_ap_score = evaluate(train_emb, train_label)
            valid_loss, valid_roc_auc, valid_ap_score = evaluate(valid_emb, valid_label)

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
                best_valid_weight_path = os.path.join(enc_epoch_dir, "model.epoch-" + str(epoch))
                torch.save(
                    {'model_weights': rbp_probe.state_dict(),
                     'opt_weights': optimizer.state_dict()},
                    best_valid_weight_path)
                print('Validation loss improved, saving current weights to path:', best_valid_weight_path)
                last_improved = 0
            else:
                last_improved += 1

        if best_valid_weight_path is not None:
            print('Loading best weights from: %s' % (best_valid_weight_path))
            rbp_probe.load_state_dict(torch.load(best_valid_weight_path)['model_weights'])

        rbp_probe.eval()
        test_emb = convert_seq_to_embeddings(test_seq, pretrain_model, mp_pool, preprocess_type=preprocess_type)
        test_loss, test_roc_auc, test_ap_score = evaluate(test_emb, test_label)

        all_test_loss.append(test_loss)
        all_test_roc.append(test_roc_auc)
        all_test_ap.append(test_ap_score)

        logger.close()

    font = {'fontname': 'Times New Roman', 'size': 14}
    plt.clf()
    ax = plt.figure(figsize=(5., 5.)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epochs_to_load, all_test_loss)
    plt.xlabel('epoch', **font)
    ax.set_xlim(xmin=epochs_to_load[0])
    plt.ylabel('test loss', **font)
    plt.savefig(os.path.join(save_dir, 'test_loss.png'), dpi=350)

    plt.clf()
    ax = plt.figure(figsize=(5., 5.)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epochs_to_load, [test_roc_auc.mean() for test_roc_auc in all_test_roc])
    plt.xlabel('epoch', **font)
    ax.set_xlim(xmin=epochs_to_load[0])
    plt.ylabel('averaged_test_roc_score', **font)
    plt.savefig(os.path.join(save_dir, 'averaged_test_roc_score.png'), dpi=350)

    plt.clf()
    ax = plt.figure(figsize=(5., 5.)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epochs_to_load, [test_ap_score.mean() for test_ap_score in all_test_ap])
    plt.xlabel('epoch', **font)
    ax.set_xlim(xmin=epochs_to_load[0])
    plt.ylabel('averaged_test_ap_score', **font)
    plt.savefig(os.path.join(save_dir, 'averaged_test_ap_score.png'), dpi=350)

    np.savetxt(os.path.join(save_dir, 'all_test_loss'), all_test_loss)
    np.savetxt(os.path.join(save_dir, 'all_test_roc_scores'), all_test_roc)
    np.savetxt(os.path.join(save_dir, 'all_test_ap_scores'), all_test_ap)

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
