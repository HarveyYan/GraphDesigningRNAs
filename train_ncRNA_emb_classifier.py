import os
import sys
import torch
import argparse
import torch.optim as optim
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from emb_models.emb_dataloader import convert_seq_to_embeddings, read_ncRNA_fasta, \
    ncRNA_all_classes, ncRNA_train_datapath, ncRNA_test_datapath
from emb_models.base_emb_model import EMB_Classifier
import lib.plot_utils, lib.logger
from baseline_models.FlowLSTMVAE import LSTMVAE
from baseline_models.GraphLSTMVAE import GraphLSTMVAE
from jtvae_models.VAE import JunctionTreeVAE

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='mlp')
parser.add_argument('--hidden_size', type=eval, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--expr_path', type=str, default='')
parser.add_argument('--mode', type=str, default='lstm')
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--epoch', type=int, default=100)


def evaluate(embedding, label):
    all_loss = 0.
    all_preds = []
    size = len(label)

    with torch.no_grad():
        for idx in range(0, size, args.batch_size):
            latent_vec = embedding[idx: idx + args.batch_size]
            batch_label = label[idx: idx + args.batch_size]
            # compute various metrics
            ret_dict = ncRNA_probe(latent_vec, batch_label)
            all_loss += ret_dict['loss'].item()

            all_preds.extend(np.argmax(ret_dict['preds'], axis=-1))

    all_loss /= size
    acc = np.sum(np.array(all_preds) == np.array(label)) / size
    f1_macro = f1_score(all_preds, label, average='macro')
    f1_micro = f1_score(all_preds, label, average='micro')
    mcc = matthews_corrcoef(all_preds, label)

    return all_loss, acc, f1_macro, f1_micro, mcc


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    preprocess_type = args.mode
    input_size = 128  # latent dimension
    output_size = len(ncRNA_all_classes)
    train_val_split_ratio = 0.1

    train_seq, train_label = read_ncRNA_fasta(ncRNA_train_datapath)
    test_seq, test_label = read_ncRNA_fasta(ncRNA_test_datapath)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    valid_idx = []
    for label in range(output_size):
        label_idx = np.where(train_label == label)[0]
        nb_label = len(label_idx)
        valid_idx.extend(np.random.choice(label_idx, int(nb_label * train_val_split_ratio), replace=False))
    valid_idx = np.array(valid_idx)
    train_idx = np.setdiff1d(np.arange(len(train_label)), valid_idx)

    # all_idx = np.random.permutation(len(train_seq))
    # train_idx = all_idx[:int(len(train_seq) * 0.9)]
    # valid_idx = all_idx[int(len(train_seq) * 0.9):]

    valid_seq = np.array(train_seq)[valid_idx]
    valid_label = np.array(train_label)[valid_idx]
    val_size = len(valid_seq)

    train_seq = np.array(train_seq)[train_idx]
    train_label = np.array(train_label)[train_idx]
    train_size = len(train_label)

    expr_investigate = args.expr_path
    assert os.path.exists(expr_investigate), '%s does not exist' % (expr_investigate)
    epochs_to_load = []
    for dirname in os.listdir(expr_investigate):
        if dirname.startswith('model'):
            epochs_to_load.append(int(dirname.split('-')[-1]))
    epochs_to_load = list(np.sort(epochs_to_load))

    save_dir = os.path.join(args.expr_path, 'ncRNA-classification-%s' % (args.save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    outfile = open(os.path.join(save_dir, '%s.out' % (args.save_dir)), "w")
    sys.stdout = outfile
    sys.stderr = outfile

    all_test_loss, all_test_acc, all_test_f1_macro, all_test_f1_micro, all_test_mcc = \
        [], [], [], [], []

    mp_pool = Pool(10)

    for enc_epoch_to_load in epochs_to_load:
        ncRNA_probe = EMB_Classifier(input_size, args.hidden_size, output_size, device=device).to(device)
        print(ncRNA_probe)
        optimizer = optim.Adam(ncRNA_probe.parameters(), lr=args.lr)

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

        all_fields = ['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc',
                      'train_f1_macro', 'valid_f1_macro', 'train_f1_micro', 'valid_f1_micro',
                      'train_mcc', 'valid_mcc']

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
            # training loop
            shuffle_idx = np.random.permutation(train_emb.size(0))
            shuffled_train_emb = train_emb[shuffle_idx]
            shuffled_train_label = np.array(train_label)[shuffle_idx]

            ncRNA_probe.train()
            for idx in range(0, train_size, args.batch_size):
                ncRNA_probe.zero_grad()
                ret_dict = ncRNA_probe(shuffled_train_emb[idx: idx + args.batch_size],
                                       shuffled_train_label[idx: idx + args.batch_size])
                loss = ret_dict['loss'] / ret_dict['nb_preds']
                loss.backward()
                optimizer.step()

            ncRNA_probe.eval()
            # validation loop
            train_loss, train_acc, train_f1_macro, train_f1_micro, train_mcc = evaluate(train_emb, train_label)
            valid_loss, valid_acc, valid_f1_macro, valid_f1_micro, valid_mcc = evaluate(valid_emb, valid_label)

            lib.plot_utils.plot('train_loss', train_loss)
            lib.plot_utils.plot('train_acc', train_acc)
            lib.plot_utils.plot('train_f1_macro', train_f1_macro)
            lib.plot_utils.plot('train_f1_micro', train_f1_micro)
            lib.plot_utils.plot('train_mcc', train_mcc)

            lib.plot_utils.plot('valid_loss', valid_loss)
            lib.plot_utils.plot('valid_acc', valid_acc)
            lib.plot_utils.plot('valid_f1_macro', valid_f1_macro)
            lib.plot_utils.plot('valid_f1_micro', valid_f1_micro)
            lib.plot_utils.plot('valid_mcc', valid_mcc)

            lib.plot_utils.set_xlabel_for_tick(index=0, label='epoch')
            lib.plot_utils.flush()
            lib.plot_utils.tick(index=0)

            print(
                'Epoch %d, train_loss: %.2f, train_acc: %2f, train_f1_macro: %.2f, train_f1_micro: %.2f, train_mcc: %.2f, '
                'valid_loss: %.2f, valid_acc: %.2f, valid_f1_macro: %.2f, valid_f1_micro: %.2f, valid_mcc: %.2f' %
                (epoch, train_loss, train_acc, train_f1_macro, train_f1_micro, train_mcc,
                 valid_loss, valid_acc, valid_f1_macro, valid_f1_micro, valid_mcc,))

            logger.update_with_dict({
                'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss,
                'train_f1_macro': train_f1_macro, 'train_f1_micro': train_f1_micro,
                'train_mcc': train_mcc, 'valid_f1_macro': valid_f1_macro,
                'valid_f1_micro': valid_f1_micro, 'valid_mcc': valid_mcc,
                'train_acc': train_acc, 'valid_acc': valid_acc
            })

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_weight_path = os.path.join(enc_epoch_dir, "model.epoch-" + str(epoch))
                torch.save(
                    {'model_weights': ncRNA_probe.state_dict(),
                     'opt_weights': optimizer.state_dict()},
                    best_valid_weight_path)
                print('Validation loss improved, saving current weights to path:', best_valid_weight_path)
                last_improved = 0
            else:
                last_improved += 1

        if best_valid_weight_path is not None:
            print('Loading best weights from: %s' % (best_valid_weight_path))
            ncRNA_probe.load_state_dict(torch.load(best_valid_weight_path)['model_weights'])

        ncRNA_probe.eval()
        test_emb = convert_seq_to_embeddings(test_seq, pretrain_model, mp_pool, preprocess_type=preprocess_type)
        test_loss, test_acc, test_f1_macro, test_f1_micro, test_mcc = evaluate(test_emb, test_label)
        print('Test acc:', test_acc)
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)
        all_test_f1_macro.append(test_f1_macro)
        all_test_f1_micro.append(test_f1_micro)
        all_test_mcc.append(test_mcc)

        logger.close()

    font = {'fontname': 'Times New Roman', 'size': 14}
    plt.clf()
    ax = plt.figure(figsize=(5., 5.)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epochs_to_load, all_test_loss)
    plt.xlabel('epoch', **font)
    ax.set_xlim(xmin=epochs_to_load[0])
    plt.ylabel('test_loss', **font)
    plt.savefig(os.path.join(save_dir, 'test_loss.png'), dpi=350)

    plt.clf()
    ax = plt.figure(figsize=(5., 5.)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epochs_to_load, all_test_acc)
    plt.xlabel('epoch', **font)
    ax.set_xlim(xmin=epochs_to_load[0])
    plt.ylabel('test_acc', **font)
    plt.savefig(os.path.join(save_dir, 'test_acc.png'), dpi=350)

    plt.clf()
    ax = plt.figure(figsize=(5., 5.)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epochs_to_load, all_test_f1_macro)
    plt.xlabel('epoch', **font)
    ax.set_xlim(xmin=epochs_to_load[0])
    plt.ylabel('test_f1_macro', **font)
    plt.savefig(os.path.join(save_dir, 'test_f1_macro.png'), dpi=350)

    plt.clf()
    ax = plt.figure(figsize=(5., 5.)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epochs_to_load, all_test_f1_micro)
    plt.xlabel('epoch', **font)
    ax.set_xlim(xmin=epochs_to_load[0])
    plt.ylabel('test_f1_micro', **font)
    plt.savefig(os.path.join(save_dir, 'test_f1_micro.png'), dpi=350)

    plt.clf()
    ax = plt.figure(figsize=(5., 5.)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epochs_to_load, all_test_mcc)
    plt.xlabel('epoch', **font)
    ax.set_xlim(xmin=epochs_to_load[0])
    plt.ylabel('test_mcc', **font)
    plt.savefig(os.path.join(save_dir, 'test_mcc.png'), dpi=350)

    np.savetxt(os.path.join(save_dir, 'all_test_metrics'),
               [all_test_loss, all_test_acc, all_test_f1_macro,
                all_test_f1_micro, all_test_mcc])

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
