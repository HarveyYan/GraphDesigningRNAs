import os
import torch
import argparse
import torch.optim as optim
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys

from emb_models.emb_dataloader import convert_seq_to_embeddings, rnacompete_s_all_rbps, \
    read_curated_rnacompete_s_dataset
from emb_models.base_emb_model import EMB_Classifier
import lib.plot_utils, lib.logger
from baseline_models.FlowLSTMVAE import LSTMVAE
from baseline_models.GraphLSTMVAE import GraphLSTMVAE
from jtvae_models.VAE import JunctionTreeVAE

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='mlp')
parser.add_argument('--rbp_name', type=str, default='PTB')
parser.add_argument('--hidden_size', type=eval, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--expr_path', type=str, default='')
parser.add_argument('--mode', type=str, default='lstm')
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--epoch', type=int, default=100)


def evaluate(embedding, targets):
    all_loss = 0.
    all_preds = []
    size = len(targets)

    with torch.no_grad():
        for idx in range(0, size, args.batch_size):
            latent_vec = embedding[idx: idx + args.batch_size]
            batch_targets = targets[idx: idx + args.batch_size]
            # compute various metrics
            ret_dict = probe(latent_vec, batch_targets)
            all_loss += ret_dict['loss'].item()
            all_preds.append(ret_dict['preds'])

    all_loss /= size
    acc = sum(targets[:, 0] == (np.concatenate(all_preds, axis=0)[:, 0] > 0.5).astype(np.int32)) / size
    roc_auc = roc_auc_score(targets[:, 0], np.concatenate(all_preds, axis=0)[:, 0])

    return all_loss, acc, roc_auc


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)
    rbp_name = args.rbp_name
    if rbp_name == 'all':
        all_rbps = rnacompete_s_all_rbps
    else:
        assert rbp_name in rnacompete_s_all_rbps
        all_rbps = [rbp_name]

    preprocess_type = args.mode
    input_size = 128  # latent dimension
    output_size = 1
    train_val_split_ratio = 0.1
    loss_type = 'binary_ce'
    mp_pool = None  # Pool(8)

    expr_investigate = args.expr_path
    assert os.path.exists(expr_investigate), '%s does not exist' % (expr_investigate)
    epochs_to_load = []
    for dirname in os.listdir(expr_investigate):
        if dirname.startswith('model'):
            epochs_to_load.append(int(dirname.split('-')[-1]))
    epochs_to_load = list(np.sort(epochs_to_load))
    # note, only probing the last epoch
    epochs_to_load = epochs_to_load[-1:]

    for rbp_name in all_rbps:

        datapath_train = 'data/RNAcompete_S/curated_dataset/' + rbp_name + '_train.fa'
        train_pos, train_neg = read_curated_rnacompete_s_dataset(datapath_train)
        train_seq = train_pos + train_neg
        train_targets = [1] * len(train_pos) + [0] * len(train_neg)

        datapath_test = 'data/RNAcompete_S/curated_dataset/' + rbp_name + '_test.fa'
        test_pos, test_neg = read_curated_rnacompete_s_dataset(datapath_test)
        test_seq = test_pos + test_neg
        test_targets = [1] * len(test_pos) + [0] * len(test_neg)

        train_targets = np.array(train_targets)[:, None]
        test_targets = np.array(test_targets)[:, None]

        valid_idx = np.random.choice(np.arange(len(train_targets)), int(len(train_targets) * train_val_split_ratio),
                                     replace=False)
        valid_idx = np.array(valid_idx)
        train_idx = np.setdiff1d(np.arange(len(train_targets)), valid_idx)

        valid_seq = np.array(train_seq)[valid_idx]
        valid_targets = np.array(train_targets)[valid_idx]
        val_size = len(valid_seq)

        train_seq = np.array(train_seq)[train_idx]
        train_targets = np.array(train_targets)[train_idx]
        train_size = len(train_seq)

        save_dir = os.path.join(args.expr_path, 'rnacompete-S-regressor', '%s-%s' % (rbp_name, args.save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        outfile = open(os.path.join(save_dir, '%s-%s.out' % (rbp_name, args.save_dir)), "w")
        sys.stdout = outfile
        sys.stderr = outfile

        for enc_epoch_to_load in epochs_to_load:

            probe = EMB_Classifier(
                input_size, args.hidden_size, output_size, device=device, loss_type=loss_type).to(device)
            print(probe)
            optimizer = optim.Adam(probe.parameters(), lr=args.lr)

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
                    512, 64, 10, 5, decode_nuc_with_lstm=True, tree_encoder_arch='baseline',
                    use_flow_prior=True, device=device).to(device)
            elif preprocess_type == 'jtvae_branched':
                pretrain_model = JunctionTreeVAE(
                    512, 64, 10, 5, decode_nuc_with_lstm=True, tree_encoder_arch='branched',
                    decoder_version='v1', use_flow_prior=True, device=device).to(device)

            pretrain_model.load_state_dict(
                torch.load(enc_epoch_weight_path, map_location=device)['model_weights'])

            if not os.path.exists(enc_epoch_dir):
                os.makedirs(enc_epoch_dir)

            from importlib import reload

            reload(lib.plot_utils)
            lib.plot_utils.set_output_dir(enc_epoch_dir)
            lib.plot_utils.suppress_stdout()

            all_fields = ['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'train_roc_auc',
                          'valid_roc_auc']

            logger = lib.logger.CSVLogger('run.csv', enc_epoch_dir, all_fields)

            best_valid_loss = np.inf
            best_valid_weight_path = None

            print('converting embeddings')
            train_emb = convert_seq_to_embeddings(train_seq, pretrain_model, mp_pool, preprocess_type=preprocess_type)
            valid_emb = convert_seq_to_embeddings(valid_seq, pretrain_model, mp_pool, preprocess_type=preprocess_type)

            print('Probing', enc_epoch_weight_path)
            last_improved = 0
            last_5_epochs = []
            for epoch in range(1, args.epoch + 1):
                # if last_improved >= 20:
                #     print('Have\'t improved for %d epochs' % (last_improved))
                #     break
                # training loop
                shuffle_idx = np.random.permutation(train_emb.size(0))
                shuffled_train_emb = train_emb[shuffle_idx]
                shuffled_train_targets = np.array(train_targets)[shuffle_idx]

                probe.train()
                for idx in range(0, train_size, args.batch_size):
                    probe.zero_grad()
                    ret_dict = probe(shuffled_train_emb[idx: idx + args.batch_size],
                                     shuffled_train_targets[idx: idx + args.batch_size])
                    loss = ret_dict['loss'] / ret_dict['nb_preds']
                    loss.backward()
                    optimizer.step()

                probe.eval()
                # validation loop
                train_loss, train_acc, train_roc_auc = evaluate(train_emb, train_targets)
                valid_loss, valid_acc, valid_roc_auc = evaluate(valid_emb, valid_targets)

                lib.plot_utils.plot('train_loss', train_loss)
                lib.plot_utils.plot('train_acc', train_acc)
                lib.plot_utils.plot('train_roc_auc', train_roc_auc)

                lib.plot_utils.plot('valid_loss', valid_loss)
                lib.plot_utils.plot('valid_acc', valid_acc)
                lib.plot_utils.plot('valid_roc_auc', valid_roc_auc)

                lib.plot_utils.set_xlabel_for_tick(index=0, label='epoch')
                lib.plot_utils.flush()
                lib.plot_utils.tick(index=0)

                print(
                    'Epoch %d, train_loss: %.2f, train_acc: %2f, train_roc_auc: %2f, '
                    'valid_loss: %.2f, valid_acc: %.2f, valid_roc_auc: %.2f' %
                    (epoch, train_loss, train_acc, train_roc_auc,
                     valid_loss, valid_acc, valid_roc_auc))

                logger.update_with_dict({
                    'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss,
                    'train_acc': train_acc, 'valid_acc': valid_acc,
                    'train_roc_auc': train_roc_auc, 'valid_roc_auc': valid_roc_auc
                })

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if len(last_5_epochs) >= 5:
                        to_remove_epoch = last_5_epochs.pop(0)
                        os.remove(os.path.join(enc_epoch_dir, "model.epoch-" + str(to_remove_epoch)))
                    last_5_epochs.append(epoch)
                    best_valid_weight_path = os.path.join(enc_epoch_dir, "model.epoch-" + str(epoch))
                    torch.save(
                        {'model_weights': probe.state_dict(),
                         'opt_weights': optimizer.state_dict()},
                        best_valid_weight_path)
                    print('Validation loss improved, saving current weights to path:', best_valid_weight_path)
                    last_improved = 0
                else:
                    last_improved += 1

            if best_valid_weight_path is not None:
                print('Loading best weights from: %s' % (best_valid_weight_path))
                probe.load_state_dict(torch.load(best_valid_weight_path)['model_weights'])

            probe.eval()
            test_emb = convert_seq_to_embeddings(test_seq, pretrain_model, mp_pool, preprocess_type=preprocess_type)
            test_loss, test_acc, test_roc_auc = evaluate(test_emb, test_targets)
            print('Test acc:', test_acc)
            print('Test roc auc:', test_roc_auc)

            logger.close()

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
