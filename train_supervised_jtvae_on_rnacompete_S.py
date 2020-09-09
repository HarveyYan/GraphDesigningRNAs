import os
import sys
import torch
import argparse
import torch.optim as optim
import numpy as np
import shutil
import inspect
import datetime
from multiprocessing import Pool
import torch.nn as nn
import math
from sklearn.metrics import roc_auc_score

import jtvae_models.GraphEncoder
import jtvae_models.TreeEncoder
import jtvae_models.BranchedTreeEncoder
import jtvae_models.ParallelAltDecoder
import jtvae_models.ParallelAltDecoderV1
import jtvae_models.jtvae_utils
from jtvae_models.jtvae_utils import evaluate_prior, evaluate_posterior

from supervised_encoder_models.task_dataloader import TaskFolder, rnacompete_s_all_rbps, \
    read_curated_rnacompete_s_dataset
from supervised_encoder_models.supervised_vae_model import SUPERVISED_VAE_Model
import lib.plot_utils, lib.logger
from lib.tree_decomp import RNAJunctionTree

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='mlp')
parser.add_argument('--rbp_name', type=str, default='PTB')
parser.add_argument('--hidden_size', type=eval, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--mode', type=str, default='lstm')

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--warmup_epoch', type=int, default=1)
parser.add_argument('--print_iter', type=int, default=1000)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--resume', type=eval, default=False, choices=[True, False])


def evaluate(loader):
    all_loss = 0.
    size = loader.size
    all_preds = []
    all_label = []
    with torch.no_grad():
        for batch_input, batch_label in loader:
            # compute various metrics
            ret_dict = model(batch_input, batch_label)
            all_loss += ret_dict['supervised_loss'].item()

            all_preds.append(ret_dict['supervised_preds'])
            all_label.extend(batch_label)
    all_loss /= size
    acc = sum(np.array(all_label)[:, 0] == (np.concatenate(all_preds, axis=0)[:, 0] > 0.5).astype(np.int32)) / size
    roc_auc = roc_auc_score(np.array(all_label)[:, 0], np.concatenate(all_preds, axis=0)[:, 0])

    return all_loss, acc, roc_auc


def evaluate_posterior_decoding(loader):
    batch_size = loader.batch_size
    nb_iters = loader.size // batch_size
    total = 0
    nb_encode, nb_decode = 5, 5

    recon_acc, post_valid, post_fe_deviation, post_fe_deviation_len_normed = 0, 0, 0., 0.
    recon_acc_noreg, post_valid_noreg, post_fe_deviation_noreg, post_fe_deviation_noreg_len_normed = 0, 0, 0., 0.
    recon_acc_noreg_det, post_valid_noreg_det, post_fe_deviation_noreg_det, post_fe_deviation_noreg_det_len_normed = 0, 0, 0., 0.
    valid_kl, valid_node_acc, valid_nuc_stop_acc, valid_nuc_ord_acc, \
    valid_nuc_acc, valid_topo_acc = 0., 0., 0., 0., 0., 0.

    valid_entropy, valid_neg_log_prior = 0., 0.
    all_means = []
    total_mi = 0.

    with torch.no_grad():
        for batch_input, batch_label in loader:
            tree_batch, graph_encoder_input, tree_encoder_input = batch_input
            graph_vectors, tree_vectors = model.vae.encode(graph_encoder_input, tree_encoder_input)

            all_seq = [''.join(tree.rna_seq) for tree in tree_batch]
            all_struct = [''.join(tree.rna_struct) for tree in tree_batch]

            ######################## evaluate posterior with regularity constraints ########################
            ret = evaluate_posterior(all_seq, all_struct, graph_vectors, tree_vectors,
                                     mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                     enforce_rna_prior=True)
            total += nb_encode * nb_decode * batch_size
            recon_acc += np.sum(ret['recon_acc'])
            post_valid += np.sum(ret['posterior_valid'])
            post_fe_deviation += np.sum(ret['posterior_fe_deviation'])
            post_fe_deviation_len_normed += np.sum(ret['posterior_fe_deviation_len_normed'])

            ######################## evaluate posterior without regularity constraints ########################
            ret = evaluate_posterior(all_seq, all_struct, graph_vectors, tree_vectors,
                                     mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                     enforce_rna_prior=False)
            recon_acc_noreg += np.sum(ret['recon_acc'])
            post_valid_noreg += np.sum(ret['posterior_valid'])
            post_fe_deviation_noreg += np.sum(ret['posterior_fe_deviation'])
            post_fe_deviation_noreg_len_normed += np.sum(ret['posterior_fe_deviation_len_normed'])

            ######################## evaluate posterior without regularity constraints and greedy ########################
            ret = evaluate_posterior(all_seq, all_struct, graph_vectors, tree_vectors,
                                     mp_pool, nb_encode=nb_encode, nb_decode=1,
                                     enforce_rna_prior=False, prob_decode=False)
            recon_acc_noreg_det += np.sum(ret['recon_acc'])
            post_valid_noreg_det += np.sum(ret['posterior_valid'])
            post_fe_deviation_noreg_det += np.sum(ret['posterior_fe_deviation'])
            post_fe_deviation_noreg_det_len_normed += np.sum(ret['posterior_fe_deviation_len_normed'])

            total_mi += model.vae.calc_mi((graph_encoder_input, tree_encoder_input),
                                          graph_latent_vec=graph_vectors, tree_latent_vec=tree_vectors)
            all_mean = torch.cat([model.vae.g_mean(graph_vectors), model.vae.t_mean(tree_vectors)], dim=-1)
            all_means.append(all_mean.cpu().detach().numpy())

            # trite accuracy measures
            (z_vecs, graph_z_vecs, tree_z_vecs), (entropy, log_pz) = model.vae.rsample(graph_vectors, tree_vectors)
            graph_z_vecs, tree_z_vecs = graph_z_vecs[:, 0, :], tree_z_vecs[:, 0, :]

            ret_dict = model.vae.decoder(tree_batch, tree_z_vecs, graph_z_vecs)

            valid_entropy += float(entropy.mean())
            valid_neg_log_prior += -float(log_pz.mean())
            valid_kl += float(- entropy.mean() - log_pz.mean())

            valid_node_acc += ret_dict['nb_hpn_pred_correct'] / ret_dict['nb_hpn_targets']
            valid_nuc_stop_acc += ret_dict['nb_stop_trans_pred_correct'] / ret_dict['nb_stop_trans_targets']
            valid_nuc_ord_acc += ret_dict['nb_ord_nuc_pred_correct'] / ret_dict['nb_ord_nuc_targets']
            valid_nuc_acc += ret_dict['nb_nuc_pred_correct'] / ret_dict['nb_nuc_targets']
            valid_topo_acc += ret_dict['nb_stop_pred_correct'] / ret_dict['nb_stop_targets']

    return {
        'nb_iters': nb_iters,
        'valid_entropy': valid_entropy,
        'valid_neg_log_prior': valid_neg_log_prior,
        'valid_kl': valid_kl,
        'valid_node_acc': valid_node_acc,
        'valid_nuc_stop_acc': valid_nuc_stop_acc,
        'valid_nuc_ord_acc': valid_nuc_ord_acc,
        'valid_nuc_acc': valid_nuc_acc,
        'valid_topo_acc': valid_topo_acc,
        'total': total,
        'recon_acc': recon_acc,
        'post_valid': post_valid,
        'post_fe_deviation': post_fe_deviation,
        'post_fe_deviation_len_normed': post_fe_deviation_len_normed,
        'recon_acc_noreg': recon_acc_noreg,
        'post_valid_noreg': post_valid_noreg,
        'post_fe_deviation_noreg': post_fe_deviation_noreg,
        'post_fe_deviation_noreg_len_normed': post_fe_deviation_noreg_len_normed,
        'recon_acc_noreg_det': recon_acc_noreg_det,
        'post_valid_noreg_det': post_valid_noreg_det,
        'post_fe_deviation_noreg_det': post_fe_deviation_noreg_det,
        'post_fe_deviation_noreg_det_len_normed': post_fe_deviation_noreg_det_len_normed,
        'total_mi': total_mi,
        'all_means': all_means,
    }


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)
    rbp_name = args.rbp_name

    preprocess_type = args.mode
    input_size = 128  # latent dimension
    output_size = 1
    train_val_split_ratio = 0.1
    loss_type = 'binary_ce'

    ############### creating models
    model = SUPERVISED_VAE_Model(input_size, args.hidden_size, output_size, device=device,
                                 vae_type=preprocess_type, loss_type=loss_type).to(device)
    print(model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        elif param.dim() >= 2:
            nn.init.xavier_normal_(param)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    valid_idx = None
    if args.resume:
        weight_path = None
        expr_dir = 'output/supervised-corrected-treeenc-jtvae-rnacompeteS'
        for dirname in os.listdir(expr_dir):
            if dirname.split('-')[2] == rbp_name:
                valid_idx = np.load(os.path.join(expr_dir, dirname, 'valid_idx.npy'), allow_pickle=True)
                weight_path = os.path.join(expr_dir, dirname, 'model.epoch-13')
                epoch_to_start = 14
                beta = 3e-3
                if not os.path.exists(weight_path):
                    weight_path = os.path.join(expr_dir, dirname, 'model.epoch-12')
                    epoch_to_start = 13
                break

        if weight_path is None:
            raise ValueError('checkpoints not found')
        else:
            all_weights = torch.load(weight_path)
            model.load_state_dict(all_weights['model_weights'])
            optimizer.load_state_dict(all_weights['opt_weights'])
            print('Loaded weights:', weight_path)
            print('Loaded beta:', beta)

    else:
        epoch_to_start = 1

    ############### creating expr dir and backing up files
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = '/'.join(args.save_dir.split('/')[:-1] + [cur_time + '-' + args.save_dir.split('/')[-1]])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    outfile = open(os.path.join(save_dir, 'run.out'), "w")
    sys.stdout = outfile
    sys.stderr = outfile

    backup_dir = os.path.join(save_dir, 'backup')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    shutil.copy(__file__, backup_dir)
    shutil.copy(inspect.getfile(SUPERVISED_VAE_Model), backup_dir)
    shutil.copy(inspect.getfile(TaskFolder), backup_dir)
    shutil.copy(inspect.getfile(jtvae_models.jtvae_utils), backup_dir)
    shutil.copy(inspect.getfile(jtvae_models.GraphEncoder), backup_dir)
    if preprocess_type == 'jtvae':
        shutil.copy(inspect.getfile(jtvae_models.TreeEncoder), backup_dir)
        shutil.copy(inspect.getfile(jtvae_models.ParallelAltDecoder), backup_dir)
    else:
        shutil.copy(inspect.getfile(jtvae_models.BranchedTreeEncoder), backup_dir)
        shutil.copy(inspect.getfile(jtvae_models.ParallelAltDecoderV1), backup_dir)

    logger = lib.logger.CSVLogger(
        'run.csv', save_dir,
        ['Epoch', 'Beta', 'Validation_Entropy', 'Validation_Neg_Log_Prior', 'Validation_KL',
         'Validation_Node_Acc', 'Validation_Nuc_Stop_Acc', 'Validation_Nuc_Ord_Acc',
         'Validation_Nuc_Acc', 'Validation_Topo_Acc', 'Validation_recon_acc_with_reg',
         'Validation_post_valid_with_reg', 'Validation_post_fe_deviation_with_reg',
         'Validation_post_fe_deviation_len_normed_with_reg',
         'Validation_recon_acc_no_reg', 'Validation_post_valid_no_reg',
         'Validation_post_fe_deviation_no_reg',
         'Validation_post_fe_deviation_len_normed_no_reg',
         'Validation_recon_acc_no_reg_det', 'Validation_post_valid_no_reg_det',
         'Validation_post_fe_deviation_no_reg_det',
         'Validation_post_fe_deviation_len_normed_no_reg_det', 'Prior_valid_with_reg',
         'Prior_fe_deviation_with_reg', 'Prior_fe_deviation_len_normed_with_reg',
         'Prior_valid_no_reg', 'Prior_fe_deviation_no_reg',
         'Prior_fe_deviation_len_normed_no_reg',
         'Prior_valid_no_reg_greedy', 'Prior_fe_deviation_no_reg_greedy',
         'Prior_fe_deviation_len_normed_no_reg_greedy',
         'Prior_uniqueness_no_reg_greedy', 'Validation_mutual_information', 'Validation_NLL_IW_100',
         'Validation_active_units', 'Train_supervised_loss', 'Train_acc', 'Train_roc_auc',
         'Validation_supervised_loss', 'Validation_acc', 'Validation_roc_auc'])

    lib.plot_utils.set_output_dir(save_dir)
    lib.plot_utils.suppress_stdout()
    lib.plot_utils.set_first_tick(epoch_to_start)

    ############### dataset loading
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

    if valid_idx is None:
        print('creating new valid idx')
        valid_idx = np.random.choice(np.arange(len(train_targets)), int(len(train_targets) * train_val_split_ratio),
                                     replace=False)
        np.save(os.path.join(save_dir, 'valid_idx'), valid_idx)

    valid_idx = np.array(valid_idx)
    train_idx = np.setdiff1d(np.arange(len(train_targets)), valid_idx)

    valid_seq = np.array(train_seq)[valid_idx]
    valid_targets = np.array(train_targets)[valid_idx]
    val_size = len(valid_seq)

    train_seq = np.array(train_seq)[train_idx]
    train_targets = np.array(train_targets)[train_idx]
    train_size = len(train_seq)

    ############### prepare expr misc
    mp_pool = Pool(8)
    jtvae_models.jtvae_utils.model = model.vae
    beta = args.beta
    total_step = 0
    meters = np.zeros(8)

    train_loader = TaskFolder(train_seq, train_targets, args.batch_size, shuffle=True,
                              preprocess_type=preprocess_type, num_workers=8)
    valid_loader = TaskFolder(valid_seq, valid_targets, args.batch_size, shuffle=False,
                              preprocess_type=preprocess_type, num_workers=8)
    test_loader = TaskFolder(test_seq, test_targets, args.batch_size, shuffle=False,
                             preprocess_type=preprocess_type, num_workers=8)

    # best_valid_loss = np.inf
    best_valid_weight_path = None
    last_improved = 0
    # best_valid_loss = np.inf
    last_5_epochs = []
    for epoch in range(epoch_to_start, args.epoch + 1):
        if last_improved >= 20:
            print('Have\'t improved for %d epochs' % (last_improved))
            break

        if epoch > args.warmup_epoch:
            beta = min(args.max_beta, beta + args.step_beta)

        # training loop
        model.train()
        for batch_input, batch_label in train_loader:
            total_step += 1
            model.zero_grad()
            ret_dict = model(batch_input, batch_label)
            loss = ret_dict['sum_hpn_pred_loss'] / ret_dict['nb_hpn_targets'] + \
                   ret_dict['sum_nuc_pred_loss'] / ret_dict['nb_nuc_targets'] + \
                   ret_dict['sum_stop_pred_loss'] / ret_dict['nb_stop_targets'] + \
                   ret_dict['supervised_loss'] / ret_dict['nb_supervised_preds'] + \
                   beta * (ret_dict['entropy_loss'] + ret_dict['prior_loss'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            neg_entropy = float(ret_dict['entropy_loss'])
            neg_log_prior = float(ret_dict['prior_loss'])
            kl_div = neg_entropy + neg_log_prior

            hpn_pred_acc, stop_translation_nuc_acc, ord_nuc_acc, all_nuc_pred_acc, stop_acc = \
                ret_dict['nb_hpn_pred_correct'] / ret_dict['nb_hpn_targets'], \
                ret_dict['nb_stop_trans_pred_correct'] / ret_dict['nb_stop_trans_targets'], \
                ret_dict['nb_ord_nuc_pred_correct'] / ret_dict['nb_ord_nuc_targets'], \
                ret_dict['nb_nuc_pred_correct'] / ret_dict['nb_nuc_targets'], \
                ret_dict['nb_stop_pred_correct'] / ret_dict['nb_stop_targets'],

            meters = meters + np.array(
                [neg_entropy, neg_log_prior, kl_div, hpn_pred_acc * 100, stop_translation_nuc_acc * 100,
                 ord_nuc_acc * 100, all_nuc_pred_acc * 100, stop_acc * 100])

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print(
                    "[%d] Beta: %.4f, Entropy: %.2f, Neg_log_prior: %.2f, KL: %.2f, Node: %.2f, Nucleotide stop: %.2f, Nucleotide ord: %.2f, Nucleotide: %.2f, Topo: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                        total_step, beta, -meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], meters[6],
                        meters[7], param_norm(model), grad_norm(model)))
                lib.plot_utils.plot('Train_Entropy', -meters[0], index=0)
                lib.plot_utils.plot('Train_Neg_Log_Prior', meters[1], index=0)
                lib.plot_utils.plot('Train_KL', meters[2], index=0)
                lib.plot_utils.plot('Train_Node_Acc', meters[3], index=0)
                lib.plot_utils.plot('Train_Nucleotide_Stop', meters[4], index=0)
                lib.plot_utils.plot('Train_Nucleotide_Ord', meters[5], index=0)
                lib.plot_utils.plot('Train_Nucleotide_All', meters[6], index=0)
                lib.plot_utils.plot('Train_Topo_Acc', meters[7], index=0)
                lib.plot_utils.flush()
                sys.stdout.flush()
                meters *= 0

            lib.plot_utils.tick(index=0)
            del loss, kl_div

        model.eval()
        # validation loop
        train_loss, train_acc, train_roc_auc = evaluate(train_loader)
        valid_loss, valid_acc, valid_roc_auc = evaluate(valid_loader)

        lib.plot_utils.plot('Train_supervised_loss', train_loss, index=1)
        lib.plot_utils.plot('Train_roc_auc', train_roc_auc, index=1)
        lib.plot_utils.plot('Train_acc', train_acc, index=1)
        lib.plot_utils.plot('Validation_supervised_loss', valid_loss, index=1)
        lib.plot_utils.plot('Validation_roc_auc', valid_roc_auc, index=1)
        lib.plot_utils.plot('Validation_acc', valid_acc, index=1)

        lib.plot_utils.set_xlabel_for_tick(index=0, label='epoch')
        lib.plot_utils.flush()
        lib.plot_utils.tick(index=0)

        print(
            'Epoch %d, train_loss: %.2f, train_acc: %2f, train_roc_auc: %2f, '
            'valid_loss: %.2f, valid_acc: %2f, valid_roc_auc: %.2f' %
            (epoch, train_loss, train_acc, train_roc_auc,
             valid_loss, valid_acc, valid_roc_auc))

        # if valid_loss < best_valid_loss:
        # best_valid_loss = valid_loss
        if len(last_5_epochs) >= 5:
            to_remove_epoch = last_5_epochs.pop(0)
            os.remove(os.path.join(save_dir, "model.epoch-" + str(to_remove_epoch)))
        last_5_epochs.append(epoch)
        # best_valid_weight_path = os.path.join(save_dir, "model.epoch-" + str(epoch))
        torch.save(
            {'model_weights': model.state_dict(),
             'opt_weights': optimizer.state_dict()},
            os.path.join(save_dir, "model.epoch-" + str(epoch)))

        # print('Validation loss improved, saving current weights to path:', best_valid_weight_path)
        #     last_improved = 0
        # else:
        #     last_improved += 1

        # validation step
        print('End of epoch %d,' % (epoch), 'starting validation')
        ret_dict = evaluate_posterior_decoding(valid_loader)

        lib.plot_utils.plot('Validation_Entropy', ret_dict['valid_entropy'] / ret_dict['nb_iters'], index=1)
        lib.plot_utils.plot('Validation_Neg_Log_Prior', ret_dict['valid_neg_log_prior'] / ret_dict['nb_iters'], index=1)
        lib.plot_utils.plot('Validation_KL', ret_dict['valid_kl'] / ret_dict['nb_iters'], index=1)
        lib.plot_utils.plot('Validation_Node_Acc', ret_dict['valid_node_acc'] / ret_dict['nb_iters'] * 100, index=1)
        lib.plot_utils.plot('Validation_Nuc_Stop_Acc',
                            ret_dict['valid_nuc_stop_acc'] / ret_dict['nb_iters'] * 100, index=1)
        lib.plot_utils.plot('Validation_Nuc_Ord_Acc',
                            ret_dict['valid_nuc_ord_acc'] / ret_dict['nb_iters'] * 100, index=1)
        lib.plot_utils.plot('Validation_Nuc_Acc', ret_dict['valid_nuc_acc'] / ret_dict['nb_iters'] * 100, index=1)
        lib.plot_utils.plot('Validation_Topo_Acc', ret_dict['valid_topo_acc'] / ret_dict['nb_iters'] * 100, index=1)

        # posterior decoding with enforced RNA regularity
        lib.plot_utils.plot('Validation_recon_acc_with_reg', ret_dict['recon_acc'] / ret_dict['total'] * 100, index=1)
        lib.plot_utils.plot('Validation_post_valid_with_reg', ret_dict['post_valid'] / ret_dict['total'] * 100, index=1)
        lib.plot_utils.plot('Validation_post_fe_deviation_with_reg',
                            ret_dict['post_fe_deviation'] / ret_dict['post_valid'], index=1)
        lib.plot_utils.plot('Validation_post_fe_deviation_len_normed_with_reg',
                            ret_dict['post_fe_deviation_noreg_len_normed'] / ret_dict['post_valid'], index=1)

        # posterior decoding without RNA regularity
        lib.plot_utils.plot('Validation_recon_acc_no_reg',
                            ret_dict['recon_acc_noreg'] / ret_dict['total'] * 100, index=1)
        lib.plot_utils.plot('Validation_post_valid_no_reg',
                            ret_dict['post_valid_noreg'] / ret_dict['total'] * 100, index=1)
        lib.plot_utils.plot('Validation_post_fe_deviation_no_reg',
                            ret_dict['post_fe_deviation_noreg'] / ret_dict['post_valid_noreg'], index=1)
        lib.plot_utils.plot('Validation_post_fe_deviation_len_normed_no_reg',
                            ret_dict['post_fe_deviation_noreg_len_normed'] / ret_dict['post_valid_noreg'], index=1)

        # posterior decoding without RNA regularity and greedy
        lib.plot_utils.plot('Validation_recon_acc_no_reg_det',
                            ret_dict['recon_acc_noreg_det'] / ret_dict['total'] * 100 * 5, index=1)
        lib.plot_utils.plot('Validation_post_valid_no_reg_det',
                            ret_dict['post_valid_noreg_det'] / ret_dict['total'] * 100 * 5, index=1)
        lib.plot_utils.plot('Validation_post_fe_deviation_no_reg_det',
                            ret_dict['post_fe_deviation_noreg_det'] / ret_dict['post_valid_noreg_det'], index=1)
        lib.plot_utils.plot('Validation_post_fe_deviation_len_normed_no_reg_det',
                            ret_dict['post_fe_deviation_noreg_det_len_normed'] / ret_dict['post_valid_noreg_det'],
                            index=1)

        ######################## sampling from the prior ########################
        sampled_g_z = torch.as_tensor(np.random.randn(1000, 64).
                                      astype(np.float32)).to(device)
        sampled_t_z = torch.as_tensor(np.random.randn(1000, 64).
                                      astype(np.float32)).to(device)
        sampled_z = torch.cat([sampled_g_z, sampled_t_z], dim=-1)

        sampled_z = model.vae.latent_cnf(sampled_z, None, reverse=True).view(
            *sampled_z.size())
        sampled_g_z = sampled_z[:, :64]
        sampled_t_z = sampled_z[:, 64:]

        ######################## evaluate prior with regularity constraints ########################
        ret = evaluate_prior(sampled_g_z, sampled_t_z, 1000, 1, mp_pool, enforce_rna_prior=True)
        lib.plot_utils.plot('Prior_valid_with_reg', np.sum(ret['prior_valid']) / 10,
                            index=1)  # /1000 * 100 = /10
        lib.plot_utils.plot('Prior_fe_deviation_with_reg',
                            np.sum(ret['prior_fe_deviation']) / np.sum(ret['prior_valid']), index=1)
        lib.plot_utils.plot('Prior_fe_deviation_len_normed_with_reg',
                            np.sum(ret['prior_fe_deviation_len_normed']) / np.sum(ret['prior_valid']), index=1)

        ######################## evaluate prior without regularity constraints ########################
        ret = evaluate_prior(sampled_g_z, sampled_t_z, 1000, 1, mp_pool, enforce_rna_prior=False)
        lib.plot_utils.plot('Prior_valid_no_reg', np.sum(ret['prior_valid']) / 10, index=1)  # /1000 * 100 = /10
        lib.plot_utils.plot('Prior_fe_deviation_no_reg',
                            np.sum(ret['prior_fe_deviation']) / np.sum(ret['prior_valid']), index=1)
        lib.plot_utils.plot('Prior_fe_deviation_len_normed_no_reg',
                            np.sum(ret['prior_fe_deviation_len_normed']) / np.sum(ret['prior_valid']), index=1)

        ######################## evaluate prior without regularity constraints and greedy ########################
        ret = evaluate_prior(sampled_g_z, sampled_t_z, 1000, 1, mp_pool,
                             enforce_rna_prior=False, prob_decode=False)
        decoded_seq = [''.join(tree.rna_seq) for tree in ret['all_parsed_trees'][:1000] if
                       type(tree) is RNAJunctionTree and tree.is_valid]
        lib.plot_utils.plot('Prior_valid_no_reg_greedy', np.sum(ret['prior_valid']) / 10,
                            index=1)  # /1000 * 100 = /10
        lib.plot_utils.plot('Prior_fe_deviation_no_reg_greedy',
                            np.sum(ret['prior_fe_deviation']) / np.sum(ret['prior_valid']), index=1)
        lib.plot_utils.plot('Prior_fe_deviation_len_normed_no_reg_greedy',
                            np.sum(ret['prior_fe_deviation_len_normed']) / np.sum(ret['prior_valid']), index=1)
        if len(decoded_seq) == 0:
            lib.plot_utils.plot('Prior_uniqueness_no_reg_greedy', 0.,
                                index=1)
        else:
            lib.plot_utils.plot('Prior_uniqueness_no_reg_greedy',
                                len(set(decoded_seq)) / len(decoded_seq) * 100,
                                index=1)

        ######################## mutual information ########################
        cur_mi = ret_dict['total_mi'] / ret_dict['nb_iters']
        lib.plot_utils.plot('Validation_mutual_information', cur_mi, index=1)

        ######################## active units ########################
        all_means = np.concatenate(ret_dict['all_means'], axis=0)
        au_mean = np.mean(all_means, axis=0, keepdims=True)
        au_var = all_means - au_mean
        ns = au_var.shape[0]
        au_var = (au_var ** 2).sum(axis=0) / (ns - 1)
        delta = 0.01
        au = (au_var >= delta).sum().item()
        lib.plot_utils.plot('Validation_active_units', au, index=1)

        lib.plot_utils.plot('Beta', beta, index=1)

        tocsv = {'Epoch': epoch}
        for name, val in lib.plot_utils._since_last_flush.items():
            if lib.plot_utils._ticker_registry[name] == 1:
                tocsv[name] = list(val.values())[0]
        logger.update_with_dict(tocsv)

        lib.plot_utils.set_xlabel_for_tick(index=1, label='epoch')
        lib.plot_utils.flush()
        lib.plot_utils.tick(index=1)

    if best_valid_weight_path is not None:
        print('Loading best weights from: %s' % (best_valid_weight_path))
        model.load_state_dict(torch.load(best_valid_weight_path)['model_weights'])

    model.eval()
    test_loss, test_acc, test_roc_auc = evaluate(test_loader)
    print('Test acc:', test_acc)
    print('Test roc auc:', test_roc_auc)

    ret_dict = evaluate_posterior_decoding(test_loader)
    # posterior decoding with enforced RNA regularity
    print('Test_recon_acc_with_reg', ret_dict['recon_acc'] / ret_dict['total'] * 100)
    print('Test_post_valid_with_reg', ret_dict['post_valid'] / ret_dict['total'] * 100)
    print('Test_post_fe_deviation_with_reg', ret_dict['post_fe_deviation'] / ret_dict['post_valid'])
    print('Test_post_fe_deviation_len_normed_with_reg',
          ret_dict['post_fe_deviation_noreg_len_normed'] / ret_dict['post_valid'])
    # posterior decoding without RNA regularity
    print('Test_recon_acc_no_reg', ret_dict['recon_acc_noreg'] / ret_dict['total'] * 100)
    print('Test_post_valid_no_reg', ret_dict['post_valid_noreg'] / ret_dict['total'] * 100)
    print('Test_post_fe_deviation_no_reg', ret_dict['post_fe_deviation_noreg'] / ret_dict['post_valid_noreg'])
    print('Test_post_fe_deviation_len_normed_no_reg',
          ret_dict['post_fe_deviation_noreg_len_normed'] / ret_dict['post_valid_noreg'])
    # posterior decoding without RNA regularity and greedy
    print('Test_recon_acc_no_reg_det', ret_dict['recon_acc_noreg_det'] / ret_dict['total'] * 100 * 5)
    print('Test_post_valid_no_reg_det', ret_dict['post_valid_noreg_det'] / ret_dict['total'] * 100 * 5)
    print('Test_post_fe_deviation_no_reg_det',
          ret_dict['post_fe_deviation_noreg_det'] / ret_dict['post_valid_noreg_det'])
    print('Test_post_fe_deviation_len_normed_no_reg_det',
          ret_dict['post_fe_deviation_noreg_det_len_normed'] / ret_dict['post_valid_noreg_det'])

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()

    logger.close()
