import os
import sys
import math
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import shutil
import inspect
from multiprocessing import Pool

from jtvae_models.VAE import JunctionTreeVAE
from lib.data_utils import JunctionTreeFolder
import jtvae_models.GraphEncoder
import jtvae_models.TreeEncoder
import jtvae_models.ParallelAltDecoder
import lib.plot_utils, lib.logger
import jtvae_models.jtvae_utils
from jtvae_models.jtvae_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', required=True)
parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=40)
parser.add_argument('--depthG', type=int, default=10)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)

parser.add_argument('--epoch', type=int, default=10)
# parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=1000)
parser.add_argument('--tree_encoder_arch', type=str, default='baseline')
parser.add_argument('--warmup_epoch', type=int, default=1)
parser.add_argument('--use_flow_prior', type=eval, default=True, choices=[True, False])
parser.add_argument('--limit_data', type=int, default=None)
parser.add_argument('--resume', type=eval, default=False, choices=[True, False])

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    model = JunctionTreeVAE(args.hidden_size, args.latent_size, args.depthT, args.depthG,
                            decode_nuc_with_lstm=True, device=device, tree_encoder_arch=args.tree_encoder_arch,
                            use_flow_prior=args.use_flow_prior).to(device)
    print(model)
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        elif param.dim() >= 2:
            nn.init.xavier_normal_(param)

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = 0
    beta = args.beta
    meters = np.zeros(8)

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = '/'.join(args.save_dir.split('/')[:-1] + [cur_time + '-' + args.save_dir.split('/')[-1]])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    backup_dir = os.path.join(save_dir, 'backup')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    shutil.copy(__file__, backup_dir)
    shutil.copy(inspect.getfile(JunctionTreeVAE), backup_dir)
    shutil.copy(inspect.getfile(JunctionTreeFolder), backup_dir)
    shutil.copy(inspect.getfile(jtvae_models.jtvae_utils), backup_dir)
    shutil.copy(inspect.getfile(jtvae_models.GraphEncoder), backup_dir)
    shutil.copy(inspect.getfile(jtvae_models.TreeEncoder), backup_dir)
    shutil.copy(inspect.getfile(jtvae_models.ParallelAltDecoder), backup_dir)

    lib.plot_utils.set_output_dir(save_dir)
    lib.plot_utils.suppress_stdout()
    logger = lib.logger.CSVLogger(
        'run.csv', save_dir,
        ['Epoch', 'Beta', 'Validation_Entropy', 'Validation_Neg_Log_Prior', 'Validation_KL',
         'Validation_Node_Acc', 'Validation_Nuc_Stop_Acc', 'Validation_Nuc_Ord_Acc',
         'Validation_Nuc_Acc', 'Validation_Topo_Acc', 'Validation_recon_acc_with_reg',
         'Validation_post_valid_with_reg', 'Validation_post_fe_deviation_with_reg',
         'Validation_recon_acc_no_reg', 'Validation_post_valid_no_reg',
         'Validation_post_fe_deviation_no_reg', 'Prior_valid_with_reg',
         'Prior_fe_deviation_with_reg', 'Prior_valid_no_reg', 'Prior_fe_deviation_no_reg',
         'Prior_valid_no_reg_greedy', 'Prior_fe_deviation_no_reg_greedy',
         'Prior_uniqueness_no_reg_greedy', 'Validation_mutual_information', 'Validation_NLL_IW_100',
         'Validation_active_units'])

    mp_pool = Pool(8)
    jtvae_models.jtvae_utils.model = model

    if args.resume:
        '''load warm-up results'''
        weight_path = '/home/zichao/scratch/JTRNA/output/20200507-182856-512-64-5-10-maxpooled-hidden-states/model.epoch-1'
        all_weights = torch.load(weight_path)
        model.load_state_dict(all_weights['model_weights'])
        optimizer.load_state_dict(all_weights['opt_weights'])
        print('Weights loaded from', weight_path)
        epoch_to_start = 2
    else:
        epoch_to_start = 1

    for epoch in range(epoch_to_start, args.epoch + 1):
        loader = JunctionTreeFolder('data/rna_jt_32-512/train-split', args.batch_size,
                                    num_workers=8, tree_encoder_arch=args.tree_encoder_arch,
                                    limit_data=args.limit_data)
        for batch in loader:
            total_step += 1
            model.zero_grad()
            ret_dict = model(batch)
            loss = ret_dict['sum_hpn_pred_loss'] / ret_dict['nb_hpn_targets'] + \
                   ret_dict['sum_nuc_pred_loss'] / ret_dict['nb_nuc_targets'] + \
                   ret_dict['sum_stop_pred_loss'] / ret_dict['nb_stop_targets'] + \
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
                        meters[7],
                        param_norm(model), grad_norm(model)))
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

        # scheduler.step(epoch)
        # print("learning rate: %.6f" % scheduler.get_lr()[0])

        if epoch >= args.warmup_epoch:
            beta = min(args.max_beta, beta + args.step_beta)

        # save model at the end of each epoch
        torch.save(
            {'model_weights': model.state_dict(),
             'opt_weights': optimizer.state_dict()},
            os.path.join(save_dir, "model.epoch-" + str(epoch)))

        # validation step
        print('End of epoch %d,' % (epoch), 'starting validation')
        valid_batch_size = 128
        loader = JunctionTreeFolder('data/rna_jt_32-512/validation-split', valid_batch_size,
                                    num_workers=8, tree_encoder_arch=args.tree_encoder_arch)

        # turns out there is a very large graph in the validation set, therefore we have to use a smaller batch size
        nb_iters = 20000 // valid_batch_size  # 20000 is the size of the validation set
        post_max_iters = min(10, nb_iters)  # for efficiency
        total = 0
        # from tqdm import trange
        # bar = trange(nb_iters, desc='', leave=True)
        # loader = loader.__iter__()
        nb_encode, nb_decode = 4, 4

        recon_acc, post_valid, post_fe_deviation = 0, 0, 0.
        recon_acc_noreg, post_valid_noreg, post_fe_deviation_noreg = 0, 0, 0.
        valid_kl, valid_node_acc, valid_nuc_stop_acc, valid_nuc_ord_acc, \
        valid_nuc_acc, valid_topo_acc = 0., 0., 0., 0., 0., 0.

        valid_entropy, valid_neg_log_prior = 0., 0.
        all_means = []
        total_mi = 0.
        nll_iw = 0.

        with torch.no_grad():

            for i, batch in enumerate(loader):
            # for i in bar:

                tree_batch, graph_encoder_input, tree_encoder_input = batch
                # tree_batch, graph_encoder_input, tree_encoder_input = next(loader)
                graph_vectors, tree_vectors = model.encode(graph_encoder_input, tree_encoder_input)

                if i < post_max_iters:
                    all_seq = [''.join(tree.rna_seq) for tree in tree_batch]
                    all_struct = [''.join(tree.rna_struct) for tree in tree_batch]
                    batch_recon_acc, batch_post_valid, batch_post_fe_deviation = \
                        evaluate_posterior(all_seq, all_struct, graph_vectors, tree_vectors,
                                           mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                           enforce_rna_prior=True)

                    total += nb_encode * nb_decode * valid_batch_size
                    recon_acc += np.sum(batch_recon_acc)
                    post_valid += np.sum(batch_post_valid)
                    post_fe_deviation += np.sum(batch_post_fe_deviation)

                    batch_recon_acc, batch_post_valid, batch_post_fe_deviation = \
                        evaluate_posterior(all_seq, all_struct, graph_vectors, tree_vectors,
                                           mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                           enforce_rna_prior=False)

                    recon_acc_noreg += np.sum(batch_recon_acc)
                    post_valid_noreg += np.sum(batch_post_valid)
                    post_fe_deviation_noreg += np.sum(batch_post_fe_deviation)

                #     bar.set_description(
                #         'streaming recon acc: %.2f, streaming post valid: %.2f, streaming post free energy deviation: %.2f'
                #         % (recon_acc / total * 100, post_valid / total * 100, post_fe_deviation / post_valid))
                #
                # bar.refresh()

                total_mi += model.calc_mi((graph_encoder_input, tree_encoder_input),
                                          graph_latent_vec=graph_vectors, tree_latent_vec=tree_vectors)
                all_mean = torch.cat([model.g_mean(graph_vectors), model.t_mean(tree_vectors)], dim=-1)
                all_means.append(all_mean.cpu().detach().numpy())

                # trite accuracy measures
                (z_vecs, graph_z_vecs, tree_z_vecs), (entropy, log_pz) = model.rsample(graph_vectors, tree_vectors)
                graph_z_vecs, tree_z_vecs = graph_z_vecs[:, 0, :], tree_z_vecs[:, 0, :]

                ret_dict = model.decoder(tree_batch, tree_z_vecs, graph_z_vecs)

                valid_entropy += float(entropy.mean())
                valid_neg_log_prior += -float(log_pz.mean())
                valid_kl += float(- entropy.mean() - log_pz.mean())

                valid_node_acc += ret_dict['nb_hpn_pred_correct'] / ret_dict['nb_hpn_targets']
                valid_nuc_stop_acc += ret_dict['nb_stop_trans_pred_correct'] / ret_dict['nb_stop_trans_targets']
                valid_nuc_ord_acc += ret_dict['nb_ord_nuc_pred_correct'] / ret_dict['nb_ord_nuc_targets']
                valid_nuc_acc += ret_dict['nb_nuc_pred_correct'] / ret_dict['nb_nuc_targets']
                valid_topo_acc += ret_dict['nb_stop_pred_correct'] / ret_dict['nb_stop_targets']

            lib.plot_utils.plot('Validation_Entropy', valid_entropy / nb_iters, index=1)
            lib.plot_utils.plot('Validation_Neg_Log_Prior', valid_neg_log_prior / nb_iters, index=1)
            lib.plot_utils.plot('Validation_KL', valid_kl / nb_iters, index=1)
            lib.plot_utils.plot('Validation_Node_Acc', valid_node_acc / nb_iters * 100, index=1)
            lib.plot_utils.plot('Validation_Nuc_Stop_Acc', valid_nuc_stop_acc / nb_iters * 100, index=1)
            lib.plot_utils.plot('Validation_Nuc_Ord_Acc', valid_nuc_ord_acc / nb_iters * 100, index=1)
            lib.plot_utils.plot('Validation_Nuc_Acc', valid_nuc_acc / nb_iters * 100, index=1)
            lib.plot_utils.plot('Validation_Topo_Acc', valid_topo_acc / nb_iters * 100, index=1)

            # posterior decoding with enforced RNA regularity
            lib.plot_utils.plot('Validation_recon_acc_with_reg', recon_acc / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_valid_with_reg', post_valid / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_with_reg', post_fe_deviation / post_valid, index=1)

            # posterior decoding without RNA regularity
            lib.plot_utils.plot('Validation_recon_acc_no_reg', recon_acc_noreg / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_valid_no_reg', post_valid_noreg / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_no_reg', post_fe_deviation_noreg / post_valid_noreg, index=1)

            ######################## sampling from the prior ########################
            sampled_g_z = torch.as_tensor(np.random.randn(1000, args.latent_size).
                                          astype(np.float32)).to(device)
            sampled_t_z = torch.as_tensor(np.random.randn(1000, args.latent_size).
                                          astype(np.float32)).to(device)
            sampled_z = torch.cat([sampled_g_z, sampled_t_z], dim=-1)
            if args.use_flow_prior:
                sampled_z = model.latent_cnf(sampled_z, None, reverse=True).view(
                    *sampled_z.size())
            sampled_g_z = sampled_z[:, :args.latent_size]
            sampled_t_z = sampled_z[:, args.latent_size:]

            ######################## evaluate prior with regularity constraints ########################
            prior_valid, prior_fe_deviation, _ = evaluate_prior(sampled_g_z, sampled_t_z, 1000, 1, mp_pool,
                                                                enforce_rna_prior=True)
            lib.plot_utils.plot('Prior_valid_with_reg', np.sum(prior_valid) / 10, index=1)  # /10000 * 100
            lib.plot_utils.plot('Prior_fe_deviation_with_reg', np.sum(prior_fe_deviation) / np.sum(prior_valid), index=1)

            ######################## evaluate prior without regularity constraints ########################
            prior_valid, prior_fe_deviation, _ = evaluate_prior(sampled_g_z, sampled_t_z, 1000, 1, mp_pool,
                                                                enforce_rna_prior=False)
            lib.plot_utils.plot('Prior_valid_no_reg', np.sum(prior_valid) / 10, index=1)  # /10000 * 100
            lib.plot_utils.plot('Prior_fe_deviation_no_reg', np.sum(prior_fe_deviation) / np.sum(prior_valid), index=1)

            ######################## evaluate prior without regularity constraints and greedy ########################
            prior_valid, prior_fe_deviation, parsed_trees = evaluate_prior(sampled_g_z, sampled_t_z, 1000, 1, mp_pool,
                                                                           enforce_rna_prior=False, prob_decode=False)
            decoded_seq = [''.join(tree.rna_seq) for tree in parsed_trees[:1000] if
                           type(tree) is RNAJunctionTree and tree.is_valid]
            lib.plot_utils.plot('Prior_valid_no_reg_greedy', np.sum(prior_valid) / 10, index=1)  # /10000 * 100
            lib.plot_utils.plot('Prior_fe_deviation_no_reg_greedy', np.sum(prior_fe_deviation) / np.sum(prior_valid),
                                index=1)
            if len(decoded_seq) == 0:
                lib.plot_utils.plot('Prior_uniqueness_no_reg_greedy', 0.,
                                    index=1)
            else:
                lib.plot_utils.plot('Prior_uniqueness_no_reg_greedy', len(set(decoded_seq)) / len(decoded_seq) * 100, index=1)

            ######################## mutual information ########################
            cur_mi = total_mi / nb_iters
            lib.plot_utils.plot('Validation_mutual_information', cur_mi, index=1)

            ######################## active units ########################
            all_means = np.concatenate(all_means, axis=0)
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

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()

    logger.close()
