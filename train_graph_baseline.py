import os
import sys
import math
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import torch.optim.lr_scheduler as lr_scheduler
from multiprocessing import Pool

from baseline_models.GraphLSTMVAE import GraphLSTMVAE, BasicLSTMVAEFolder
import lib.plot_utils, lib.logger
import baseline_models.baseline_metrics
from baseline_models.baseline_metrics import evaluate_posterior, evaluate_prior

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', required=True)
parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depth', type=int, default=2)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.02)
parser.add_argument('--max_beta', type=float, default=1.0)

parser.add_argument('--epoch', type=int, default=10)
# parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=1000)
parser.add_argument('--warmup_epoch', type=int, default=1)
parser.add_argument('--use_flow_prior', type=eval, default=True, choices=[True, False])
parser.add_argument('--limit_data', type=int, default=None)
parser.add_argument('--resume', type=eval, default=False, choices=[True, False])

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    model = GraphLSTMVAE(args.hidden_size, args.latent_size, args.depth,
                         device=device, use_flow_prior=args.use_flow_prior, use_aux_regressor=False).to(device)
    print(model)
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        elif param.dim() >= 2:
            nn.init.xavier_normal_(param)

    baseline_models.baseline_metrics.model = model
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    beta = args.beta
    total_step = 0
    meters = np.zeros(7)

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = '/'.join(args.save_dir.split('/')[:-1] + [cur_time + '-' + args.save_dir.split('/')[-1]])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lib.plot_utils.set_output_dir(save_dir)
    lib.plot_utils.suppress_stdout()
    logger = lib.logger.CSVLogger('run.csv', save_dir,
                                  ['Epoch', 'Beta', 'Validation_Entropy', 'Validation_Neg_Log_Prior', 'Validation_KL',
                                   'Validation_Stop_Symbol', 'Validation_Nuc_Symbol', 'Validation_Struct_Symbol',
                                   'Validation_all_Symbol',
                                   'Validation_recon_acc_with_reg', 'Validation_post_valid_with_reg',
                                   'Validation_post_fe_deviation_with_reg',
                                   'Validation_post_fe_deviation_len_normed_with_reg',
                                   'Validation_recon_acc_no_reg', 'Validation_post_valid_no_reg',
                                   'Validation_post_fe_deviation_no_reg',
                                   'Validation_post_fe_deviation_len_normed_no_reg',
                                   'Prior_valid_with_reg', 'Prior_fe_deviation_with_reg',
                                   'Prior_fe_deviation_len_normed_with_reg',
                                   'Prior_valid_no_reg', 'Prior_fe_deviation_no_reg',
                                   'Prior_fe_deviation_len_normed_no_reg',
                                   'Prior_valid_no_reg_greedy', 'Prior_fe_deviation_no_reg_greedy',
                                   'Prior_fe_deviation_len_normed_no_reg_greedy',
                                   'Prior_uniqueness_no_reg_greedy', 'Validation_mutual_information',
                                   'Validation_NLL_IW_100', 'Validation_active_units'])

    mp_pool = Pool(8)

    if args.resume:
        '''load warm-up results'''
        if args.use_flow_prior:
            if args.limit_data is not None:
                weight_path = '/home/zichao/scratch/JTRNA/graph-baseline-output/20200515-140602-512-128-5-maxpooled-hidden-states-mb-3e-3-sb-5e-4-amsgrad-limit-data-30/model.epoch-17'
                epoch_to_start = 18
                beta = 0.0012
            else:
                weight_path = '/home/zichao/scratch/JTRNA/graph-baseline-output/20200515-140604-512-128-5-maxpooled-hidden-states-mb-3e-3-sb-5e-4-amsgrad/model.epoch-11'
                epoch_to_start = 12
                beta = 0.0030
        else:
            raise ValueError('Cannot resume GraphLSTMVAE without flow prior')

        all_weights = torch.load(weight_path)
        model.load_state_dict(all_weights['model_weights'])
        optimizer.load_state_dict(all_weights['opt_weights'])
        print('Loaded weights:', weight_path)
        print('Loaded beta:', beta)
    else:
        epoch_to_start = 1

    lib.plot_utils.set_first_tick(epoch_to_start)

    for epoch in range(epoch_to_start, args.epoch + 1):

        if epoch > args.warmup_epoch:
            beta = min(args.max_beta, beta + args.step_beta)

        loader = BasicLSTMVAEFolder('data/rna_jt_32-512/train-split', args.batch_size, num_workers=4,
                                    limit_data=args.limit_data)

        model.train()
        for batch in loader:
            original_data, batch_sequence, batch_label, batch_fe, batch_graph_input = batch
            total_step += 1

            model.zero_grad()
            ret_dict = model(batch_sequence, batch_label, batch_fe, batch_graph_input)

            loss = ret_dict['sum_nuc_pred_loss'] / ret_dict['nb_nuc_targets'] + \
                   beta * (ret_dict['entropy_loss'] + ret_dict['prior_loss'])

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            optimizer.step()

            neg_entropy = float(ret_dict['entropy_loss'])
            neg_log_prior = float(ret_dict['prior_loss'])
            kl_div = neg_entropy + neg_log_prior
            stop_symbol_acc, nuc_pred_acc, struct_pred_acc, all_acc = \
                ret_dict['nb_stop_symbol_correct'] / ret_dict['nb_stop_symbol'], \
                ret_dict['nb_ord_symbol_correct'] / ret_dict['nb_ord_symbol'], \
                ret_dict['nb_struct_symbol_correct'] / ret_dict['nb_ord_symbol'], \
                ret_dict['nb_all_correct'] / ret_dict['nb_nuc_targets'],

            meters = meters + np.array([neg_entropy, neg_log_prior, kl_div, stop_symbol_acc * 100,
                                        nuc_pred_acc * 100, struct_pred_acc * 100, all_acc * 100])

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print(
                    "[%d] Entropy: %.2f, Neg_log_prior: %.2f, KL: %.2f, Stop symbol: %.2f, Nucleotide symbol: %.2f, Structural symbol: %.2f, All symbol: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                        total_step, -meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], meters[6],
                        param_norm(model), grad_norm(model)))
                lib.plot_utils.plot('Train_Entropy', -meters[0], index=0)
                lib.plot_utils.plot('Train_Neg_Log_Prior', meters[1], index=0)
                lib.plot_utils.plot('Train_KL', meters[2], index=0)
                lib.plot_utils.plot('Train_Stop_Symbol', meters[3], index=0)
                lib.plot_utils.plot('Train_Nucleotide_Symbol', meters[4], index=0)
                lib.plot_utils.plot('Train_Structural_Symbol', meters[5], index=0)
                lib.plot_utils.plot('Train_All_Symbol', meters[6], index=0)
                lib.plot_utils.flush()
                sys.stdout.flush()
                meters *= 0

            lib.plot_utils.tick(index=0)
            del loss, kl_div, all_acc

        # scheduler.step(epoch)
        # print("learning rate: %.6f" % scheduler.get_lr()[0])

        # save model at the end of each epoch
        torch.save(
            {'model_weights': model.state_dict(),
             'opt_weights': optimizer.state_dict()},
            os.path.join(save_dir, "model.epoch-" + str(epoch)))

        ''' validation step '''
        print('End of epoch %d,' % (epoch), 'starting validation')

        valid_batch_size = 128
        loader = BasicLSTMVAEFolder('data/rna_jt_32-512/validation-split', valid_batch_size, num_workers=4)
        nb_iters = 20000 // valid_batch_size  # 20000 is the size of the validation set
        post_max_iters = min(10, nb_iters)  # for efficiency
        total = 0
        # bar = trange(nb_iters, desc='', leave=True)
        # loader = loader.__iter__()
        nb_encode, nb_decode = 4, 4

        recon_acc, post_valid, post_fe_deviation, post_fe_deviation_len_normed = 0, 0, 0., 0.
        recon_acc_noreg, post_valid_noreg, post_fe_deviation_noreg, post_fe_deviation_noreg_len_normed = 0, 0, 0., 0.
        valid_kl, valid_stop_symbol, \
        valid_nuc_symbol, valid_struct_symbol, valid_all_symbol = 0., 0., 0., 0., 0.
        valid_entropy, valid_neg_log_prior = 0., 0.
        all_means = []
        total_mi = 0.
        nll_iw = 0.

        model.eval()
        with torch.no_grad():
            # for i in bar:
            for i, (original_data, batch_sequence, batch_label, batch_fe, batch_graph_input) in enumerate(loader):
                latent_vec = model.encode(batch_graph_input)

                if i < post_max_iters:
                    ret = evaluate_posterior(list(np.array(original_data)[:, 0]), list(np.array(original_data)[:, 1]),
                                             latent_vec, mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                             enforce_rna_prior=True)

                    total += nb_encode * nb_decode * valid_batch_size
                    recon_acc += np.sum(ret['recon_acc'])
                    post_valid += np.sum(ret['posterior_valid'])
                    post_fe_deviation += np.sum(ret['posterior_fe_deviation'])
                    post_fe_deviation_len_normed += np.sum(ret['posterior_fe_deviation_len_normed'])

                    ret = evaluate_posterior(list(np.array(original_data)[:, 0]), list(np.array(original_data)[:, 1]),
                                             latent_vec, mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                             enforce_rna_prior=False)

                    recon_acc_noreg += np.sum(ret['recon_acc'])
                    post_valid_noreg += np.sum(ret['posterior_valid'])
                    post_fe_deviation_noreg += np.sum(ret['posterior_fe_deviation'])
                    post_fe_deviation_noreg_len_normed += np.sum(ret['posterior_fe_deviation_len_normed'])

                #     bar.set_description(
                #         'streaming recon acc: %.2f, streaming post valid: %.2f, streaming post free energy deviation: %.2f'
                #         % (recon_acc / total * 100, post_valid / total * 100, post_fe_deviation / post_valid))
                #
                # bar.refresh()

                total_mi += model.calc_mi(batch_graph_input, latent_vec=latent_vec)
                nll_iw += model.nll_iw(batch_sequence, batch_label, batch_graph_input, 100,
                                       ns=25, latent_vec=latent_vec).sum().item()
                all_means.append(model.mean(latent_vec).cpu().detach().numpy())

                # trite accuracy measures
                latent_vec, (entropy, log_pz) = model.rsample(latent_vec, nsamples=1)
                latent_vec = latent_vec[:, 0, :]
                ret_dict = model.decoder(batch_sequence, latent_vec, batch_label)

                # averaged batch accuracies
                valid_entropy += float(entropy.mean())
                valid_neg_log_prior += -float(log_pz.mean())
                valid_kl += float(- entropy.mean() - log_pz.mean())
                valid_stop_symbol += ret_dict['nb_stop_symbol_correct'] / ret_dict['nb_stop_symbol']
                valid_nuc_symbol += ret_dict['nb_ord_symbol_correct'] / ret_dict['nb_ord_symbol']
                valid_struct_symbol += ret_dict['nb_struct_symbol_correct'] / ret_dict['nb_ord_symbol']
                valid_all_symbol += ret_dict['nb_all_correct'] / ret_dict['nb_nuc_targets']

            lib.plot_utils.plot('Validation_Entropy', valid_entropy / nb_iters, index=1)
            lib.plot_utils.plot('Validation_Neg_Log_Prior', valid_neg_log_prior / nb_iters, index=1)
            lib.plot_utils.plot('Validation_KL', valid_kl / nb_iters, index=1)
            lib.plot_utils.plot('Validation_Stop_Symbol', valid_stop_symbol / nb_iters * 100, index=1)
            lib.plot_utils.plot('Validation_Nuc_Symbol', valid_nuc_symbol / nb_iters * 100, index=1)
            lib.plot_utils.plot('Validation_Struct_Symbol', valid_struct_symbol / nb_iters * 100, index=1)
            lib.plot_utils.plot('Validation_all_Symbol', valid_all_symbol / nb_iters * 100, index=1)

            # posterior decoding with enforced RNA regularity
            lib.plot_utils.plot('Validation_recon_acc_with_reg', recon_acc / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_valid_with_reg', post_valid / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_with_reg',
                                post_fe_deviation / post_valid, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_len_normed_with_reg',
                                post_fe_deviation_noreg_len_normed / post_valid, index=1)

            # posterior decoding without RNA regularity
            lib.plot_utils.plot('Validation_recon_acc_no_reg', recon_acc_noreg / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_valid_no_reg', post_valid_noreg / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_no_reg',
                                post_fe_deviation_noreg / post_valid_noreg, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_len_normed_no_reg',
                                post_fe_deviation_noreg_len_normed / post_valid_noreg, index=1)

            ######################## sampling from the prior ########################
            sampled_latent_prior = torch.as_tensor(np.random.randn(1000, args.latent_size).astype(np.float32)).to(
                device)
            if args.use_flow_prior:
                sampled_latent_prior = model.latent_cnf(sampled_latent_prior, None, reverse=True).view(
                    *sampled_latent_prior.size())

            ######################## evaluate prior with regularity constraints ########################
            ret = evaluate_prior(sampled_latent_prior, 1000, 10, mp_pool, enforce_rna_prior=True)
            lib.plot_utils.plot('Prior_valid_with_reg', np.sum(ret['prior_valid']) / 100,
                                index=1)  # /10000 * 100 = /100
            lib.plot_utils.plot('Prior_fe_deviation_with_reg',
                                np.sum(ret['prior_fe_deviation']) / np.sum(ret['prior_valid']), index=1)
            lib.plot_utils.plot('Prior_fe_deviation_len_normed_with_reg',
                                np.sum(ret['prior_fe_deviation_len_normed']) / np.sum(ret['prior_valid']), index=1)

            ######################## evaluate prior without regularity constraints ########################
            ret = evaluate_prior(sampled_latent_prior, 1000, 10, mp_pool, enforce_rna_prior=False)
            lib.plot_utils.plot('Prior_valid_no_reg', np.sum(ret['prior_valid']) / 100, index=1)  # /10000 * 100 = /100
            lib.plot_utils.plot('Prior_fe_deviation_no_reg',
                                np.sum(ret['prior_fe_deviation']) / np.sum(ret['prior_valid']), index=1)
            lib.plot_utils.plot('Prior_fe_deviation_len_normed_no_reg',
                                np.sum(ret['prior_fe_deviation_len_normed']) / np.sum(ret['prior_valid']), index=1)

            ######################## evaluate prior without regularity constraints and greedy ########################
            ret = evaluate_prior(sampled_latent_prior, 1000, 1, mp_pool, enforce_rna_prior=False, prob_decode=False)
            decoded_seq = ret['decoded_seq'][:1000]
            lib.plot_utils.plot('Prior_valid_no_reg_greedy', np.sum(ret['prior_valid']) / 10,
                                index=1)  # /1000 * 100 = /10
            lib.plot_utils.plot('Prior_fe_deviation_no_reg_greedy',
                                np.sum(ret['prior_fe_deviation']) / np.sum(ret['prior_valid']), index=1)
            lib.plot_utils.plot('Prior_fe_deviation_len_normed_no_reg_greedy',
                                np.sum(ret['prior_fe_deviation_len_normed']) / np.sum(ret['prior_valid']), index=1)
            lib.plot_utils.plot('Prior_uniqueness_no_reg_greedy',
                                len(set(list(
                                    np.array(decoded_seq)[np.where(np.array(ret['prior_valid']) > 0)[0]]))) / np.sum(
                                    ret['prior_valid']) * 100, index=1)

            cur_mi = total_mi / nb_iters
            lib.plot_utils.plot('Validation_mutual_information', cur_mi, index=1)

            cur_nll_iw = nll_iw / nb_iters / valid_batch_size
            lib.plot_utils.plot('Validation_NLL_IW_100', cur_nll_iw, index=1)

            all_means = np.concatenate(all_means, axis=0)
            au_mean = np.mean(all_means, axis=0, keepdims=True)
            au_var = all_means - au_mean
            ns = au_var.shape[0]
            au_var = (au_var ** 2).sum(axis=0) / (ns - 1)
            delta = 0.01
            au = (au_var >= delta).sum().item()
            lib.plot_utils.plot('Validation_active_units', au, index=1)

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
