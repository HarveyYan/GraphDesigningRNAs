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

from baseline_models.LSTMVAE import LSTMVAE, BasicLSTMVAEFolder
import lib.plot_utils
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
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)

parser.add_argument('--epoch', type=int, default=10)
# parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=1000)


if __name__ == "__main__":

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    model = LSTMVAE(args.hidden_size, args.latent_size, args.depth,
                    device=device, use_attention=True, nb_heads=4).to(device)
    print(model)
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    baseline_models.baseline_metrics.model = model
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = 0
    beta = args.beta
    meters = np.zeros(6)

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = '/'.join(args.save_dir.split('/')[:-1] + [cur_time + '-' + args.save_dir.split('/')[-1]])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lib.plot_utils.set_output_dir(save_dir)
    lib.plot_utils.suppress_stdout()

    mp_pool = Pool(8)

    for epoch in range(args.epoch):

        loader = BasicLSTMVAEFolder('data/rna_jt_32-512/train-split', args.batch_size, num_workers=4)

        # training iterations
        for batch in loader:
            original_data, batch_sequence, batch_label, batch_fe = batch
            total_step += 1
            model.zero_grad()
            loss, kl_div, aux_loss, all_acc = model(batch_sequence, batch_label, batch_fe, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            pearson_corr, stop_symbol_acc, nuc_pred_acc, struct_pred_acc, all_acc = all_acc
            meters = meters + np.array([float(kl_div), pearson_corr * 100, stop_symbol_acc * 100,
                                        nuc_pred_acc * 100, struct_pred_acc * 100, all_acc * 100])
            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print(
                    "[%d] Beta: %.3f, KL: %.2f, Pearson Corr: %.2f, Stop symbol: %.2f, Nucleotide symbol: %.2f, Structural symbol: %.2f, All symbol: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                        total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5],
                        param_norm(model), grad_norm(model)))
                lib.plot_utils.plot('Train_KL', meters[0], index=0)
                lib.plot_utils.plot('Train_Pearson_Corr', meters[1], index=0)
                lib.plot_utils.plot('Train_Stop_Symbol', meters[2], index=0)
                lib.plot_utils.plot('Train_Nucleotide_Symbol', meters[3], index=0)
                lib.plot_utils.plot('Train_Structural_Symbol', meters[4], index=0)
                lib.plot_utils.plot('Train_All_Symbol', meters[5], index=0)
                lib.plot_utils.flush()
                sys.stdout.flush()
                meters *= 0

            lib.plot_utils.tick(index=0)
            del loss, kl_div, all_acc

        # scheduler.step(epoch)
        # print("learning rate: %.6f" % scheduler.get_lr()[0])

        if epoch >= 19:
            # warm up 20 epochs
            beta = min(args.max_beta, beta + args.step_beta)

        # save model at the end of each epoch
        torch.save(
            {'model_weights': model.state_dict(), 'opt_weights': optimizer.state_dict()},
            os.path.join(save_dir, "model.epoch-" + str(epoch + 1)))

        # validation step
        print('End of epoch %d,' % (epoch), 'starting validation')

        valid_batch_size = 128
        loader = BasicLSTMVAEFolder('data/rna_jt_32-512/validation-split', valid_batch_size, num_workers=2)
        nb_iters = 20000 // valid_batch_size  # 20000 is the size of the validation set
        max_iters = min(10, nb_iters)  # for efficiency
        recon_acc, post_valid, post_fe_deviation = 0, 0, 0.
        valid_kl, valid_corr, valid_stop_symbol, \
        valid_nuc_symbol, valid_struct_symbol, valid_all_symbol = 0., 0., 0., 0., 0., 0.
        total = 0
        bar = trange(nb_iters, desc='', leave=True)
        loader = loader.__iter__()
        nb_encode, nb_decode = 4, 4

        with torch.no_grad():

            for i in bar:
                if i >= max_iters:
                    break
                original_data, batch_sequence, batch_label, batch_fe = next(loader)
                latent_vec = model.encode(batch_sequence)

                batch_recon_acc, batch_post_valid, batch_post_fe_deviation = \
                    evaluate_posterior(list(np.array(original_data)[:, 0]), list(np.array(original_data)[:, 1]), latent_vec,
                                       mp_pool, nb_encode=nb_encode, nb_decode=nb_decode, enforce_rna_prior=True)

                total += nb_encode * nb_decode * valid_batch_size
                recon_acc += np.sum(batch_recon_acc)
                post_valid += np.sum(batch_post_valid)
                post_fe_deviation += np.sum(batch_post_fe_deviation)

                bar.set_description(
                    'streaming recon acc: %.2f, streaming post valid: %.2f, streaming post free energy deviation: %.2f'
                    % (recon_acc / total * 100, post_valid / total * 100, post_fe_deviation / post_valid))

                bar.refresh()

                normed_fe_loss, normed_fe_corr = model.aux_regressor(latent_vec, batch_fe)

                # trite accuracy measures
                latent_vec, kl_loss = model.rsample(latent_vec)
                all_loss, stop_symbol_acc, nuc_pred_acc, struct_pred_acc, all_acc = \
                    model.decoder(batch_sequence, latent_vec, batch_label)

                # averaged batch accuracies
                valid_kl += float(kl_loss)
                valid_corr += normed_fe_corr
                valid_stop_symbol += stop_symbol_acc
                valid_nuc_symbol += nuc_pred_acc
                valid_struct_symbol += struct_pred_acc
                valid_all_symbol += all_acc

            lib.plot_utils.plot('Validation_KL', valid_kl / max_iters, index=1)
            lib.plot_utils.plot('Validation_Correlation', valid_corr / max_iters * 100, index=1)
            lib.plot_utils.plot('Validation_Stop_Symbol', valid_stop_symbol / max_iters * 100, index=1)
            lib.plot_utils.plot('Validation_Nuc_Symbol', valid_nuc_symbol / max_iters * 100, index=1)
            lib.plot_utils.plot('Validation_Struct_Symbol', valid_struct_symbol / max_iters * 100, index=1)
            lib.plot_utils.plot('Validation_all_Symbol', valid_all_symbol / max_iters * 100, index=1)

            lib.plot_utils.plot('Validation_recon_acc', recon_acc / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_valid', post_valid / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation', post_fe_deviation / post_valid, index=1)

            sampled_latent_prior = torch.as_tensor(np.random.randn(1000, args.latent_size).astype(np.float32)).to(
                device)
            prior_valid, prior_fe_deviation = evaluate_prior(sampled_latent_prior, 1000, 10, mp_pool, enforce_rna_prior=True)

            lib.plot_utils.plot('Prior_valid', np.sum(prior_valid) / 100, index=1)  # /10000 * 100
            lib.plot_utils.plot('Prior_fe_deviation', np.sum(prior_fe_deviation) / np.sum(prior_valid), index=1)

            lib.plot_utils.set_xlabel_for_tick(index=1, label='epoch')
            lib.plot_utils.flush()
            lib.plot_utils.tick(index=1)

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
