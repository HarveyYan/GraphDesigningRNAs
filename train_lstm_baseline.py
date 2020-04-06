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
import re
import RNA
import forgi.graph.bulge_graph as fgb
import torch.optim.lr_scheduler as lr_scheduler
from multiprocessing import Pool

from baseline_models.LSTMVAE import LSTMVAE, BasicLSTMVAEFolder
import lib.plot_utils

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


def isvalid(dotbracket_struct):
    # todo, check basepairing validity
    # check that hairpin has at least 3 nucleotides
    for match in re.finditer(r'\([.]*\)', dotbracket_struct):
        if match.end() - match.start() < 5:
            return False
    try:
        fgb.BulgeGraph.from_dotbracket(dotbracket_struct)
    except ValueError:
        return False

    return True


def posterior_check_subroutine(args):
    o_seq, o_struct, d_seq, d_struct = args
    ret = [0, 0, 0]  # recon_acc, post_valid, post_fe_dev
    if isvalid(d_struct):
        ret[1] = 1
        if d_seq == o_seq and d_struct == o_struct:
            ret[0] = 1
        else:
            mfe_struct, mfe = RNA.fold(d_seq)
            decoded_free_energy = RNA.eval_structure_simple(d_seq, d_struct)
            ret[2] = np.abs(mfe - decoded_free_energy)
    return ret


def evaluate_posterior(original_sequence, original_structure, latent_vector, mp_pool, nb_encode=10, nb_decode=10,
                       enforce_rna_prior=True):
    batch_size = len(original_sequence)
    recon_acc = [0] * batch_size
    posterior_valid = [0] * batch_size
    posterior_fe_deviation = [0] * batch_size
    batch_idx = list(range(batch_size))

    original_sequence = original_sequence * nb_encode
    original_structure = original_structure * nb_encode
    batch_idx = batch_idx * nb_encode
    to_encode_latent = torch.cat([latent_vector] * nb_encode, dim=0)

    # batch_size x nb_encode
    sampled_latent, _ = model.rsample(to_encode_latent)

    original_sequence = original_sequence * nb_decode
    original_structure = original_structure * nb_decode
    batch_idx = batch_idx * nb_decode
    to_decode_latent = torch.cat([sampled_latent] * nb_decode, dim=0)

    decoded_seq, decoded_struct = model.decoder.decode(to_decode_latent, prob_decode=True,
                                                       enforce_rna_prior=enforce_rna_prior)

    ret = np.array(list(mp_pool.imap(posterior_check_subroutine,
                                     list(zip(original_sequence, original_structure,
                                              decoded_seq, decoded_struct)))))

    for i, r in enumerate(ret):
        recon_acc[batch_idx[i]] += r[0]
        posterior_valid[batch_idx[i]] += r[1]
        posterior_fe_deviation[batch_idx[i]] += r[2]

    recon_acc = np.array(recon_acc)
    posterior_valid = np.array(posterior_valid)
    posterior_fe_deviation = np.array(posterior_fe_deviation)

    return recon_acc, posterior_valid, posterior_fe_deviation


def prior_check_subroutine(args):
    d_seq, d_struct = args
    ret = [0, 0]  # prior_valid, prior_fe_dev
    if isvalid(d_struct):
        ret[0] = 1
        mfe_struct, mfe = RNA.fold(d_seq)
        decoded_free_energy = RNA.eval_structure_simple(d_seq, d_struct)
        ret[1] = np.abs(mfe - decoded_free_energy)
    return ret


def evaluate_prior(nb_samples, nb_decode, latent_size, mp_pool, enforce_rna_prior=True):
    sampled_latent_vector = torch.as_tensor(np.random.randn(nb_samples, latent_size).astype(np.float32)).to(device)
    prior_valid = [0] * nb_samples
    prior_fe_deviation = [0] * nb_samples
    batch_idx = list(range(nb_samples))

    batch_idx = batch_idx * nb_decode
    to_decode_latent = torch.cat([sampled_latent_vector] * nb_decode, dim=0)

    decoded_seq, decoded_struct = model.decoder.decode(to_decode_latent, prob_decode=True,
                                                       enforce_rna_prior=enforce_rna_prior)

    ret = np.array(list(mp_pool.imap(prior_check_subroutine,
                                     list(zip(decoded_seq, decoded_struct)))))

    for i, r in enumerate(ret):
        prior_valid[batch_idx[i]] += r[0]
        prior_fe_deviation[batch_idx[i]] += r[1]

    prior_valid = np.array(prior_valid)
    prior_fe_deviation = np.array(prior_fe_deviation)

    return prior_valid, prior_fe_deviation


if __name__ == "__main__":

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

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

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = 0
    beta = args.beta
    meters = np.zeros(5)

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = '/'.join(args.save_dir.split('/')[:-1] + [cur_time + '-' + args.save_dir.split('/')[-1]])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lib.plot_utils.set_output_dir(save_dir)
    lib.plot_utils.suppress_stdout()

    mp_pool = None

    for epoch in range(args.epoch):

        loader = BasicLSTMVAEFolder('data/rna_jt_32-512/train-split', args.batch_size, num_workers=4)

        # training iterations
        for batch in loader:
            original_data, batch_sequence, batch_label = batch
            total_step += 1
            model.zero_grad()
            loss, kl_div, all_acc = model(batch_sequence, batch_label, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            stop_symbol_acc, nuc_pred_acc, struct_pred_acc, all_acc = all_acc
            meters = meters + np.array([float(kl_div), stop_symbol_acc * 100,
                                        nuc_pred_acc * 100, struct_pred_acc * 100, all_acc * 100])

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print(
                    "[%d] Beta: %.3f, KL: %.2f, Stop symbol: %.2f, Nucleotide symbol: %.2f, Structural symbol: %.2f, All symbol: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                        total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4],
                        param_norm(model), grad_norm(model)))
                lib.plot_utils.plot('kl', meters[0], index=0)
                lib.plot_utils.plot('stop_symbol', meters[1], index=0)
                lib.plot_utils.plot('nucleotide_symbol', meters[2], index=0)
                lib.plot_utils.plot('structural_symbol', meters[3], index=0)
                lib.plot_utils.plot('all_symbol', meters[4], index=0)
                lib.plot_utils.flush()
                sys.stdout.flush()
                meters *= 0

            lib.plot_utils.tick(index=0)
            del loss, kl_div, all_acc

        # scheduler.step(epoch)
        # print("learning rate: %.6f" % scheduler.get_lr()[0])

        if epoch >= 9:
            # warm up 5 epochs
            beta = min(args.max_beta, beta + args.step_beta)

        # save model at the end of each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, "model.epoch-" + str(epoch + 1)))

        # validation step
        print('End of epoch %d,' % (epoch), 'starting validation')

        valid_batch_size = 128
        loader = BasicLSTMVAEFolder('data/rna_jt_32-512/validation-split', valid_batch_size, num_workers=4)
        nb_iters = 20000 // valid_batch_size  # 20000 is the size of the validation set
        recon_acc, post_valid, post_fe_deviation = 0, 0, 0.
        valid_kl, valid_stop_symbol, valid_nuc_symbol, valid_struct_symbol, valid_all_symbol = 0., 0., 0., 0., 0.
        total = 0
        bar = trange(nb_iters, desc='', leave=True)
        loader = loader.__iter__()
        nb_encode, nb_decode = 4, 4

        with torch.no_grad():

            for i in bar:
                original_data, batch_sequence, batch_label = next(loader)
                latent_vec = model.encode(batch_sequence)

                # reconstruction_acc_measure
                # may require a lot of time
                if mp_pool is None:
                    mp_pool = Pool(8)
                batch_recon_acc, batch_post_valid, batch_post_fe_deviation = \
                    evaluate_posterior(list(np.array(original_data)[:, 0]), list(np.array(original_data)[:, 1]), latent_vec,
                                       mp_pool, nb_encode=nb_encode, nb_decode=nb_decode)

                total += nb_encode * nb_decode * valid_batch_size
                recon_acc += np.sum(batch_recon_acc)
                post_valid += np.sum(batch_post_valid)
                post_fe_deviation += np.sum(batch_post_fe_deviation)

                bar.set_description(
                    'streaming recon acc: %.2f, streaming post valid: %.2f, streaming post free energy deviation: %.2f'
                    % (recon_acc / total * 100, post_valid / total * 100, post_fe_deviation / post_valid))

                bar.refresh()

                # trite accuracy measures
                latent_vec, kl_loss = model.rsample(latent_vec)
                all_loss, stop_symbol_acc, nuc_pred_acc, struct_pred_acc, all_acc = \
                    model.decoder(batch_sequence, latent_vec, batch_label)

                valid_kl += float(kl_loss)
                valid_stop_symbol += stop_symbol_acc
                valid_nuc_symbol += nuc_pred_acc
                valid_struct_symbol += struct_pred_acc
                valid_all_symbol += all_acc

            lib.plot_utils.plot('validation_kl', valid_kl / nb_iters, index=1)
            lib.plot_utils.plot('validation_stop_symbol_acc', valid_stop_symbol / nb_iters * 100, index=1)
            lib.plot_utils.plot('validation_nuc_symbol_acc', valid_nuc_symbol / nb_iters * 100, index=1)
            lib.plot_utils.plot('validation_struct_symbol_acc', valid_struct_symbol / nb_iters * 100, index=1)
            lib.plot_utils.plot('validation_all_symbol_acc', valid_all_symbol / nb_iters * 100, index=1)

            lib.plot_utils.plot('validation_recon_acc', recon_acc / nb_encode / nb_decode / nb_iters * 100, index=1)
            lib.plot_utils.plot('validation_post_valid', post_valid / nb_encode / nb_decode / nb_iters * 100, index=1)
            lib.plot_utils.plot('validation_post_fe_deviation', post_fe_deviation / post_valid)

            prior_valid, prior_fe_deviation = evaluate_prior(1000, 10, args.latent_size, mp_pool)

            lib.plot_utils.plot('prior_valid', np.sum(prior_valid) / 100, index=1)  # /10000 * 100
            lib.plot_utils.plot('prior_fe_deviation', np.sum(prior_fe_deviation) / np.sum(prior_valid), index=1)

            lib.plot_utils.set_xlabel_for_tick(index=1, label='epoch')
            lib.plot_utils.flush()
            lib.plot_utils.tick(index=1)

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
