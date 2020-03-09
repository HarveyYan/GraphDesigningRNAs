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

from model.VAE import JunctionTreeVAE
from lib.data_utils import JunctionTreeFolder
import lib.plot

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
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=1000)

def compute_recon_acc(tree_batch, graph_vectors, tree_vectors, nb_encode=10, nb_decode=10, verbose=False):
    batch_size = len(tree_batch)
    recon_acc = [0] * batch_size
    posterior_valid = [0] * batch_size
    posterior_stability = [0] * batch_size

    # for each molecule encode 10 times
    for _ in range(nb_encode):
        graph_latent_vec, graph_kl_loss = \
            model.rsample(graph_vectors, model.g_mean, model.g_var)

        tree_latent_vec, tree_kl_loss = \
            model.rsample(tree_vectors, model.t_mean, model.t_var)

        for i in range(len(tree_batch)):

            # for each encoding decode 10 times

            for _ in range(nb_decode):

                try:
                    rna = model.decoder.decode(tree_latent_vec[i:i + 1, :], graph_latent_vec[i:i + 1, :], prob_decode=False,
                                         verbose=False)

                    if ''.join(rna.rna_seq) == ''.join(tree_batch[i].rna_seq) \
                            and rna.rna_struct == tree_batch[i].rna_struct:
                        recon_acc[i] += 1

                    posterior_valid[i] += 1

                    if rna.is_mfe or (rna.mfe_range is not None and rna.mfe_range < 0.01):
                        posterior_stability[i] += 1

                    if verbose:
                        print('original sequence:', ''.join(tree_batch[i].rna_seq))
                        print('decoded sequence', ''.join(rna.rna_seq))
                        print('decoded structure:', rna.rna_struct)
                        print('decoded structure free energy:', rna.free_energy)

                        if rna.is_mfe:
                            print('mfe achieved!')
                        else:
                            print('actual mfe structure:', rna.mfe_struct)
                            print('actual mfe:', rna.mfe)
                            print('hamming distance', rna.struct_hamming_dist)
                            print('mfe range', rna.mfe_range)
                except ValueError as e:
                    if verbose:
                        print(e)
                    continue
            if verbose:
                print('=' * 50)
        recon_acc = np.array(recon_acc)
        posterior_valid = np.array(posterior_valid)
        posterior_stability = np.array(posterior_stability)

        return recon_acc, posterior_valid, posterior_stability

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    model = JunctionTreeVAE(args.hidden_size, args.latent_size, args.depthT, args.depthG,
                            decode_nuc_with_lstm=True, device=device).to(device)
    print(model)
    # for param in model.parameters():
    #     print(param)
    # exit()
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = 0
    beta = args.beta
    meters = np.zeros(4)

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = '/'.join(args.save_dir.split('/')[:-1] + [cur_time + '-' + args.save_dir.split('/')[-1]])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lib.plot.set_output_dir(save_dir)
    lib.plot.suppress_stdout()

    for epoch in range(args.epoch):
        loader = JunctionTreeFolder('data/rna_jt_32-512/train-split', args.batch_size, num_workers=8)
        for batch in loader:
            total_step += 1
            model.zero_grad()
            loss, kl_div, all_acc = model(batch, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            meters = meters + np.array([float(kl_div), float(all_acc[0]) * 100, float(all_acc[1]) * 100, float(all_acc[2]) * 100])

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print("[%d] Beta: %.3f, KL: %.2f, Node: %.2f, Nucleotide: %.2f, Topo: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
                lib.plot.plot('node acc', meters[1])
                lib.plot.plot('nucleotide acc', meters[2])
                lib.plot.plot('topo acc', meters[3])
                lib.plot.flush()
                sys.stdout.flush()
                meters *= 0

            lib.plot.tick()

            if total_step % args.anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
                beta = min(args.max_beta, beta + args.step_beta)

        # save model at the end of each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, "model.epoch-" + str(epoch + 1)))

        del loader

        # validation step
        print('End of epoch %d,' % (epoch), 'starting validation')
        loader = JunctionTreeFolder('data/rna_jt_32-512/validation-split', args.batch_size, num_workers=8)
        valid_kl, valid_node_acc, valid_nuc_acc, valid_topo_acc = 0., 0., 0., 0.
        # recon_acc, post_valid, post_stab = 0., 0., 0.
        size = 0
        for batch in loader:
            size += len(batch)

            tree_batch, graph_encoder_input, tree_encoder_input = batch
            graph_vectors, tree_vectors, enc_tree_messages = \
                model.encode(graph_encoder_input, tree_encoder_input)

            # trite accuracy measures
            graph_latent_vec, graph_kl_loss = model.rsample(graph_vectors, model.g_mean, model.g_var)
            tree_latent_vec, tree_kl_loss = model.rsample(tree_vectors, model.t_mean, model.t_var)

            all_kl_loss = graph_kl_loss + tree_kl_loss

            all_loss, all_acc, tree_messages, tree_traces = \
                model.decoder(tree_batch, tree_latent_vec, graph_latent_vec)

            valid_kl += float(all_kl_loss)
            valid_node_acc += float(all_acc[0])
            valid_nuc_acc += float(all_acc[1])
            valid_topo_acc += float(all_acc[2])

            # # reconstruction_acc_measure
            # batch_recon_acc, batch_post_valid, batch_post_stability = \
            #     compute_recon_acc(tree_batch, graph_vectors, tree_vectors, nb_encode=10, nb_decode=10)
            #
            # recon_acc += np.sum(batch_recon_acc)
            # post_valid += np.sum(batch_post_valid)
            # post_stab += np.sum(batch_post_stability)


        lib.plot.plot('validation_kl', valid_kl / size)
        lib.plot.plot('validation_node_acc', valid_node_acc / size * 100)
        lib.plot.plot('validation_nuc_acc', valid_nuc_acc / size * 100)
        lib.plot.plot('validation_topo_acc', valid_topo_acc / size * 100)
        # lib.plot.plot('validation_reconstruction_acc', recon_acc / size)
        # lib.plot.plot('validation_posterior_validity', post_valid / size)
        # lib.plot.plot('validation_posterior_stability', post_stab / size)

        lib.plot.flush()

        del loader