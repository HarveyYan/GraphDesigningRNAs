import os
import torch
import numpy as np
from multiprocessing import Pool
import csv
import argparse
import shutil

from baseline_models.FlowLSTMVAE import LSTMVAE, BasicLSTMVAEFolder
from baseline_models.GraphLSTMVAE import GraphLSTMVAE
from baseline_models.GraphLSTMVAE import BasicLSTMVAEFolder as BasicGraphLSTMVAEFolder
from jtvae_models.VAE import JunctionTreeVAE
from lib.data_utils import JunctionTreeFolder
from lib.tree_decomp import RNAJunctionTree

import baseline_models.baseline_metrics
from baseline_models.baseline_metrics import evaluate_prior as baseline_evaluate_prior
from baseline_models.baseline_metrics import evaluate_posterior as baseline_evaluate_posterior

import jtvae_models.jtvae_utils
from jtvae_models.jtvae_utils import evaluate_prior as jt_evaluate_prior
from jtvae_models.jtvae_utils import evaluate_posterior as jt_evaluate_posterior

import lib.plot_utils, lib.logger

parser = argparse.ArgumentParser()
parser.add_argument('--expr_dir', required=True)
parser.add_argument('--use_flow_prior', type=eval, default=True, choices=[True, False])
parser.add_argument('--focus_epoch', type=int, default=None)
parser.add_argument('--mode', required=True)


def write_baseline_seq_struct(filename_stub, batch_idx, is_valid, decoded_seq, decoded_struct):
    with open(filename_stub.format('seq'), 'a') as file:
        for idx, is_valid_ in enumerate(is_valid):
            if is_valid_ == True:
                file.write('>batch-%d-idx-%d\n%s\n' % (batch_idx, idx, decoded_seq[idx]))

    with open(filename_stub.format('struct'), 'a') as file:
        for idx, is_valid_ in enumerate(is_valid):
            if is_valid_ == True:
                file.write('>batch-%d-idx-%d\n%s\n' % (batch_idx, idx, decoded_struct[idx]))


def write_jt_seq_struct(filename_stub, batch_idx, parsed_trees):
    with open(filename_stub.format('seq'), 'a') as file:
        for idx, tree in enumerate(parsed_trees):
            if type(tree) is RNAJunctionTree and tree.is_valid:
                file.write('>batch-%d-idx-%d\n%s\n' % (batch_idx, idx, ''.join(tree.rna_seq)))

    with open(filename_stub.format('struct'), 'a') as file:
        for idx, tree in enumerate(parsed_trees):
            if type(tree) is RNAJunctionTree and tree.is_valid:
                file.write('>batch-%d-idx-%d\n%s\n' % (batch_idx, idx, ''.join(tree.rna_struct)))


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    mode = args.mode
    assert mode in ['lstm', 'graph_lstm', 'jtvae'], \
        'mode must be one of {}'.format(['lstm', 'graph_lstm', 'jtvae'])

    all_fields = ['Epoch',
                  'Validation_recon_acc_with_reg', 'Validation_post_valid_with_reg',
                  'Validation_post_fe_deviation_with_reg',
                  'Validation_post_fe_deviation_len_normed_with_reg',
                  'Validation_recon_acc_no_reg', 'Validation_post_valid_no_reg',
                  'Validation_post_fe_deviation_no_reg',
                  'Validation_post_fe_deviation_len_normed_no_reg',
                  'Validation_recon_acc_no_reg_greedy', 'Validation_post_valid_no_reg_greedy',
                  'Validation_post_fe_deviation_no_reg_greedy',
                  'Validation_post_fe_deviation_len_normed_no_reg_greedy',
                  'Prior_valid_with_reg', 'Prior_fe_deviation_with_reg',
                  'Prior_fe_deviation_len_normed_with_reg',
                  'Prior_valid_no_reg', 'Prior_fe_deviation_no_reg',
                  'Prior_fe_deviation_len_normed_no_reg',
                  'Prior_valid_no_reg_greedy', 'Prior_fe_deviation_no_reg_greedy',
                  'Prior_fe_deviation_len_normed_no_reg_greedy',
                  'Prior_uniqueness_no_reg_greedy']

    expr_dir = args.expr_dir
    if args.focus_epoch is not None:
        save_dir = os.path.join(expr_dir, 'rigorosity-epoch-%d' % (args.focus_epoch))

        epochs_to_load = [args.focus_epoch]
    else:
        save_dir = os.path.join(expr_dir, 'rigorosity')

        epochs_to_load = []
        for dirname in os.listdir(expr_dir):
            if dirname.startswith('model'):
                epochs_to_load.append(int(dirname.split('-')[-1]))
        epochs_to_load = list(np.sort(epochs_to_load))

    lib.plot_utils.set_output_dir(save_dir)
    lib.plot_utils.suppress_stdout()
    lib.plot_utils.set_first_tick(int(epochs_to_load[0]))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger = lib.logger.CSVLogger('run.csv', save_dir, all_fields)
    else:
        # load existing results
        all_lines = []
        reader = csv.DictReader(open(os.path.join(save_dir, 'run.csv'), 'r'))
        for i, line in enumerate(reader):
            if i == 0:
                lib.plot_utils.set_first_tick(int(line['Epoch']))
            new_line = {}
            for key in line.keys():
                if key == 'Epoch':
                    epochs_to_load.remove(int(line[key]))
                    new_line[key] = int(line[key])
                    continue
                new_line[key] = float(line[key])
                lib.plot_utils.plot(key, float(line[key]), index=1)
            all_lines.append(new_line)

            lib.plot_utils.set_xlabel_for_tick(index=1, label='epoch')
            lib.plot_utils.flush()
            lib.plot_utils.tick(index=1)

        logger = lib.logger.CSVLogger('run.csv', save_dir, all_fields)
        for row in all_lines:
            logger.update_with_dict(row)

    epochs_to_load = list(np.sort(epochs_to_load))
    print(epochs_to_load)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mp_pool = Pool(10)

    for enc_epoch_to_load in epochs_to_load:
        if mode == 'lstm':
            model = LSTMVAE(512, 128, 2, device=device, use_attention=True, use_aux_regressor=False).to(device)
        elif mode == 'graph_lstm':
            model = GraphLSTMVAE(
                512, 128, 5, use_aux_regressor=False,
                use_flow_prior=args.use_flow_prior, device=device).to(device)
        else:
            model = JunctionTreeVAE(
                512, 64, 5, 10, decode_nuc_with_lstm=True, use_flow_prior=args.use_flow_prior,
                tree_encoder_arch='baseline', device=device).to(device)

        weight_path = os.path.join(expr_dir, 'model.epoch-%d' % (enc_epoch_to_load))
        print('Loading', weight_path)
        model.load_state_dict(
            torch.load(weight_path, map_location=device)['model_weights'])

        if mode != 'jtvae':
            baseline_models.baseline_metrics.model = model
        else:
            jtvae_models.jtvae_utils.model = model

        valid_batch_size = 256
        if mode == 'lstm':
            loader = BasicLSTMVAEFolder('data/rna_jt_32-512/validation-split', valid_batch_size, num_workers=4,
                                        shuffle=False)
        elif mode == 'graph_lstm':
            loader = BasicGraphLSTMVAEFolder('data/rna_jt_32-512/validation-split', valid_batch_size, num_workers=4,
                                             shuffle=False)
        else:
            loader = JunctionTreeFolder('data/rna_jt_32-512/validation-split', valid_batch_size, num_workers=4,
                                        shuffle=False)

        nb_iters = 20000 // valid_batch_size  # 20000 is the size of the validation set
        total = 0
        # from tqdm import trange
        # bar = trange(nb_iters, desc='', leave=True)
        # loader = loader.__iter__()
        nb_encode, nb_decode = 5, 5

        recon_acc, post_valid, post_fe_deviation, post_fe_deviation_len_normed = 0, 0, 0., 0.
        recon_acc_noreg, post_valid_noreg, post_fe_deviation_noreg, post_fe_deviation_noreg_len_normed = 0, 0, 0., 0.
        recon_acc_noreg_det, post_valid_noreg_det, post_fe_deviation_noreg_det, post_fe_deviation_noreg_det_len_normed = 0, 0, 0., 0.

        epoch_dir = os.path.join(save_dir, 'epoch-%d' % (enc_epoch_to_load))
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        else:
            shutil.rmtree(epoch_dir)
            os.makedirs(epoch_dir)

        with torch.no_grad():

            # for i in bar:
            for i, batch_input in enumerate(loader):
                # batch_input = next(loader)

                if mode == 'lstm':
                    original_data, batch_sequence, batch_label, batch_fe = batch_input
                    latent_vec = model.encode(batch_sequence)
                elif mode == 'graph_lstm':
                    original_data, batch_sequence, batch_label, batch_fe, batch_graph_input = batch_input
                    latent_vec = model.encode(batch_graph_input)
                else:
                    tree_batch, graph_encoder_input, tree_encoder_input = batch_input
                    graph_vectors, tree_vectors = model.encode(graph_encoder_input, tree_encoder_input)
                    all_seq = [''.join(tree.rna_seq) for tree in tree_batch]
                    all_struct = [''.join(tree.rna_struct) for tree in tree_batch]

                ####################### evaluate posterior with regularity constraints ########################
                if mode != 'jtvae':
                    ret_dict = baseline_evaluate_posterior(list(np.array(original_data)[:, 0]),
                                                           list(np.array(original_data)[:, 1]),
                                                           latent_vec, mp_pool, nb_encode=nb_encode,
                                                           nb_decode=nb_decode,
                                                           enforce_rna_prior=True)
                    write_baseline_seq_struct(
                        os.path.join(epoch_dir, 'valid-post-sto-reg-{}.fa')
                        , i, np.array(ret_dict['ret'])[:, 1], ret_dict['decoded_seq'], ret_dict['decoded_struct'])

                else:
                    ret_dict = jt_evaluate_posterior(all_seq, all_struct, graph_vectors, tree_vectors,
                                                     mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                                     enforce_rna_prior=True)

                    write_jt_seq_struct(
                        os.path.join(epoch_dir, 'valid-post-sto-reg-{}.fa'), i, ret_dict['all_parsed_trees'])

                total += nb_encode * nb_decode * valid_batch_size
                recon_acc += np.sum(ret_dict['recon_acc'])
                post_valid += np.sum(ret_dict['posterior_valid'])
                post_fe_deviation += np.sum(ret_dict['posterior_fe_deviation'])
                post_fe_deviation_len_normed += np.sum(ret_dict['posterior_fe_deviation_len_normed'])

                ####################### evaluate posterior without regularity constraints ########################
                if mode != 'jtvae':
                    ret_dict = baseline_evaluate_posterior(list(np.array(original_data)[:, 0]),
                                                           list(np.array(original_data)[:, 1]),
                                                           latent_vec, mp_pool, nb_encode=nb_encode,
                                                           nb_decode=nb_decode,
                                                           enforce_rna_prior=False)

                    write_baseline_seq_struct(
                        os.path.join(epoch_dir, 'valid-post-sto-noreg-{}.fa')
                        , i, np.array(ret_dict['ret'])[:, 1], ret_dict['decoded_seq'], ret_dict['decoded_struct'])

                else:
                    ret_dict = jt_evaluate_posterior(all_seq, all_struct, graph_vectors, tree_vectors,
                                                     mp_pool, nb_encode=nb_encode, nb_decode=nb_decode,
                                                     enforce_rna_prior=False)

                    write_jt_seq_struct(
                        os.path.join(epoch_dir, 'valid-post-sto-noreg-{}.fa'), i, ret_dict['all_parsed_trees'])

                recon_acc_noreg += np.sum(ret_dict['recon_acc'])
                post_valid_noreg += np.sum(ret_dict['posterior_valid'])
                post_fe_deviation_noreg += np.sum(ret_dict['posterior_fe_deviation'])
                post_fe_deviation_noreg_len_normed += np.sum(ret_dict['posterior_fe_deviation_len_normed'])

                ####################### evaluate posterior without regularity constraints and greedy ########################
                if mode != 'jtvae':
                    ret_dict = baseline_evaluate_posterior(list(np.array(original_data)[:, 0]),
                                                           list(np.array(original_data)[:, 1]),
                                                           latent_vec, mp_pool, nb_encode=nb_encode,
                                                           nb_decode=1,
                                                           prob_decode=False, enforce_rna_prior=False)

                    write_baseline_seq_struct(
                        os.path.join(epoch_dir, 'valid-post-det-noreg-{}.fa')
                        , i, np.array(ret_dict['ret'])[:, 1], ret_dict['decoded_seq'], ret_dict['decoded_struct'])

                else:
                    ret_dict = jt_evaluate_posterior(all_seq, all_struct, graph_vectors, tree_vectors,
                                                     mp_pool, nb_encode=nb_encode, nb_decode=1,
                                                     enforce_rna_prior=False, prob_decode=False)

                    write_jt_seq_struct(
                        os.path.join(epoch_dir, 'valid-post-det-noreg-{}.fa'), i, ret_dict['all_parsed_trees'])

                recon_acc_noreg_det += np.sum(ret_dict['recon_acc'])
                post_valid_noreg_det += np.sum(ret_dict['posterior_valid'])
                post_fe_deviation_noreg_det += np.sum(ret_dict['posterior_fe_deviation'])
                post_fe_deviation_noreg_det_len_normed += np.sum(ret_dict['posterior_fe_deviation_len_normed'])

            #     bar.set_description(
            #         'streaming recon acc: %.2f, streaming post valid: %.2f, streaming post free energy deviation: %.2f, streaming post free energy deviation length normalized: %.2f'
            #         % (recon_acc / total * 100, post_valid / total * 100, post_fe_deviation / post_valid,
            #            post_fe_deviation_len_normed / post_valid))
            #
            # bar.refresh()

            # posterior decoding with enforced RNA regularity
            lib.plot_utils.plot('Validation_recon_acc_with_reg', recon_acc / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_valid_with_reg', post_valid / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_with_reg', post_fe_deviation / post_valid, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_len_normed_with_reg',
                                post_fe_deviation_len_normed / post_valid, index=1)

            # posterior decoding without RNA regularity
            lib.plot_utils.plot('Validation_recon_acc_no_reg', recon_acc_noreg / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_valid_no_reg', post_valid_noreg / total * 100, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_no_reg', post_fe_deviation_noreg / post_valid_noreg, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_len_normed_no_reg',
                                post_fe_deviation_noreg_len_normed / post_valid_noreg, index=1)

            # posterior decoding without RNA regularity and deterministic
            lib.plot_utils.plot('Validation_recon_acc_no_reg_greedy',
                                recon_acc_noreg_det / total * nb_decode * 100, index=1)  # only decoded once
            lib.plot_utils.plot('Validation_post_valid_no_reg_greedy', post_valid_noreg_det / total * nb_decode * 100, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_no_reg_greedy',
                                post_fe_deviation_noreg_det / post_valid_noreg_det, index=1)
            lib.plot_utils.plot('Validation_post_fe_deviation_len_normed_no_reg_greedy',
                                post_fe_deviation_noreg_det_len_normed / post_valid_noreg_det, index=1)

            prior_valid_reg_sto, prior_fe_deviation_reg_sto, prior_fe_deviation_reg_sto_len_normed = 0., 0., 0.
            prior_valid_noreg_sto, prior_fe_deviation_noreg_sto, prior_fe_deviation_noreg_sto_len_normed = 0., 0., 0.
            prior_valid_noreg_det, prior_fe_deviation_noreg_det, prior_fe_deviation_noreg_det_len_normed, prior_uniqueness_noreg_det = 0., 0., 0., 0

            ####################### sampling from the prior ########################
            if mode != 'jtvae':
                sampled_latent_prior = torch.as_tensor(np.random.randn(10000, 128).astype(np.float32)).to(
                    device)
                if args.use_flow_prior:
                    sampled_latent_prior = model.latent_cnf(sampled_latent_prior, None, reverse=True).view(
                        *sampled_latent_prior.size())
            else:
                sampled_g_z = torch.as_tensor(np.random.randn(10000, 64).
                                              astype(np.float32)).to(device)
                sampled_t_z = torch.as_tensor(np.random.randn(10000, 64).
                                              astype(np.float32)).to(device)
                sampled_z = torch.cat([sampled_g_z, sampled_t_z], dim=-1)
                if args.use_flow_prior:
                    sampled_z = model.latent_cnf(sampled_z, None, reverse=True).view(
                        *sampled_z.size())
                sampled_g_z = sampled_z[:, :64]
                sampled_t_z = sampled_z[:, 64:]

            ######################## evaluate prior with regularity constraints ########################
            if mode != 'jtvae':
                ret_dict = baseline_evaluate_prior(
                    sampled_latent_prior, 10000, 10, mp_pool,
                    enforce_rna_prior=True)

                write_baseline_seq_struct(
                    os.path.join(epoch_dir, 'prior-sto-reg-{}.fa')
                    , 0, np.array(ret_dict['ret'])[:, 0], ret_dict['decoded_seq'], ret_dict['decoded_struct'])

                prior_valid_reg_sto += np.sum(ret_dict['prior_valid'])
                prior_fe_deviation_reg_sto += np.sum(ret_dict['prior_fe_deviation'])
                prior_fe_deviation_reg_sto_len_normed += np.sum(ret_dict['prior_fe_deviation_len_normed'])

            else:
                for decode_idx in range(10):
                    ret_dict = jt_evaluate_prior(
                        sampled_g_z, sampled_t_z, 10000, 1, mp_pool,
                        enforce_rna_prior=True)

                    write_jt_seq_struct(
                        os.path.join(epoch_dir, 'prior-sto-reg-{}.fa'), decode_idx, ret_dict['all_parsed_trees'])

                    prior_valid_reg_sto += np.sum(ret_dict['prior_valid'])
                    prior_fe_deviation_reg_sto += np.sum(ret_dict['prior_fe_deviation'])
                    prior_fe_deviation_reg_sto_len_normed += np.sum(ret_dict['prior_fe_deviation_len_normed'])

            ######################## evaluate prior without regularity constraints ########################
            if mode != 'jtvae':
                ret_dict = baseline_evaluate_prior(
                    sampled_latent_prior, 10000, 10, mp_pool,
                    enforce_rna_prior=False)

                write_baseline_seq_struct(
                    os.path.join(epoch_dir, 'prior-sto-noreg-{}.fa')
                    , 0, np.array(ret_dict['ret'])[:, 0], ret_dict['decoded_seq'], ret_dict['decoded_struct'])

                prior_valid_noreg_sto += np.sum(ret_dict['prior_valid'])
                prior_fe_deviation_noreg_sto += np.sum(ret_dict['prior_fe_deviation'])
                prior_fe_deviation_noreg_sto_len_normed += np.sum(ret_dict['prior_fe_deviation_len_normed'])

            else:
                for decode_idx in range(10):
                    ret_dict = jt_evaluate_prior(
                        sampled_g_z, sampled_t_z, 10000, 1, mp_pool,
                        enforce_rna_prior=False)

                    write_jt_seq_struct(
                        os.path.join(epoch_dir, 'prior-sto-noreg-{}.fa'), decode_idx, ret_dict['all_parsed_trees'])

                    prior_valid_noreg_sto += np.sum(ret_dict['prior_valid'])
                    prior_fe_deviation_noreg_sto += np.sum(ret_dict['prior_fe_deviation'])
                    prior_fe_deviation_noreg_sto_len_normed += np.sum(ret_dict['prior_fe_deviation_len_normed'])

            ######################## evaluate prior without regularity constraints and greedy ########################
            if mode != 'jtvae':
                ret_dict = baseline_evaluate_prior(
                    sampled_latent_prior, 10000, 1, mp_pool,
                    enforce_rna_prior=False, prob_decode=False)

                prior_uniqueness_noreg_det += len(
                    set(list(np.array(ret_dict['decoded_seq'])[np.where(np.array(ret_dict['prior_valid']) > 0)[0]])))

                write_baseline_seq_struct(
                    os.path.join(epoch_dir, 'prior-det-noreg-{}.fa')
                    , 0, np.array(ret_dict['ret'])[:, 0], ret_dict['decoded_seq'], ret_dict['decoded_struct'])

            else:
                ret_dict = jt_evaluate_prior(
                    sampled_g_z, sampled_t_z, 10000, 1, mp_pool,
                    enforce_rna_prior=False, prob_decode=False)

                decoded_seq = [''.join(tree.rna_seq) for tree in ret_dict['all_parsed_trees'][:10000] if
                               type(tree) is RNAJunctionTree and tree.is_valid]
                prior_uniqueness_noreg_det += len(set(decoded_seq))

                write_jt_seq_struct(
                    os.path.join(epoch_dir, 'prior-det-noreg-{}.fa'), 0, ret_dict['all_parsed_trees'])

            prior_valid_noreg_det += np.sum(ret_dict['prior_valid'])
            prior_fe_deviation_noreg_det += np.sum(ret_dict['prior_fe_deviation'])
            prior_fe_deviation_noreg_det_len_normed += np.sum(ret_dict['prior_fe_deviation_len_normed'])

            lib.plot_utils.plot('Prior_valid_with_reg', prior_valid_reg_sto / 1000, index=1)
            lib.plot_utils.plot('Prior_fe_deviation_with_reg', prior_fe_deviation_reg_sto / prior_valid_reg_sto, index=1)
            lib.plot_utils.plot('Prior_fe_deviation_len_normed_with_reg',
                                prior_fe_deviation_reg_sto_len_normed / prior_valid_reg_sto, index=1)

            lib.plot_utils.plot('Prior_valid_no_reg', prior_valid_noreg_sto / 1000, index=1)
            lib.plot_utils.plot('Prior_fe_deviation_no_reg', prior_fe_deviation_noreg_sto / prior_valid_noreg_sto, index=1)
            lib.plot_utils.plot('Prior_fe_deviation_len_normed_no_reg',
                                prior_fe_deviation_noreg_sto_len_normed / prior_valid_noreg_sto, index=1)

            lib.plot_utils.plot('Prior_valid_no_reg_greedy', prior_valid_noreg_det / 100, index=1)
            lib.plot_utils.plot('Prior_fe_deviation_no_reg_greedy',
                                prior_fe_deviation_noreg_det / prior_valid_noreg_det, index=1)
            lib.plot_utils.plot('Prior_fe_deviation_len_normed_no_reg_greedy',
                                prior_fe_deviation_noreg_det_len_normed / prior_valid_noreg_det, index=1)
            lib.plot_utils.plot('Prior_uniqueness_no_reg_greedy',
                                prior_uniqueness_noreg_det / prior_valid_noreg_det * 100, index=1)

            tocsv = {'Epoch': enc_epoch_to_load}
            for name, val in lib.plot_utils._since_last_flush.items():
                tocsv[name] = list(val.values())[0]
            logger.update_with_dict(tocsv)

            lib.plot_utils.set_xlabel_for_tick(index=1, label='epoch')
            lib.plot_utils.flush()
            lib.plot_utils.tick(index=1)

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
