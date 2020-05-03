import numpy as np
import torch
from lib.tree_decomp import RNAJunctionTree

model = None


def posterior_check_subroutine(args):
    o_seq, o_struct, d_tree = args
    ret = [0, 0, 0]  # recon_acc, post_valid, post_fe_dev
    if type(d_tree) is RNAJunctionTree and d_tree.is_valid:
        ret[1] = 1
        if ''.join(d_tree.rna_seq) == o_seq and ''.join(d_tree.rna_struct) == o_struct:
            ret[0] = 1
        else:
            ret[2] = d_tree.fe_deviation
    return ret


def evaluate_posterior(original_sequence, original_structure, graph_latent_vec, tree_latent_vec, mp_pool,
                       nb_encode=10, nb_decode=10, enforce_rna_prior=True):
    batch_size = len(original_sequence)
    recon_acc = [0] * batch_size
    posterior_valid = [0] * batch_size
    posterior_fe_deviation = [0] * batch_size
    batch_idx = list(range(batch_size))

    original_sequence = original_sequence * nb_encode
    original_structure = original_structure * nb_encode
    batch_idx = batch_idx * nb_encode

    # batch_size x nb_encode
    (z_vec, g_z_vec, t_z_vec), _ = model.rsample(graph_latent_vec, tree_latent_vec, nsamples=nb_encode)
    g_z_vec = g_z_vec.transpose(0, 1).reshape(batch_size * nb_encode, -1)
    t_z_vec = t_z_vec.transpose(0, 1).reshape(batch_size * nb_encode, -1)

    original_sequence = original_sequence * nb_decode
    original_structure = original_structure * nb_decode
    batch_idx = batch_idx * nb_decode
    g_z_vec = torch.cat([g_z_vec] * nb_decode, dim=0)
    t_z_vec = torch.cat([t_z_vec] * nb_decode, dim=0)

    all_rna_seq, all_trees, is_successful = model.decoder.decode(
        t_z_vec, g_z_vec, prob_decode=True, enforce_topo_prior=enforce_rna_prior,
        enforce_hpn_prior=enforce_rna_prior, enforce_dec_prior=enforce_rna_prior)

    all_parsed_trees = model.decoder.assemble_trees(all_rna_seq, all_trees, is_successful, mp_pool)

    ret = list(mp_pool.imap(posterior_check_subroutine,
                            list(zip(original_sequence, original_structure,
                                     all_parsed_trees))))

    for i, r in enumerate(ret):
        recon_acc[batch_idx[i]] += r[0]
        posterior_valid[batch_idx[i]] += r[1]
        posterior_fe_deviation[batch_idx[i]] += r[2]

    return recon_acc, posterior_valid, posterior_fe_deviation


def prior_check_subroutine(d_tree):
    ret = [0, 0]  # prior_valid, prior_fe_dev
    if type(d_tree) is RNAJunctionTree and d_tree.is_valid:
        ret[0] = 1
        ret[1] = d_tree.fe_deviation
    return ret


def evaluate_prior(g_z_vec, t_z_vec, nb_samples, nb_decode, mp_pool, enforce_rna_prior=True, prob_decode=True):
    prior_valid = [0] * nb_samples
    prior_fe_deviation = [0] * nb_samples
    batch_idx = list(range(nb_samples))

    batch_idx = batch_idx * nb_decode
    g_z_vec = torch.cat([g_z_vec] * nb_decode, dim=0)
    t_z_vec = torch.cat([t_z_vec] * nb_decode, dim=0)

    all_rna_seq, all_trees, is_successful = model.decoder.decode(
        t_z_vec, g_z_vec, prob_decode=prob_decode, enforce_topo_prior=enforce_rna_prior,
        enforce_hpn_prior=enforce_rna_prior, enforce_dec_prior=enforce_rna_prior)

    all_parsed_trees = model.decoder.assemble_trees(all_rna_seq, all_trees, is_successful, mp_pool)

    ret = np.array(list(mp_pool.imap(prior_check_subroutine,
                                     all_parsed_trees)))

    for i, r in enumerate(ret):
        prior_valid[batch_idx[i]] += r[0]
        prior_fe_deviation[batch_idx[i]] += r[1]

    return prior_valid, prior_fe_deviation, all_parsed_trees