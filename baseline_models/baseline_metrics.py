import re
import forgi.graph.bulge_graph as fgb
import RNA
import numpy as np
import torch
import gc

model = None

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

    ret = list(mp_pool.imap(posterior_check_subroutine,
                            list(zip(original_sequence, original_structure,
                                     decoded_seq, decoded_struct))))

    for i, r in enumerate(ret):
        recon_acc[batch_idx[i]] += r[0]
        posterior_valid[batch_idx[i]] += r[1]
        posterior_fe_deviation[batch_idx[i]] += r[2]

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


def evaluate_prior(sampled_latent_vector, nb_samples, nb_decode, mp_pool, enforce_rna_prior=True):
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

    return prior_valid, prior_fe_deviation
