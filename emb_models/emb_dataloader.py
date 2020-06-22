import os
import sys
import torch
from functools import partial
import numpy as np
import RNA
import itertools
import h5py

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_models.GraphLSTMVAE import GraphEncoder
from jtvae_models.GraphEncoder import GraphEncoder as jtvae_GraphEncoder
from jtvae_models.TreeEncoder import TreeEncoder
from jtvae_models.BranchedTreeEncoder import BranchedTreeEncoder
from lib.tree_decomp import RNAJunctionTree

NUC_VOCAB = ['A', 'C', 'G', 'U']
LEN_NUC_VOCAB = len(NUC_VOCAB)
STRUCT_VOCAB = ['(', ')', '.']
LEN_STRUCT_VOCAB = len(STRUCT_VOCAB)

JOINT_VOCAB = [''.join(cand) for cand in itertools.product(NUC_VOCAB, STRUCT_VOCAB)]
FDIM_JOINT_VOCAB = len(JOINT_VOCAB)

# RNAcompete dataset information
rnacompete_train_datapath = os.path.join(basedir, 'data', 'RNAcompete_derived_datasets', 'full', '{}_data_full_A.txt')
rnacompete_test_datapath = os.path.join(basedir, 'data', 'RNAcompete_derived_datasets', 'full', '{}_data_full_B.txt')
rnacompete_all_rbps = ['Fusip', 'HuR', 'PTB', 'RBM4', 'SF2', 'SLM2', 'U1A', 'VTS1', 'YB1']

# ncRNA dataset information
ncRNA_train_datapath = os.path.join(basedir, 'data', 'ncRNA', 'dataset_Rfam_6320_13classes.fasta')
ncRNA_test_datapath = os.path.join(basedir, 'data', 'ncRNA', 'dataset_Rfam_validated_2600_13classes.fasta')
ncRNA_all_classes = ['miRNA', '5S_rRNA', '5_8S_rRNA', 'ribozyme', 'CD-box', 'HACA-box', 'scaRNA', 'tRNA', 'Intron_gpI',
                     'Intron_gpII', 'IRES', 'leader', 'riboswitch']

# in-vivo RBP binding dataset information
rbp_datapath = os.path.join(basedir, 'data', 'rbpdata', '{}')
rbp_dataset_options = {'data_RBPslow.h5', 'data_RBPsmed.h5', 'data_RBPshigh.h5'}


def read_rnacompete_datafile(filepath):
    seqs, targets = [], []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.rstrip()
            target, seq = line.split('\t')
            targets.append(float(target))
            seqs.append(seq.replace('T', 'U'))
    return seqs, targets


def read_ncRNA_fasta(filepath):
    seqs, labels = [], []
    seq = ''
    with open(filepath, 'r') as file:
        for line in file:
            line = line.rstrip()
            if line.startswith('>'):
                if len(seq) > 0:
                    seqs.append(seq)
                    seq = ''
                labels.append(ncRNA_all_classes.index(line.split(' ')[-1]))
            else:
                seq += line.replace('T', 'U')
        seqs.append(seq)
    return seqs, labels


def read_rbp_h5py(filepath, mode='train'):
    with h5py.File(filepath, 'r') as file:
        all_np_seq = np.array(file['%s_in_seq' % (mode)]).transpose(0, 2, 1)
        all_label = np.array(file['%s_out' % (mode)])
    return all_np_seq, all_label


def convert_seq_to_embeddings(all_seq, model, mp_pool, preprocess_type='lstm'):
    size = len(all_seq)
    if preprocess_type == 'lstm':
        # obtain all secondary structures
        if mp_pool is None:
            all_joint_encodings = []
            for seq in all_seq:
                all_joint_encodings.append(lstm_joint_encoding_subroutine(seq))
        else:
            all_joint_encodings = list(mp_pool.imap(lstm_joint_encoding_subroutine, all_seq))
        batches = [all_joint_encodings[i: i + 32] for i in range(0, size, 32)]

        all_latent_vec = []
        with torch.no_grad():
            for joint_encoding in batches:
                latent_vec = model.encode(joint_encoding)
                z_vec = model.mean(latent_vec)
                all_latent_vec.append(z_vec)

    elif preprocess_type == 'graph_lstm':

        batches = [all_seq[i: i + 32] for i in range(0, size, 32)]
        if mp_pool is None:
            batch_graph_encoder_input = []
            for abatch in batches:
                batch_graph_encoder_input.append(graph_encoding_subroutine(abatch))
        else:
            batch_graph_encoder_input = list(mp_pool.imap(graph_encoding_subroutine, batches))

        all_latent_vec = []
        with torch.no_grad():
            for graph_encoder_input in batch_graph_encoder_input:
                latent_vec = model.encode(graph_encoder_input)
                z_vec = model.mean(latent_vec)
                all_latent_vec.append(z_vec)

    elif preprocess_type == 'jtvae':
        from tqdm import tqdm
        batches = [all_seq[i: i + 32] for i in range(0, size, 32)]
        if mp_pool is None:
            all_inputs = []
            for batch in tqdm(batches):
                all_inputs.append(jtvae_encoding_subroutine(batch))
        else:
            all_inputs = list(mp_pool.imap(jtvae_encoding_subroutine, batches))

        all_latent_vec = []
        with torch.no_grad():
            for pair_inputs in all_inputs:
                graph_vectors, tree_vectors = model.encode(*pair_inputs)
                z_vec = torch.cat([model.g_mean(graph_vectors),
                                   model.t_mean(tree_vectors)], dim=-1)
                all_latent_vec.append(z_vec)

    elif preprocess_type == 'jtvae_branched':

        batches = [all_seq[i: i + 1] for i in range(0, size, 1)]
        func = partial(jtvae_encoding_subroutine, tree_enc_type='branched')
        if mp_pool is None:
            all_inputs = []
            for batch in batches:
                all_inputs.append(func(batch))
        else:
            all_inputs = list(mp_pool.imap(func, batches))

        all_latent_vec = []
        with torch.no_grad():
            for pair_inputs in all_inputs:
                graph_vectors, tree_vectors = model.encode(*pair_inputs)
                z_vec = torch.cat([model.g_mean(graph_vectors),
                                   model.t_mean(tree_vectors)], dim=-1)
                all_latent_vec.append(z_vec)
    else:
        raise ValueError('preprocess_type must be one of: [lstm, graph_lstm, jtvae, jtvae_branched]')

    return torch.cat(all_latent_vec, dim=0)


def lstm_joint_encoding_subroutine(seq):
    if type(seq) is np.ndarray:
        # convert numpy sequence to string
        seq = ''.join([NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(seq, axis=-1)])

    struct = RNA.fold(seq)[0]
    joint_encoding = []
    for seq_char, struct_char in zip(seq, struct):
        onehot_enc = np.array(list(map(lambda x: x == seq_char + struct_char, JOINT_VOCAB)),
                              dtype=np.float32)
        joint_encoding.append(onehot_enc)
    return joint_encoding


def graph_encoding_subroutine(batch_seq):
    all_pairs = []
    for seq in batch_seq:
        if type(seq) is np.ndarray:
            # convert numpy sequence to string
            seq = ''.join([NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(seq, axis=-1)])

        struct = RNA.fold(seq)[0]
        all_pairs.append([seq, struct])
    graph_encoder_input = GraphEncoder.prepare_batch_data(all_pairs)
    return graph_encoder_input


def jtvae_encoding_subroutine(batch_seq, tree_enc_type='baseline'):
    all_trees = []
    for seq in batch_seq:
        if type(seq) is np.ndarray:
            # convert numpy sequence to string
            seq = ''.join([NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(seq, axis=-1)])

        struct, mfe = RNA.fold(seq)
        tree = RNAJunctionTree(seq, struct, free_energy=float(mfe))
        all_trees.append(tree)
    graph_encoder_input = jtvae_GraphEncoder.prepare_batch_data(
        [(tree.rna_seq, tree.rna_struct) for tree in all_trees])
    if tree_enc_type == 'baseline':
        tree_encoder_input = TreeEncoder.prepare_batch_data(all_trees)
    else:
        tree_encoder_input = BranchedTreeEncoder.prepare_batch_data(all_trees)
    return (graph_encoder_input, tree_encoder_input)
