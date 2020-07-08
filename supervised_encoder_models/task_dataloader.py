import os
import sys
import torch
from functools import partial
import numpy as np
import RNA
import itertools
import h5py
from torch.utils.data import Dataset, DataLoader

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


def lstm_seqonly_encoding_subroutine(seq):
    if type(seq) is np.ndarray:
        # convert numpy sequence to string
        seq = ''.join([NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(seq, axis=-1)])

    encoding = []
    label = []
    for seq_char in seq:
        onehot_enc = np.array(list(map(lambda x: x == seq_char, NUC_VOCAB)),
                              dtype=np.float32)
        encoding.append(onehot_enc)
        label.append(np.argmax(onehot_enc))
    return encoding, label


def lstm_joint_encoding_subroutine(seq):
    if type(seq) is np.ndarray:
        # convert numpy sequence to string
        seq = ''.join([NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(seq, axis=-1)])

    struct = RNA.fold(seq)[0]
    joint_encoding = []
    label = []
    for seq_char, struct_char in zip(seq, struct):
        onehot_enc = np.array(list(map(lambda x: x == seq_char + struct_char, JOINT_VOCAB)),
                              dtype=np.float32)
        joint_encoding.append(onehot_enc)
        label.append(np.argmax(onehot_enc))
    return joint_encoding, label


def graph_encoding_subroutine(batch_seq):
    all_joint_encoding = []
    all_label = []
    all_pairs = []
    for seq in batch_seq:
        if type(seq) is np.ndarray:
            # convert numpy sequence to string
            seq = ''.join([NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(seq, axis=-1)])

        struct = RNA.fold(seq)[0]
        all_pairs.append([seq, struct])

        joint_encoding = []
        label = []
        for seq_char, struct_char in zip(seq, struct):
            onehot_enc = np.array(list(map(lambda x: x == seq_char + struct_char, JOINT_VOCAB)), dtype=np.float32)
            joint_encoding.append(onehot_enc)
            label.append(np.argmax(onehot_enc))
        all_joint_encoding.append(joint_encoding)
        all_label.append(label)

    graph_encoder_input = GraphEncoder.prepare_batch_data(all_pairs)

    return (graph_encoder_input, all_joint_encoding, all_label)


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
    if tree_enc_type == 'jtvae':
        tree_encoder_input = TreeEncoder.prepare_batch_data(all_trees)
    elif tree_enc_type == 'jtvae_branched':
        tree_encoder_input = BranchedTreeEncoder.prepare_batch_data(all_trees)

    return (all_trees, graph_encoder_input, tree_encoder_input)


class TaskFolder:

    def __init__(self, all_seq, all_labels, batch_size, num_workers=4, shuffle=True,
                 mode='train', preprocess_type='lstm'):
        assert preprocess_type in ['lstm', 'lstm_seqonly', 'graph_lstm', 'jtvae', 'jtvae_branched'], \
            'preprocess type {} not found in {}'.format(preprocess_type, ['lstm', 'lstm_seqonly', 'graph_lstm', 'jtvae', 'jtvae_branched'])

        self.all_seq = all_seq
        self.all_labels = all_labels
        self.size = len(self.all_seq)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.mode = mode
        self.preprocess_type = preprocess_type

    def __iter__(self):

        if self.shuffle:
            shuffled_idx = np.random.permutation(self.size)
            all_seq = self.all_seq[shuffled_idx]
            all_labels = self.all_labels[shuffled_idx]
        else:
            all_seq = self.all_seq
            all_labels = self.all_labels

        batches = [[all_seq[i: i + self.batch_size], all_labels[i: i + self.batch_size]] for i in
                   range(0, self.size, self.batch_size)]

        dataset = TaskDataset(batches, self.preprocess_type)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                collate_fn=lambda x: x[0])

        for b in dataloader:
            yield b

        del batches, dataset, dataloader


class TaskDataset(Dataset):

    def __init__(self, data, preprocess_type):
        self.data = data
        self.preprocess_type = preprocess_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        batch_seq, batch_label = self.data[idx]
        with torch.no_grad():

            if self.preprocess_type == 'lstm':

                batch_joint_encodings = []
                batch_seq_label = []
                for seq in batch_seq:
                    joint_encoding, label = lstm_joint_encoding_subroutine(seq)
                    batch_joint_encodings.append(joint_encoding)
                    batch_seq_label.append(label)
                return (batch_joint_encodings, batch_seq_label), batch_label

            elif self.preprocess_type == 'lstm_seqonly':

                batch_encodings = []
                batch_seq_label = []
                for seq in batch_seq:
                    encoding, label = lstm_seqonly_encoding_subroutine(seq)
                    batch_encodings.append(encoding)
                    batch_seq_label.append(label)
                return (batch_encodings, batch_seq_label), batch_label

            elif self.preprocess_type == 'graph_lstm':

                batch_graph_input = graph_encoding_subroutine(batch_seq)
                return batch_graph_input, batch_label

            elif self.preprocess_type == 'jtvae' or self.preprocess_type == 'jtvae_branched':

                batch_jtvae_inputs = jtvae_encoding_subroutine(batch_seq, self.preprocess_type)
                return batch_jtvae_inputs, batch_label
