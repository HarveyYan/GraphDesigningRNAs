import os
import sys
import torch
import h5py
import numpy as np
import RNA
import itertools
from torch.utils.data import Dataset, DataLoader

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_models.GraphLSTMVAE import GraphEncoder
from jtvae_models.GraphEncoder import GraphEncoder as JTVAE_GraphEncoder
from jtvae_models.TreeEncoder import TreeEncoder
from lib.tree_decomp import RNAJunctionTree

datapath = os.path.join(basedir, 'data', 'rbpdata', '{}')
dataset_options = {'data_RBPslow.h5', 'data_RBPsmed.h5', 'data_RBPshigh.h5'}

RBP_DATASET_NUC_VOCAB = ['A', 'C', 'G', 'U']

NUC_VOCAB = ['A', 'C', 'G', 'U']
LEN_NUC_VOCAB = len(NUC_VOCAB)
STRUCT_VOCAB = ['(', ')', '.']
LEN_STRUCT_VOCAB = len(STRUCT_VOCAB)

JOINT_VOCAB = [''.join(cand) for cand in itertools.product(NUC_VOCAB, STRUCT_VOCAB)]
FDIM_JOINT_VOCAB = len(JOINT_VOCAB)


class RBPFolder:

    def __init__(self, dataset_name, batch_size, num_workers=4, shuffle=True,
                 mode='train', preprocess_type='lstm'):
        assert dataset_name in dataset_options, \
            'dataset name {} not found in {}'.format(dataset_name, dataset_options)
        assert mode in ['train', 'valid', 'test'], \
            'mode {} not found in {}'.format(mode, ['train', 'valid', 'test'])
        assert preprocess_type in ['lstm', 'graph_lstm', 'jtvae'], \
            'preprocess type {} not found in {}'.format(preprocess_type, ['lstm', 'graph_lstm', 'jtvae'])
        self.data_file = datapath.format(dataset_name)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.mode = mode
        self.preprocess_type = preprocess_type

    def __iter__(self):

        with h5py.File(self.data_file, 'r') as file:
            all_np_seq = np.array(file['%s_in_seq' % (self.mode)]).transpose(0, 2, 1)
            all_label = np.array(file['%s_out' % (self.mode)])
        self.size = all_np_seq.shape[0]

        if self.shuffle:
            shuffled_idx = np.random.permutation(self.size)
            all_np_seq = all_np_seq[shuffled_idx]
            all_label = all_label[shuffled_idx]

        batches = [[all_np_seq[i: i + self.batch_size], all_label[i: i + self.batch_size]] for i in
                   range(0, self.size, self.batch_size)]
        # if len(batches[-1]) < self.batch_size:
        #     batches.pop()

        dataset = RBPDataset(batches, self.preprocess_type)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                collate_fn=lambda x: x[0])

        for b in dataloader:
            yield b

        del batches, dataset, dataloader


class RBPDataset(Dataset):

    def __init__(self, data, preprocess_type):
        self.data = data
        self.preprocess_type = preprocess_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        with torch.no_grad():

            if self.preprocess_type == 'lstm':
                all_joint_encoding = []
                all_np_seq, all_label = self.data[idx]
                for np_seq in all_np_seq:
                    seq = ''.join([RBP_DATASET_NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(np_seq, axis=-1)])
                    struct = RNA.fold(seq)[0]

                    joint_encoding = []
                    for seq_char, struct_char in zip(seq, struct):
                        onehot_enc = np.array(list(map(lambda x: x == seq_char + struct_char, JOINT_VOCAB)),
                                              dtype=np.float32)
                        joint_encoding.append(onehot_enc)
                    all_joint_encoding.append(joint_encoding)

                return all_joint_encoding, all_label

            elif self.preprocess_type == 'graph_lstm':
                all_np_seq, all_label = self.data[idx]
                all_pairs = []
                for np_seq in all_np_seq:
                    seq = ''.join([RBP_DATASET_NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(np_seq, axis=-1)])
                    struct = RNA.fold(seq)[0]
                    all_pairs.append([seq, struct])
                graph_encoder_input = GraphEncoder.prepare_batch_data(all_pairs)

                return graph_encoder_input, all_label

            elif self.preprocess_type == 'jtvae':
                all_np_seq, all_label = self.data[idx]
                tree_batch = []
                for np_seq in all_np_seq:
                    seq = ''.join([RBP_DATASET_NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(np_seq, axis=-1)])
                    struct, mfe = RNA.fold(seq)
                    tree = RNAJunctionTree(seq, struct, free_energy=float(mfe))
                    tree_batch.append(tree)
                graph_encoder_input = JTVAE_GraphEncoder.prepare_batch_data(
                    [(tree.rna_seq, tree.rna_struct) for tree in tree_batch])
                tree_encoder_input = TreeEncoder.prepare_batch_data(tree_batch)

                return (graph_encoder_input, tree_encoder_input), all_label
