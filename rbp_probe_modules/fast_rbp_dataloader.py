import os
import sys
import torch
import h5py
import numpy as np
import RNA
import itertools

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_models.FlowLSTMVAE import LSTMVAE
from baseline_models.GraphLSTMVAE import GraphEncoder, GraphLSTMVAE
from jtvae_models.VAE import JunctionTreeVAE
from jtvae_models.GraphEncoder import GraphEncoder as jtvae_GraphEncoder
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

model = None


class RBPFolder:

    def __init__(self, dataset_name, batch_size, num_workers=4, shuffle=True,
                 mode='train', preprocess_type='lstm', weight_path=None, mp_pool=None):
        assert dataset_name in dataset_options, \
            'dataset name {} not found in {}'.format(dataset_name, dataset_options)
        assert mode in ['train', 'valid', 'test'], \
            'mode {} not found in {}'.format(mode, ['train', 'valid', 'test'])
        assert preprocess_type in ['lstm', 'graph_lstm', 'jtvae'], \
            'preprocess type {} not found in {}'.format(preprocess_type, ['lstm', 'graph_lstm', 'jtvae'])
        assert os.path.exists(weight_path), '%s doesn\'t exists'
        self.data_file = datapath.format(dataset_name)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.mode = mode
        self.mp_pool = mp_pool

        self.preprocess_type = preprocess_type
        self.weight_path = weight_path

        global model
        if model is None:
            cpu_device = torch.device('cpu')
            print('Data Utils Loading %s Model' % (self.preprocess_type))
            if self.preprocess_type == 'lstm':
                model = LSTMVAE(
                    512, 128, 2, device=cpu_device, use_attention=True,
                    use_flow_prior=True, use_aux_regressor=False).to(cpu_device)
            elif self.preprocess_type == 'graph_lstm':
                model = GraphLSTMVAE(
                    512, 128, 10, device=cpu_device, use_attention=False,
                    use_flow_prior=True, use_aux_regressor=False).to(cpu_device)
            elif self.preprocess_type == 'jtvae':
                model = JunctionTreeVAE(
                    512, 64, 5, 10, decode_nuc_with_lstm=True, tree_encoder_arch='baseline',
                    use_flow_prior=True, device=cpu_device).to(cpu_device)

            model.load_state_dict(
                torch.load(self.weight_path, map_location=cpu_device)['model_weights'])

        self.model = model

    @staticmethod
    def lstm_joint_encoding_subroutine(np_seq):
        seq = ''.join([RBP_DATASET_NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(np_seq, axis=-1)])
        struct = RNA.fold(seq)[0]
        joint_encoding = []
        for seq_char, struct_char in zip(seq, struct):
            onehot_enc = np.array(list(map(lambda x: x == seq_char + struct_char, JOINT_VOCAB)),
                                  dtype=np.float32)
            joint_encoding.append(onehot_enc)
        return joint_encoding

    @staticmethod
    def graph_encoding_subroutine(batch_np_seq):
        all_pairs = []
        for np_seq in batch_np_seq:
            seq = ''.join([RBP_DATASET_NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(np_seq, axis=-1)])
            struct = RNA.fold(seq)[0]
            all_pairs.append([seq, struct])
        graph_encoder_input = GraphEncoder.prepare_batch_data(all_pairs)
        return graph_encoder_input

    @staticmethod
    def jtvae_encoding_subroutine(batch_np_seq):
        all_trees = []
        for np_seq in batch_np_seq:
            seq = ''.join([RBP_DATASET_NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(np_seq, axis=-1)])
            struct, mfe = RNA.fold(seq)
            tree = RNAJunctionTree(seq, struct, free_energy=float(mfe))
            all_trees.append(tree)
        graph_encoder_input = jtvae_GraphEncoder.prepare_batch_data(
            [(tree.rna_seq, tree.rna_struct) for tree in all_trees])
        tree_encoder_input = TreeEncoder.prepare_batch_data(all_trees)
        return (graph_encoder_input, tree_encoder_input)

    def __iter__(self):

        if not hasattr(self, 'all_latent_vec'):
            print('Converting raw RNAs to embeddings')
            with h5py.File(self.data_file, 'r') as file:
                all_np_seq = np.array(file['%s_in_seq' % (self.mode)]).transpose(0, 2, 1)
                all_label = np.array(file['%s_out' % (self.mode)])
            self.size = all_np_seq.shape[0]

            if self.preprocess_type == 'lstm':
                # obtain all secondary structures
                all_joint_encodings = list(self.mp_pool.imap(RBPFolder.lstm_joint_encoding_subroutine, all_np_seq))
                batches = [all_joint_encodings[i: i + 1000] for i in range(0, self.size, 1000)]

                all_latent_vec = []
                with torch.no_grad():
                    for joint_encoding in batches:
                        latent_vec = self.model.encode(joint_encoding)
                        z_vec = self.model.mean(latent_vec)
                        all_latent_vec.append(z_vec)

            elif self.preprocess_type == 'graph_lstm':

                batches = [all_np_seq[i: i + 1000] for i in range(0, self.size, 1000)]
                batch_graph_encoder_input = list(self.mp_pool.imap(RBPFolder.graph_encoding_subroutine, batches))

                all_latent_vec = []
                with torch.no_grad():
                    for graph_encoder_input in batch_graph_encoder_input:
                        latent_vec = self.model.encode(graph_encoder_input)
                        z_vec = self.model.mean(latent_vec)
                        all_latent_vec.append(z_vec)

            elif self.preprocess_type == 'jtvae':

                batches = [all_np_seq[i: i + 1000] for i in range(0, self.size, 1000)]
                all_inputs = list(self.mp_pool.imap(RBPFolder.jtvae_encoding_subroutine, batches))

                all_latent_vec = []
                with torch.no_grad():
                    for pair_inputs in all_inputs:
                        graph_vectors, tree_vectors = self.model.encode(*pair_inputs)
                        z_vec = torch.cat([self.model.g_mean(graph_vectors),
                                           self.model.t_mean(tree_vectors)], dim=-1)
                        all_latent_vec.append(z_vec)

            self.all_latent_vec = torch.cat(all_latent_vec, dim=0)
            self.all_label = all_label

        if self.shuffle:
            shuffled_idx = np.random.permutation(self.size)
            all_latent_vec = self.all_latent_vec[shuffled_idx]
            all_label = self.all_label[shuffled_idx]
        else:
            all_latent_vec = self.all_latent_vec
            all_label = self.all_label

        for i in range(0, self.size, self.batch_size):
            yield all_latent_vec[i: i + self.batch_size], all_label[i: i + self.batch_size]
