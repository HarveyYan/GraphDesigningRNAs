import os
import sys
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import RNA
import itertools

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_models.FlowLSTMVAE import LSTMVAE

datapath = os.path.join(basedir, 'data', 'rbpdata', '{}')
dataset_options = {'data_RBPslow.h5', 'data_RBPsmed.h5', 'data_RBPshigh.h5'}

model = None

RBP_DATASET_NUC_VOCAB = ['A', 'C', 'G', 'U']

NUC_VOCAB = ['A', 'C', 'G', 'U']
LEN_NUC_VOCAB = len(NUC_VOCAB)
STRUCT_VOCAB = ['(', ')', '.']
LEN_STRUCT_VOCAB = len(STRUCT_VOCAB)

JOINT_VOCAB = [''.join(cand) for cand in itertools.product(NUC_VOCAB, STRUCT_VOCAB)]
FDIM_JOINT_VOCAB = len(JOINT_VOCAB)


class RBPFolder:

    def __init__(self, dataset_name, batch_size, num_workers=4, shuffle=True,
                 mode='train', preprocess_type='lstm', weight_path=None):
        assert dataset_name in dataset_options, \
            'dataset name {} not found in {}'.format(dataset_name, dataset_options)
        assert mode in ['train', 'valid', 'test'], \
            'mode {} not found in {}'.format(mode, ['train', 'valid', 'test'])
        assert preprocess_type in ['lstm', 'graph_lstm', 'jtvae'], \
            'preprocess type {} not found in {}'.format(preprocess_type, ['lstm', 'graph_lstm', 'jtvae'])
        assert os.path.exists(self.weight_path), '%s doesn\'t exists'
        self.data_file = datapath.format(dataset_name)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.mode = mode

        self.preprocess_type = preprocess_type
        self.weight_path = weight_path

        global model
        if model is None:
            if self.preprocess_type == 'lstm':
                print('Data Utils Loading LSTM Baseline Model')
                model = LSTMVAE(512, 128, 2, device=torch.device('cpu'), use_attention=True).to(torch.device('cpu'))
            else:
                raise ValueError('mark')

    def __iter__(self):

        with h5py.File(self.data_file, 'r') as file:
            all_np_seq = file['%s_in_seq' % (self.mode)].transpose(0, 2, 1)
            all_label = file['%s_out' % (self.mode)]
        self.size = all_np_seq.shape[0]

        if self.shuffle:
            shuffled_idx = np.random.permutation(self.size)
            all_np_seq = all_np_seq[shuffled_idx]
            all_label = all_label[shuffled_idx]

        batches = [[all_np_seq[i: i + self.batch_size], all_label[i: i + self.batch_size]] for i in
                   range(0, self.size, self.batch_size)]
        if len(batches[-1]) < self.batch_size:
            batches.pop()

        dataset = RBPDataset(batches, self.preprocess_type, self.weight_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                collate_fn=lambda x: x[0])

        for b in dataloader:
            yield b


class RBPDataset(Dataset):

    def __init__(self, data, preprocess_type, weight_path):
        self.data = data
        self.preprocess_type = preprocess_type
        self.weight_path = weight_path
        global encoder
        encoder.load_state_dict(
            torch.load(self.weight_path,
                       map_location=torch.device('cpu'))['model_weights'])
        self.encoder = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        with torch.no_grad():

            if self.preprocess_type == 'lstm':
                all_joint_encoding = []
                for all_np_seq, all_label in self.data[idx]:
                    seq = ''.join([RBP_DATASET_NUC_VOCAB[nuc_idx] for nuc_idx in np.argmax(all_np_seq, axis=-1)])
                    struct = RNA.fold(seq)[0]

                    joint_encoding = []
                    for seq_char, struct_char in zip(seq, struct):
                        onehot_enc = np.array(list(map(lambda x: x == seq_char + struct_char, JOINT_VOCAB)), dtype=np.float32)
                        joint_encoding.append(onehot_enc)
                    all_joint_encoding.append(joint_encoding)

                latent_vec = model.encode(all_joint_encoding)
            else:
                raise ValueError('Not supported yet..')

        return latent_vec, all_label
