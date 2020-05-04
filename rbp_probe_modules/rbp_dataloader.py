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

    def __init__(self, dataset_name, batch_size, num_workers=4, shuffle=True, device=None,
                 mode='train', preprocess_type='lstm', weight_path=None, mp_pool=None):
        assert dataset_name in dataset_options, \
            'dataset name {} not found in {}'.format(dataset_name, dataset_options)
        assert mode in ['train', 'valid', 'test'], \
            'mode {} not found in {}'.format(mode, ['train', 'valid', 'test'])
        assert preprocess_type in ['lstm', 'graph_lstm', 'jtvae'], \
            'preprocess type {} not found in {}'.format(preprocess_type, ['lstm', 'graph_lstm', 'jtvae'])
        assert os.path.exists(weight_path), '%s doesn\'t exists'
        self.data_file = datapath.format(dataset_name)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.mode = mode
        self.mp_pool = mp_pool

        self.preprocess_type = preprocess_type
        self.weight_path = weight_path

        global model
        if model is None:
            if self.preprocess_type == 'lstm':
                print('Data Utils Loading LSTM Baseline Model')
                model = LSTMVAE(512, 128, 2, device=self.device, use_attention=True).to(self.device)
                model.load_state_dict(
                    torch.load(self.weight_path)['model_weights'])
            else:
                raise ValueError('mark')
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

    def __iter__(self):

        if not hasattr(self, 'all_latent_vec'):
            print('Converting raw RNAs to embeddings')
            with h5py.File(self.data_file, 'r') as file:
                all_np_seq = np.array(file['%s_in_seq' % (self.mode)]).transpose(0, 2, 1)
                all_label = np.array(file['%s_out' % (self.mode)])
            self.size = all_np_seq.shape[0]

            if self.preprocess_type == 'lstm':
                # obtain all secondary structures
                from tqdm import tqdm
                all_joint_encodings = list(tqdm(self.mp_pool.imap(RBPFolder.lstm_joint_encoding_subroutine, all_np_seq), total=self.size))
                batches = [all_joint_encodings[i: i + 1000] for i in range(0, self.size, 1000)]

                all_latent_vec = []
                with torch.no_grad():
                    from tqdm import tqdm
                    for joint_encoding in tqdm(batches, total=len(batches)):
                        latent_vec = self.model.encode(joint_encoding)
                        z_vec = self.model.mean(latent_vec)
                        all_latent_vec.append(z_vec)
            else:
                raise ValueError('mark')

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

