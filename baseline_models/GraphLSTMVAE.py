import os
import sys
import itertools
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from scipy.stats import pearsonr
import forgi.graph.bulge_graph as fgb

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.nn_utils import log_sum_exp, index_select_ND
from cnf_models.flow import get_latent_cnf

NUC_VOCAB = ['A', 'C', 'G', 'U']
LEN_NUC_VOCAB = len(NUC_VOCAB)
STRUCT_VOCAB = ['(', ')', '.']
LEN_STRUCT_VOCAB = len(STRUCT_VOCAB)

JOINT_VOCAB = [''.join(cand) for cand in itertools.product(NUC_VOCAB, STRUCT_VOCAB)]
FDIM_JOINT_VOCAB = len(JOINT_VOCAB)

FDIM_JOINT_VOCAB_DECODING = FDIM_JOINT_VOCAB + 1  # 1 extra dimension for the stop symbol
MAX_DECODE_LENGTH = 1000
MIN_HAIRPIN_LEN = 3
MAX_FE = 0.85

allowed_basepairs = np.array([[False, False, False, True],
                              [False, False, True, False],
                              [False, True, False, True],
                              [True, False, True, False]])

NUC_FDIM = 4
BOND_FDIM = 4
# 5' to 3' covalent bond,
# 3' to 5' covalent bond,
# 5' to 3' bp bond,
# 3' to 5' bp bond

MAX_NB = 3


# maximal number of incoming messages


class BasicLSTMVAEFolder:

    def __init__(self, data_folder, batch_size, num_workers=4, shuffle=True, limit_data=None):
        self.data_folder = data_folder
        self.limit_data = limit_data
        if self.limit_data:
            assert type(self.limit_data) is int, '\'limit_data\' should either be None or an integer'
            self.data_files = [fn for fn in os.listdir(data_folder) if
                               int(fn.split('-')[-1].split('.')[0]) <= self.limit_data]
        else:
            self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)

            batches = [[(''.join(rna.rna_seq), ''.join(rna.rna_struct), rna.free_energy)
                        for rna in data[i: i + self.batch_size]]
                       for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = LSTMBaselineDataset(batches)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                    collate_fn=lambda x: x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class LSTMBaselineDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  # joint encoding of the structure and sequence
        graph_encoder_input = GraphEncoder.prepare_batch_data([data[:2] for data in self.data[idx]])
        all_joint_encoding = []
        all_label = []
        all_free_energy = []
        for seq, struct, free_energy in self.data[idx]:
            joint_encoding = []
            label = []
            for seq_char, struct_char in zip(seq, struct):
                onehot_enc = np.array(list(map(lambda x: x == seq_char + struct_char, JOINT_VOCAB)), dtype=np.float32)
                joint_encoding.append(onehot_enc)
                label.append(np.argmax(onehot_enc))
            all_joint_encoding.append(joint_encoding)
            all_label.append(label)
            all_free_energy.append(np.abs(free_energy / len(seq)) / MAX_FE)  # length normalized minimum free energy
        return self.data[idx], all_joint_encoding, all_label, all_free_energy, graph_encoder_input


class GraphEncoder(nn.Module):

    def __init__(self, hidden_size, depth, **kwargs):
        super(GraphEncoder, self).__init__()
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_size = hidden_size
        self.depth = depth

        self.w_local = nn.Linear(NUC_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.w_msg = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_node_emb = nn.Linear(hidden_size + NUC_FDIM, hidden_size, bias=False)
        self.update_gru = nn.GRUCell(hidden_size, hidden_size)

        self.nuc_order_lstm = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)
        # bidirectional hence hidden_size//2

    def send_to_device(self, *args):
        ret = []
        for item in args:
            ret.append(item.to(self.device))
        return ret

    def forward(self, f_nuc, f_bond, node_graph, message_graph, all_bonds, scope):

        # f_nuc is included in f_bond
        f_nuc, f_bond, node_graph, message_graph = \
            self.send_to_device(f_nuc, f_bond, node_graph, message_graph)

        local_potentials = self.w_local(f_bond)
        # messages from the first iteration
        messages = torch.relu(local_potentials)
        mask = torch.ones(messages.size(0), 1).to(self.device)
        mask[0] = 0  # first vector is padding

        for i in range(1, self.depth):
            nei_message = index_select_ND(messages, 0, message_graph)
            sum_nei_message = nei_message.sum(dim=1)
            nb_clique_msg_prop = self.w_msg(sum_nei_message)
            new_msg = torch.relu(local_potentials + nb_clique_msg_prop)
            messages = self.update_gru(new_msg, messages)
            # new_msg = torch.relu(self.w_msg(torch.cat([local_potentials, sum_nei_message], dim=-1)))
            # messages = self.update_gru(new_msg, messages)
            messages = messages * mask

        nuc_nb_msg = index_select_ND(messages, 0, node_graph).sum(dim=1)
        nuc_embedding = torch.relu(self.w_node_emb(torch.cat([f_nuc, nuc_nb_msg], dim=1)))

        ''' bilstm to add order information into the learnt node embeddings '''
        batch_size = len(scope)
        all_len = list(np.array(scope)[:, 1])
        max_len = max(all_len)
        all_pre_padding_idx = np.concatenate(
            [np.array(list(range(length))) + i * max_len for i, length in enumerate(all_len)]).astype(np.long)

        batch_rna_vec = []
        for start_idx, length in scope:
            batch_rna_vec.append(nuc_embedding[start_idx: start_idx + length])

        # [batch_size, max_len, hidden_size]
        padded_rna_vec = nn.utils.rnn.pad_sequence(batch_rna_vec, batch_first=True)
        packed_rna_vec = nn.utils.rnn.pack_padded_sequence(
            padded_rna_vec, all_len, enforce_sorted=False, batch_first=True)

        output, (hn, cn) = self.nuc_order_lstm(packed_rna_vec)
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        nuc_embedding = output.reshape(batch_size * max_len, self.hidden_size). \
            index_select(0, torch.as_tensor(all_pre_padding_idx).to(self.device))

        representation = []
        start = 0
        for length in all_len:
            representation.append(torch.max(nuc_embedding[start: start + length], dim=0)[0])
            start += length

        return torch.stack(representation, dim=0)

    @staticmethod
    def prepare_batch_data(rna_mol_batch):
        # sparse encoding, index operations

        nuc_offset = 0

        f_nuc, f_bond = [], [torch.zeros(NUC_FDIM + BOND_FDIM)]
        # nucleotide features and bond features, merely one hot encoding at this stage

        in_bonds, all_bonds = [], [(-1, -1)]
        # keeps the indices of incoming messages

        scope = []

        max_len = max([len(pair[0]) for pair in rna_mol_batch])
        '''positional encoding'''
        pe = torch.zeros(max_len, LEN_NUC_VOCAB, dtype=torch.float32)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, LEN_NUC_VOCAB, 2) *
                             -(np.log(10000.0) / LEN_NUC_VOCAB))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        for rna_seq, rna_struct in rna_mol_batch:
            len_seq = len(rna_seq)
            for i, nuc in enumerate(rna_seq):
                onehot_enc = np.array(list(map(lambda x: x == nuc, NUC_VOCAB)), dtype=np.float32)
                f_nuc.append(torch.as_tensor(onehot_enc) + pe[i])
                in_bonds.append([])

            # authentic molecular graph
            bg = fgb.BulgeGraph.from_dotbracket(rna_struct)
            for i, st_ele in enumerate(rna_struct):
                # covalent bonds
                cb_from = i + nuc_offset
                if i < len_seq - 1:  # 5' to 3' covalent bond
                    cb_to = cb_from + 1
                    idx_ref = len(all_bonds)
                    all_bonds.append([cb_from, cb_to])
                    f_bond.append(torch.cat(
                        [f_nuc[cb_from], torch.as_tensor(
                            np.array([1., 0., 0., 0.], dtype=np.float32))]))
                    in_bonds[cb_to].append(idx_ref)
                if i > 0:  # 3' to 5' covalent bond
                    cb_to = cb_from - 1
                    idx_ref = len(all_bonds)
                    all_bonds.append([cb_from, cb_to])
                    f_bond.append(torch.cat(
                        [f_nuc[cb_from], torch.as_tensor(
                            np.array([0., 1., 0., 0.], dtype=np.float32))]))
                    in_bonds[cb_to].append(idx_ref)

                # base-pairing
                if st_ele != '.':
                    bp_from = i + nuc_offset
                    bp_to = bg.pairing_partner(i + 1) - 1 + nuc_offset
                    idx_ref = len(all_bonds)
                    all_bonds.append([bp_from, bp_to])
                    if bp_to > bp_from:
                        onehot_enc = torch.as_tensor(
                            np.array([0., 0., 1., 0.], dtype=np.float32))
                    else:
                        onehot_enc = torch.as_tensor(
                            np.array([0., 0., 0., 1.], dtype=np.float32))
                    f_bond.append(torch.cat([f_nuc[bp_from], onehot_enc]))
                    in_bonds[bp_to].append(idx_ref)

            scope.append((nuc_offset, len_seq))
            nuc_offset += len_seq

        total_nuc = nuc_offset
        total_bonds = len(all_bonds)
        f_nuc = torch.stack(f_nuc)
        f_bond = torch.stack(f_bond)

        node_graph = torch.zeros(total_nuc, MAX_NB, dtype=torch.long)
        # keeps a list of indices of incoming messages for the update of a node-embedding

        message_graph = torch.zeros(total_bonds, MAX_NB, dtype=torch.long)
        # indices needed for the update of each message

        for nuc_idx in range(total_nuc):
            for i, msg_idx in enumerate(in_bonds[nuc_idx]):
                node_graph[nuc_idx, i] = msg_idx

        for bond_idx in range(1, total_bonds):
            nuc_idx_from, nuc_idx_to = all_bonds[bond_idx]
            for i, msg_idx in enumerate(in_bonds[nuc_idx_from]):
                # if all_bonds[msg_idx][0] != nuc_idx_to:
                message_graph[bond_idx, i] = msg_idx

        return f_nuc, f_bond, node_graph, message_graph, all_bonds, scope


class LSTMDecoder(nn.Module):

    def __init__(self, hidden_size, latent_size, **kwargs):
        super(LSTMDecoder, self).__init__()
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.latent_vec_linear = nn.Linear(self.latent_size, self.hidden_size)
        self.lstm_cell = nn.LSTMCell(FDIM_JOINT_VOCAB + 1, hidden_size)
        self.W_nonlinear = nn.Linear(hidden_size + latent_size, hidden_size)
        self.W = nn.Linear(hidden_size, FDIM_JOINT_VOCAB_DECODING)
        self.nuc_pred_loss = nn.CrossEntropyLoss(reduction='none')

    def aggregate(self, hiddens):
        return self.W(torch.relu(self.W_nonlinear(hiddens)))

    def forward(self, sequence_batch, latent_vec, sequence_label):
        all_seq_input = []
        for sequence in sequence_batch:
            dim_expanded_seq = np.concatenate([np.array(sequence), np.zeros((len(sequence), 1), dtype=np.float32)],
                                              axis=1)
            # length x FDIM
            all_seq_input.append(np.concatenate([
                np.zeros((1, FDIM_JOINT_VOCAB_DECODING), dtype=np.float32),
                dim_expanded_seq], axis=0))
            # for the start decoding symbol

        all_len = [len(s_in) for s_in in all_seq_input]
        all_hidden_states = self.teacher_forced_decoding(all_seq_input, latent_vec)

        all_seq_label = []
        for label in sequence_label:
            all_seq_label.extend(label)
            all_seq_label.append(FDIM_JOINT_VOCAB_DECODING - 1)

        all_seq_label = np.array(all_seq_label)
        all_seq_label_deviced = torch.as_tensor(all_seq_label).to(self.device)
        all_hidden_states = self.aggregate(all_hidden_states)

        # Predict nucleotides
        nb_nuc_targets = all_seq_label_deviced.size(0)
        nuc_pred_loss = self.nuc_pred_loss(all_hidden_states, all_seq_label_deviced)
        batch_loss = []
        start = 0
        for inc in all_len:
            batch_loss.append(torch.sum(nuc_pred_loss[start: start + inc]))
            start += inc

        _, preds = torch.max(all_hidden_states, dim=1)
        preds = preds.cpu().numpy()

        # stop translation (segment) symbol
        stop_symbol_loc = all_seq_label == FDIM_JOINT_VOCAB_DECODING - 1
        nb_stop_symbol = np.sum(stop_symbol_loc)
        nb_stop_symbol_correct = np.sum(preds[stop_symbol_loc] == FDIM_JOINT_VOCAB_DECODING - 1)

        # nucleotide accuracy
        ord_symbol_loc = np.logical_not(stop_symbol_loc)
        nb_ord_symbol = np.sum(ord_symbol_loc)
        nb_ord_symbol_correct = np.sum(preds[ord_symbol_loc] // LEN_STRUCT_VOCAB ==
                                       all_seq_label[ord_symbol_loc] // LEN_STRUCT_VOCAB)

        # structural accuracy
        nb_struct_symbol_correct = np.sum(preds[ord_symbol_loc] % LEN_STRUCT_VOCAB ==
                                          all_seq_label[ord_symbol_loc] % LEN_STRUCT_VOCAB)

        # all translation predictions
        nb_all_correct = np.sum(preds == all_seq_label)

        return {
            'sum_nuc_pred_loss': torch.sum(nuc_pred_loss),
            'batch_nuc_pred_loss': torch.as_tensor(batch_loss).to(self.device),
            'nb_nuc_targets': nb_nuc_targets,
            'nb_stop_symbol_correct': nb_stop_symbol_correct,
            'nb_stop_symbol': nb_stop_symbol,
            'nb_ord_symbol_correct': nb_ord_symbol_correct,
            'nb_ord_symbol': nb_ord_symbol,
            'nb_struct_symbol_correct': nb_struct_symbol_correct,
            'nb_all_correct': nb_all_correct
        }

    def teacher_forced_decoding(self, all_seq_input, latent_vec):
        batch_size = len(all_seq_input)
        all_len = [len(seq_input) for seq_input in all_seq_input]
        max_len = max(all_len)

        padded_seq_input = []
        for seq_input in all_seq_input:
            # paddings
            padded_seq_input.append(
                np.concatenate(
                    [seq_input, np.zeros((max_len - len(seq_input), FDIM_JOINT_VOCAB_DECODING), dtype=np.float32)],
                    axis=0))
        all_seq_input = torch.as_tensor(np.array(padded_seq_input)).to(self.device)

        all_pre_padding_idx = np.concatenate(
            [np.array(list(range(length))) + i * max_len for i, length in enumerate(all_len)]).astype(np.long)

        batch_idx = np.concatenate([[i] * length for i, length in enumerate(all_len)]).astype(np.long)

        all_hidden_states = []
        cell_memory = torch.zeros(batch_size, self.hidden_size).to(self.device)
        # hidden_states = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32).to(self.device)
        hidden_states = torch.relu(self.latent_vec_linear(latent_vec))
        for len_idx in range(max_len):
            hidden_states, cell_memory = self.lstm_cell(all_seq_input[:, len_idx, :],
                                                        (hidden_states, cell_memory))
            all_hidden_states.append(hidden_states)
        all_hidden_states = torch.stack(all_hidden_states, dim=1).view(-1, self.hidden_size)

        all_hidden_states = all_hidden_states.index_select(0, torch.as_tensor(all_pre_padding_idx).to(self.device))
        all_latent_vec = latent_vec.index_select(0, torch.as_tensor(batch_idx).to(self.device))

        all_hidden_states = torch.cat([all_hidden_states, all_latent_vec], dim=1)

        return all_hidden_states

    def decode(self, latent_vector, prob_decode=False, enforce_rna_prior=False):
        # enforce_rna_prior here means three things:
        # 1. closed left and right brackets
        # 2. closed left and right brackets shall be reversely complementary
        # 3. hairpins should have at least 3 nucleotides
        decode_step = 0
        batch_size = latent_vector.size(0)
        last_token = torch.zeros(batch_size, FDIM_JOINT_VOCAB_DECODING).to(self.device)
        cell_memory = torch.zeros(batch_size, self.hidden_size).to(self.device)

        # hidden_state = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32).to(self.device)
        hidden_state = torch.relu(self.latent_vec_linear(latent_vector))
        decoded_sequence = [''] * batch_size
        decoded_structure = [''] * batch_size
        batch_idx = np.array(list(range(batch_size)), dtype=np.long)

        nonclosed_nuc_stack = []
        for _ in range(batch_size):
            nonclosed_nuc_stack.append([])

        while decode_step < MAX_DECODE_LENGTH:

            hidden_state, cell_memory = self.lstm_cell(last_token, (hidden_state, cell_memory))
            all_hidden_states = torch.cat([hidden_state, latent_vector], dim=1)
            nuc_pred_score = self.aggregate(all_hidden_states)

            if enforce_rna_prior:
                mask = np.zeros((batch_size, FDIM_JOINT_VOCAB_DECODING), dtype=np.float32)
                for i, idx in enumerate(batch_idx):
                    if len(nonclosed_nuc_stack[idx]) > 0:
                        last_nonclosed_nuc_item, last_decode_step = nonclosed_nuc_stack[idx][-1]
                        if decode_step - last_decode_step <= MIN_HAIRPIN_LEN:
                            for j in range(LEN_NUC_VOCAB):
                                mask[i][j * LEN_STRUCT_VOCAB + 1] = -np.inf
                        else:
                            for disallowed_nuc_idx in np.where(allowed_basepairs[last_nonclosed_nuc_item] == False)[
                                0]:  # blunder corrected
                                mask[i][disallowed_nuc_idx * LEN_STRUCT_VOCAB + 1] = -np.inf
                                # intuition: if you have to choose right bracket, don't select those nucleotides that
                                # can't be paired with the last non-closed nucleotide
                        mask[i][-1] = -np.inf  # forbid end of decoding
                    else:
                        for j in range(LEN_NUC_VOCAB):
                            mask[i][j * LEN_STRUCT_VOCAB + 1] = -np.inf
                            # forbid any right brackets
                mask = torch.as_tensor(mask).to(self.device)

                if prob_decode:
                    nuc_idx = torch.multinomial(torch.softmax(nuc_pred_score + mask, dim=1), num_samples=1)[:, 0]
                else:
                    _, nuc_idx = torch.max(nuc_pred_score + mask, dim=1)

                cont_translation_idx = torch.where(nuc_idx != FDIM_JOINT_VOCAB_DECODING - 1)[0]

                nuc_idx = nuc_idx.cpu().numpy()

                for i, idx in enumerate(nuc_idx):
                    if idx == FDIM_JOINT_VOCAB_DECODING - 1:
                        continue
                    if idx % LEN_STRUCT_VOCAB == 0:  # a left bracket
                        nonclosed_nuc_stack[batch_idx[i]].append((idx // LEN_STRUCT_VOCAB, decode_step))
                    elif idx % LEN_STRUCT_VOCAB == 1:
                        nonclosed_nuc_stack[batch_idx[i]].pop()
            else:
                if prob_decode:
                    nuc_idx = torch.multinomial(torch.softmax(nuc_pred_score, dim=1), num_samples=1)[:, 0]
                else:
                    _, nuc_idx = torch.max(nuc_pred_score, dim=1)

                # identify sequences that should continue the translation
                cont_translation_idx = torch.where(nuc_idx != FDIM_JOINT_VOCAB_DECODING - 1)[0]

                nuc_idx = nuc_idx.cpu().numpy()

            if cont_translation_idx.size(0) == 0:
                break

            # those that remains in the translation
            hidden_state = hidden_state.index_select(0, cont_translation_idx)
            cell_memory = cell_memory.index_select(0, cont_translation_idx)
            latent_vector = latent_vector.index_select(0, cont_translation_idx)
            batch_size = cont_translation_idx.size(0)

            cont_translation_idx = cont_translation_idx.cpu().numpy()
            batch_idx = batch_idx[cont_translation_idx]

            for i, idx in enumerate(cont_translation_idx):  # ignore the stop symbols
                decoded_sequence[batch_idx[i]] += JOINT_VOCAB[nuc_idx[idx]][0]
                decoded_structure[batch_idx[i]] += JOINT_VOCAB[nuc_idx[idx]][1]

            last_token = np.array([list(map(lambda x: x == idx, range(FDIM_JOINT_VOCAB_DECODING)))
                                   for idx in nuc_idx[cont_translation_idx]], dtype=np.float32)
            last_token = torch.as_tensor(last_token).to(self.device)

            decode_step += 1

        return decoded_sequence, decoded_structure


class GraphLSTMVAE(nn.Module):

    def __init__(self, hidden_size, latent_size, depthEncoder, **kwargs):
        super(GraphLSTMVAE, self).__init__()

        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depthEncoder = depthEncoder
        self.use_aux_regressor = kwargs.get('use_aux_regressor', True)
        self.use_flow_prior = kwargs.get('use_flow_prior', True)

        self.encoder = GraphEncoder(self.hidden_size, self.depthEncoder, **kwargs)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

        self.decoder = LSTMDecoder(self.hidden_size, self.latent_size, **kwargs)
        if self.use_aux_regressor:
            self.regressor_nonlinear = nn.Linear(hidden_size, hidden_size)
            self.regressor_output = nn.Linear(hidden_size, 1)
            self.normed_fe_loss = nn.BCEWithLogitsLoss(reduction='sum')

        if self.use_flow_prior:
            self.flow_args = {
                'latent_dims': "256-256",
                'latent_num_blocks': 1,
                'zdim': latent_size,
                'layer_type': 'concatsquash',
                'nonlinearity': 'tanh',
                'time_length': 0.5,
                'train_T': True,
                'solver': 'dopri5',
                'use_adjoint': True,
                'atol': 1e-5,
                'rtol': 1e-5,
                'batch_norm': False,
                'bn_lag': 0,
                'sync_bn': False,
                'device': self.device
            }
            self.latent_cnf = get_latent_cnf(self.flow_args)

    def encode(self, batch_graph_input):
        latent_vec = self.encoder(*batch_graph_input)
        return latent_vec

    def rsample(self, latent_vec, nsamples=1):
        batch_size = latent_vec.size(0)
        z_mean = self.mean(latent_vec)
        z_log_var = -torch.abs(self.var(latent_vec))  # Following Mueller et al.

        entropy = self.gaussian_entropy(z_log_var)  # batch_size,
        z_vecs = self.reparameterize(z_mean, z_log_var, nsamples).reshape(batch_size * nsamples, self.latent_size)

        if self.use_flow_prior:
            w, delta_log_pw = self.latent_cnf(z_vecs, None, torch.zeros(batch_size * nsamples, 1).to(z_vecs))
            log_pw = self.standard_normal_logprob(w).reshape(batch_size, nsamples, 1)
            delta_log_pw = delta_log_pw.reshape(batch_size, nsamples, 1)
            log_pz = log_pw - delta_log_pw
        else:
            log_pz = self.standard_normal_logprob(z_vecs).reshape(batch_size, nsamples, 1)

        return z_vecs.reshape(batch_size, nsamples, self.latent_size), (entropy, log_pz)

    def reparameterize(self, mean, logvar, nsamples=1):
        batch_size, nz = mean.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mean.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_().to(self.device)

        return mu_expd + torch.mul(eps, std_expd)

    def standard_normal_logprob(self, z):
        dim = z.size(-1)
        log_z = -0.5 * dim * np.log(2 * np.pi)
        return log_z - 0.5 * torch.sum(z.pow(2), dim=-1, keepdim=True)

    def gaussian_entropy(self, logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def calc_mi(self, batch_graph_input, latent_vec=None):
        """Approximate the mutual information between x and z under the approximate posterior
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
        Returns: Float
        """
        # [x_batch, nz]
        if latent_vec is None:
            latent_vec = self.encode(*batch_graph_input)
        z_mean = self.mean(latent_vec)
        z_log_var = -torch.abs(self.var(latent_vec))  # Following Mueller et al.
        x_batch, nz = z_mean.size()
        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * np.log(2 * np.pi) - 0.5 * (1 + z_log_var).sum(-1)).mean()
        # [z_batch, 1, nz]
        z_samples = self.reparameterize(z_mean, z_log_var, nsamples=1)
        # [1, x_batch, nz]
        mu, logvar = z_mean.unsqueeze(0), z_log_var.unsqueeze(0)
        var = logvar.exp()
        # (z_batch, x_batch, nz)
        dev = z_samples - mu
        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * np.log(2 * np.pi) + logvar.sum(-1))
        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - np.log(x_batch)
        return (neg_entropy - log_qz.mean(-1)).item()

    def eval_inference_dist(self, batch_graph_input, z_vec, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z_vec.size(2)

        if not param:
            latent_vec = self.encode(*batch_graph_input)
            mu, logvar = self.mean(latent_vec), -torch.abs(self.var(latent_vec))
        else:
            mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z_vec - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * np.log(2 * np.pi) + logvar.sum(-1))

        return log_density

    def nll_iw(self, sequence_batch, sequence_label, batch_graph_input, nsamples, ns=100, latent_vec=None):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10
        tmp = []
        batch_size = len(sequence_batch)
        if latent_vec is None:
            latent_vec = self.encode(*batch_graph_input)
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]
            z_vec, (entropy, log_pz) = self.rsample(latent_vec, ns)

            # [batch, ns], log p(x,z)
            z_vec_reshaped = z_vec.reshape(batch_size * ns, self.latent_size)
            rep_seq_batch = [i for seq in sequence_batch for i in [seq] * ns]
            rep_label_batch = [i for label in sequence_label for i in [label] * ns]
            ret_dict = self.decoder(rep_seq_batch, z_vec_reshaped, rep_label_batch)
            recon_log_prob = - ret_dict['batch_nuc_pred_loss'].reshape(batch_size, ns)
            log_comp_ll = log_pz[:, :, 0] + recon_log_prob

            # log q(z|x)
            log_infer_ll = self.eval_inference_dist(batch_graph_input, z_vec,
                                                    param=(self.mean(latent_vec),
                                                           -torch.abs(self.var(latent_vec))))

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - np.log(nsamples)

        return -ll_iw

    def aux_regressor(self, latent_vec, fe_target):
        batch_size = len(fe_target)
        '''note: nonlinearity added'''
        predicted_fe = self.regressor_output(torch.relu(self.regressor_nonlinear(latent_vec)))[:, 0]
        normed_fe_loss = self.normed_fe_loss(
            predicted_fe,
            torch.as_tensor(np.array(fe_target, dtype=np.float32)).to(self.device)) \
                         / batch_size
        preds = torch.sigmoid(predicted_fe).cpu().detach().numpy()
        if np.any(np.isfinite(preds) == False):
            print('NAN/inf in pearson correlation!', flush=True)
            torch.save(self.state_dict(), 'lstm_baseline_output/blunder')
            import pdb
            pdb.set_trace()
            valid_idx = np.isfinite(preds)
            if sum(valid_idx) > 0:
                normed_fe_corr = pearsonr(preds[valid_idx], np.array(fe_target)[valid_idx])[0]
            else:
                normed_fe_corr = 0.
        else:
            if len(set(preds)) == 1:
                normed_fe_corr = 0.
                print('pearson warning', flush=True)
                torch.save(self.state_dict(), 'lstm_baseline_output/new_blunder')
                import pdb
                pdb.set_trace()
            else:
                normed_fe_corr = pearsonr(preds, fe_target)[0]

        return normed_fe_loss, normed_fe_corr

    def forward(self, sequence_batch, sequence_label, fe_target, batch_graph_input):
        latent_vec = self.encode(batch_graph_input)

        if self.use_aux_regressor:
            normed_fe_loss, normed_fe_corr = self.aux_regressor(latent_vec, fe_target)

        latent_vec, (entropy, log_pz) = self.rsample(latent_vec, nsamples=1)
        latent_vec = latent_vec[:, 0, :]  # squeeze

        ret_dict = self.decoder(sequence_batch, latent_vec, sequence_label)

        ret_dict['entropy_loss'] = -entropy.mean()
        ret_dict['prior_loss'] = -log_pz.mean()
        if self.use_aux_regressor:
            ret_dict['normed_fe_loss'] = normed_fe_loss
            ret_dict['normed_fe_corr'] = normed_fe_corr

        return ret_dict
