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

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

allowed_basepairs = [[False, False, False, True],
                     [False, False, True, False],
                     [False, True, False, True],
                     [True, False, True, False]]


class BasicLSTMVAEFolder:

    def __init__(self, data_folder, batch_size, num_workers=4, shuffle=True):
        self.data_folder = data_folder
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
        return self.data[idx], all_joint_encoding, all_label, all_free_energy


class LSTMEncoder(nn.Module):

    def __init__(self, hidden_size, depth, **kwargs):
        super(LSTMEncoder, self).__init__()
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_size = hidden_size
        self.depth = depth
        self.use_attention = kwargs.get('use_attention', False)

        self.lstm = torch.nn.LSTM(FDIM_JOINT_VOCAB, hidden_size, bidirectional=True, batch_first=True,
                                  num_layers=self.depth)

        if self.use_attention:
            self.nb_heads = kwargs.get('nb_heads', 4)
            self.attention_layer = nn.MultiheadAttention(self.hidden_size * 2, self.nb_heads)

    def forward(self, sequence_batch):
        batch_size = len(sequence_batch)
        all_len = [len(seq) for seq in sequence_batch]
        max_len = max(all_len)
        all_pre_padding_idx = np.concatenate(
            [np.array(list(range(length))) + i * max_len for i, length in enumerate(all_len)]).astype(np.long)

        padded_seq_vec = nn.utils.rnn.pad_sequence([torch.as_tensor(np.array(seq)) for seq in sequence_batch],
                                                   batch_first=True).to(self.device)
        # [batch_size, max_len, FDIM_JOINT_VOCAB]

        packed_seq_vec = nn.utils.rnn.pack_padded_sequence(
            padded_seq_vec, all_len, enforce_sorted=False, batch_first=True)

        output, _ = self.lstm(packed_seq_vec)

        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        if self.use_attention:
            batch_second_output = output.transpose(0, 1)
            key_padding_mask = torch.as_tensor(
                [[False] * all_len[i] + [True] * (max_len - all_len[i]) for i in range(batch_size)]).to(self.device)
            batch_second_output = self.attention_layer(
                batch_second_output, batch_second_output, batch_second_output,
                key_padding_mask=key_padding_mask)[0]
            output = batch_second_output.transpose(0, 1)

        nuc_embedding = output.reshape(batch_size * max_len, self.hidden_size * 2). \
            index_select(0, torch.as_tensor(all_pre_padding_idx).to(self.device))

        representation = []
        start = 0
        for length in all_len:
            representation.append(torch.mean(nuc_embedding[start: start + length], dim=0))
            start += length

        return torch.stack(representation, dim=0)


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
        self.nuc_pred_loss = nn.CrossEntropyLoss(reduction='sum')

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
        nuc_pred_loss = self.nuc_pred_loss(all_hidden_states, all_seq_label_deviced) / nb_nuc_targets
        _, preds = torch.max(all_hidden_states, dim=1)
        preds = preds.cpu().numpy()

        # stop translation (segment) symbol
        stop_symbol_loc = all_seq_label == FDIM_JOINT_VOCAB_DECODING - 1
        nb_stop_symbol = np.sum(stop_symbol_loc)
        nb_stop_symbol_correct = np.sum(preds[stop_symbol_loc] == FDIM_JOINT_VOCAB_DECODING - 1)
        stop_symbol_acc = nb_stop_symbol_correct / nb_stop_symbol

        # nucleotide accuracy
        ord_symbol_loc = np.logical_not(stop_symbol_loc)
        nb_ord_symbol = np.sum(ord_symbol_loc)
        nuc_pred_acc = np.sum(preds[ord_symbol_loc] // LEN_STRUCT_VOCAB ==
                              all_seq_label[ord_symbol_loc] // LEN_STRUCT_VOCAB) / nb_ord_symbol

        # structural accuracy
        struct_pred_acc = np.sum(preds[ord_symbol_loc] % LEN_STRUCT_VOCAB ==
                                 all_seq_label[ord_symbol_loc] % LEN_STRUCT_VOCAB) / nb_ord_symbol

        # all translation predictions
        all_acc = np.sum(preds == all_seq_label) / nb_nuc_targets

        return nuc_pred_loss, stop_symbol_acc, nuc_pred_acc, struct_pred_acc, all_acc

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
                            for disallowed_nuc_idx in np.where(allowed_basepairs[last_nonclosed_nuc_item] is False):
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


class LSTMVAE(nn.Module):

    def __init__(self, hidden_size, latent_size, depthEncoder, **kwargs):
        super(LSTMVAE, self).__init__()

        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depthEncoder = depthEncoder

        self.encoder = LSTMEncoder(self.hidden_size, self.depthEncoder, **kwargs)
        self.mean = nn.Linear(2 * hidden_size, latent_size)
        self.var = nn.Linear(2 * hidden_size, latent_size)

        self.decoder = LSTMDecoder(self.hidden_size, self.latent_size, **kwargs)
        self.regressor_nonlinear = nn.Linear(2 * hidden_size, hidden_size)
        self.regressor_output = nn.Linear(hidden_size, 1)
        self.normed_fe_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def encode(self, sequence_batch):
        latent_vec = self.encoder(sequence_batch)
        return latent_vec

    def rsample(self, latent_vec):
        batch_size = latent_vec.size(0)
        z_mean = self.mean(latent_vec)
        z_log_var = -torch.abs(self.var(latent_vec))  # Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).to(self.device)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss


    def aux_regressor(self, latent_vec, fe_target):
        batch_size = len(fe_target)
        predicted_fe = self.regressor_output(self.regressor_nonlinear(latent_vec))[:, 0]
        normed_fe_loss = self.normed_fe_loss(
            predicted_fe,
            torch.as_tensor(np.array(fe_target, dtype=np.float32)).to(self.device)) \
                         / batch_size
        normed_fe_corr = pearsonr(predicted_fe.cpu().detach().numpy(), fe_target)[0]
        return normed_fe_loss, normed_fe_corr


    def forward(self, sequence_batch, sequence_label, fe_target, beta):
        latent_vec = self.encode(sequence_batch)

        normed_fe_loss, normed_fe_corr = self.aux_regressor(latent_vec, fe_target)

        latent_vec, kl_loss = self.rsample(latent_vec)

        nuc_pred_loss, stop_symbol_acc, nuc_pred_acc, struct_pred_acc, all_acc = \
            self.decoder(sequence_batch, latent_vec, sequence_label)

        return nuc_pred_loss + 0.1 * normed_fe_loss + beta * kl_loss, kl_loss, normed_fe_loss, \
               (normed_fe_corr, stop_symbol_acc, nuc_pred_acc, struct_pred_acc, all_acc)
