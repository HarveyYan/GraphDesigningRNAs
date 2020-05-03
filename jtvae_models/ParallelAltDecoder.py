# This model simultaneously decodes the junction tree
# as well as the nucleotides associated within each subgraph

import torch
import torch.nn as nn
import numpy as np
from lib.tree_decomp import RNAJunctionTree, RNAJTNode

# '<' to signal stop translation
NUC_VOCAB = ['A', 'C', 'G', 'U', '<']
HYPERGRAPH_VOCAB = ['H', 'I', 'M', 'S']

# G-U pairs are allowed, A-A pairs are not allowed
allowed_basepairs = [[False, False, False, True],
                     [False, False, True, False],
                     [False, True, False, True],
                     [True, False, True, False]]

# hairpin is always leaf
# other types of loops must invariably transit to stem
# stem can transit to any type of loop but not to another stem
allowed_hpn_transition = [[False, False, False, False],
                          [False, False, False, True],
                          [False, False, False, True],
                          [True, True, True, False]]

MAX_TREE_DECODE_STEPS = 300
MAX_SEGMENT_LENGTH = 100
MIN_HAIRPIN_LENGTH = 3


def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx:
            continue
        stack.append((x, y, 1))
        dfs(stack, y, x.idx)
        stack.append((y, x, 0))


def GRU(x, h, W_z, W_r, W_h):
    # a normal GRU cell
    z_input = r_input = torch.cat([x, h], dim=1)
    z = torch.sigmoid(W_z(z_input))
    r = torch.sigmoid(W_r(r_input))
    gated_h = r * h
    h_input = torch.cat([x, gated_h], dim=1)
    pre_h = torch.tanh(W_h(h_input))
    new_h = (1.0 - z) * h + z * pre_h
    return new_h


def GraphGRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = W_r.out_features
    sum_h = h_nei.sum(dim=1)
    z_input = torch.cat([x, sum_h], dim=1)
    z = torch.sigmoid(W_z(z_input))

    r_1 = W_r(x).view(-1, 1, hidden_size)
    r_2 = U_r(h_nei)
    r = torch.sigmoid(r_1 + r_2)

    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x, sum_gated_h], dim=1)
    pre_h = torch.tanh(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h


class UnifiedDecoder(nn.Module):

    def __init__(self, hidden_size, latent_size, **kwargs):
        super(UnifiedDecoder, self).__init__()
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # GRU Weights for message passing
        self.W_z_mp = nn.Linear(hidden_size * 2 + len(HYPERGRAPH_VOCAB), hidden_size)
        self.U_r_mp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r_mp = nn.Linear(len(HYPERGRAPH_VOCAB) + hidden_size, hidden_size)
        self.W_h_mp = nn.Linear(hidden_size * 2 + len(HYPERGRAPH_VOCAB), hidden_size)

        # self.tree_vector_linear = nn.Linear(self.latent_size, self.hidden_size // 2)
        # self.graph_vector_linear = nn.Linear(self.latent_size, self.hidden_size // 2)

        self.concat_squash_linear = nn.Linear(self.hidden_size + self.latent_size, self.hidden_size)
        self.decode_nuc_with_lstm = kwargs.get('decode_nuc_with_lstm', False)
        if not self.decode_nuc_with_lstm:
            # GRU Weights for nucleotide decoding
            self.W_z_nuc = nn.Linear(hidden_size + len(NUC_VOCAB), hidden_size)
            self.W_r_nuc = nn.Linear(hidden_size + len(NUC_VOCAB), hidden_size)
            self.W_h_nuc = nn.Linear(hidden_size + len(NUC_VOCAB), hidden_size)
        else:
            self.lstm_cell = nn.LSTMCell(len(NUC_VOCAB), hidden_size)

        # hypernode label prediction
        self.W_hpn = nn.Linear(hidden_size, len(HYPERGRAPH_VOCAB))
        self.W_hpn_nonlinear = nn.Linear(hidden_size, hidden_size)

        # nucleotide prediction
        self.W_nuc = nn.Linear(hidden_size, len(NUC_VOCAB))
        self.W_nuc_nonlinear = nn.Linear(hidden_size + 2 * latent_size, hidden_size)

        # topological prediction
        self.W_topo = nn.Linear(hidden_size, 1)
        self.W_topo_nonlinear = nn.Linear(hidden_size, hidden_size)

        # Loss Functions
        self.hpn_pred_loss = nn.CrossEntropyLoss(reduction='none')
        self.nuc_pred_loss = nn.CrossEntropyLoss(reduction='none')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='none')

    def aggregate(self, hiddens, mode):
        # topological predictions: stop
        # hypernode label prediction: word_hpn
        # subgraph nucleotide preidiction: word_nuc
        if mode == 'word_hpn':
            return self.W_hpn(torch.relu(self.W_hpn_nonlinear(hiddens)))
        elif mode == 'word_nuc':
            return self.W_nuc(torch.relu(self.W_nuc_nonlinear(hiddens)))
        elif mode == 'stop':
            return self.W_topo(torch.relu(self.W_topo_nonlinear(hiddens)))
        else:
            raise ValueError('aggregation mode is not understood')

    def teacher_forced_decoding(self, all_seq_input, hidden_states, tree_latent_vec, graph_latent_vec):
        ''' decode segments as well as unrolling new tree messages '''
        all_len = [len(seq_input) for seq_input in all_seq_input]
        max_len = max(all_len)
        for seq_input in all_seq_input:
            # paddings
            seq_input += [np.zeros(len(NUC_VOCAB), dtype=np.float32)] * (max_len - len(seq_input))
        all_seq_input = torch.as_tensor(np.array(all_seq_input)).to(self.device)

        pre_padding_idx = (np.array(list(range(0, len(all_len) * max_len, max_len)))
                           + np.array(all_len) - 1).astype(np.long)
        all_pre_padding_idx = np.concatenate(
            [np.array(list(range(length))) + i * max_len for i, length in enumerate(all_len)]).astype(np.long)

        '''concat the volatile hidden states with a reference graph latent vector'''
        hidden_states = torch.relu(self.concat_squash_linear(torch.cat([hidden_states, graph_latent_vec], dim=-1)))
        cell_memory_ph = torch.zeros_like(hidden_states)

        all_hidden_states = []
        if self.decode_nuc_with_lstm:
            for len_idx in range(max_len):
                hidden_states, cell_memory_ph = self.lstm_cell(
                    all_seq_input[:, len_idx, :],
                    (hidden_states, cell_memory_ph))
                all_hidden_states.append(hidden_states)
            all_hidden_states = torch.stack(all_hidden_states, dim=1).view(-1, self.hidden_size)
        else:
            for len_idx in range(max_len):
                hidden_states = GRU(
                    all_seq_input[:, len_idx, :], hidden_states,
                    self.W_z_nuc, self.W_r_nuc, self.W_h_nuc)
                all_hidden_states.append(hidden_states)
            all_hidden_states = torch.stack(all_hidden_states, dim=1).view(-1, self.hidden_size)

        # the last hidden state at each segment
        new_h = all_hidden_states.index_select(0, torch.as_tensor(pre_padding_idx).to(self.device))
        all_hidden_states = all_hidden_states.index_select(0, torch.as_tensor(all_pre_padding_idx).to(self.device))

        segment_representation = []
        start = 0
        for length in all_len:
            segment_representation.append(torch.max(all_hidden_states[start: start + length], dim=0)[0])
            start += length

        batch_idx = [c for i, length in enumerate(all_len) for c in [i] * length]
        tensor_batch_idx = torch.as_tensor(np.array(batch_idx, dtype=np.long)).to(self.device)
        all_hidden_states = torch.cat([
            all_hidden_states,
            tree_latent_vec.index_select(0, tensor_batch_idx),
            graph_latent_vec.index_select(0, tensor_batch_idx),
        ], dim=1)

        return new_h, all_hidden_states, segment_representation

    def forward(self, rna_tree_batch, tree_latent_vec, graph_latent_vec):
        '''
        the training function utilizing teacher forcing
        hypernode label prediction: label of the hypergraph node
        nucleotide label prediction: segments of nucleotides
        topological prediction: more children or not for the current node
        '''
        hpn_pred_hiddens, hpn_pred_targets, hpn_batch_idx = [], [], []
        nuc_pred_hiddens, nuc_pred_targets, nuc_batch_idx = [], [], []
        stop_hiddens, stop_targets, stop_batch_idx = [], [], []

        ''' gather all messages from the ground truth structure '''
        traces = []
        for tree in rna_tree_batch:
            s = []
            dfs(s, tree.nodes[1], 0)  # skipping the pseudo node
            traces.append(s)
            for node in tree.nodes:
                if node.idx == 1:
                    # to ensure the first non pseudo node receives an incoming message
                    node.neighbors = [tree.nodes[0]]
                else:
                    node.neighbors = []
                node.segment_features = [None]

        batch_size = len(rna_tree_batch)
        initial_tree_latent_vec = tree_latent_vec
        tree_latent_vec = torch.cat([
            tree_latent_vec,
            torch.zeros(batch_size, self.hidden_size - self.latent_size).to(self.device)
        ], dim=1)

        ''' predict root node label '''
        hpn_pred_hiddens.append(tree_latent_vec)
        hpn_batch_idx.extend(range(batch_size))
        hpn_pred_targets.extend([HYPERGRAPH_VOCAB.index(tree.nodes[1].hpn_label) for tree in rna_tree_batch])

        depth_tree_batch = [len(tree.nodes) for tree in rna_tree_batch]
        max_iter = max([len(tr) for tr in traces])
        msg_padding = torch.zeros(self.hidden_size).to(self.device)
        nuc_padding = np.zeros(len(NUC_VOCAB), dtype=np.float32)
        h = {}

        ''' append initial tree messages '''
        for batch_idx in range(batch_size):
            offset = sum(depth_tree_batch[:batch_idx])
            h[(offset, offset + 1)] = tree_latent_vec[batch_idx]

        for t in range(max_iter):

            prop_list = []
            batch_list = []

            for i, plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            hpn_label = []
            node_incoming_msg = []
            all_seq_input = []
            all_nuc_label = []
            segment_local_field = []

            ''' assemble tree messages and relevant node features as well as predictive targets '''
            for i, (node_x, real_y, _) in enumerate(prop_list):
                batch_idx = batch_list[i]
                offset = sum(depth_tree_batch[:batch_idx])

                # messages flowing into a node for topological prediction
                incoming_msg = [h[(node_y.idx + offset, node_x.idx + offset)] for node_y in node_x.neighbors]
                nb_effective_msg = len(incoming_msg)
                node_incoming_msg.append(incoming_msg)

                # local node features of decoded segment
                if node_x.segment_features[nb_effective_msg - 1] is None:
                    # to decode a brand new segment
                    segment_local_field.append(torch.zeros(self.hidden_size, dtype=torch.float32).to(self.device))
                else:
                    # history segment embeddings
                    # todo, better than sum
                    segment_local_field.append(sum(node_x.segment_features[1:nb_effective_msg]))

                # teacher forcing the ground truth node label
                onehot_enc = np.array(list(map(lambda x: x == node_x.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
                hpn_label.append(onehot_enc)

                # decode a segment of nucleotides for the current hypernode
                if node_x.hpn_label != 'H':
                    node_nt_idx = node_x.nt_idx_assignment[nb_effective_msg - 1]
                else:
                    node_nt_idx = node_x.nt_idx_assignment
                if t == 0:
                    # start token for the first segment of the non pseudo root node will be zeros
                    seq_input = [nuc_padding]
                else:
                    seq_input = []

                for nuc_idx, nuc in enumerate([rna_tree_batch[batch_idx].rna_seq[nt_idx] for nt_idx in node_nt_idx]):
                    onehot_enc = np.array(list(map(lambda x: x == nuc, NUC_VOCAB)), dtype=np.float32)
                    seq_input.append(onehot_enc)
                    if nuc_idx == 0 and t > 0:
                        continue
                    all_nuc_label.append(NUC_VOCAB.index(nuc))
                all_nuc_label.append(NUC_VOCAB.index('<'))
                all_seq_input.append(seq_input)
                nuc_batch_idx.extend([batch_idx] * len(seq_input))

            tensor_batch_list = torch.as_tensor(np.array(batch_list, dtype=np.long)).to(self.device)
            batch_graph_latent_vec = graph_latent_vec.index_select(0, tensor_batch_list)
            batch_tree_latent_vec = initial_tree_latent_vec.index_select(0, tensor_batch_list)

            ''' unrolling new tree messages given current node and its segment features '''
            hpn_label = torch.as_tensor(np.array(hpn_label)).to(self.device)
            segment_local_field = torch.stack(segment_local_field, dim=0)
            local_field = torch.cat([hpn_label, segment_local_field], dim=-1)
            all_len = [len(incoming_msg) for incoming_msg in node_incoming_msg]
            max_len = max(all_len)
            for incoming_msg in node_incoming_msg:
                incoming_msg += [msg_padding] * (max_len - len(incoming_msg))
            node_incoming_msg = torch.stack([msg for incoming_msg in node_incoming_msg for msg in incoming_msg],
                                            dim=0).view(-1, max_len, self.hidden_size)
            node_incoming_msg = GraphGRU(local_field, node_incoming_msg,
                                         self.W_z_mp, self.W_r_mp, self.U_r_mp, self.W_h_mp)

            ''' decode a new segment '''
            new_h, all_hidden_states, segment_representation = self.teacher_forced_decoding(
                all_seq_input, node_incoming_msg, batch_tree_latent_vec, batch_graph_latent_vec)
            nuc_pred_hiddens.append(all_hidden_states)
            nuc_pred_targets.extend(all_nuc_label)

            '''to predict new hypernodes and topology'''
            pred_target, pred_list, stop_target = [], [], []
            for i, m in enumerate(prop_list):
                # some messages in the prop_list will be used to make new hypernode prediction
                batch_idx = batch_list[i]
                offset = sum(depth_tree_batch[:batch_idx])
                node_x, node_y, direction = m
                x, y = node_x.idx + offset, node_y.idx + offset
                h[(x, y)] = new_h[i]
                node_x.segment_features.append(segment_representation[i])
                node_y.neighbors.append(node_x)
                if direction == 1:
                    # direction where we are expanding (relative to backtracking)
                    # for these we make a prediction about the expanded hypernode's label
                    pred_target.append(HYPERGRAPH_VOCAB.index(node_y.hpn_label))
                    hpn_batch_idx.append(batch_idx)
                    pred_list.append(i)
                stop_target.append(direction)
                stop_batch_idx.append(batch_idx)

            # hidden states for stop prediction
            # stop_hidden = torch.cat([hpn_label, node_incoming_msg], dim=1)
            stop_hiddens.append(node_incoming_msg)
            stop_targets.extend(stop_target)

            # hidden states for label prediction
            if len(pred_list) > 0:
                # list where we make label predictions
                cur_pred = torch.as_tensor(np.array(pred_list, dtype=np.long)).to(self.device)
                hpn_pred_hiddens.append(node_incoming_msg.index_select(0, cur_pred))
                hpn_pred_targets.extend(pred_target)

        ''' 
        last stop at the non-pseudo root node
        topological prediction --> no more children 
        '''
        hpn_label, segment_local_field, node_incoming_msg = [], [], []
        all_seq_input, all_nuc_label = [], []

        for batch_idx, tree in enumerate(rna_tree_batch):
            offset = sum(depth_tree_batch[:batch_idx])
            node_x = tree.nodes[1]
            onehot_enc = np.array(list(map(lambda x: x == node_x.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
            hpn_label.append(torch.as_tensor(onehot_enc).to(self.device))
            incoming_msg = [h[(node_y.idx + offset, node_x.idx + offset)] for node_y in node_x.neighbors]
            node_incoming_msg.append(incoming_msg)
            if node_x.segment_features[-1] is None:
                segment_local_field.append(torch.zeros(self.hidden_size, dtype=torch.float32).to(self.device))
            else:
                segment_local_field.append(sum(node_x.segment_features[1:]))

            # decode the last segment of the non pseudo root node
            if node_x.hpn_label != 'H':
                root_node_nt_idx = node_x.nt_idx_assignment[-1]
            else:
                root_node_nt_idx = node_x.nt_idx_assignment
            if node_x.hpn_label == 'H':
                seq_input = [nuc_padding]  # start token
            else:
                seq_input = []
            for nuc_idx, nuc in enumerate([rna_tree_batch[batch_idx].rna_seq[nt_idx] for nt_idx in root_node_nt_idx]):
                onehot_enc = np.array(list(map(lambda x: x == nuc, NUC_VOCAB)), dtype=np.float32)
                seq_input.append(onehot_enc)
                if nuc_idx == 0 and node_x.hpn_label != 'H':
                    continue
                all_nuc_label.append(NUC_VOCAB.index(nuc))
            all_nuc_label.append(NUC_VOCAB.index('<'))
            all_seq_input.append(seq_input)
            nuc_batch_idx.extend([batch_idx] * len(seq_input))

        hpn_label = torch.stack(hpn_label, dim=0)
        segment_local_field = torch.stack(segment_local_field, dim=0)
        local_field = torch.cat([hpn_label, segment_local_field], dim=-1)
        all_len = [len(incoming_msg) for incoming_msg in node_incoming_msg]
        max_len = max(all_len)
        for incoming_msg in node_incoming_msg:
            incoming_msg += [msg_padding] * (max_len - len(incoming_msg))
        node_incoming_msg = torch.stack(
            [msg for incoming_msg in node_incoming_msg for msg in incoming_msg], dim=0). \
            view(-1, max_len, self.hidden_size)
        node_incoming_msg = GraphGRU(local_field, node_incoming_msg,
                                     self.W_z_mp, self.W_r_mp, self.U_r_mp, self.W_h_mp)

        new_h, all_hidden_states, _ = self.teacher_forced_decoding(
            all_seq_input, node_incoming_msg, initial_tree_latent_vec, graph_latent_vec)
        nuc_pred_hiddens.append(all_hidden_states)
        nuc_pred_targets.extend(all_nuc_label)

        ''' here topological predictions should always be backtrack '''
        stop_hiddens.append(node_incoming_msg)
        stop_targets.extend([0] * batch_size)
        stop_batch_idx.extend(range(batch_size))

        ''' objectives for hypernode prediction '''
        nb_hpn_targets = len(hpn_pred_targets)
        hpn_pred_hiddens = torch.cat(hpn_pred_hiddens, dim=0)
        hpn_pred_scores = self.aggregate(hpn_pred_hiddens, 'word_hpn')
        hpn_pred_targets = np.array(hpn_pred_targets, dtype=np.long)
        hpn_pred_loss = self.hpn_pred_loss(
            hpn_pred_scores,
            torch.as_tensor(hpn_pred_targets).to(self.device))

        hpn_preds = torch.max(hpn_pred_scores, dim=1)[1].cpu().detach().numpy()
        nb_hpn_pred_correct = np.equal(hpn_preds, hpn_pred_targets).astype(np.float32).sum()

        batch_hpn_loss = []
        hpn_batch_idx = np.array(hpn_batch_idx)
        for batch_idx in range(batch_size):
            batch_hpn_loss.append(torch.sum(hpn_pred_loss[hpn_batch_idx == batch_idx]))

        hpn_ret_dict = {
            'sum_hpn_pred_loss': torch.sum(hpn_pred_loss),
            'batch_hpn_pred_loss': torch.as_tensor(batch_hpn_loss),
            'nb_hpn_targets': nb_hpn_targets,
            'nb_hpn_pred_correct': nb_hpn_pred_correct
        }

        ''' objectives for nucleotide prediction '''
        nb_nuc_targets = len(nuc_pred_targets)
        nuc_pred_hiddens = torch.cat(nuc_pred_hiddens, dim=0)
        nuc_pred_scores = self.aggregate(nuc_pred_hiddens, 'word_nuc')
        nuc_pred_targets = np.array(nuc_pred_targets, dtype=np.long)
        nuc_pred_loss = self.nuc_pred_loss(
            nuc_pred_scores,
            torch.as_tensor(nuc_pred_targets).to(self.device))

        nuc_preds = torch.max(nuc_pred_scores, dim=1)[1].cpu().detach().numpy()
        nuc_pred_correct = np.equal(nuc_preds, nuc_pred_targets).astype(np.float32)
        nb_nuc_pred_correct = nuc_pred_correct.sum()
        stop_trans_targets = nuc_pred_targets == len(NUC_VOCAB) - 1
        nb_stop_trans_pred_correct = np.sum(stop_trans_targets * nuc_pred_correct)
        ord_nuc_targets = 1 - stop_trans_targets
        nb_ord_nuc_pred_correct = np.sum(ord_nuc_targets * nuc_pred_correct)

        batch_nuc_loss = []
        nuc_batch_idx = np.array(nuc_batch_idx)
        for batch_idx in range(batch_size):
            batch_nuc_loss.append(torch.sum(nuc_pred_loss[nuc_batch_idx == batch_idx]))

        nuc_ret_dict = {
            'sum_nuc_pred_loss': torch.sum(nuc_pred_loss),
            'batch_nuc_pred_loss': torch.as_tensor(batch_nuc_loss),
            'nb_nuc_targets': nb_nuc_targets,
            'nb_nuc_pred_correct': nb_nuc_pred_correct,
            'nb_stop_trans_pred_correct': nb_stop_trans_pred_correct,
            'nb_stop_trans_targets': np.sum(stop_trans_targets),
            'nb_ord_nuc_pred_correct': nb_ord_nuc_pred_correct,
            'nb_ord_nuc_targets': np.sum(ord_nuc_targets)
        }

        ''' objectives for topology prediction '''
        nb_stop_targets = len(stop_targets)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_scores = self.aggregate(stop_hiddens, 'stop')
        stop_scores = stop_scores.squeeze(-1)
        stop_targets = np.array(stop_targets, dtype=np.float32)
        stop_loss = self.stop_loss(stop_scores, torch.as_tensor(stop_targets).to(self.device))

        stops = torch.ge(stop_scores, 0).float().cpu().detach().numpy()
        nb_stop_pred_correct = np.equal(stops, stop_targets).astype(np.float32).sum()

        batch_stop_loss = []
        stop_batch_idx = np.array(stop_batch_idx)
        for batch_idx in range(batch_size):
            batch_stop_loss.append(torch.sum(stop_loss[stop_batch_idx == batch_idx]))

        stop_ret_dict = {
            'sum_stop_pred_loss': torch.sum(stop_loss),
            'batch_stop_pred_loss': torch.as_tensor(batch_stop_loss).to(self.device),
            'nb_stop_targets': nb_stop_targets,
            'nb_stop_pred_correct': nb_stop_pred_correct
        }

        return {**hpn_ret_dict, **nuc_ret_dict, **stop_ret_dict}

    ########################################
    # decoding RNA with regularity constraint
    ########################################

    def decode_segment_with_constraint(self, last_token, hidden_state, tree_latent_vec, graph_latent_vec, **kwargs):
        batch_size = last_token.size(0)
        prob_decode = kwargs.get('prob_decode', False)
        batch_minimal_length = kwargs.get('minimal_length', [0] * batch_size)
        batch_last_nuc_complement_to_idx = kwargs.get('last_nuc_complement_to_idx', [None] * batch_size)
        batch_second_stem_segment_complement_to_idx = kwargs.get('second_stem_segment_complement_to_idx',
                                                                 [None] * batch_size)
        # properly reversed

        decoded_nuc_idx = []
        for _ in range(batch_size):
            decoded_nuc_idx.append([])

        batch_list = []
        successful = [True] * batch_size
        all_batch_list = []
        all_hidden_states = []
        final_last_token, final_hidden_state, final_graph_latent_vec = \
            [None] * batch_size, [None] * batch_size, [None] * batch_size
        segment_representation = [None] * batch_size
        first_nuc_idx = np.argmax(last_token.cpu().detach().numpy(), axis=-1)
        stem_max_len = 0

        for batch_idx, second_stem_segment_complement_to_idx in enumerate(batch_second_stem_segment_complement_to_idx):
            if second_stem_segment_complement_to_idx is None:
                batch_list.append(batch_idx)
            else:
                if allowed_basepairs[first_nuc_idx[batch_idx]][second_stem_segment_complement_to_idx[0]] is False:
                    successful[batch_idx] = 'SECOND_STEM_INITIAL_COND_FAILED'  # these have failed
                    # final_last_token[batch_idx] = last_token[batch_idx]
                    # final_hidden_state[batch_idx] = hidden_state[batch_idx]
                    # final_graph_latent_vec[batch_idx] = graph_latent_vec[batch_idx]
                    continue
                stem_max_len = max(stem_max_len, len(second_stem_segment_complement_to_idx))
                del second_stem_segment_complement_to_idx[0]  # the first reference is passed
                batch_list.append(batch_idx)

        batch_list = np.array(batch_list)
        max_iteration = max(MAX_SEGMENT_LENGTH, stem_max_len)
        stop_symbol_mask = np.array([0., 0., 0., 0., -99999.], dtype=np.float32)
        non_stop_symbol_mask = np.array([-99999., -99999., -99999., -99999., 0.], dtype=np.float32)

        # initial selection to ditch unsatisfiable second stem segment initial decoding criterion
        '''tensor_batch_list should never be empty, as \'SECOND_STEM_INITIAL_COND_FAILED\' should never happen'''
        tensor_batch_list = torch.as_tensor(batch_list).to(self.device)
        last_token = last_token.index_select(0, tensor_batch_list)
        hidden_state = torch.relu(self.concat_squash_linear(torch.cat([hidden_state, graph_latent_vec], dim=-1)))
        hidden_state = hidden_state.index_select(0, tensor_batch_list)
        tree_latent_vec = tree_latent_vec.index_select(0, tensor_batch_list)
        graph_latent_vec = graph_latent_vec.index_select(0, tensor_batch_list)

        decode_step = 0
        while decode_step < max_iteration:

            '''prepare proper mask'''
            all_mask = []
            for i, batch_idx in enumerate(batch_list):
                second_stem_segment_complement_to_idx = batch_second_stem_segment_complement_to_idx[batch_idx]
                minimal_length = batch_minimal_length[batch_idx]
                last_nuc_complement_to_idx = batch_last_nuc_complement_to_idx[batch_idx]

                if second_stem_segment_complement_to_idx is not None:
                    if decode_step < len(second_stem_segment_complement_to_idx):
                        mask = ((np.array(allowed_basepairs[second_stem_segment_complement_to_idx[decode_step]] + [
                            False]) - 1) * 99999).astype(np.float32)
                        all_mask.append(mask)
                    else:
                        all_mask.append(non_stop_symbol_mask)
                    continue  # if this constraint is on then we won't need to check the others

                mask = np.zeros(len(NUC_VOCAB), dtype=np.float32)

                if minimal_length is not None and decode_step < minimal_length:
                    mask = stop_symbol_mask

                if decode_step > 0 and last_nuc_complement_to_idx is not None and \
                        allowed_basepairs[last_nuc_complement_to_idx][decoded_nuc_idx[batch_idx][-1]] is False:
                    mask = stop_symbol_mask

                all_mask.append(mask)

            all_mask = torch.as_tensor(np.array(all_mask)).to(self.device)

            '''decode one position'''
            if self.decode_nuc_with_lstm:
                hidden_state, graph_latent_vec = self.lstm_cell(last_token, (hidden_state, graph_latent_vec))
            else:
                hidden_state = GRU(last_token, hidden_state, self.W_z_nuc, self.W_r_nuc, self.W_h_nuc)

            score_state = torch.cat([hidden_state, tree_latent_vec, graph_latent_vec], dim=-1)
            nuc_pred_score = self.aggregate(score_state, 'word_nuc')
            nuc_pred_score = nuc_pred_score + all_mask

            if prob_decode:
                nuc_idx = torch.multinomial(torch.softmax(nuc_pred_score, dim=1), num_samples=1)[:, 0]
            else:
                nuc_idx = torch.max(nuc_pred_score, dim=1)[1]
            nuc_idx = nuc_idx.cpu().detach().numpy()

            '''inspect decoded nucleotides'''
            tmp_last_token = []
            for i, cur_dec_nuc_idx in enumerate(nuc_idx):
                batch_idx = batch_list[i]
                if cur_dec_nuc_idx == len(NUC_VOCAB) - 1:  # < for stop translation
                    last_nuc_complement_to_idx = batch_last_nuc_complement_to_idx[batch_idx]
                    if last_nuc_complement_to_idx is not None and \
                            allowed_basepairs[last_nuc_complement_to_idx][decoded_nuc_idx[batch_idx][-1]] is False:
                        successful[batch_idx] = 'LAST_NUC_COND_FAILED'
                    else:
                        final_last_token[batch_idx] = last_token[i]
                        final_hidden_state[batch_idx] = hidden_state[i]
                        final_graph_latent_vec[batch_idx] = graph_latent_vec[i]

                else:
                    decoded_nuc_idx[batch_idx].append(cur_dec_nuc_idx)
                    tmp_last_token.append(np.array(list(map(lambda x: x == cur_dec_nuc_idx, range(len(NUC_VOCAB)))),
                                                   dtype=np.float32))

            # cont_translation_idx = np.where(nuc_idx != len(NUC_VOCAB) - 1 or np.array(successful) is not True)[0]
            cont_translation_idx = np.where(nuc_idx != len(NUC_VOCAB) - 1)[0]
            tensor_cont_translation_idx = torch.as_tensor(cont_translation_idx).to(self.device)

            if len(cont_translation_idx) == 0:
                break

            last_token = torch.as_tensor(tmp_last_token).to(self.device)
            hidden_state = hidden_state.index_select(0, tensor_cont_translation_idx)
            tree_latent_vec = tree_latent_vec.index_select(0, tensor_cont_translation_idx)
            graph_latent_vec = graph_latent_vec.index_select(0, tensor_cont_translation_idx)
            batch_list = batch_list[cont_translation_idx]

            all_hidden_states.append(hidden_state)
            all_batch_list.extend(batch_list)

            decode_step += 1

        if decode_step == max_iteration:

            for i, batch_idx in enumerate(batch_list):
                # last_nuc_complement_to_idx = batch_last_nuc_complement_to_idx[batch_idx]
                # if last_nuc_complement_to_idx is not None and \
                #         allowed_basepairs[last_nuc_complement_to_idx][decoded_nuc_idx[batch_idx][-1]] is False:
                #     successful[batch_idx] = False
                # else:
                #     final_last_token[batch_idx] = last_token[i]
                #     final_hidden_state[batch_idx] = hidden_state[i]
                #     final_graph_latent_vec[batch_idx] = graph_latent_vec[i]
                '''those that won't finish decoding would fail here'''
                '''a flat RNA should be always be fairly short, therefore the global maximum iterations suffice'''
                successful[batch_idx] = 'SEGMENT_ITER_EXCEED_MAXIMUM'

        all_batch_list = np.array(all_batch_list)
        if len(all_hidden_states) > 0:
            all_hidden_states = torch.cat(all_hidden_states, dim=0)
        for batch_idx in range(batch_size):
            all_idx = np.where(all_batch_list == batch_idx)[0]
            if len(all_idx) > 0:
                segment_representation[batch_idx] = torch.max(all_hidden_states[all_idx], dim=0)[0]
            else:
                segment_representation[batch_idx] = torch.zeros(self.hidden_size).to(self.device)

        return final_hidden_state, decoded_nuc_idx, final_last_token, successful, segment_representation

    def decode_segment(self, list_current_node, last_token, hidden_state, tree_latent_vec, graph_latent_vec,
                       prob_decode, is_backtrack, enforce_dec_prior=True):

        batch_size = len(list_current_node)
        np_last_token = last_token.cpu().detach().numpy()
        minimal_length = [None] * batch_size
        last_nuc_complement_to_idx = [None] * batch_size
        second_stem_segment_complement_to_idx = [None] * batch_size

        if enforce_dec_prior:

            for i, current_node in enumerate(list_current_node):

                if current_node.hpn_label == 'H':

                    if current_node.idx > 1:
                        # parent node is a stem, thus current node is
                        # somewhere in the middle of this RNA structure
                        start_nuc_idx = int(np.argmax(np_last_token[i]))
                        minimal_length[i] = MIN_HAIRPIN_LENGTH + 1
                        last_nuc_complement_to_idx[i] = start_nuc_idx
                    else:
                        # the first non pseudo root node
                        # hence no complementarity constraint
                        # todo, length constraints for completely single stranded RNA
                        # have at least 1 nucleotide
                        minimal_length[i] = 1
                        # minimal_length.append(None)
                        # last_nuc_complement_to_idx.append(None)

                elif current_node.hpn_label == 'I':

                    if len(current_node.nt_idx_assignment) == 0:
                        minimal_length[i] = 1
                    else:
                        # the second segment of internal loop
                        if current_node.idx > 1:  # connected by a stem
                            if len(current_node.nt_idx_assignment[0]) == 2:
                                # two nucleotides on two closing stems
                                # the first segment of internal loop is empty, therefore the second segment must not be empty
                                min_internal_loop_length = 2
                            else:
                                min_internal_loop_length = 1
                            # at the starting position in the 5' end
                            start_nuc_idx = NUC_VOCAB.index(current_node.decoded_segment[0][0])
                        else:  # dangling end
                            if len(current_node.nt_idx_assignment[0]) == 1:
                                min_internal_loop_length = 1
                            else:
                                min_internal_loop_length = 0
                            start_nuc_idx = None

                        minimal_length[i] = min_internal_loop_length
                        last_nuc_complement_to_idx[i] = start_nuc_idx

                elif current_node.hpn_label == 'M':

                    if current_node.idx == 1 and is_backtrack[i]:  # dangling end
                        min_length = 0
                        start_nuc_idx = None
                    elif current_node.idx > 1 and is_backtrack[i]:
                        min_length = 1
                        start_nuc_idx = NUC_VOCAB.index(current_node.decoded_segment[0][0])
                    else:
                        min_length = 1
                        start_nuc_idx = None

                    minimal_length[i] = min_length
                    last_nuc_complement_to_idx[i] = start_nuc_idx

                elif current_node.hpn_label == 'S':

                    if len(current_node.nt_idx_assignment) == 0:  # the first segment
                        # minimal length is zero, as the first starting nucleotide is a basepair
                        if np.max(np_last_token[i]) == 0:
                            # the first nucleotide to be decoded, then
                            min_length = 1
                        else:
                            # todo, lonely basepairs
                            min_length = 0
                        minimal_length[i] = min_length
                    else:
                        second_stem_segment_complement_to_idx[i] = [NUC_VOCAB.index(nuc) for nuc in
                                                                    reversed(current_node.decoded_segment[0])]

                else:
                    raise ValueError('Unknown hypernode')

        hidden_state, decoded_nuc_idx, last_token, is_successful, segment_representation = \
            self.decode_segment_with_constraint(
                last_token, hidden_state, tree_latent_vec, graph_latent_vec,
                prob_decode=prob_decode,
                minimal_length=minimal_length,
                last_nuc_complement_to_idx=last_nuc_complement_to_idx,
                second_stem_segment_complement_to_idx=second_stem_segment_complement_to_idx)

        '''some segments in \'decoded_nuc_idx\' may be empty'''
        return hidden_state, decoded_nuc_idx, last_token, is_successful, segment_representation

    def decode(self, tree_latent_vec, graph_latent_vec, prob_decode,
               verbose=False, enforce_topo_prior=True, enforce_hpn_prior=True, enforce_dec_prior=True):
        '''
        A function for parallel decoding
        Regularity(validity) constraint for stem:
          - topology constraint:
              exactly two segments (one parent and one child)
          - hpn word constraint:
              must be preceded by either P(pseudo start), H, I or M
          - nucleotide constraint:
              - reverse complement of both segments
              - equivalent segment sizes

        Regularity(validity) constraint for hairpin loop:
          - topology constraint:
              exactly one segment (one parent and no children), a.k.a a leaf node
          - hpn word constraint:
              preceded by P or S
          - nucleotide constraint:
              - minimum size of 3
              - if not the first non pseudo root node,
              the last decoded nucleotide must be able to
              form base-pair with the starting nucleotide

        Regularity(validity) constraint for internal loop:
          - topology constraint:
              exactly two segments (one parent and one child)
          - hpn word constraint:
              preceded by P or S
          - nucleotide constraint:
              - at least one segment is not empty
              - if not the first non pseudo root node
              the last decoded nucleotide on the second segment must be able to
              form base-pair with the starting nucleotide on the first segment

        Regularity(validity) constraint for multi loop:
          - topology constraint:
              more than two segments (one parent and more than one child)
          - hpn word constraint:
              preceded by P or S
        '''
        batch_size = tree_latent_vec.size(0)

        # decoding starts from the 5' end in a depth first topological order
        all_rna_seq = [''] * batch_size

        all_trees, all_h = [], []
        for _ in range(batch_size):
            # always start from a hypothetical pseudo node
            pseudo_node = RNAJTNode('P', [])
            pseudo_node.idx = 0
            all_trees.append([pseudo_node])
            all_h.append({})

        all_stacks = []
        for i in range(batch_size):
            stack = [(all_trees[i][0], None, None)]  # pseudo-node
            all_stacks.append(stack)

        # Root Prediction
        initial_tree_latent_vec = tree_latent_vec
        tree_latent_vec = torch.cat([
            tree_latent_vec,
            torch.zeros(batch_size, self.hidden_size - self.latent_size).to(self.device)
        ], dim=1)

        root_hpn_pred_score = self.aggregate(tree_latent_vec, 'word_hpn')

        # we can basically decode anything at the first non pseudo root node
        if prob_decode:
            all_root_label_idx = torch.multinomial(torch.softmax(root_hpn_pred_score, dim=1), num_samples=1).view(
                batch_size)
        else:
            _, all_root_label_idx = torch.max(root_hpn_pred_score, dim=1)
        all_root_label_idx = all_root_label_idx.cpu().detach().numpy()

        for i in range(batch_size):
            pseudo_node = all_trees[i][0]
            root = RNAJTNode(HYPERGRAPH_VOCAB[all_root_label_idx[i]], [], neighbors=[pseudo_node])
            root.idx = 1
            root.decoded_segment = []
            root.segment_features = [None]

            pseudo_node.neighbors.append(root)
            all_stacks[i].append((root, torch.zeros(len(NUC_VOCAB)).to(self.device), 0))
            all_h[i].update({(0, 1): tree_latent_vec[i]})
            all_trees[i].append(root)

        msg_padding = torch.zeros(self.hidden_size).to(self.device)
        is_successful = [True] * batch_size

        for step in range(MAX_TREE_DECODE_STEPS):

            batch_list = []  # to mark if a tree has been finished
            node_incoming_msg = []
            segment_local_field = []
            hpn_label = []

            list_current_node = []
            list_last_token = []
            list_last_token_idx = []

            for i in range(batch_size):
                node_x, last_token, last_token_idx = all_stacks[i][-1]

                if node_x.hpn_label == 'P' or is_successful[i] is not True:
                    '''this one is finished'''
                    continue

                list_current_node.append(node_x)
                list_last_token.append(last_token)
                list_last_token_idx.append(last_token_idx)

                batch_list.append(i)

                incoming_msg = [all_h[i][(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
                nb_effective_msg = len(incoming_msg)
                node_incoming_msg.append(incoming_msg)

                if node_x.segment_features[nb_effective_msg - 1] is None:
                    # to decode a brand new segment
                    segment_local_field.append(torch.zeros(self.hidden_size, dtype=torch.float32).to(self.device))
                else:
                    # history segment embeddings
                    # todo, better than sum here, perhaps max? Also should change segment features
                    segment_local_field.append(sum(node_x.segment_features[1:nb_effective_msg]))

                onehot_enc = np.array(list(map(lambda x: x == node_x.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
                hpn_label.append(onehot_enc)

            if len(batch_list) == 0:
                break

            '''out here you use \'batch_list\''''
            tensor_batch_list = torch.as_tensor(np.array(batch_list, dtype=np.long)).to(self.device)
            batch_graph_latent_vec = graph_latent_vec.index_select(0, tensor_batch_list)
            batch_tree_latent_vec = initial_tree_latent_vec.index_select(0, tensor_batch_list)

            hpn_label = torch.as_tensor(np.array(hpn_label)).to(self.device)
            segment_local_field = torch.stack(segment_local_field, dim=0)
            local_field = torch.cat([hpn_label, segment_local_field], dim=-1)
            all_len = [len(incoming_msg) for incoming_msg in node_incoming_msg]
            max_len = max(all_len)
            for incoming_msg in node_incoming_msg:
                incoming_msg += [msg_padding] * (max_len - len(incoming_msg))
            node_incoming_msg = torch.stack([msg for incoming_msg in node_incoming_msg for msg in incoming_msg],
                                            dim=0).view(-1, max_len, self.hidden_size)

            node_incoming_msg = GraphGRU(local_field, node_incoming_msg,
                                         self.W_z_mp, self.W_r_mp, self.U_r_mp, self.W_h_mp)

            '''topological prediction'''
            stop_score = self.aggregate(node_incoming_msg, 'stop')
            if prob_decode:
                backtrack = (torch.bernoulli(torch.sigmoid(stop_score)).cpu().detach().numpy() == 0)[:, 0]
            else:
                backtrack = (stop_score.cpu().detach().numpy() < 0)[:, 0]
            backtrack = list(backtrack)

            if enforce_topo_prior:
                for i, batch_idx in enumerate(batch_list):
                    node_x, _, _ = all_stacks[batch_idx][-1]
                    # topology regularity constraint
                    if node_x.hpn_label == 'H':
                        backtrack[i] = True
                    elif node_x.hpn_label == 'I' or node_x.hpn_label == 'S':
                        if len(node_x.nt_idx_assignment) == 0:
                            backtrack[i] = False
                        else:
                            backtrack[i] = True
                    else:  # a multiloop segment
                        if len(node_x.nt_idx_assignment) <= 1:
                            backtrack[i] = False
            else:
                # check if topological constrains have been met
                for i, batch_idx in enumerate(batch_list):
                    node_x, _, _ = all_stacks[batch_idx][-1]
                    # topology regularity constraint
                    if node_x.hpn_label == 'H':
                        if backtrack[i] is not True:
                            is_successful[batch_idx] = 'TOPO_FAILURE'
                    elif node_x.hpn_label == 'I' or node_x.hpn_label == 'S':
                        if len(node_x.nt_idx_assignment) == 0:
                            if backtrack[i] is not False:
                                is_successful[batch_idx] = 'TOPO_FAILURE'
                        else:
                            if backtrack[i] is not True:
                                is_successful[batch_idx] = 'TOPO_FAILURE'
                    else:  # a multiloop segment
                        if len(node_x.nt_idx_assignment) <= 1:
                            if backtrack[i] is not False:
                                is_successful[batch_idx] = 'TOPO_FAILURE'

            '''segment decoding'''
            last_token = torch.stack(list_last_token, dim=0)
            new_h, decoded_nuc_idx, last_token, list_is_successful, segment_representation = \
                self.decode_segment(
                    list_current_node, last_token, node_incoming_msg,
                    batch_tree_latent_vec, batch_graph_latent_vec, prob_decode, backtrack,
                    enforce_dec_prior=enforce_dec_prior)

            '''hypernode prediction'''
            if False in backtrack:
                expand = [not is_backtrack for is_backtrack in backtrack]
                hpn_pred_score = self.aggregate(node_incoming_msg[expand], 'word_hpn')
                if enforce_hpn_prior:
                    hpn_transit_mask = []
                    for i, is_backtrack in enumerate(backtrack):
                        if not is_backtrack:
                            node_x = list_current_node[i]
                            mask = ((np.array(allowed_hpn_transition[HYPERGRAPH_VOCAB.index(node_x.hpn_label)]) - 1) *
                                    99999).astype(np.float32)
                            hpn_transit_mask.append(mask)
                    hpn_transit_mask = torch.as_tensor(np.array(hpn_transit_mask)).to(self.device)
                    hpn_pred_score += hpn_transit_mask

                if prob_decode:
                    hpn_label_idx = torch.multinomial(torch.softmax(hpn_pred_score, dim=1), num_samples=1)[:, 0]
                else:
                    hpn_label_idx = torch.max(hpn_pred_score, dim=1)[1]
                hpn_label_idx = hpn_label_idx.cpu().detach().numpy()

                if not enforce_hpn_prior:
                    start = 0
                    for i, is_backtrack in enumerate(backtrack):
                        if not is_backtrack:
                            node_x = list_current_node[i]
                            batch_idx = batch_list[i]
                            if allowed_hpn_transition[HYPERGRAPH_VOCAB.index(node_x.hpn_label)][
                                hpn_label_idx[start]] is False:
                                if is_successful[batch_idx] is True:
                                    is_successful[batch_idx] = 'HPN_TRANSIT_FAILURE'
                                else:
                                    is_successful[batch_idx] += '|HPN_TRANSIT_FAILURE'
                            start += 1

            start = 0
            for i, is_backtrack in enumerate(backtrack):
                batch_idx = batch_list[i]
                if list_is_successful[i] is not True:  # only update failures
                    if is_successful[batch_idx] is True:
                        is_successful[batch_idx] = list_is_successful[i]
                    else:
                        is_successful[batch_idx] += '|' + list_is_successful[i]
                if is_successful[batch_idx] is not True:
                    continue
                node_x = list_current_node[i]
                node_x.segment_features.append(segment_representation[i])
                if not is_backtrack:
                    node_y = RNAJTNode(HYPERGRAPH_VOCAB[hpn_label_idx[start]], [])
                    node_y.decoded_segment = []
                    node_y.segment_features = [None]
                    node_y.idx = len(all_trees[batch_idx])
                    node_y.neighbors.append(node_x)
                    all_h[batch_idx][(node_x.idx, node_y.idx)] = new_h[i]

                    last_token_idx = list_last_token_idx[i]
                    node_x.nt_idx_assignment.append(list(range(last_token_idx - 1 if last_token_idx > 0 else 0,
                                                               last_token_idx + len(decoded_nuc_idx[i]))))
                    last_token_idx += len(decoded_nuc_idx[i])

                    all_stacks[batch_idx].append((node_y, last_token[i], last_token_idx))
                    all_trees[batch_idx].append(node_y)
                    start += 1
                else:
                    node_fa, _, _ = all_stacks[batch_idx][-2]
                    all_h[batch_idx][(node_x.idx, node_fa.idx)] = new_h[i]
                    node_fa.neighbors.append(node_x)
                    all_stacks[batch_idx].pop()
                    # modify node_fa

                    last_token_idx = list_last_token_idx[i]
                    if len(node_x.nt_idx_assignment) != 0:
                        node_x.nt_idx_assignment.append(list(range(last_token_idx - 1 if last_token_idx > 0 else 0,
                                                                   last_token_idx + len(decoded_nuc_idx[i]))))
                    else:
                        # a hairpin node
                        node_x.nt_idx_assignment = list(range(last_token_idx - 1 if last_token_idx > 0 else 0,
                                                              last_token_idx + len(decoded_nuc_idx[i])))
                    # ready to rollout to the next level
                    ''' blunder solved yesterday midnight... '''
                    last_token_idx += len(decoded_nuc_idx[i])

                    all_stacks[batch_idx][-1] = (node_fa, last_token[i], last_token_idx)

                all_rna_seq[batch_idx] += ''.join(map(lambda idx: NUC_VOCAB[idx], decoded_nuc_idx[i]))
                if node_x.hpn_label == 'H':
                    node_x.decoded_segment.extend(
                        list(map(lambda idx: all_rna_seq[batch_idx][idx], node_x.nt_idx_assignment)))
                else:
                    node_x.decoded_segment.append(
                        list(map(lambda idx: all_rna_seq[batch_idx][idx], node_x.nt_idx_assignment[-1])))

            if verbose:
                print('=' * 40, 'Iteration:', step, '=' * 40)
                print('Active examples:', len(batch_list))
                for i, batch_idx in enumerate(batch_list):
                    print('+' * 40, 'batch_idx:', batch_idx, '+' * 40)
                    print('successful:', is_successful[batch_idx])
                    if is_successful[batch_idx] is True:
                        node_x = list_current_node[i]
                        print('node index: {}, label: {}, total decoded segments: {}, expanding: {}'.
                              format(node_x.idx, node_x.hpn_label,
                                     len(node_x.nt_idx_assignment) if node_x.hpn_label != 'H' else 1, not backtrack[i]))
                        print('last decoded segment:',
                              ''.join(node_x.decoded_segment) if node_x.hpn_label == 'H' else ''.join(
                                  node_x.decoded_segment[-1]))
                        print('all seq:', all_rna_seq[batch_idx])
                        print('size of stack:', len(all_stacks[batch_idx]))

        for i in range(batch_size):
            node_x, last_token, last_token_idx = all_stacks[i][-1]

            if node_x.hpn_label == 'P' or is_successful[i] is not True:
                '''this one is finished'''
                continue
            else:
                is_successful[i] = 'TREE_ITER_EXCEED_MAXIMUM'

        for all_nodes, successful in zip(all_trees, is_successful):
            if successful is True:
                for node in all_nodes:
                    if hasattr(node, 'segment_features'):
                        del node.segment_features

        return all_rna_seq, all_trees, is_successful

    @staticmethod
    def assemble_subroutine(args):
        rna_seq, all_nodes, successful = args
        if successful is True:
            tree = RNAJunctionTree(rna_seq, None, nodes=all_nodes)
            if tree.is_valid is False:
                return 'INVALID'
            else:
                return tree
        else:
            return successful

    def assemble_trees(self, all_rna_seq, all_trees, is_successful, mp_pool=None):

        if mp_pool is None:
            all_parsed_trees = []
            for rna_seq, all_nodes, successful in zip(all_rna_seq, all_trees, is_successful):
                if successful is True:
                    all_parsed_trees.append(RNAJunctionTree(rna_seq, None, nodes=all_nodes))
                else:
                    all_parsed_trees.append(successful)
        else:
            all_parsed_trees = list(mp_pool.imap(UnifiedDecoder.assemble_subroutine,
                                                 zip(all_rna_seq, all_trees, is_successful)))

        return all_parsed_trees
