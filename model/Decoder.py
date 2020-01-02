# This model simultaneously decodes the junction tree
# as well as the nucleotides associated within each subgraph

import torch
import torch.nn as nn
import numpy as np
from lib.tree_decomp import RNAJunctionTree, RNAJTNode

# '<' to signal stop translation
NUC_VOCAB = ['A', 'C', 'G', 'U', '<']
HYPERGRAPH_VOCAB = ['H', 'I', 'M', 'S']

MAX_NB = 10
MAX_DECODE_LEN = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


class UnifiedDecoder(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super(UnifiedDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # GRU Weights for message passing
        # self.W_z_mp = nn.Linear(hidden_size + len(HYPERGRAPH_VOCAB), hidden_size)
        # self.U_r_mp = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.W_r_mp = nn.Linear(len(HYPERGRAPH_VOCAB), hidden_size)
        # self.W_h_mp = nn.Linear(hidden_size + len(HYPERGRAPH_VOCAB), hidden_size)

        # GRU Weights for nucleotide decoding
        self.W_z_nuc = nn.Linear(hidden_size + len(NUC_VOCAB), hidden_size)
        self.W_r_nuc = nn.Linear(hidden_size + len(NUC_VOCAB), hidden_size)
        self.W_h_nuc = nn.Linear(hidden_size + len(NUC_VOCAB), hidden_size)

        # hypernode label prediction
        self.W_hpn = nn.Linear(hidden_size, len(HYPERGRAPH_VOCAB))
        self.W_hpn_nonlinear = nn.Linear(hidden_size, hidden_size)

        # nucleotide prediction
        self.W_nuc = nn.Linear(hidden_size, len(NUC_VOCAB))
        self.W_nuc_nonlinear = nn.Linear(hidden_size + latent_size, hidden_size)

        # topological prediction
        self.W_topo = nn.Linear(hidden_size, 1)
        self.W_topo_nonlinear = nn.Linear(hidden_size + len(HYPERGRAPH_VOCAB), hidden_size)

        # Loss Functions
        self.hpn_pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.nuc_pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

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
            raise ValueError('aggregate mode is wrong')

    def forward(self, rna_tree_batch, tree_latent_vec, graph_latent_vec):
        # the training procedure which requires teacher forcing
        # actual inference procedure is implemented separately in a decode function

        # hypernode label prediction --> label of the hypergraph node
        hpn_pred_hiddens, hpn_pred_targets = [], []
        # nucleotide label prediction
        nuc_pred_hiddens, nuc_pred_targets = [], []
        # topological prediction --> whether or not a clique has more children
        stop_hiddens, stop_targets = [], []

        # gather all messages from the ground truth structure
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

        # predict root node label
        batch_size = len(rna_tree_batch)
        hpn_pred_hiddens.append(
            torch.cat([tree_latent_vec, torch.zeros(batch_size, self.hidden_size - self.latent_size).to(device)],
                      dim=1))
        hpn_pred_targets.extend([HYPERGRAPH_VOCAB.index(tree.nodes[1].hpn_label) for tree in rna_tree_batch])

        depth_tree_batch = [len(tree.nodes) for tree in rna_tree_batch]
        max_iter = max([len(tr) for tr in traces])
        padding = torch.zeros(self.hidden_size).to(device)
        h = {}

        for batch_idx in range(batch_size):
            offset = sum(depth_tree_batch[:batch_idx])
            h[(offset, offset + 1)] = torch.cat(
                [tree_latent_vec[batch_idx], torch.zeros(self.hidden_size - self.latent_size).to(device)])

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

            for i, (node_x, real_y, _) in enumerate(prop_list):
                batch_idx = batch_list[i]
                offset = sum(depth_tree_batch[:batch_idx])

                # messages flowing into a node for topological prediction
                incoming_msg = [h[(node_y.idx + offset, node_x.idx + offset)] for node_y in node_x.neighbors]
                nb_effective_msg = len(incoming_msg)
                pad_len = MAX_NB - nb_effective_msg
                node_incoming_msg.extend(incoming_msg)
                node_incoming_msg.extend([padding] * pad_len)

                # teacher forcing the ground truth node label
                onehot_enc = np.array(list(map(lambda x: x == node_x.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
                hpn_label.append(onehot_enc)

                # decode a segment of nucleotides in the current hypernode
                if node_x.hpn_label != 'H':
                    try:
                        node_nt_idx = node_x.nt_idx_assignment[nb_effective_msg - 1]
                    except IndexError:
                        exit()
                else:
                    node_nt_idx = node_x.nt_idx_assignment
                if t == 0:
                    seq_input = [np.zeros(len(NUC_VOCAB), dtype=np.float32)]  # start token
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

            # messages for hyper node topological prediction
            hpn_label = torch.as_tensor(np.array(hpn_label)).to(device)
            node_incoming_msg = torch.stack(node_incoming_msg, dim=0).view(-1, MAX_NB, self.hidden_size)
            node_incoming_msg = node_incoming_msg.sum(dim=1)  # [batch_size, hidden_dim]

            # decode a segment as well as compute a new message
            all_len = [len(seq_input) for seq_input in all_seq_input]
            max_len = max(all_len)
            for seq_input in all_seq_input:
                # paddings
                seq_input += [np.zeros(len(NUC_VOCAB), dtype=np.float32)] * (max_len - len(seq_input))
            all_seq_input = torch.as_tensor(np.array(all_seq_input)).to(device)

            all_hidden_states = []
            hidden_states = node_incoming_msg
            for len_idx in range(max_len):
                hidden_states = GRU(all_seq_input[:, len_idx, :], hidden_states,
                                    self.W_z_nuc, self.W_r_nuc, self.W_h_nuc)
                all_hidden_states.append(hidden_states)

            all_hidden_states = torch.stack(all_hidden_states, dim=1).view(-1, self.hidden_size)
            pre_padding_idx = (
                    np.array(list(range(0, len(all_len) * max_len, max_len))) + np.array(all_len) - 1).astype(
                np.long)
            new_h = all_hidden_states.index_select(0, torch.as_tensor(pre_padding_idx).to(device))

            pre_padding_idx = np.concatenate(
                [np.array(list(range(length))) + i * max_len for i, length in enumerate(all_len)]).astype(np.long)
            all_hidden_states = all_hidden_states.index_select(0, torch.as_tensor(pre_padding_idx).to(device))

            # hidden states for nucleotide reconstruction
            batch_idx = [c for i, length in enumerate(all_len) for c in [i] * length]
            all_hidden_states = torch.cat([
                all_hidden_states,
                graph_latent_vec.index_select(0, torch.as_tensor(np.array(batch_idx, dtype=np.long)).to(device))
            ], dim=1)
            nuc_pred_hiddens.append(all_hidden_states)
            nuc_pred_targets.extend(all_nuc_label)

            pred_target, pred_list = [], []
            stop_target = []
            for i, m in enumerate(prop_list):
                # some messages in the prop_list will be used to make new hypernode prediction
                batch_idx = batch_list[i]
                offset = sum(depth_tree_batch[:batch_idx])
                node_x, node_y, direction = m
                x, y = node_x.idx + offset, node_y.idx + offset
                h[(x, y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    # direction where we are expanding (relative to backtracking)
                    # for these we make a prediction about the expanded hypernode's label
                    pred_target.append(HYPERGRAPH_VOCAB.index(node_y.hpn_label))
                    pred_list.append(i)
                stop_target.append(direction)

            # hidden states for stop prediction
            stop_hidden = torch.cat([hpn_label, node_incoming_msg], dim=1)
            stop_hiddens.append(stop_hidden)
            stop_targets.extend(stop_target)

            # hidden states for label prediction
            if len(pred_list) > 0:
                # list where we make label predictions
                cur_pred = torch.as_tensor(np.array(pred_list, dtype=np.long)).to(device)
                hpn_pred_hiddens.append(new_h.index_select(0, cur_pred))
                hpn_pred_targets.extend(pred_target)

        # last stop at the non pseudo root node
        # topological prediction --> no more children
        hpn_label, node_incoming_msg = [], []
        for batch_idx, tree in enumerate(rna_tree_batch):
            offset = sum(depth_tree_batch[:batch_idx])
            node_x = tree.nodes[1]
            onehot_enc = np.array(list(map(lambda x: x == node_x.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
            hpn_label.append(torch.as_tensor(onehot_enc).to(device))
            incoming_msg = [h[(node_y.idx + offset, node_x.idx + offset)] for node_y in node_x.neighbors]
            nb_effective_msg = len(incoming_msg)
            pad_len = MAX_NB - nb_effective_msg
            node_incoming_msg.extend(incoming_msg)
            node_incoming_msg.extend([padding] * pad_len)

        hpn_label = torch.stack(hpn_label, dim=0)
        node_incoming_msg = torch.stack(node_incoming_msg, dim=0).view(-1, MAX_NB, self.hidden_size)
        node_incoming_msg = node_incoming_msg.sum(dim=1)

        stop_hidden = torch.cat([hpn_label, node_incoming_msg], dim=1)
        stop_hiddens.append(stop_hidden)
        stop_targets.extend([0] * batch_size)

        # decode the last segment of the non pseudo root node
        all_seq_input = []
        all_nuc_label = []
        for i in range(batch_size):
            root_node = rna_tree_batch[i].nodes[1]
            if root_node.hpn_label != 'H':
                root_node_nt_idx = root_node.nt_idx_assignment[-1]
            else:
                root_node_nt_idx = root_node.nt_idx_assignment
            if root_node.hpn_label == 'H':
                seq_input = [np.zeros(len(NUC_VOCAB), dtype=np.float32)]  # start token
            else:
                seq_input = []
            for nuc_idx, nuc in enumerate([rna_tree_batch[i].rna_seq[nt_idx] for nt_idx in root_node_nt_idx]):
                onehot_enc = np.array(list(map(lambda x: x == nuc, NUC_VOCAB)), dtype=np.float32)
                seq_input.append(onehot_enc)
                if nuc_idx == 0 and root_node.hpn_label != 'H':
                    continue
                all_nuc_label.append(NUC_VOCAB.index(nuc))
            all_nuc_label.append(NUC_VOCAB.index('<'))
            all_seq_input.append(seq_input)

        # decode a segment as well as compute a new message
        all_len = [len(seq_input) for seq_input in all_seq_input]
        max_len = max(all_len)
        for seq_input in all_seq_input:
            # paddings
            seq_input += [np.zeros(len(NUC_VOCAB), dtype=np.float32)] * (max_len - len(seq_input))
        all_seq_input = torch.as_tensor(np.array(all_seq_input)).to(device)

        all_hidden_states = []
        hidden_states = node_incoming_msg
        for len_idx in range(max_len):
            hidden_states = GRU(all_seq_input[:, len_idx, :], hidden_states,
                                self.W_z_nuc, self.W_r_nuc, self.W_h_nuc)
            all_hidden_states.append(hidden_states)

        all_hidden_states = torch.stack(all_hidden_states, dim=1).view(-1, self.hidden_size)
        pre_padding_idx = np.concatenate(
            [np.array(list(range(length))) + i * max_len for i, length in enumerate(all_len)])
        all_hidden_states = all_hidden_states.index_select(0, torch.as_tensor(pre_padding_idx).to(device))

        # hidden states for segment reconstruction
        batch_idx = [c for i, length in enumerate(all_len) for c in [i] * length]
        all_hidden_states = torch.cat([
            all_hidden_states,
            graph_latent_vec.index_select(0, torch.as_tensor(np.array(batch_idx, dtype=np.long)).to(device))
        ], dim=1)
        nuc_pred_hiddens.append(all_hidden_states)
        nuc_pred_targets.extend(all_nuc_label)

        '''building objective functions'''

        # Predict next clique
        hpn_pred_hiddens = torch.cat(hpn_pred_hiddens, dim=0)
        hpn_pred_scores = self.aggregate(hpn_pred_hiddens, 'word_hpn')
        hpn_pred_targets = torch.as_tensor(np.array(hpn_pred_targets, dtype=np.long)).to(device)
        hpn_pred_loss = self.hpn_pred_loss(hpn_pred_scores, hpn_pred_targets) / len(rna_tree_batch)
        _, hpn_preds = torch.max(hpn_pred_scores, dim=1)
        hpn_pred_acc = torch.eq(hpn_preds, hpn_pred_targets).float()
        hpn_pred_acc = torch.sum(hpn_pred_acc) / hpn_pred_targets.nelement()

        # Predict nucleotides
        nuc_pred_hiddens = torch.cat(nuc_pred_hiddens, dim=0)
        nuc_pred_scores = self.aggregate(nuc_pred_hiddens, 'word_nuc')
        nuc_pred_targets = torch.as_tensor(np.array(nuc_pred_targets, dtype=np.long)).to(device)
        nuc_pred_loss = self.nuc_pred_loss(nuc_pred_scores, nuc_pred_targets) / len(rna_tree_batch)
        _, nuc_preds = torch.max(nuc_pred_scores, dim=1)
        nuc_pred_acc = torch.eq(nuc_preds, nuc_pred_targets).float()
        nuc_pred_acc = torch.sum(nuc_pred_acc) / nuc_pred_targets.nelement()

        # Predict stop
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_scores = self.aggregate(stop_hiddens, 'stop')
        stop_scores = stop_scores.squeeze(-1)
        stop_targets = torch.as_tensor(np.array(stop_targets, dtype=np.float32)).to(device)
        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(rna_tree_batch)
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return (hpn_pred_loss, nuc_pred_loss, stop_loss), \
               (hpn_pred_acc.item(), nuc_pred_acc.item(), stop_acc.item()), \
               h, traces

    def decode_segment(self, last_token, hidden_state, graph_latent_vec):
        decoded_nuc_idx = []
        while True:
            hidden_state = GRU(last_token, hidden_state, self.W_z, self.W_r, self.W_h)
            nuc_pred_score = self.aggregate(torch.cat([hidden_state, graph_latent_vec], dim=1), 'word_nuc')
            _, nuc_idx = torch.max(nuc_pred_score, dim=1)
            nuc_idx = nuc_idx.item()
            if nuc_idx == len(NUC_VOCAB) - 1:
                # < for stop translation
                break
            decoded_nuc_idx.append(nuc_idx)
            last_token = np.array(list(map(lambda x: x == decoded_nuc_idx[-1], range(len(NUC_VOCAB)))),
                                  dtype=np.float32)
            last_token = torch.as_tensor(last_token).to(device)

        return hidden_state, last_token, decoded_nuc_idx

    def decode(self, tree_latent_vec, graph_latent_vec, prob_decode):
        # decode one graph at a time
        # decoding starts from the 5' in a depth first topological order
        assert (tree_latent_vec.size(0) == 1)
        assert (graph_latent_vec.size(0) == 1)

        rna_seq = ''
        # a pseudo node
        pseudo_node = RNAJTNode('P', [])
        pseudo_node.idx = 0
        stack = []
        zero_pad = torch.zeros(1, 1, self.hidden_size).to(device)

        # Root Prediction
        tree_latent_vec = torch.cat([
            tree_latent_vec,
            torch.zeros(1, self.hidden_size - self.latent_size).to(device)], dim=1)
        root_hpn_pred_score = self.aggregate(tree_latent_vec, 'word_hpn')
        _, root_label_idx = torch.max(root_hpn_pred_score, dim=1)
        root_label_idx = root_label_idx.item()

        root = RNAJTNode(HYPERGRAPH_VOCAB[root_label_idx], [pseudo_node])
        root.idx = 1
        pseudo_node.neighbors.append(root)
        stack.append((root, torch.zeros(1, len(NUC_VOCAB)), 0))
        h = {(0, 1): tree_latent_vec}
        all_nodes = [pseudo_node, root]

        for step in range(MAX_DECODE_LEN):
            node_x, last_token, last_token_idx = stack[-1]
            node_incoming_msg = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            if len(node_incoming_msg) > 0:
                node_incoming_msg = torch.stack(node_incoming_msg, dim=0).view(1, -1, self.hidden_size)
            else:
                node_incoming_msg = zero_pad

            hpn_label = np.array(list(map(lambda x: x == node_x.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
            hpn_label = torch.as_tensor(hpn_label).to(device).view(1, len(HYPERGRAPH_VOCAB))

            # Predict stop
            node_incoming_msg = node_incoming_msg.sum(dim=1)
            stop_hidden = torch.cat([hpn_label, node_incoming_msg], dim=1)
            stop_score = self.aggregate(stop_hidden, 'stop')

            if prob_decode:
                backtrack = (torch.bernoulli(torch.sigmoid(stop_score)).item() == 0)
            else:
                backtrack = (stop_score.item() < 0)

            if not backtrack:  # expand the graph: decode a segment and to predict next clique

                hidden_state, last_token, decoded_nuc_idx = \
                    self.decode_segment(last_token, node_incoming_msg, graph_latent_vec)

                # todo, ensure that decoded_nuc_idx is not empty
                if len(decoded_nuc_idx) == 0:
                    return None

                node_x.nt_idx_assignment.append(list(range(last_token_idx - 1 if last_token_idx > 0 else 0,
                                                           last_token_idx + len(decoded_nuc_idx))))
                last_token_idx += len(decoded_nuc_idx)

                new_h = hidden_state
                hpn_pred_score = self.aggregate(new_h, 'word_hpn')
                _, hpn_label_idx = torch.max(hpn_pred_score, dim=1)
                hpn_label_idx = hpn_label_idx.item()

                # todo, eligbility check
                node_y = RNAJTNode(HYPERGRAPH_VOCAB[hpn_label_idx], [])
                node_y.idx = len(all_nodes)
                node_y.neighbors.append(node_x)
                h[(node_x.idx, node_y.idx)] = new_h[0]
                stack.append((node_y, last_token, last_token_idx))
                all_nodes.append(node_y)

            if backtrack:  # Backtrack, use if instead of else
                if len(stack) == 1:
                    break  # At root, terminate

                node_fa, _, _ = stack[-2]

                hidden_state, last_token, decoded_nuc_idx = \
                    self.decode_segment(last_token, node_incoming_msg, graph_latent_vec)

                # todo, ensure that decoded_nuc_idx is not empty
                if len(decoded_nuc_idx) == 0:
                    return None

                if len(node_x.nt_idx_assignment) != 0:
                    node_x.nt_idx_assignment.append(list(range(last_token_idx - 1,
                                                               last_token_idx + len(decoded_nuc_idx))))
                else:
                    # a hairpin node
                    node_x.nt_idx_assignment = list(range(last_token_idx - 1,
                                                          last_token_idx + len(decoded_nuc_idx)))
                # ready to rollout to the next level
                last_token_idx += len(decoded_nuc_idx)

                new_h = hidden_state
                h[(node_x.idx, node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()
                # modify node_fa
                stack[-1] = (node_fa, last_token, last_token_idx)

            rna_seq += ''.join(map(lambda idx: NUC_VOCAB[idx], decoded_nuc_idx))

        return RNAJunctionTree(rna_seq, None, all_nodes)


