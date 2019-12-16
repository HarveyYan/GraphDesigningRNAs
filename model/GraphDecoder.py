# This model only decodes the junction tree

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUC_VOCAB = ['A', 'C', 'G', 'U']
HYPERGRAPH_VOCAB = ['F', 'T', 'H', 'I', 'M', 'S']

MAX_NB = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def GRU(x, memory, W_z, W_r, W_h):

    input = torch.cat([x, memory], dim=1)
    z = F.sigmoid(W_z(input))
    r = F.sigmoid(W_r(input))

    gated_h = r * memory
    h_input = torch.cat([x, gated_h], dim=1)
    pre_h = F.tanh(W_h(h_input))
    new_h = (1.0 - z) * memory + z * pre_h

    return new_h


class GraphDecoder(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super(GraphDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # GRU Weights
        self.W_z = nn.Linear(hidden_size + len(HYPERGRAPH_VOCAB), hidden_size)
        self.W_r = nn.Linear(len(HYPERGRAPH_VOCAB), hidden_size)
        self.W_h = nn.Linear(hidden_size + len(HYPERGRAPH_VOCAB), hidden_size)

        # Word Prediction Weights
        self.W = nn.Linear(hidden_size + latent_size, hidden_size)

        # Stop Prediction Weights
        self.U = nn.Linear(hidden_size + latent_size, hidden_size)
        self.U_i = nn.Linear(hidden_size + len(HYPERGRAPH_VOCAB), hidden_size)

        # Output Weights
        self.W_o = nn.Linear(hidden_size, len(HYPERGRAPH_VOCAB))
        self.U_o = nn.Linear(hidden_size, 1)

        # Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        if mode == 'word':
            V, V_o = self.W, self.W_o
        elif mode == 'stop':
            V, V_o = self.U, self.U_o
        else:
            raise ValueError('aggregate mode is wrong')

        tree_contexts = x_tree_vecs.index_select(0, contexts)
        input_vec = torch.cat([hiddens, tree_contexts], dim=-1)
        output_vec = F.relu(V(input_vec))
        return V_o(output_vec)

    def forward(self, rna_tree_batch, tree_messages, traces):
        # this is the training procedure which requires teacher forcing
        # actual inference procedure is implemented separately in a decode function

        # hypernode label prediction --> label of the hypergraph node
        pred_hiddens, pred_contexts, pred_targets = [], [], []

        # predict root node label
        batch_size = len(rna_tree_batch)
        # start token of the decoding procedure, which is usually an empty vector
        pred_hiddens.append(torch.zeros(batch_size, self.hidden_size).to(device))
        # the first hypergraph node, which is usually a dangling start
        pred_targets.extend([HYPERGRAPH_VOCAB.index(tree.nodes[0].hpn_label) for tree in rna_tree_batch])
        pred_contexts.append(torch.as_tensor(
            np.array(list(range(batch_size)), dtype=np.long)).to(device))

        depth_tree_batch = [len(tr) for tr in traces]
        max_iter = max(depth_tree_batch)
        padding = torch.zeros(self.hidden_size).to(device)
        h = {}

        for t in range(max_iter):

            prop_list = []
            batch_list = []

            for i, plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei, cur_o_nei = [], []

            for i, (node_x, real_y, _) in enumerate(prop_list):
                batch_idx = batch_list[i]
                offset = sum(depth_tree_batch[:batch_idx])

                # Neighbors for message passing (target not included)
                cur_nei = [h[(node_y.idx + offset, node_x.idx + offset)] for node_y in node_x.neighbors if
                           node_y.idx != real_y.idx]
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                # Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx + offset, node_x.idx + offset)] for node_y in node_x.neighbors]
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                # teacher forcing, input to the gru should be the ground truth hypergraph tree node
                onehot_enc = np.array(list(map(lambda x: x == node_x.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
                cur_x.append(torch.as_tensor(onehot_enc).to(device))

            cur_x = torch.stack(cur_x, dim=0)
            # Message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0). \
                view(-1, MAX_NB, self.hidden_size)  # [batch_size, max_nei, hidden_size]
            # updated messages from node x to one of its neighbor
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            # Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)  # [batch_size, hidden_dim]

            # Gather targets
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
                    pred_target.append(HYPERGRAPH_VOCAB.index(node_y.hpn_label))
                    pred_list.append(i)
                stop_target.append(direction)

            # Hidden states for stop prediction
            cur_batch = torch.as_tensor(np.array(batch_list, dtype=np.long)).to(device)
            stop_hidden = torch.cat([cur_x, cur_o], dim=1)
            stop_hiddens.append(stop_hidden)
            stop_contexts.append(cur_batch)
            stop_targets.extend(stop_target)

            # Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                # list where we make label predictions
                cur_batch = torch.as_tensor(np.array(batch_list, dtype=np.long)).to(device)
                pred_contexts.append(cur_batch)

                cur_pred = torch.as_tensor(np.array(pred_list, dtype=np.long)).to(device)
                pred_hiddens.append(new_h.index_select(0, cur_pred))
                pred_targets.extend(pred_target)

        # Last stop at root
        # toplogical prediction --> no more children
        cur_x, cur_o_nei = [], []
        for batch_idx, tree in enumerate(rna_tree_batch):
            offset = sum(depth_tree_batch[:batch_idx])
            node_x = tree.nodes[0]
            onehot_enc = np.array(list(map(lambda x: x == node_x.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
            cur_x.append(torch.as_tensor(onehot_enc).to(device))
            cur_nei = [h[(node_y.idx + offset, node_x.idx + offset)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = torch.stack(cur_x, dim=0)
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x, cur_o], dim=1)
        stop_hiddens.append(stop_hidden)
        stop_contexts.append(torch.as_tensor(
            np.array(list(range(batch_size)), dtype=np.long)).to(device))
        stop_targets.extend([0] * len(rna_tree_batch))

        # parts for building the objective function
        # Predict next clique
        pred_contexts = torch.cat(pred_contexts, dim=0)
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_scores = self.aggregate(pred_hiddens, pred_contexts, tree_latent_vec, 'word')
        pred_targets = torch.as_tensor(np.array(pred_targets, dtype=np.long)).to(device)

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(rna_tree_batch)
        _, preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        # Predict stop
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_hiddens = F.relu(self.U_i(stop_hiddens))
        stop_scores = self.aggregate(stop_hiddens, stop_contexts, tree_latent_vec, 'stop')
        stop_scores = stop_scores.squeeze(-1)
        stop_targets = torch.as_tensor(np.array(stop_targets, dtype=np.float32)).to(device)

        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(rna_tree_batch)
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item(), h, traces
