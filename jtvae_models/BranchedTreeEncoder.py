# a composite encoder that simultaneously encodes graph level and tree level structures

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import networkx as nx
from lib.nn_utils import index_select_ND
from lib.tree_decomp import dfs

HYPERGRAPH_VOCAB = ['H', 'I', 'M', 'S']
# there ain't no F/T anymore
# there is no need to encode/decode the pseudo start node
HPN_FDIM = len(['H', 'I', 'M', 'S'])

NUC_VOCAB = ['A', 'C', 'G', 'U']
NUC_FDIM = len(NUC_VOCAB)


class BranchedTreeEncoder(nn.Module):

    def __init__(self, hidden_size, depth, **kwargs):
        super(BranchedTreeEncoder, self).__init__()
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_w = nn.Linear(HPN_FDIM + hidden_size * 2, hidden_size)
        self.GRU = GraphGRU(HPN_FDIM + hidden_size, hidden_size, depth=depth, **kwargs)

        # bidirectional hence hidden_size//2
        # scan the tree nodes in the order it will be traversed later in decoding
        self.jt_order_lstm = torch.nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)

    def send_to_device(self, *args):
        ret = []
        for item in args:
            ret.append(item.to(self.device))
        return ret

    def forward(self, nuc_embedding, f_node_label, f_node_assignment, f_message, node_graph, message_graph, scope,
                all_dfs_idx):

        f_node_label, f_node_assignment, f_message, node_graph, message_graph = \
            self.send_to_device(f_node_label, f_node_assignment, f_message, node_graph, message_graph)

        nuc_emb = torch.cat([nuc_embedding, torch.zeros(1, self.hidden_size).to(self.device)], dim=0)
        f_node_assignment = index_select_ND(nuc_emb, 0, f_node_assignment).max(dim=1)[0]  # [nb_nodes, hidden_size]
        f_node = torch.cat([f_node_label, f_node_assignment], dim=1)

        ''' begin tree messages iteration'''
        messages = torch.zeros(message_graph.size(0), self.hidden_size).to(self.device)
        f_message = index_select_ND(
            torch.cat([f_node, torch.zeros(1, HPN_FDIM + self.hidden_size).to(self.device)], dim=0),
            0, f_message).sum(dim=1)  # [nb_msg, hidden_size]
        messages = self.GRU(messages, f_message, message_graph)  # bottom-up and top-down phases

        incoming_msg = index_select_ND(messages, 0, node_graph).sum(1)
        f_node = index_select_ND(f_node, 0, node_graph).sum(1)
        hpn_embedding = torch.relu(self.output_w(torch.cat([f_node, incoming_msg], dim=-1)))

        ''' bilstm to add order information into the learnt node embeddings '''
        '''depth first ordering of the nodes'''
        all_len = list([len(trace) for trace in all_dfs_idx])
        max_len = max(all_len)
        batch_size = len(all_dfs_idx)
        all_pre_padding_idx = np.concatenate(
            [np.array(list(range(length))) + i * max_len for i, length in enumerate(all_len)]).astype(np.long)

        batch_jt_vec = []
        for dfs_idx in all_dfs_idx:
            batch_jt_vec.append(hpn_embedding[torch.as_tensor(dfs_idx).to(self.device)])  # skip the pseudo node

        # [batch_size, max_len, hidden_size]
        padded_jt_vec = nn.utils.rnn.pad_sequence(batch_jt_vec, batch_first=True)
        packed_jt_vec = nn.utils.rnn.pack_padded_sequence(
            padded_jt_vec, all_len, enforce_sorted=False, batch_first=True)

        output, _ = self.jt_order_lstm(packed_jt_vec)
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        output = output.reshape(batch_size * max_len, self.hidden_size). \
            index_select(0, torch.as_tensor(all_pre_padding_idx).to(self.device))

        representation = []
        start = 0
        for length in all_len:
            representation.append(torch.max(output[start: start + length], dim=0)[0])
            start += length

        return torch.stack(representation, dim=0)

    @staticmethod
    def prepare_batch_data(rna_tree_batch):
        scope = []
        all_dfs_idx = []

        messages, message_dict = [None], {}
        f_node_label = []
        f_node_assignment = []
        all_nb_branches = []

        tree_node_offset = 0
        graph_nuc_offset = 0

        for tree in rna_tree_batch:

            s = []
            dfs(s, tree.nodes[1], 0)
            all_traces = [tup[0].idx + tree_node_offset for tup in s]
            if len(all_traces) > 0:
                all_dfs_idx.append(all_traces)
            else:
                all_dfs_idx.append([1 + tree_node_offset])  # blunder solved here!

            for node in np.array(tree.nodes):
                onehot_enc = np.array(list(map(lambda x: x == node.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)

                if type(node.nt_idx_assignment) is list:
                    nt_idx_assignment = node.nt_idx_assignment
                else:
                    nt_idx_assignment = node.nt_idx_assignment.tolist()

                if type(nt_idx_assignment[0]) is int:
                    nb_branches = 1
                    f_node_assignment.append([nt_idx + graph_nuc_offset for nt_idx in nt_idx_assignment])
                    f_node_label.append(torch.as_tensor(onehot_enc))
                else:
                    nb_branches = len(nt_idx_assignment)
                    f_node_assignment.extend(
                        [[nt_idx + graph_nuc_offset for nt_idx in segment] for segment in
                         nt_idx_assignment])
                    f_node_label.extend([torch.as_tensor(onehot_enc)] * len(nt_idx_assignment))

                all_nb_branches.append(nb_branches)

                for nb_idx, neighbor_node in enumerate(node.neighbors):
                    message_dict[(node.idx + tree_node_offset, neighbor_node.idx + tree_node_offset)] = len(messages)
                    messages.append((node, neighbor_node, nb_idx, tree_node_offset))

            scope.append([tree_node_offset, len(tree.nodes)])
            tree_node_offset += len(tree.nodes)
            graph_nuc_offset += len(tree.rna_seq)

        total_nucleotides = graph_nuc_offset
        total_nodes = tree_node_offset
        total_messages = len(messages)
        node_graph = [[] for _ in range(total_nodes)]
        message_graph = [[] for _ in range(total_messages)]
        f_message = [[] for _ in range(total_messages)]

        for node_from, node_to, nb_idx, tree_node_offset in messages[1:]:

            if node_from.idx < node_to.idx:
                is_backtrack = False
            else:
                is_backtrack = True

            # nb_idx is the index of node_to in the neighborhood of node_from
            msg_idx = message_dict[(node_from.idx + tree_node_offset, node_to.idx + tree_node_offset)]

            branch_offset = sum(all_nb_branches[:node_from.idx + tree_node_offset])
            # todo, nb_idx is no longer quite useful as the parent node may appear the first in the neighbors list
            if is_backtrack:
                f_message[msg_idx] = list(np.array(range(len(node_from.neighbors))) + branch_offset)
            else:
                all_nb_idx = [node.idx for node in node_from.neighbors]

                if node_from.hpn_label != 'P':
                    # remove the parent node index
                    # and determine node_to's position relative to its sibling
                    par_idx = min(all_nb_idx)
                    all_nb_idx.remove(par_idx)

                node_to_nb_idx = all_nb_idx.index(node_to.idx)
                f_message[msg_idx] = list(np.array(range(node_to_nb_idx + 1)) + branch_offset)
            # message passed from node_from to node_to
            node_graph[node_to.idx + tree_node_offset].append(msg_idx)

            if node_to.hpn_label == 'P':
                continue

            for neighbor_node in node_to.neighbors:
                # neighbor_node.idx != min([node.idx for node in node_to.neighbors])
                # detects if neighbor_node is the parent node of node_to
                if is_backtrack and neighbor_node.idx <= node_from.idx and neighbor_node.idx != min([node.idx for node in node_to.neighbors]):
                    continue

                necessary_msg_idx = message_dict[
                    (node_to.idx + tree_node_offset, neighbor_node.idx + tree_node_offset)]
                # that the computation of this message depends on the message from (node_from, node_to)
                message_graph[necessary_msg_idx].append(msg_idx)

        max_len = max([len(t) for t in node_graph] + [1])
        # 1 is for the special case where we have only one clique
        for t in node_graph:
            t.extend([0] * (max_len - len(t)))

        max_len = max([len(t) for t in message_graph] + [1])
        for t in message_graph:
            t.extend([0] * (max_len - len(t)))

        max_len = max([len(t) for t in f_node_assignment])
        for t in f_node_assignment:
            t.extend([total_nucleotides] * (max_len - len(t)))

        max_len = max([len(t) for t in f_message])
        for t in f_message:
            t.extend([sum(all_nb_branches)] * (max_len - len(t)))

        node_graph = torch.as_tensor(np.array(node_graph, dtype=np.long))
        message_graph = torch.as_tensor(np.array(message_graph, dtype=np.long))
        f_node_label = torch.stack(f_node_label)
        f_node_assignment = torch.as_tensor(np.array(f_node_assignment, dtype=np.long))
        f_message = torch.as_tensor(np.array(f_message, dtype=np.long))

        # cannot pass large list to pytorch dataloader in multiprocessing setting
        return f_node_label, f_node_assignment, f_message, node_graph, message_graph, scope, all_dfs_idx


class GraphGRU(nn.Module):

    def __init__(self, input_size, hidden_size, depth, **kwargs):
        super(GraphGRU, self).__init__()
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth  # a suggestive time of unrolling GRU

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, messages, local_field, mess_graph, unroll_depth=None):
        mask = torch.ones(messages.size(0), 1).to(self.device)
        mask[0] = 0  # first vector is padding

        for it in range(self.depth if unroll_depth is None else unroll_depth):
            msg_nei = index_select_ND(messages, 0, mess_graph)
            # [nb_msg, nb_neighbors, hidden_dim]
            sum_msg = msg_nei.sum(dim=1)
            z_input = torch.cat([local_field, sum_msg], dim=1)
            z = torch.sigmoid(self.W_z(z_input))

            r_1 = self.W_r(local_field).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(msg_nei)
            r = torch.sigmoid(r_1 + r_2)

            gated_msg = r * msg_nei
            sum_gated_msg = gated_msg.sum(dim=1)
            msg_input = torch.cat([local_field, sum_gated_msg], dim=1)
            pre_msg = torch.tanh(self.W_h(msg_input))
            messages = (1.0 - z) * sum_msg + z * pre_msg
            messages = messages * mask

        return messages