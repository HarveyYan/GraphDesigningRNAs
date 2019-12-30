# a composite encoder that simultaneously encodes graph level and tree level structures

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib.nnutils import index_select_ND

HYPERGRAPH_VOCAB = ['H', 'I', 'M', 'S']
# there ain't no F/T anymore
# there is no need to encode/decode the pseudo start node
HPN_FDIM = len(['H', 'I', 'M', 'S'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def send_to_device(*args):
    ret = []
    for item in args:
        ret.append(item.to(device))
    return ret

class TreeEncoder(nn.Module):

    def __init__(self, hidden_size, depth):
        super(TreeEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_w = nn.Linear(2 * self.hidden_size + len(HYPERGRAPH_VOCAB), self.hidden_size)
        self.GRU = GraphGRU(hidden_size + HPN_FDIM, hidden_size, depth=depth)

    def forward(self, nuc_emebedding, f_node_label, f_node_assignment, f_message, node_graph, message_graph, scope):
        nuc_emebedding, f_node_label, f_node_assignment, f_message, node_graph, message_graph = \
            send_to_device(nuc_emebedding, f_node_label, f_node_assignment, f_message, node_graph, message_graph)
        nuc_emb = torch.cat([nuc_emebedding, torch.zeros(1, nuc_emebedding.size(1)).to(device)], dim=0)
        f_node_assignment = index_select_ND(nuc_emb, 0, f_node_assignment).sum(dim=1)  # [nb_nodes, hidden_size]
        f_node = torch.cat([f_node_label, f_node_assignment], dim=1)

        messages = torch.zeros(message_graph.size(0), self.hidden_size).to(device)
        f_message = index_select_ND(f_node, 0, f_message)  # [nb_msg, nb_neighbors, hidden_size+4]
        messages = self.GRU(messages, f_message, message_graph)

        incoming_msg = index_select_ND(messages, 0, node_graph).sum(1)
        hpn_embedding = torch.relu(self.output_w(torch.cat([f_node, incoming_msg], dim=-1)))

        batch_hpn_vec = []
        for start_idx, length in scope:
            # skip the first pseudo node
            # only the root vector is kept
            # todo, does including other nodes really confuse the decoding stage?
            batch_hpn_vec.append(hpn_embedding[start_idx + 1])

        return messages, torch.stack(batch_hpn_vec)

    @staticmethod
    def prepare_batch_data(rna_tree_batch):
        scope = []

        messages, message_dict = [None], {}
        f_node_label = []
        f_node_assignment = []

        tree_node_offset = 0
        graph_nuc_offset = 0

        for tree in rna_tree_batch:
            for node in tree.nodes:
                onehot_enc = np.array(list(map(lambda x: x == node.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
                f_node_label.append(torch.as_tensor(onehot_enc))

                if type(node.nt_idx_assignment) is list:
                    nt_idx_assignment = node.nt_idx_assignment
                else:
                    nt_idx_assignment = node.nt_idx_assignment.tolist()
                # the nucleotides embeddings inside a subgraph are simply averaged
                # todo better pooling tactics
                if type(nt_idx_assignment[0]) is int:
                    f_node_assignment.append([nt_idx + graph_nuc_offset for nt_idx in nt_idx_assignment])
                else:
                    f_node_assignment.append(list(np.concatenate(nt_idx_assignment, axis=0) + graph_nuc_offset))

                if node.hpn_label != 'P':
                    for neighbor_node in node.neighbors:
                        if neighbor_node.hpn_label == 'P':  # skip the pseudo start node
                            continue
                        message_dict[(node.idx + tree_node_offset, neighbor_node.idx + tree_node_offset)] = len(messages)
                        messages.append((node, neighbor_node, tree_node_offset))

            scope.append([tree_node_offset, len(tree.nodes)])
            tree_node_offset += len(tree.nodes)
            graph_nuc_offset += len(tree.rna_seq)

        total_nucleotides = graph_nuc_offset
        total_nodes = tree_node_offset
        total_messages = len(messages)
        node_graph = [[] for i in range(total_nodes)]
        message_graph = [[] for i in range(total_messages)]
        f_message = [0] * len(messages)

        for node_from, node_to, tree_node_offset in messages[1:]:
            msg_idx = message_dict[(node_from.idx + tree_node_offset, node_to.idx + tree_node_offset)]
            f_message[msg_idx] = node_from.idx + tree_node_offset
            # message passed from node_from to node_to
            node_graph[node_to.idx + tree_node_offset].append(msg_idx)
            for neighbor_node in node_to.neighbors:
                if neighbor_node.idx == node_from.idx or neighbor_node.hpn_label == 'P':
                    continue
                else:
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

        node_graph = torch.as_tensor(np.array(node_graph, dtype=np.long))
        message_graph = torch.as_tensor(np.array(message_graph, dtype=np.long))
        f_node_label = torch.stack(f_node_label)
        f_node_assignment = torch.as_tensor(np.array(f_node_assignment, dtype=np.long))
        f_message = torch.as_tensor(np.array(f_message, dtype=np.long))

        return f_node_label, f_node_assignment, f_message, node_graph, message_graph, scope


class GraphGRU(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, messages, local_field, mess_graph):
        mask = torch.ones(messages.size(0), 1).to(device)
        mask[0] = 0  # first vector is padding

        for it in range(self.depth):

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
