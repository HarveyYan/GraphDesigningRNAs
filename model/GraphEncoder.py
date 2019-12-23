# a composite encoder that simultaneously encodes graph level and tree level structures

import torch
import torch.nn as nn
import forgi.graph.bulge_graph as fgb
import torch.nn.functional as F
import numpy as np
from lib.nnutils import index_select_ND

NUC_VOCAB = ['A', 'C', 'G', 'U']
NUC_FDIM = 4

BOND_FDIM = 4
# 5' to 3' covalent bond,
# 3' to 5' covalent bond,
# 5' to 3' bp bond,
# 3' to 5' bp bond

MAX_NB = 3
# maximal number of incoming messages

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def send_to_device(*args):
    ret = []
    for item in args:
        ret.append(item.to(device))
    return ret


class GraphEncoder(nn.Module):

    def __init__(self, hidden_size, depth):
        super(GraphEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.w_local = nn.Linear(NUC_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.w_msg = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_node_emb = nn.Linear(NUC_FDIM + hidden_size, hidden_size, bias=False)

    def forward(self, f_nuc, f_bond, node_graph, message_graph, scope):
        f_nuc, f_bond, node_graph, message_graph = \
            send_to_device(f_nuc, f_bond, node_graph, message_graph)
        local_potentials = self.w_local(f_bond)
        # messages from the first iteration
        messages = F.relu(local_potentials)

        for i in range(1, self.depth):
            nei_message = index_select_ND(messages, 0, message_graph)
            sum_nei_message = nei_message.sum(dim=1)
            nb_clique_msg_prop = self.w_msg(sum_nei_message)
            messages = F.relu(local_potentials + nb_clique_msg_prop)

        nuc_nb_msg = index_select_ND(messages, 0, node_graph).sum(dim=1)
        nuc_embedding = F.relu(self.w_node_emb(torch.cat([f_nuc, nuc_nb_msg], dim=1)))

        # global averaging pooling to obtain the graph level embedding
        batch_rna_vec = []
        for start_idx, length in scope:
            batch_rna_vec.append(nuc_embedding[start_idx: start_idx + length].mean(dim=0))

        return nuc_embedding, torch.stack(batch_rna_vec)

    @staticmethod
    def prepare_batch_data(rna_mol_batch):
        # sparse encoding, index operations

        nuc_offset = 0

        f_nuc, f_bond = [], [torch.zeros(NUC_FDIM + BOND_FDIM)]
        # nucleotide features and bond features, merely one hot encoding at this stage

        in_bonds, all_bonds = [], [(-1, -1)]
        # keeps the indices of incoming messages

        scope = []

        for rna_seq, rna_struct in rna_mol_batch:
            len_seq = len(rna_seq)
            for nuc in rna_seq:
                onehot_enc = np.array(list(map(lambda x: x == nuc, NUC_VOCAB)), dtype=np.float32)
                f_nuc.append(torch.as_tensor(onehot_enc))
                in_bonds.append([])

            # authentic molecular graph
            bg = fgb.BulgeGraph.from_dotbracket(rna_struct)
            for i, st_ele in enumerate(rna_struct):
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
                else:
                    cb_from = i + nuc_offset

                    if i < len_seq - 1:
                        cb_to = cb_from + 1
                        idx_ref = len(all_bonds)
                        all_bonds.append([cb_from, cb_to])
                        f_bond.append(torch.cat(
                            [f_nuc[cb_from], torch.as_tensor(
                                np.array([1., 0., 0., 0.], dtype=np.float32))]))
                        in_bonds[cb_to].append(idx_ref)

                    if i > 0:
                        cb_to = cb_from - 1
                        idx_ref = len(all_bonds)
                        all_bonds.append([cb_from, cb_to])
                        f_bond.append(torch.cat(
                            [f_nuc[cb_from], torch.as_tensor(
                                np.array([0., 1., 0., 0.], dtype=np.float32))]))
                        in_bonds[cb_to].append(idx_ref)

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
            try:
                nuc_idx_from, nuc_idx_to = all_bonds[bond_idx]
            except ValueError:
                print('yikes')
            for i, msg_idx in enumerate(in_bonds[nuc_idx_from]):
                if all_bonds[msg_idx][0] != nuc_idx_to:
                    message_graph[bond_idx, i] = msg_idx

        return f_nuc, f_bond, node_graph, message_graph, scope
