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

MAX_TREE_DECODE_STEPS = 500
MAX_SEGMENT_LENGTH = 100
MIN_HAIRPIN_LENGTH = 3
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

    def __init__(self, hidden_size, latent_size):
        super(UnifiedDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # GRU Weights for message passing
        self.W_z_mp = nn.Linear(hidden_size + len(HYPERGRAPH_VOCAB), hidden_size)
        self.U_r_mp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r_mp = nn.Linear(len(HYPERGRAPH_VOCAB), hidden_size)
        self.W_h_mp = nn.Linear(hidden_size + len(HYPERGRAPH_VOCAB), hidden_size)

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
        # self.W_topo_nonlinear = nn.Linear(hidden_size + len(HYPERGRAPH_VOCAB), hidden_size)
        self.W_topo_nonlinear = nn.Linear(hidden_size, hidden_size)

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

    def teacher_forced_decoding(self, all_seq_input, hidden_states, graph_latent_vec):
        # decode a segment as well as compute a new message
        all_len = [len(seq_input) for seq_input in all_seq_input]
        max_len = max(all_len)
        for seq_input in all_seq_input:
            # paddings
            seq_input += [np.zeros(len(NUC_VOCAB), dtype=np.float32)] * (max_len - len(seq_input))
        all_seq_input = torch.as_tensor(np.array(all_seq_input)).to(device)

        all_hidden_states = []
        for len_idx in range(max_len):
            hidden_states = GRU(all_seq_input[:, len_idx, :], hidden_states,
                                self.W_z_nuc, self.W_r_nuc, self.W_h_nuc)
            all_hidden_states.append(hidden_states)

        all_hidden_states = torch.stack(all_hidden_states, dim=1).view(-1, self.hidden_size)
        pre_padding_idx = (np.array(list(range(0, len(all_len) * max_len, max_len)))
                           + np.array(all_len) - 1).astype(np.long)
        # the last hidden state at each segment
        new_h = all_hidden_states.index_select(0, torch.as_tensor(pre_padding_idx).to(device))

        pre_padding_idx = np.concatenate(
            [np.array(list(range(length))) + i * max_len for i, length in enumerate(all_len)]).astype(np.long)
        all_hidden_states = all_hidden_states.index_select(0, torch.as_tensor(pre_padding_idx).to(device))

        # hidden states to reconstruct segments of nucleotides
        batch_idx = [c for i, length in enumerate(all_len) for c in [i] * length]
        all_hidden_states = torch.cat([
            all_hidden_states,
            graph_latent_vec.index_select(0, torch.as_tensor(np.array(batch_idx, dtype=np.long)).to(device))
        ], dim=1)

        return new_h, all_hidden_states

    def forward(self, rna_tree_batch, tree_latent_vec, graph_latent_vec):
        # the training procedure which requires teacher forcing
        # actual inference procedure is implemented separately in a decode function

        # hypernode label prediction: label of the hypergraph node
        hpn_pred_hiddens, hpn_pred_targets = [], []
        # nucleotide label prediction: segments of nucleotides
        nuc_pred_hiddens, nuc_pred_targets = [], []
        # topological prediction: more children or not for the current node
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
        tree_latent_vec = torch.cat([
            tree_latent_vec,
            torch.zeros(batch_size, self.hidden_size - self.latent_size).to(device)
        ], dim=1)
        hpn_pred_hiddens.append(tree_latent_vec)
        hpn_pred_targets.extend([HYPERGRAPH_VOCAB.index(tree.nodes[1].hpn_label) for tree in rna_tree_batch])

        depth_tree_batch = [len(tree.nodes) for tree in rna_tree_batch]
        max_iter = max([len(tr) for tr in traces])
        msg_padding = torch.zeros(self.hidden_size).to(device)
        nuc_padding = np.zeros(len(NUC_VOCAB), dtype=np.float32)
        h = {}

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

            for i, (node_x, real_y, _) in enumerate(prop_list):
                batch_idx = batch_list[i]
                offset = sum(depth_tree_batch[:batch_idx])

                # messages flowing into a node for topological prediction
                incoming_msg = [h[(node_y.idx + offset, node_x.idx + offset)] for node_y in node_x.neighbors]
                nb_effective_msg = len(incoming_msg)
                node_incoming_msg.append(incoming_msg)

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

            # messages for hyper node topological prediction
            hpn_label = torch.as_tensor(np.array(hpn_label)).to(device)
            all_len = [len(incoming_msg) for incoming_msg in node_incoming_msg]
            max_len = max(all_len)
            for incoming_msg in node_incoming_msg:
                incoming_msg += [msg_padding] * (max_len - len(incoming_msg))
            node_incoming_msg = torch.stack([msg for incoming_msg in node_incoming_msg for msg in incoming_msg],
                                            dim=0).view(-1, max_len, self.hidden_size)
            # node_incoming_msg = node_incoming_msg.sum(dim=1)  # [batch_size, hidden_dim]

            node_incoming_msg = GraphGRU(hpn_label, node_incoming_msg,
                                         self.W_z_mp, self.W_r_mp, self.U_r_mp, self.W_h_mp)

            new_h, all_hidden_states = self.teacher_forced_decoding(
                all_seq_input, node_incoming_msg, graph_latent_vec)
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
            # stop_hidden = torch.cat([hpn_label, node_incoming_msg], dim=1)
            stop_hiddens.append(node_incoming_msg)
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
            node_incoming_msg.append(incoming_msg)

        hpn_label = torch.stack(hpn_label, dim=0)
        all_len = [len(incoming_msg) for incoming_msg in node_incoming_msg]
        max_len = max(all_len)
        for incoming_msg in node_incoming_msg:
            incoming_msg += [msg_padding] * (max_len - len(incoming_msg))
        node_incoming_msg = torch.stack(
            [msg for incoming_msg in node_incoming_msg for msg in incoming_msg], dim=0). \
            view(-1, max_len, self.hidden_size)

        node_incoming_msg = GraphGRU(hpn_label, node_incoming_msg,
                                     self.W_z_mp, self.W_r_mp, self.U_r_mp, self.W_h_mp)

        # stop_hidden = torch.cat([hpn_label, node_incoming_msg], dim=1)
        stop_hiddens.append(node_incoming_msg)
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
                seq_input = [nuc_padding]  # start token
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

        new_h, all_hidden_states = self.teacher_forced_decoding(
            all_seq_input, node_incoming_msg, graph_latent_vec)
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

    ########################################
    # decoding RNA with regularity constraint
    ########################################

    def decode_segment_with_constraint(self, last_token, hidden_state, graph_latent_vec, **kwargs):
        prob_decode = kwargs.get('prob_deocde', False)
        minimal_length = kwargs.get('minimal_length', 0)
        last_nuc_complement_to_idx = kwargs.get('last_nuc_complement_to_idx', None)
        second_stem_segment_complement_to_idx = kwargs.get('second_stem_segment_complement_to_idx', None)
        # properly reversed

        decoded_nuc_idx = []
        stop_symbol_mask = torch.as_tensor(np.array([0., 0., 0., 0., -99999.], dtype=np.float32)).to(device)
        first_nuc_idx = int(np.argmax(last_token.data.numpy()))

        if second_stem_segment_complement_to_idx is not None:
            # decode the second segment of stem completely complementarily to the given first segment

            if allowed_basepairs[first_nuc_idx][second_stem_segment_complement_to_idx[0]] is False:
                # check that we have the right nucleotide to begin with
                print('stem decoding error')
                return None
                # raise ValueError('First nucleotide complementarity not met while decoding stem.')

            for first_seg_nuc_idx in second_stem_segment_complement_to_idx[1:]:
                hidden_state = GRU(last_token, hidden_state, self.W_z_nuc, self.W_r_nuc, self.W_h_nuc)
                nuc_pred_score = self.aggregate(torch.cat([hidden_state, graph_latent_vec], dim=1), 'word_nuc')
                mask = (torch.as_tensor(((np.array(allowed_basepairs[first_seg_nuc_idx] + [False]) - 1) *
                                         99999).astype(np.float32))).to(device)
                nuc_pred_score = nuc_pred_score + mask

                if prob_decode:
                    nuc_idx = torch.multinomial(torch.softmax(nuc_pred_score, dim=1), num_samples=1)
                else:
                    _, nuc_idx = torch.max(nuc_pred_score, dim=1)
                nuc_idx = nuc_idx.item()

                decoded_nuc_idx.append(nuc_idx)
                last_token = np.array(list(map(lambda x: x == nuc_idx, range(len(NUC_VOCAB)))),
                                      dtype=np.float32)[None, :]
                last_token = torch.as_tensor(last_token).to(device)

            # this one is for the stop token
            hidden_state = GRU(last_token, hidden_state, self.W_z_nuc, self.W_r_nuc, self.W_h_nuc)
        else:
            decode_step = 0
            while decode_step < MAX_SEGMENT_LENGTH:
                hidden_state = GRU(last_token, hidden_state, self.W_z_nuc, self.W_r_nuc, self.W_h_nuc)
                nuc_pred_score = self.aggregate(torch.cat([hidden_state, graph_latent_vec], dim=1), 'word_nuc')

                # length constraint
                # sometimes we want a hairpin to be at least 3 nucleotides long
                # or an internal loop segment is not empty
                if decode_step < minimal_length:
                    nuc_pred_score = nuc_pred_score + stop_symbol_mask

                # complementarity constraint
                # when decoding a non pseudo hairpin loop or the second segment of internal loop
                # we want the last nucleotide decoded to be reversely complementary to an earlier one
                elif decode_step > 0 and last_nuc_complement_to_idx is not None and \
                        allowed_basepairs[last_nuc_complement_to_idx][decoded_nuc_idx[-1]] is False:
                    # the last decoded nucleotide cannot complement the start nucleotide
                    # hence we will not stop at this iteration
                    nuc_pred_score = nuc_pred_score + stop_symbol_mask

                if prob_decode:
                    nuc_idx = torch.multinomial(torch.softmax(nuc_pred_score, dim=1), num_samples=1)
                else:
                    _, nuc_idx = torch.max(nuc_pred_score, dim=1)
                nuc_idx = nuc_idx.item()

                if nuc_idx == len(NUC_VOCAB) - 1:
                    # < for stop translation
                    break
                decoded_nuc_idx.append(nuc_idx)
                last_token = np.array(list(map(lambda x: x == nuc_idx, range(len(NUC_VOCAB)))),
                                      dtype=np.float32)[None, :]
                last_token = torch.as_tensor(last_token).to(device)
                decode_step += 1

            if last_nuc_complement_to_idx is not None and \
                    allowed_basepairs[last_nuc_complement_to_idx][decoded_nuc_idx[-1]] is False:
                # raise ValueError('Last nucleotide complementarity not met while decoding loops (H/I/M).')
                print(''.join([NUC_VOCAB[idx] for idx in decoded_nuc_idx]))
                return None

        return hidden_state, decoded_nuc_idx, last_token

    def decode_segment(self, current_node, last_token, hidden_state, graph_latent_vec, prob_decode, **kwargs):

        if current_node.hpn_label == 'H':

            if current_node.idx > 1:
                # parent node is a stem, thus current node is
                # somewhere in the middle of this RNA structure
                start_nuc_idx = int(np.argmax(last_token.data.numpy()))

                res = self.decode_segment_with_constraint(
                    last_token, hidden_state, graph_latent_vec, prob_deocde=prob_decode,
                    minimal_length=MIN_HAIRPIN_LENGTH, last_nuc_complement_to_idx=start_nuc_idx)

                if res is None:
                    raise ValueError('Hairpin — last decoded nucleotide complementarity failed to hold.')
                else:
                    hidden_state, decoded_nuc_idx, last_token = res
            else:
                # the first non pseudo root node
                # hence no complementarity constraint
                # todo, length constraints for completely single stranded RNA
                res = self.decode_segment_with_constraint(
                    last_token, hidden_state, graph_latent_vec, prob_deocde=prob_decode)

                hidden_state, decoded_nuc_idx, last_token = res

        elif current_node.hpn_label == 'I':

            if len(current_node.nt_idx_assignment) == 0:

                res = self.decode_segment_with_constraint(
                    last_token, hidden_state, graph_latent_vec,
                    prob_deocde=prob_decode, minimal_length=1)

                hidden_state, decoded_nuc_idx, last_token = res
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

                res = self.decode_segment_with_constraint(
                    last_token, hidden_state, graph_latent_vec, prob_deocde=prob_decode,
                    minimal_length=min_internal_loop_length, last_nuc_complement_to_idx=start_nuc_idx)

                if res is None:
                    raise ValueError('Internal loop — last decoded nucleotide complementarity failed to hold.')
                else:
                    hidden_state, decoded_nuc_idx, last_token = res

        elif current_node.hpn_label == 'M':

            if current_node.idx == 1 and kwargs['is_backtrack']:  # dangling end
                min_length = 0
                start_nuc_idx = None
            elif current_node.idx > 1 and kwargs['is_backtrack']:
                min_length = 1
                start_nuc_idx = NUC_VOCAB.index(current_node.decoded_segment[0][0])
            else:
                min_length = 1
                start_nuc_idx = None

            res = self.decode_segment_with_constraint(
                last_token, hidden_state, graph_latent_vec,
                prob_deocde=prob_decode, minimal_length=min_length, last_nuc_complement_to_idx=start_nuc_idx)

            if res is None:
                raise ValueError('Multiloop — last decoded nucleotide complementarity failed to hold.')
            else:
                hidden_state, decoded_nuc_idx, last_token = res

        elif current_node.hpn_label == 'S':
            if len(current_node.nt_idx_assignment) == 0:  # the first segment
                # minimal length is zero, as the first starting nucleotide is a basepair

                if int(torch.max(last_token).item()) == 0:
                    # the first nucleotide to be decoded, then
                    min_length = 1
                else:
                    min_length = 0

                res = self.decode_segment_with_constraint(
                    last_token, hidden_state, graph_latent_vec, minimal_length=min_length)

            else:
                res = self.decode_segment_with_constraint(
                    last_token, hidden_state, graph_latent_vec, prob_deocde=prob_decode,
                    second_stem_segment_complement_to_idx=
                    [NUC_VOCAB.index(nuc) for nuc in reversed(current_node.decoded_segment[0])])

            if res is None:
                raise ValueError('Stem — the first nucleotide on the second segment is '
                                 'not complementary to what has been decoded.')
            else:
                hidden_state, decoded_nuc_idx, last_token = res
        else:
            raise ValueError('Unknown hypernode')

        # decoded_nuc_idx may be an empty list
        return hidden_state, last_token, decoded_nuc_idx

    def decode(self, tree_latent_vec, graph_latent_vec, prob_decode, verbose=False):
        '''
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

        # only one graph is decoded per call to this function
        assert (tree_latent_vec.size(0) == 1)
        assert (graph_latent_vec.size(0) == 1)

        # decoding starts from the 5' end in a depth first topological order
        rna_seq = ''
        # always start from a hypothetical pseudo node
        pseudo_node = RNAJTNode('P', [])
        pseudo_node.idx = 0
        stack = [(pseudo_node, None, None)]
        zero_pad = torch.zeros(1, 1, self.hidden_size).to(device)

        # Root Prediction
        tree_latent_vec = torch.cat([
            tree_latent_vec,
            torch.zeros(1, self.hidden_size - self.latent_size).to(device)], dim=1)
        root_hpn_pred_score = self.aggregate(tree_latent_vec, 'word_hpn')

        # we can basically decode anything at the first non pseudo root node
        if prob_decode:
            root_label_idx = torch.multinomial(torch.softmax(root_hpn_pred_score, dim=1), num_samples=1)
        else:
            _, root_label_idx = torch.max(root_hpn_pred_score, dim=1)
        root_label_idx = root_label_idx.item()

        root = RNAJTNode(HYPERGRAPH_VOCAB[root_label_idx], [], neighbors=[pseudo_node])
        root.idx = 1
        pseudo_node.neighbors.append(root)
        stack.append((root, torch.zeros(1, len(NUC_VOCAB)), 0))
        root.decoded_segment = []
        h = {(0, 1): tree_latent_vec}
        all_nodes = [pseudo_node, root]

        for step in range(MAX_TREE_DECODE_STEPS):
            node_x, last_token, last_token_idx = stack[-1]
            if node_x.hpn_label == 'P':
                break
            node_incoming_msg = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            if len(node_incoming_msg) > 0:
                node_incoming_msg = torch.stack(node_incoming_msg, dim=0).view(1, -1, self.hidden_size)
            else:
                node_incoming_msg = zero_pad

            hpn_label = np.array(list(map(lambda x: x == node_x.hpn_label, HYPERGRAPH_VOCAB)), dtype=np.float32)
            hpn_label = torch.as_tensor(hpn_label).to(device).view(1, len(HYPERGRAPH_VOCAB))

            node_incoming_msg = GraphGRU(hpn_label, node_incoming_msg,
                                         self.W_z_mp, self.W_r_mp, self.U_r_mp, self.W_h_mp)

            # topology regularity constraint
            if node_x.hpn_label == 'H':
                backtrack = 1
            elif node_x.hpn_label == 'I' or node_x.hpn_label == 'S':
                if len(node_x.nt_idx_assignment) == 0:
                    backtrack = 0
                else:
                    backtrack = 1
            else:  # a multiloop segment
                if len(node_x.nt_idx_assignment) <= 1:
                    backtrack = 0
                else:
                    # Predict stop
                    stop_score = self.aggregate(node_incoming_msg, 'stop')
                    if prob_decode:
                        backtrack = (torch.bernoulli(torch.sigmoid(stop_score)).item() == 0)
                    else:
                        backtrack = (stop_score.item() < 0)

            if not backtrack:  # expand the graph: decode a segment and to predict next clique
                hidden_state, last_token, decoded_nuc_idx = \
                    self.decode_segment(node_x, last_token, node_incoming_msg, graph_latent_vec, prob_decode, is_backtrack=backtrack)
                node_x.nt_idx_assignment.append(list(range(last_token_idx - 1 if last_token_idx > 0 else 0,
                                                           last_token_idx + len(decoded_nuc_idx))))
                last_token_idx += len(decoded_nuc_idx)

                new_h = hidden_state
                hpn_pred_score = self.aggregate(new_h, 'word_hpn')

                mask = (
                    torch.as_tensor(((np.array(allowed_hpn_transition[HYPERGRAPH_VOCAB.index(node_x.hpn_label)]) - 1) *
                                     99999).astype(np.float32))).to(device)

                if prob_decode:
                    hpn_label_idx = torch.multinomial(torch.softmax(hpn_pred_score + mask, dim=1), num_samples=1)
                else:
                    _, hpn_label_idx = torch.max(hpn_pred_score + mask, dim=1)
                hpn_label_idx = hpn_label_idx.item()

                node_y = RNAJTNode(HYPERGRAPH_VOCAB[hpn_label_idx], [])
                node_y.decoded_segment = []
                node_y.idx = len(all_nodes)
                node_y.neighbors.append(node_x)
                h[(node_x.idx, node_y.idx)] = new_h
                stack.append((node_y, last_token, last_token_idx))
                all_nodes.append(node_y)
            else:

                node_fa, _, _ = stack[-2]

                hidden_state, last_token, decoded_nuc_idx = \
                    self.decode_segment(node_x, last_token, node_incoming_msg, graph_latent_vec, prob_decode, is_backtrack=backtrack)

                if len(node_x.nt_idx_assignment) != 0:
                    node_x.nt_idx_assignment.append(list(range(last_token_idx - 1 if last_token_idx > 0 else 0,
                                                               last_token_idx + len(decoded_nuc_idx))))
                else:
                    # a hairpin node
                    node_x.nt_idx_assignment = list(range(last_token_idx - 1 if last_token_idx > 0 else 0,
                                                          last_token_idx + len(decoded_nuc_idx)))
                # ready to rollout to the next level
                last_token_idx += len(decoded_nuc_idx)

                new_h = hidden_state
                h[(node_x.idx, node_fa.idx)] = new_h
                node_fa.neighbors.append(node_x)
                stack.pop()
                # modify node_fa
                stack[-1] = (node_fa, last_token, last_token_idx)

            rna_seq += ''.join(map(lambda idx: NUC_VOCAB[idx], decoded_nuc_idx))
            if node_x.hpn_label == 'H':
                node_x.decoded_segment.extend(list(map(lambda idx: rna_seq[idx], node_x.nt_idx_assignment)))
            else:
                node_x.decoded_segment.append(list(map(lambda idx: rna_seq[idx], node_x.nt_idx_assignment[-1])))

            if verbose:
                print('node index: {}, label: {}, total decoded segments: {}, expanding: {}'.
                      format(node_x.idx, node_x.hpn_label,
                             len(node_x.nt_idx_assignment) if node_x.hpn_label != 'H' else 1, not backtrack))
                print('last decoded segment:', ''.join(node_x.decoded_segment) if node_x.hpn_label == 'H' else ''.join(
                    node_x.decoded_segment[-1]))
                print('size of stack:', len(stack))

        return RNAJunctionTree(rna_seq, None, nodes=all_nodes)
