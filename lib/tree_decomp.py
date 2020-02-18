import RNA
import numpy as np
from collections import OrderedDict
import forgi.graph.bulge_graph as fgb
import scipy.sparse as sp

NUC_VOCAB = ['A', 'C', 'G', 'U']
HYPERGRAPH_VOCAB = ['H', 'I', 'M', 'S', 'P']
allowed_basepairs = [[False, False, False, True],
                     [False, False, True, False],
                     [False, True, False, True],  # allow G-U
                     [True, False, True, False]]


# hairpin loop, internal loop, multiloop, stem, pseudo root node

class RNAJTNode:

    def __init__(self, hpn_label, nt_idx_assignment, **kwargs):
        self.hpn_label = hpn_label
        assert self.hpn_label in HYPERGRAPH_VOCAB, 'hypergraph node label must be one from {}'.format(HYPERGRAPH_VOCAB)

        # nucleotide assignments to this hypergraph node
        self.nt_idx_assignment = nt_idx_assignment

        if 'neighbors' in kwargs:
            self.neighbors = kwargs['neighbors']
        else:
            self.neighbors = []


class RNAJunctionTree:

    # [pseudo node, (non pseudo) root node, ...]

    def __init__(self, rna_seq, rna_struct, **kwargs):
        self.rna_seq = list(rna_seq)

        if rna_struct is not None and type(rna_struct) is str:
            # ground truth junction tree decomposition
            self.rna_struct = rna_struct

            # hypergraph and connectivity between hypernodes
            hp_adjmat, hpn_labels, hpn_assignment = decompose(self.rna_struct)

            # root is always the first node
            self.nodes = []
            for i, label in enumerate(hpn_labels):
                node = RNAJTNode(label, hpn_assignment[i])
                node.idx = i
                self.nodes.append(node)

            for row_idx, col_idx in zip(*np.nonzero(hp_adjmat)):
                self.nodes[row_idx].neighbors.append(self.nodes[col_idx])

            self.free_energy = kwargs.get('free_energy', None)

            if self.free_energy is None:
                self.free_energy = RNA.eval_structure_simple(self.rna_seq, self.rna_struct)

            self.is_mfe = True
        else:
            # reconstructed from the decoder
            self.nodes = kwargs.get('nodes')
            is_valid = self.isvalid()
            if not is_valid:
                raise ValueError('Decoded RNA structure is not valid')

            self.free_energy = RNA.eval_structure_simple(self.rna_seq, self.rna_struct)
            mfe_struct, mfe = RNA.fold(self.rna_seq)

            if np.abs(self.free_energy - mfe) < 1e-6:
                self.is_mfe = True
            else:
                self.is_mfe = False
                self.mfe_struct = mfe_struct
                self.mfe = mfe
                self.struct_hamming_dist = np.sum(
                    np.array(list(self.rna_struct)) - np.array(list(mfe)))
                self.mfe_range = (mfe - self.free_energy) / mfe

    def isvalid(self):
        # check:
        # 1. equivalent amount of nucleotides on both sides of the stem
        # 2. valid base pairing
        # - canonical base pairs
        # - G-U pairs and A-A pairs
        # 3. valid number of branches in each hypernode element
        self.rna_struct = ['.'] * len(self.rna_seq)
        for node in self.nodes:
            if node.hpn_label == 'S':
                nb_segments = len(node.nt_idx_assignment)
                if nb_segments != 2:
                    return False
                if len(node.nt_idx_assignment[0]) != len(node.nt_idx_assignment[1]):
                    return False
                if len(node.nt_idx_assignment[0]) == 0 or len(node.nt_idx_assignment[1]) == 0:
                    return False
                for nt_l_idx, nt_r_idx in zip(node.nt_idx_assignment[0], reversed(node.nt_idx_assignment[1])):
                    if allowed_basepairs[nt_l_idx][nt_r_idx] is False:
                        return False
                for nt_idx in node.nt_idx_assignment[0]:
                    self.rna_struct[nt_idx] = '('
                for nt_idx in node.nt_idx_assignment[1]:
                    self.rna_struct[nt_idx] = ')'
            elif node.hpn_label == 'I':
                nb_segments = len(node.nt_idx_assignment)
                if nb_segments != 2:
                    return False
            elif node_labels == 'M':
                nb_segments = len(node.nt_idx_assignment)
                if nb_segments < 3:
                    return False
        self.rna_struct = ''.join(self.rna_struct)
        return True


def decompose(dotbracket_struct):
    bg = fgb.BulgeGraph.from_dotbracket(dotbracket_struct)

    # hypergraph decomposition
    raw_hpgraph = bg.to_bg_string().rstrip().upper().split('\n')
    hypernodes = OrderedDict()
    for line in raw_hpgraph:

        if line.startswith('DEFINE'):
            tokens = line.split()[1:]
            hp_node_id = tokens[0]
            if hp_node_id.startswith('F'):
                # dangling start, closed by 1 stem;
                # extending 1 nucleotide, so that the sepset contains one nucleotide with the stem
                hypernodes[hp_node_id] = list(range(int(tokens[1]) - 1, int(tokens[2]) + 1))
            elif hp_node_id.startswith('T'):
                # dangling end, closed by 1 stem
                # extending 1 nucleotide, so that the sepset contains one nucleotide with the stem
                hypernodes[hp_node_id] = list(range(int(tokens[1]) - 2, int(tokens[2])))
            elif hp_node_id.startswith('S'):
                hypernodes[hp_node_id] = [
                    list(range(int(tokens[1]) - 1, int(tokens[2]))),
                    list(range(int(tokens[3]) - 1, int(tokens[4])))]
            elif hp_node_id.startswith('H'):
                # hairpin loop. closed by 1 stem
                # extending 2 nucleotides in both ends, so that the sepsets contain two nucleotides with the stem
                hypernodes[hp_node_id] = list(range(int(tokens[1]) - 2, int(tokens[2]) + 1))
            elif hp_node_id.startswith('I'):
                # single stranded internal loop closed by two stems
                if len(tokens[1:]) == 2:
                    # todo, double check
                    stem_id = bg.connections(hp_node_id.lower())[0].upper()
                    if hypernodes[stem_id][0][-1] == int(tokens[1]) - 2:
                        # on the 3'UTR end
                        hypernodes[hp_node_id] = [
                            list(range(int(tokens[1]) - 2, int(tokens[2]) + 1)),
                            [hypernodes[stem_id][1][0] - 1, hypernodes[stem_id][1][0]]]
                    elif hypernodes[stem_id][1][0] == int(tokens[2]):
                        # one the 5'UTR end
                        hypernodes[hp_node_id] = [
                            [hypernodes[stem_id][0][-1], hypernodes[stem_id][0][-1] + 1],
                            list(range(int(tokens[1]) - 2, int(tokens[2]) + 1))]
                    else:
                        raise ValueError('Internal loop parsing error')
                else:
                    hypernodes[hp_node_id] = [
                        list(range(int(tokens[1]) - 2, int(tokens[2]) + 1)),  # closer to 5'UTR
                        list(range(int(tokens[3]) - 2, int(tokens[4]) + 1))]  # closer to 3'UTR
            else:
                # multiloop closed by more than two stems, or
                # one on the opposite side to the dangling start/end
                if len(tokens) == 1:
                    # todo: double check
                    # todo: external regions really
                    # todo: replace with new hypernode id
                    # stem_id = bg.connections(hp_node_id.lower())[0].upper()
                    # side = int(np.argmax(bg.get_sides(stem_id.lower(), hp_node_id.lower())))
                    # hypernodes[hp_node_id] = [hypernodes[stem_id][side][-1], hypernodes[stem_id][side][-1] + 1]

                    stem_ids = [stem_id.upper() for stem_id in bg.connections(hp_node_id.lower())]
                    e_3_idx, e_5_idx = hypernodes[stem_ids[0]]
                    l_3_idx, l_5_idx = hypernodes[stem_ids[1]]
                    # e_3_idx always comes before l_3_idx, and e_5_idx always comes after l_5_idx
                    # except on the external region
                    if e_3_idx[-1] + 1 == l_3_idx[0]:
                        hypernodes[hp_node_id] = [e_3_idx[-1], l_3_idx[0]]
                    elif e_5_idx[0] - 1 == l_5_idx[-1]:
                        hypernodes[hp_node_id] = [l_5_idx[-1], e_5_idx[0]]
                    elif e_5_idx[-1] + 1 == l_3_idx[0]:
                        # this one is on the external region
                        # alternate criterion: max(e_5_idx) < max(l_5_idx)
                        hypernodes[hp_node_id] = [e_5_idx[-1], l_3_idx[0]]
                    else:
                        raise ValueError('Multiloop parsing error:%s\n%s' % (hp_node_id, dotbracket_struct))
                else:
                    hypernodes[hp_node_id] = list(range(int(tokens[1]) - 2, int(tokens[2]) + 1))

    all_hpn_ids = list(hypernodes.keys())
    hpn_neighbors = []
    for _ in range(len(all_hpn_ids)):
        hpn_neighbors.append([])

    for line in raw_hpgraph:
        if line.startswith('CONNECT'):
            tokens = line.split()[1:]
            all_idx = [all_hpn_ids.index(hpn_id) for hpn_id in tokens]
            hpn_neighbors[all_idx[0]] += all_idx[1:]
            for idx in all_idx[1:]:
                hpn_neighbors[idx] += [all_idx[0]]

    for i in range(len(all_hpn_ids)):
        hpn_neighbors[i] = list(sorted(set(hpn_neighbors[i])))

    nb_mloop = len(list(bg.mloop_iterator()))
    external_region_ids = []
    # Merging multiloop segments
    for mloops in bg.junctions:
        mloop_checker = list(map(lambda x: x.upper().startswith('M'), mloops))
        external_checker = list(map(lambda x: x.upper().startswith('F') or x.upper().startswith('T'), mloops))

        if sum(mloop_checker) <= 2 or sum(external_checker) >= 1:
            # an easy check, but there are examples that can slip off
            external_region_ids.extend([loop_id.upper() for loop_id in mloops])
            continue

        contains_external_segment = False
        for mloop_id in mloops:
            stem_ids = [stem_id.upper() for stem_id in bg.connections(mloop_id)]
            e_3_idx, e_5_idx = hypernodes[stem_ids[0]]
            l_3_idx, l_5_idx = hypernodes[stem_ids[1]]
            if e_5_idx[-1] < l_3_idx[0]:
                contains_external_segment = True
            else:
                contains_external_segment = False
                break

        if contains_external_segment:
            external_region_ids.extend([loop_id.upper() for loop_id in mloops])
            print(dotbracket_struct)
            continue

        if sum(mloop_checker) > 2:
            all_nuc_idx = []
            all_neighbors_idx = []
            for mloop_id in mloops:
                mloop_id = mloop_id.upper()
                # nucleotide indices assigned to a multiloop segment
                # note: do not remove anything from the dict or the list
                # all_nuc_idx += hypernodes[mloop_id]
                all_nuc_idx.append(hypernodes[mloop_id])

                # gather neighbors of that multiloop segment
                mloop_idx = all_hpn_ids.index(mloop_id)
                all_neighbors_idx += hpn_neighbors[mloop_idx]
                for mloop_nei_idx in hpn_neighbors[mloop_idx]:
                    hpn_neighbors[mloop_nei_idx].remove(mloop_idx)
                hpn_neighbors[mloop_idx] = []

            # all_nuc_idx = list(sorted(set(all_nuc_idx)))
            all_neighbors_idx = list(sorted(set(all_neighbors_idx)))
            hypernodes['M%d' % (nb_mloop)] = all_nuc_idx
            nb_mloop += 1
            hpn_neighbors.append(all_neighbors_idx)
            for neighbor_idx in all_neighbors_idx:
                hpn_neighbors[neighbor_idx] += [len(hpn_neighbors) - 1]

    if len(all_hpn_ids) == 1 and all_hpn_ids[0] == 'F0':
        # replace with hairpin...
        hypernodes['H'] = list(range(len(dotbracket_struct)))
        hpn_neighbors.append([2])
        hypernodes['P'] = [[0], [len(dotbracket_struct) - 1]]
        hpn_neighbors.append([1])
    else:
        # Introducing a pseudo root node
        # Note: the root node shall not start with a stem
        if len(external_region_ids) == 0:
            if 'F0' in hypernodes:
                external_region_ids.append('F0')
            if 'T0' in hypernodes:
                external_region_ids.append('T0')

        external_idx = []
        ext_loop_checker = list(map(lambda x: x.startswith('M'), external_region_ids))
        ext_dangling_checker = list(map(lambda x: x.startswith('F') or x.startswith('T'), external_region_ids))
        if sum(ext_loop_checker) == 0:
            # no external loops
            first_stem_idx = all_hpn_ids.index('S0')
            if sum(ext_dangling_checker) == 0:
                # attach pseudo node directly to the first stem of this RNA
                hypernodes['P'] = [[0], [len(dotbracket_struct) - 1]]
                hpn_neighbors.append([first_stem_idx])
                hpn_neighbors[first_stem_idx] += [len(hpn_neighbors) - 1]
            else:
                # insert an internal loop between the stem and the pseudo start node
                nb_iloop = len(list(bg.iloop_iterator()))

                if 'F0' in external_region_ids:
                    dangling_start_idx = all_hpn_ids.index('F0')
                    external_idx.append(hypernodes['F0'])
                    hpn_neighbors[dangling_start_idx] = []
                    hpn_neighbors[first_stem_idx].remove(dangling_start_idx)
                else:
                    external_idx.append([0])

                if 'T0' in external_region_ids:
                    dangling_end_idx = all_hpn_ids.index('T0')
                    external_idx.append(hypernodes['T0'])
                    hpn_neighbors[dangling_end_idx] = []
                    hpn_neighbors[first_stem_idx].remove(dangling_end_idx)
                else:
                    external_idx.append([len(dotbracket_struct) - 1])

                hypernodes['I%d' % (nb_iloop)] = external_idx
                hpn_neighbors.append([first_stem_idx])
                attached_iloop_idx = len(hpn_neighbors) - 1
                hpn_neighbors[first_stem_idx] += [attached_iloop_idx]

                # now attach pseudo-node
                hypernodes['P'] = [[0], [len(dotbracket_struct) - 1]]
                hpn_neighbors.append([attached_iloop_idx])
                hpn_neighbors[attached_iloop_idx] += [len(hpn_neighbors) - 1]
        else:
            all_neighbors_idx = []
            if 'F0' not in external_region_ids:
                external_idx.append([0])

            for ext_id in external_region_ids:
                external_idx.append(hypernodes[ext_id])

                # gather neighbors of that multiloop segment
                ext_idx = all_hpn_ids.index(ext_id)
                all_neighbors_idx += hpn_neighbors[ext_idx]
                for ext_nei_idx in hpn_neighbors[ext_idx]:
                    hpn_neighbors[ext_nei_idx].remove(ext_idx)
                hpn_neighbors[ext_idx] = []

            if 'T0' not in external_region_ids:
                external_idx.append([len(dotbracket_struct) - 1])

            all_neighbors_idx = list(sorted(set(all_neighbors_idx)))
            hypernodes['M%d' % (nb_mloop)] = external_idx
            nb_mloop += 1
            hpn_neighbors.append(all_neighbors_idx)
            for neighbor_idx in all_neighbors_idx:
                hpn_neighbors[neighbor_idx] += [len(hpn_neighbors) - 1]
            attcached_mloop = len(hpn_neighbors) - 1
            # now attach pseudo-node
            hypernodes['P'] = [[0], [len(dotbracket_struct) - 1]]
            hpn_neighbors.append([attcached_mloop])
            hpn_neighbors[attcached_mloop] += [len(hpn_neighbors) - 1]

    all_hpn_ids = np.array(list(hypernodes.keys()))
    # breadth first search to find the spanning tree
    try:
        row, col, data = zip(*[[i, t, 1] for i, row in enumerate(hpn_neighbors) for t in row])
    except ValueError as e:
        # a secondary structure where there is no base pairs
        # print(dotbracket_struct, e)
        row, col, data = [], [], []
    clique_graph = sp.csr_matrix((data, (row, col)), shape=(len(hpn_neighbors), len(hpn_neighbors)))
    # clique graph is already a junction tree

    # junction_tree = sp.csgraph.minimum_spanning_tree(clique_graph)
    # junction_tree = ((junction_tree + junction_tree.T) > 0).astype(np.int32)
    # print(np.sum(clique_graph - junction_tree))

    breadth_first_order = sp.csgraph.breadth_first_order(
        clique_graph, i_start=len(all_hpn_ids) - 1, directed=False, return_predecessors=False)
    junction_tree = clique_graph[breadth_first_order, :][:, breadth_first_order]
    hpn_id = [hid[0] for hid in all_hpn_ids[breadth_first_order]]
    hpn_nodes_assignment = np.array(list(hypernodes.values()))[breadth_first_order]

    return junction_tree, hpn_id, hpn_nodes_assignment


def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx:
            continue
        stack.append((x.hpn_label, x.nt_idx_assignment, y.hpn_label, y.nt_idx_assignment, 1))
        dfs(stack, y, x.idx)
        stack.append((y.hpn_label, y.nt_idx_assignment, x.hpn_label, x.nt_idx_assignment, 0))


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000, )

    rna_seq = 'A' * 128
    rna_struct = "(((..((((.(.............).)).))..))).((((((.....))))))..((((.(((((((.......))))))).))))...((((...((((((((((......)))))))))).))))"

    adjmat, node_labels, hpn_nodes_assignment = decompose(rna_struct)
    print(adjmat.todense())
    print(list(zip(node_labels, hpn_nodes_assignment)))
    node_labels = np.array(node_labels).astype('<U15')
    node_labels[node_labels == 'S'] = "Stem"
    # node_labels[node_labels == 'F'] = "Dangling Start"
    # node_labels[node_labels == 'T'] = "Dangling End"
    node_labels[node_labels == 'M'] = "Multiloop"
    node_labels[node_labels == 'H'] = "Hairpin"
    node_labels[node_labels == 'I'] = "Internal loop"
    print(node_labels)
    from lib.plot import draw_graph

    draw_graph(np.array(adjmat.todense()), node_labels=node_labels)

    tree = RNAJunctionTree(rna_seq, rna_struct)
    stack = []
    dfs(stack, tree.nodes[1], 0)
    for line in stack:
        print(line)
