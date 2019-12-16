import numpy as np
from collections import OrderedDict
import forgi.graph.bulge_graph as fgb
import scipy.sparse as sp

NUC_VOCAB = ['A', 'C', 'G', 'U']
HYPERGRAPH_VOCAB = ['F', 'T', 'H', 'I', 'M', 'S']
# dangling start, dangling end, hairpin loop, internal loop, multiloop, stem

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

    def __init__(self, rna_seq, rna_struct):
        self.rna_seq = list(rna_seq)
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
                        raise ValueError()
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
                    stem_id = bg.connections(hp_node_id.lower())[0].upper()
                    side = int(np.argmax(bg.get_sides(stem_id.lower(), hp_node_id.lower())))
                    hypernodes[hp_node_id] = [hypernodes[stem_id][side][-1], hypernodes[stem_id][side][-1] + 1]
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
    merged_mloops = []

    for mloops in bg.junctions:
        mloop_checker = list(map(lambda x: x.upper().startswith('M'), mloops))
        if sum(mloop_checker) > 2:
            all_nuc_idx = []
            all_neighbors_idx = []
            for mloop_id in mloops:
                mloop_id = mloop_id.upper()
                # nucleotide indices assigned to a multiloop segment
                # note: do not remove anything from the dict or the list
                all_nuc_idx += hypernodes[mloop_id]

                # gather neighbors of that multiloop segment
                mloop_idx = all_hpn_ids.index(mloop_id)
                all_neighbors_idx += hpn_neighbors[mloop_idx]
                for mloop_nei_idx in hpn_neighbors[mloop_idx]:
                    hpn_neighbors[mloop_nei_idx].remove(mloop_idx)
                hpn_neighbors[mloop_idx] = []

            all_nuc_idx = list(sorted(set(all_nuc_idx)))
            all_neighbors_idx = list(sorted(set(all_neighbors_idx)))
            merged_mloops.append('M%d' % (nb_mloop))
            hypernodes[merged_mloops[-1]] = all_nuc_idx
            hpn_neighbors.append(all_neighbors_idx)
            for neighbor_idx in all_neighbors_idx:
                hpn_neighbors[neighbor_idx] += [len(hpn_neighbors) - 1]

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

    breadth_first_order = sp.csgraph.breadth_first_order(clique_graph, i_start=0, directed=False,
                                                         return_predecessors=False)
    junction_tree = clique_graph[breadth_first_order, :][:, breadth_first_order]
    hpn_id = [hid[0] for hid in all_hpn_ids[breadth_first_order]]
    hpn_nodes_assignment = np.array(list(hypernodes.values()))[breadth_first_order]

    return junction_tree, hpn_id, hpn_nodes_assignment


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000, )
    adjmat, node_labels, hpn_nodes_assignment = decompose(
        "....((((((....((.......((((.((((.(((...(((((..........)))))...((.......))....)))......))))))))......))...)).))))......(((....((((((((...))))))))...)))........")
    print(adjmat.todense())
    print(list(zip(node_labels, hpn_nodes_assignment)))
    exit()
    node_labels = np.array(node_labels).astype('<U15')
    node_labels[node_labels == 'S'] = "Stem"
    node_labels[node_labels == 'F'] = "Dangling Start"
    node_labels[node_labels == 'T'] = "Dangling End"
    node_labels[node_labels == 'M'] = "Multiloop"
    node_labels[node_labels == 'H'] = "Hairpin"
    node_labels[node_labels == 'I'] = "Internal loop"
    print(node_labels)
    from lib.plot import draw_graph

    draw_graph(np.array(adjmat.todense()), node_labels=node_labels)
