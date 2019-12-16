import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(adj=None, G = None, marginals=None,
               draw_edge_color=False, title=None,
               node_size=300, node_labels=None):

    node_color = marginals
    if G is None:
        assert adj is not None, "you have to provide either the adjacency matrix or the graph"
        G = nx.from_numpy_array(adj)
    edge_color = G.number_of_edges()*[1]
    n = G.number_of_nodes()
    if adj is not None:
        edges = adj[np.triu_indices(n,1)]  # strict upper triangle inds
        if draw_edge_color:
            edge_color = edges[edges != 0].ravel().astype(float).tolist()
    if node_labels is not None:
        node_dict = dict([(i, str(node_labels[i])) for i in range(n)])
    else: node_dict = None
    nx.draw(G, node_color=marginals, edge_color = edge_color,
                     label=title, node_size = node_size,
                     labels=node_dict)
    plt.savefig('tmp.png', dpi=300)