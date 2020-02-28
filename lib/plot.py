import numpy as np
import networkx as nx
import matplotlib
import os
import matplotlib.pyplot as plt

def draw_graph(adj=None, G = None, marginals=None,
               draw_edge_color=False, title=None,
               node_size=300, node_labels=None, saveto=None):

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
    if saveto is None:
        saveto = 'tmp.png'
    plt.savefig(saveto, dpi=350)


plt.style.use('classic')
matplotlib.rcParams.update({'figure.figsize': [10.0, 10.0], 'font.family': 'Times New Roman', 'figure.dpi': 350})
matplotlib.rcParams['agg.path.chunksize'] = 1e10
import collections

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]

_output_dir = ''
_stdout = True


def suppress_stdout():
    global _stdout
    _stdout = False


def set_output_dir(output_dir):
    global _output_dir
    _output_dir = output_dir


def tick():
    _iter[0] += 1


def plot(name, value):
    if type(value) is tuple:
        _since_last_flush[name][_iter[0]] = np.array(value)
    else:
        _since_last_flush[name][_iter[0]] = value


def flush():
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}: {}\t".format(name, np.mean(np.array(list(vals.values())), axis=0)))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = np.array([_since_beginning[name][x] for x in x_vals])

        plt.clf()
        if len(y_vals.shape) == 1:
            plt.plot(x_vals, y_vals)
        else:  # with standard deviation
            plt.plot(x_vals, y_vals[:, 0])
            plt.fill_between(x_vals, y_vals[:, 0] - y_vals[:, 1], y_vals[:, 0] + y_vals[:, 1], alpha=0.5)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(_output_dir, name.replace(' ', '_') + '.jpg'), dpi=350)

    if _stdout:
        print("iteration {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()


def reset():
    global _since_beginning, _since_last_flush, _iter, _output_dir, _stdout
    _since_beginning = collections.defaultdict(lambda: {})
    _since_last_flush = collections.defaultdict(lambda: {})

    _iter = [0]

    _output_dir = ''
    _stdout = True
