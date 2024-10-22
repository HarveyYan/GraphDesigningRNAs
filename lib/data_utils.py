import os
import sys
import pickle
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jtvae_models.TreeEncoder import TreeEncoder
from jtvae_models.BranchedTreeEncoder import BranchedTreeEncoder
from jtvae_models.GraphEncoder import GraphEncoder


class JunctionTreeFolder:

    # data loading entry
    def __init__(self, data_folder, batch_size, num_workers=4, shuffle=True, **kwargs):
        self.data_folder = data_folder
        self.limit_data = kwargs.get('limit_data', None)
        if self.limit_data:
            assert type(self.limit_data) is int, '\'limit_data\' should either be None or an integer'
            self.data_files = [fn for fn in os.listdir(data_folder) if
                               int(fn.split('-')[-1].split('.')[0]) <= self.limit_data]
        else:
            self.data_files = [fn for fn in os.listdir(data_folder)]
        self.is_test = 'test-split' in data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.tree_encoder_arch = kwargs.get('tree_encoder_arch', 'baseline')

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            # limit test data examples to 20,000
            if self.is_test:
                data = data[:20000]

            if self.shuffle:
                random.shuffle(data)
                # data = np.array(data)[np.argsort([len(rna.rna_seq)for rna in data])[::-1]]

            batches = [data[i: i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = JunctionTreeDataset(batches, tree_encoder_arch=self.tree_encoder_arch)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                    collate_fn=lambda x: x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class JunctionTreeDataset(Dataset):

    def __init__(self, data, **kwargs):
        self.data = data
        self.tree_encoder_arch = kwargs.get('tree_encoder_arch', 'baseline')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.tree_encoder_arch == 'baseline':
            return tensorize(self.data[idx], TreeEncoder)
        elif self.tree_encoder_arch == 'branched':
            return tensorize(self.data[idx], BranchedTreeEncoder)
        else:
            raise ValueError('Unknown %s tree encoder architecture' % (self.tree_encoder_arch))


def tensorize(tree_batch, TreeEncoder):
    graph_encoder_input = GraphEncoder.prepare_batch_data([(tree.rna_seq, tree.rna_struct) for tree in tree_batch])
    tree_encoder_input = TreeEncoder.prepare_batch_data(tree_batch)

    return tree_batch, graph_encoder_input, tree_encoder_input


if __name__ == "__main__":
    from lib.tree_decomp import dfs_nt_traversal_check, get_tree_height, dfs
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np

    # sanity check on the traversal orders
    loader = JunctionTreeFolder('../data/rna_jt_32-512/train-split', 32, num_workers=8, shuffle=False)
    all_trace, max_trace = [], 0
    total_batch = 0
    for batch in loader:
        tree_batch, graph_encoder_input, tree_encoder_input = batch

        for tree in tree_batch:
            s = []
            dfs(s, tree.nodes[1], 0)  # skipping the pseudo node
            all_trace.append(len(s))
            if len(s) > max_trace:
                max_trace = len(s)

        total_batch += 1

        if total_batch % 10000 == 0:
            plt.hist(all_trace, bins=1000)
            plt.title('max trace: %d' % (max_trace))
            plt.xlabel('trace size')
            plt.ylabel('count')
            plt.savefig('all-trace-size.jpg', dpi=350)
            plt.close()

    print('max trace:', max_trace)
        # for tree in tree_batch:
        #     is_valid = dfs_nt_traversal_check(tree)
        #     if not is_valid:
        #         print('parsed tree invalid:', ''.join(tree.rna_seq))
    exit()

    total_batch = 0
    all_length, all_tree_height, all_nb_nodes = [], [], []
    loader = JunctionTreeFolder(os.path.join(basedir, 'data/rna_jt_32-512/train-split'), 32, num_workers=8,
                                shuffle=False)
    for batch in tqdm(loader):
        tree_batch, graph_encoder_input, tree_encoder_input = batch
        f_nuc, f_bond, node_graph, message_graph, scope = graph_encoder_input
        # print('total number of nucleotides:', f_nuc.shape[0])
        # print('total number of messages on the original RNA graph:', f_bond.shape[0])
        # f_node_label, f_node_assignment, f_message, hp_node_graph, hp_message_graph, hp_scope = tree_encoder_input
        # print('total number of subgraphs:', f_node_label.shape[0])
        # print('total number of messages on the hypergraph:', f_message.shape[0])
        # print('maximum incoming messages to a node:', hp_node_graph.shape[1])
        # print('#' * 30)

        for tree in tree_batch:
            tree_height = get_tree_height(np.array(tree.hp_adjmat.todense()))
            all_tree_height.append(tree_height)
            all_nb_nodes.append(len(tree.nodes) - 1)
            all_length.append(len(tree.rna_seq))

        total_batch += 1

        if total_batch % 10000 == 0:
            plt.hist(all_tree_height, bins=1000)
            plt.xlabel('tree height')
            plt.ylabel('count')
            plt.savefig('all-tree-height.jpg', dpi=350)
            plt.close()

            plt.hist(all_nb_nodes, bins=1000)
            plt.xlabel('num hypernodes')
            plt.ylabel('count')
            plt.savefig('all-nb-nodes.jpg', dpi=350)
            plt.close()

            unique_length = np.unique(all_length)
            mean_depth, mean_nb_nodes = [], []
            for ulen in unique_length:
                mean_depth.append(np.mean(np.array(all_tree_height)[all_length == ulen]))
                mean_nb_nodes.append(np.mean(np.array(all_nb_nodes)[all_length == ulen]))

            plt.bar(unique_length, mean_depth)
            plt.xlabel('sequence length')
            plt.ylabel('mean tree height')
            plt.savefig('length-vs-height.jpg', dpi=350)
            plt.close()

            plt.bar(unique_length, mean_nb_nodes)
            plt.xlabel('sequence length')
            plt.ylabel('mean num hypernodes ')
            plt.savefig('length-vs-nb-nodes.jpg', dpi=350)
            plt.close()

    all_length = np.array(all_length)
    all_tree_height = np.array(all_tree_height)
    all_nb_nodes = np.array(all_nb_nodes)

    plt.hist(all_tree_height, bins=1000)
    plt.xlabel('tree height')
    plt.ylabel('count')
    plt.savefig('all-tree-height.jpg', dpi=350)
    plt.close()

    plt.hist(all_nb_nodes, bins=1000)
    plt.xlabel('num hypernodes')
    plt.ylabel('count')
    plt.savefig('all-nb-nodes.jpg', dpi=350)
    plt.close()

    unique_length = np.unique(all_length)
    mean_depth, mean_nb_nodes = [], []
    for ulen in unique_length:
        mean_depth.append(np.mean(all_tree_height[all_length == ulen]))
        mean_nb_nodes.append(np.mean(all_nb_nodes[all_length == ulen]))

    plt.bar(unique_length, mean_depth)
    plt.xlabel('sequence length')
    plt.ylabel('mean tree height')
    plt.savefig('length-vs-height.jpg', dpi=350)
    plt.close()

    plt.bar(unique_length, mean_nb_nodes)
    plt.xlabel('sequence length')
    plt.ylabel('mean num hypernodes ')
    plt.savefig('length-vs-nb-nodes.jpg', dpi=350)
    plt.close()
