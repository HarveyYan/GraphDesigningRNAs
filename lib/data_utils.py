import os
import sys
import pickle
import os
import random
from torch.utils.data import Dataset, DataLoader

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.TreeEncoder import TreeEncoder
from model.GraphEncoder import GraphEncoder


class JunctionTreeFolder:

    # data loading entry
    def __init__(self, data_folder, batch_size, num_workers=4, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)

            batches = [data[i: i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = JunctionTreeDataset(batches)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                    collate_fn=lambda x: x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class JunctionTreeDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx])


def tensorize(tree_batch):
    graph_encoder_input = GraphEncoder.prepare_batch_data([(tree.rna_seq, tree.rna_struct) for tree in tree_batch])
    tree_encoder_input = TreeEncoder.prepare_batch_data(tree_batch)

    return tree_batch, graph_encoder_input, tree_encoder_input


if __name__ == "__main__":
    from lib.tree_decomp import dfs_nt_traversal_check, get_tree_height
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np

    # sanity check on the traversal orders
    # loader = JunctionTreeFolder('../data/rna_jt_32-512/train-split', 32, num_workers=8, shuffle=False)
    # for batch in loader:
    #     tree_batch, graph_encoder_input, tree_encoder_input = batch
    #
    #     for tree in tree_batch:
    #         is_valid = dfs_nt_traversal_check(tree)
    #         if not is_valid:
    #             print('parsed tree invalid:', ''.join(tree.rna_seq))

    all_length, all_tree_height, all_nb_nodes = [], [], []
    loader = JunctionTreeFolder(os.path.join(basedir, 'data/rna_jt_32-512/train-split'), 32, num_workers=8, shuffle=False)
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

    all_length = np.array(all_length)
    all_tree_height = np.array(all_tree_height)
    all_nb_nodes = np.array(all_nb_nodes)

    plt.hist(all_tree_height, bins=1000)
    plt.xlabel('tree height')
    plt.ylabel('count')
    plt.savefig('all-tree-height.jpg', dpi=350)

    plt.hist(all_nb_nodes, bins=1000)
    plt.xlabel('num hypernodes')
    plt.ylabel('count')
    plt.savefig('all-nb-nodes.jpg', dpi=350)

    unique_length = np.unique(all_length)
    mean_depth, mean_nb_nodes = [], []
    for ulen in unique_length:
        mean_depth.append(np.mean(all_tree_height[all_length == ulen]))
        mean_nb_nodes.append(np.mean(all_nb_nodes[all_length == ulen]))

    plt.bar(unique_length, mean_depth)
    plt.xlabel('sequence length')
    plt.ylabel('mean tree height')
    plt.savefig('length-vs-height.jpg', dpi=350)

    plt.bar(unique_length, mean_nb_nodes)
    plt.xlabel('sequence length')
    plt.ylabel('mean num hypernodes ')
    plt.savefig('length-vs-nb-nodes.jpg', dpi=350)


