from torch.utils.data import Dataset, DataLoader
from lib.tree_decomp import RNAJunctionTree
from model.TreeEncoder import TreeEncoder
from model.GraphEncoder import GraphEncoder

import pickle
import os
import random


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
    loader = JunctionTreeFolder('../data/rna_jt_32-512', 32, num_workers=0)
    for batch in loader:
        tree_batch, graph_encoder_input, tree_encoder_input = batch
        f_nuc, f_bond, node_graph, message_graph, scope = graph_encoder_input
        print('total number of nucleotides:', f_nuc.shape[0])
        print('total number of messages on the original RNA graph:', f_bond.shape[0])
        f_node_label, f_node_assignment, f_message, hp_node_graph, hp_message_graph, hp_scope = tree_encoder_input
        print('total number of subgraphs:', f_node_label.shape[0])
        print('total number of messages on the hypergraph:', f_message.shape[0])
        print('maximum incoming messages to a node:', hp_node_graph.shape[1])
        print('#' * 30)

        from lib.plot import draw_graph
        for i, tree in enumerate(tree_batch):
            print(tree.rna_seq)
            print(tree.rna_struct)
            print('\n')
            tree.get_junction_tree()
            node_labels = tree.node_labels
            node_labels[node_labels == 'S'] = "Stem"
            # node_labels[node_labels == 'F'] = "Dangling Start"
            # node_labels[node_labels == 'T'] = "Dangling End"
            node_labels[node_labels == 'M'] = "Multiloop"
            node_labels[node_labels == 'H'] = "Hairpin"
            node_labels[node_labels == 'I'] = "Internal loop"
            draw_graph(tree.hp_adjmat, node_labels, saveto='%d.jpg'%(i))

        break
