import torch
import torch.nn as nn
import numpy as np
import os
import sys
import pickle
import random
from torch.utils.data import Dataset, DataLoader
import forgi.graph.bulge_graph as fgb

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jtvae_models.GraphEncoder import GraphEncoder

NUC_VOCAB = ['A', 'C', 'G', 'U', '<']

class BasicGraphRNNFolder:

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

            dataset = GraphRNNDataset(batches)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                    collate_fn=lambda x: x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class GraphRNNDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph_encoder_input = GraphEncoder.prepare_batch_data([(tree.rna_seq, tree.rna_struct) for tree in self.data[idx]])
        return graph_encoder_input


class GraphRNNDecoder(nn.Module):

    def __init__(self, hidden_size, latent_size, **kwargs):
        super(GraphRNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size


    def forward(self, latent_encoding, ):
        pass

    @staticmethod
    def prepare_batch_data(rna_mol_batch):

        for rna_seq, rna_struct in rna_mol_batch:
            decoding_step = []
            bg = fgb.BulgeGraph.from_dotbracket(rna_struct)
            for idx, (nuc, struct) in enumerate(zip(rna_seq, rna_struct)):
                nuc_encoding = np.array(list(map(lambda x: x == nuc, NUC_VOCAB)), dtype=np.float32)
                edge_label = np.zeros(idx, dtype=np.int32)
                if struct == ')':
                    bp_to = bg.pairing_partner(idx + 1) - 1
                    edge_label[bp_to] = 1
                decoding_step.append((nuc, nuc_encoding, edge_label))




class GraphRNN(nn.Module):

    def __init__(self, hidden_size, latent_size, encoding_depth, **kwargs):
        super(GraphRNN, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encoding_depth = encoding_depth

        self.encoder = GraphEncoder(self.hidden_size, self.encoding_depth, **kwargs)