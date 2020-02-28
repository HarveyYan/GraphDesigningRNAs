import os
from lib.forgi_utils import generate_seq_dataset
import multiprocessing as mp
import pandas as pd
import pickle
from lib.tree_decomp import RNAJunctionTree
from tqdm import tqdm
import numpy as np

def parse_tree(args):
    rna_seq, rna_struct, mfe = args
    obj = RNAJunctionTree(rna_seq, rna_struct, free_energy=float(mfe))
    return obj


def create_dataset():
    if not os.path.exists('data/rna_dataset_32-512.csv'):
        # after removing the duplicate RNA examples
        # the actual size of the dataset might be smaller than 100000
        generate_seq_dataset(5000000, None, variable_length=True, min_length=32, max_length=512)

    file = pd.read_csv('data/rna_dataset_32-512.csv')
    size = len(file)

    pool = mp.Pool(12)
    all_rna_jt = list(tqdm(pool.imap(parse_tree, zip(file['seq'], file['struct'], file['MFE']))))

    if not os.path.exists('./data/rna_jt_32-512'):
        os.makedirs('./data/rna_jt_32-512')

    filename = './data/rna_jt_32-512/rna_jt_32-512_{}.obj'.format(size)
    with open(filename, 'wb') as file:
        pickle.dump(all_rna_jt, file)

    split_dataset(filename, size)

def split_dataset(filename, size):
    data = pickle.load(open(filename, 'rb'))

    test_size = int(size*0.1)
    train_size = size - test_size
    all_idx = np.random.permutation(size)
    test_idx = all_idx[:test_size]
    train_idx = all_idx[test_size:]

    test_data = list(np.array(data)[test_idx])
    train_data = list(np.array(data)[train_idx])

    pickle.dump(test_data, open('./data/rna_jt_32-512/test-32-512-%d.obj' %(test_size), 'wb'))

    splits = list(range(0, train_size, 20000))
    for i, (split_a, split_b) in enumerate(zip(splits[:-1], splits[1:])):
        pickle.dump(train_data[split_a: split_b], open('./data/rna_jt_32-512/train-32-512-split-%d.obj' %(i), 'wb'))

    if splits[-1] < train_size:
        pickle.dump(train_data[splits[-1]: ], open('./data/rna_jt_32-512/train-32-512-split-%d.obj' % (i+1), 'wb'))

if __name__ == "__main__":
    # create_dataset()
    split_dataset('data/rna_jt_32-512/rna_jt_32-512_1277621.obj', 1277621)







