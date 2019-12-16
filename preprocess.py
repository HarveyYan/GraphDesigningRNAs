import os
from lib.forgi_utils import generate_seq_dataset
import multiprocessing as mp
import pandas as pd
import pickle
from lib.tree_decomp import RNAJunctionTree
from tqdm import tqdm

def preprocess(args):
    return RNAJunctionTree(*args)

if __name__ == "__main__":
    if not os.path.exists('data/rna_dataset_32.csv'):
        # after removing the duplicate RNA examples
        # the actual size of the dataset might be smaller than 100000
        generate_seq_dataset(size=100000, length=32)

    file = pd.read_csv('data/rna_dataset_32.csv')

    # all_rna_jt = []
    # for seq, struct in zip(file['seq'], file['struct']):
    #     all_rna_jt.append(preprocess((seq, struct)))

    pool = mp.Pool(8)
    all_rna_jt = list(tqdm(pool.map(preprocess, zip(file['seq'], file['struct']))))

    if not os.path.exists('./data/rna_jt'):
        os.makedirs('./data/rna_jt')

    with open('./data/rna_jt/rna_jt.obj', 'wb') as file:
        pickle.dump(all_rna_jt, file)







