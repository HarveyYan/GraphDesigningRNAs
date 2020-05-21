import os
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--fasta_path', type=str, required=True)
parser.add_argument('--mer_size', type=int, default=10)

VOCAB = ['A', 'C', 'G', 'U']


def return_kmer_content(seq, mer_size):
    size = len(seq)
    readout_vec = np.zeros(4 ** mer_size)
    if size < mer_size:
        return readout_vec
    for i in range(0, size - mer_size + 1):
        mer = seq[i: i + mer_size]
        if 'N' in mer:
            continue
        indexed_c = [VOCAB.index(c) for c in mer]
        index = sum([indexed_c[j] * 4 ** j for j in range(mer_size)])
        readout_vec[index] += 1
    return readout_vec


if __name__ == "__main__":
    args = parser.parse_args()
    # assert os.path.exists(args.fasta_path)
    # fasta_path = args.fasta_path

    base_dir = 'lstm_baseline_output/cached-solutions-[20200429-222820-flow-prior-resumed-5e-4-1e-2]'

    # for epoch_dirname in os.listdir(base_dir):
    epoch_dir = os.path.join(base_dir, 'epoch-14')

    for filename in os.listdir(epoch_dir):
        if filename.endswith('seq.fa'):
            fasta_path = os.path.join(epoch_dir, filename)
        else:
            continue
        all_seq = []
        with open(fasta_path, 'r') as file:
            seq = ''
            for line in file:
                if line.startswith('>'):
                    if len(seq) > 0:
                        all_seq.append(seq)
                        seq = ''
                else:
                    seq += line.strip().upper().replace('T', 'U')
            all_seq.append(seq)
        print('All sequences loaded from', fasta_path)
        pool = Pool(10)

        func = partial(return_kmer_content, mer_size=args.mer_size)
        all_readouts = np.array(list(tqdm(pool.imap(func, all_seq), total=len(all_seq))))
        all_readouts = all_readouts.sum(axis=0)
        all_readouts /= np.sum(all_readouts)

        fasta_dir = os.sep.join(fasta_path.split(os.sep)[:-1])
        fasta_filename = fasta_path.split(os.sep)[-1]

        np.save(os.path.join(fasta_dir, '%d-mer-content-[%s]' %
                                  (args.mer_size, fasta_filename)), all_readouts)

        from scipy.stats import entropy
        print('entropy of kmer features vector', entropy(all_readouts))
