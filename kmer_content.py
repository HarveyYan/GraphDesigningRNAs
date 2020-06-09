import os
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from scipy.stats import entropy
import lib.logger

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

    base_dir = 'graph-baseline-output/20200525-215003-resumed-[20200515-140604-512-128-5-maxpooled-hidden-states-mb-3e-3-sb-5e-4-amsgrad]/rigorosity'
    csv_file = lib.logger.CSVLogger(
        '%d-mer-diversity.csv' % (args.mer_size), base_dir,
        ['epoch', 'prior-det-noreg', 'prior-sto-noreg', 'prior-sto-reg', 'valid-post-det-noreg', 'valid-post-sto-noreg',
         'valid-post-sto-reg'])

    epochs_to_load = []
    for dirname in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, dirname)):
            epochs_to_load.append(int(dirname.split('-')[-1]))
    epochs_to_load = list(np.sort(epochs_to_load))

    print(epochs_to_load)

    pool = Pool(10)

    for epoch in epochs_to_load:
        epoch_dir = os.path.join(base_dir, 'epoch-%d' % (epoch))

        to_dict = {
            'epoch': epoch,
        }

        for filename in os.listdir(epoch_dir):
            if not filename.endswith('seq.fa'):
                continue

            fasta_path = os.path.join(epoch_dir, filename)
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

            func = partial(return_kmer_content, mer_size=args.mer_size)
            all_readouts = np.array(list(tqdm(pool.imap(func, all_seq), total=len(all_seq))))
            all_readouts = all_readouts.sum(axis=0)
            all_readouts /= np.sum(all_readouts)

            fasta_dir = os.sep.join(fasta_path.split(os.sep)[:-1])
            fasta_filename = fasta_path.split(os.sep)[-1]

            np.save(os.path.join(fasta_dir, '%d-mer-content-[%s]' %
                                 (args.mer_size, fasta_filename)), all_readouts)

            print('entropy of %d-mer features vector in %s:' % (args.mer_size, fasta_path)
                  , entropy(all_readouts))

            to_dict[filename.split('seq')[0][:-1]] = entropy(all_readouts)

        csv_file.update_with_dict(to_dict)

    csv_file.close()
