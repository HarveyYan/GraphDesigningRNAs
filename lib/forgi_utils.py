import forgi.graph.bulge_graph as fgb
import csv
import os
import subprocess as sp
import re
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
import numpy as np
import sys
from RNA import fold

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LETTERS = ['A', 'C', 'G', 'U']

MAPPING = {
    'A': 'U',
    'C': 'G',
    'G': 'C',
    'T': 'A',
}


def plot(name, seq, struct):
    sp.call('echo ">%s\n%s\n%s" | RNAplot' % (name, seq, struct), shell=True)


def count_multiloops(bg):
    # count multi-loop, connect always starts with a stem
    multi = []
    for line in bg.get_connect_str().split('\n'):
        ml_seg_ids = list(map(lambda x: int(line[x.start() + 1]), re.finditer('m', line)))  # indices of the 'm'
        if len(ml_seg_ids) > 0:
            # 'connect s2 xx xx xx xx', hairpin and interior take two positions
            elements = line.split(' ')[2:]
            if len(ml_seg_ids) == 1:
                continue
            elif len(ml_seg_ids) == 2:
                flag = True
                for loop in multi:
                    if ml_seg_ids[0] in loop:
                        loop.append(ml_seg_ids[1])
                        flag = False
                    elif ml_seg_ids[1] in loop:
                        loop.append(ml_seg_ids[0])
                        flag = False
                if flag:
                    multi.append([ml_seg_ids[0], ml_seg_ids[1]])
            elif len(ml_seg_ids) == 3:
                ml_seg_indices = [i for i, id in enumerate(elements) if id.startswith('m')]
                if 0 in ml_seg_indices and 3 in ml_seg_indices:
                    id_0 = ml_seg_ids[ml_seg_indices.index(0)]
                    id_1 = ml_seg_ids[ml_seg_indices.index(3)]
                elif 1 in ml_seg_indices and 2 in ml_seg_indices:
                    id_0 = ml_seg_ids[ml_seg_indices.index(1)]
                    id_1 = ml_seg_ids[ml_seg_indices.index(2)]
                else:
                    print('Error processing multi-loops')
                    print(bg.seq)
                flag = True
                for loop in multi:
                    if id_0 in loop:
                        loop.append(id_1)
                        flag = False
                    elif id_1 in loop:
                        loop.append(id_0)
                        flag = False
                if flag:
                    multi.append([id_0, id_1])
            elif len(ml_seg_ids) == 4:
                for id_0, id_1 in [(ml_seg_ids[0], ml_seg_ids[3]), (ml_seg_ids[1], ml_seg_ids[2])]:
                    flag = True
                    for loop in multi:
                        if id_0 in loop:
                            loop.append(id_1)
                            flag = False
                        elif id_1 in loop:
                            loop.append(id_0)
                            flag = False
                    if flag:
                        multi.append([id_0, id_1])

    return len(list(filter(lambda x: len(x) >= 3, multi)))


def mloop_check(x):
    for item in x:
        if not item.startswith('m'):
            return False
    if len(x) <= 2:
        return False
    return True


def annotate(rna_seq, **kwargs):
    flags = kwargs.get('flags', '--noPS')
    cmd = 'echo "%s" | RNAfold %s' % (rna_seq, flags)
    output = sp.check_output(cmd, shell=True)
    output = str(output, 'UTF8').strip().split('\n')

    bg = fgb.BulgeGraph.from_dotbracket(output[1].split(' ')[0])
    res = {
        'seq': output[0],
        'struct': output[1].split(' ')[0],
        'MFE': ' '.join(output[1].split(' ')[1:]),
        'stem': len(list(filter(lambda x: 's' in x, list(bg.defines.keys())))),
        'hairpin': len(list(filter(lambda x: 'h' in x, list(bg.defines.keys())))),
        'inter': len(list(filter(lambda x: 'i' in x, list(bg.defines.keys())))),
        # multi-loop segments
        'multi_seg': len(list(filter(lambda x: 'm' in x and len(bg.defines[x]) != 0, list(bg.defines.keys())))),
        # multi-loop cycles, soft
        'multi_cycle_soft': len(bg.find_mlonly_multiloops()),
        # closed, 3 segments ar least multi-loop cycles, strict
        'multi_cycle_strict': len(list(filter(mloop_check, bg.find_mlonly_multiloops())))
    }
    return res


def generate_seq_dataset(size, length):
    if not os.path.exists(os.path.join(basedir, 'data', 'all_cdna_{}.txt'.format(length))):
        print('Preparing cDNA from ensembl file')
        prepare_RNA_seq(size, length)
    with open(os.path.join(basedir, 'data', 'all_cdna_{}.txt'.format(length)), 'r') as file, \
            open(os.path.join(basedir, 'data', 'rna_dataset_{}.csv'.format(length)), 'w') as csv_file:
        writer = csv.DictWriter(csv_file,
                                fieldnames=['seq', 'struct', 'MFE', 'stem', 'hairpin', 'inter', 'multi_seg',
                                            'multi_cycle_soft', 'multi_cycle_strict'])
        writer.writeheader()
        all_rna = [''.join(list(map(MAPPING.get, seq.rstrip()))) for seq in file]
        pool = mp.Pool(12)
        outcomes = list(tqdm(pool.imap(annotate, all_rna)))
        writer.writerows(outcomes)


def prepare_RNA_seq(size, length):
    all_cDNA = []
    with open(os.path.join(basedir, 'data', 'ensembl_cDNA.fa'), 'r') as file:
        for line in file:
            if line.startswith('>'):
                continue
            else:
                seq = line.rstrip()
                cut_indices = list(range(0, len(seq), length))
                for i in range(len(cut_indices)):
                    if i != len(cut_indices) - 1:
                        all_cDNA.append(seq[cut_indices[i]:cut_indices[i + 1]])
                        size -= 1
                    elif len(seq[cut_indices[i]:]) == length:
                        all_cDNA.append(seq[cut_indices[i]:])
                        size -= 1
                if len(all_cDNA) >= size:
                    break
        all_cDNA = set(all_cDNA)
        print('All unique RNA seqs,', len(all_cDNA))
    with open(os.path.join(basedir, 'data', 'all_cdna_{}.txt'.format(length)), 'w') as tofile:
        tofile.writelines('\n'.join(all_cDNA))


def check_reverse_rnafold(**kwargs):
    '''
    RNAfold is directional aware. Therefore generated rna graphs need to consider both directions.
    :param kwargs:
    :return:
    '''
    length = kwargs.get('length', 32)
    size = kwargs.get('size', 2e7)
    if not os.path.exists(os.path.join(basedir, 'data', 'rna_dataset_%d.csv' % (length))):
        generate_seq_dataset(size, length)

    with open(os.path.join(basedir, 'data', 'rna_dataset_%d.csv' % (length)), 'r') as f:
        reader = pd.read_csv(f)
        seq_list = reader['seq']
        struct_list = reader['struct']
        for seq, struct in zip(seq_list, struct_list):
            reversed_struct = fold(seq[::-1])[0]
            if struct[::-1] != reversed_struct:
                print(seq, struct, reversed_struct)



if __name__ == "__main__":
    generate_seq_dataset(size=100000, length=32)