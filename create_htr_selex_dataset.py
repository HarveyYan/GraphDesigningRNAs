import os
import glob
import subprocess
import numpy as np

fastq_path = 'data/ena_files/'
rbp_folders = {
    'BOLL': ['ERR2504060', 'ERR2504061', 'ERR2504062', 'ERR2504063'],
    'CARHSP1': ['ERR2504065', 'ERR2504066', 'ERR2504067', 'ERR2504068'],
    'CELF1': ['ERR2504070', 'ERR2504071', 'ERR2504072', 'ERR2504073'],
    'CELF3': ['ERR2504074', 'ERR2504075', 'ERR2504076', 'ERR2504077'],
    'CELF4': ['ERR2504080', 'ERR2504081', 'ERR2504082', 'ERR2504083'],
    'CSDA': ['ERR2504091', 'ERR2504092', 'ERR2504093', 'ERR2504094'],
    'CSTF2T': ['ERR2504098', 'ERR2504099', 'ERR2504100', 'ERR2504101'],
    'DAZ1': ['ERR2504108', 'ERR2504109', 'ERR2504110', 'ERR2504111'],
    'DAZAP1': ['ERR2504138', 'ERR2504139', 'ERR2504140', 'ERR2504141'],
    'DAZL': ['ERR2504146', 'ERR2504147', 'ERR2504148', 'ERR2504149'],
    'ELAVL3': ['ERR2504155', 'ERR2504156', 'ERR2504157', 'ERR2504158'],
    'ESRP1': ['ERR2504162', 'ERR2504163', 'ERR2504164', 'ERR2504165'],  # construct
    'HEXIM1': ['ERR2504170', 'ERR2504171', 'ERR2504172'],
    'HNRNPA0': ['ERR2504176', 'ERR2504177', 'ERR2504178', 'ERR2504179'],
    'IGF2BP1': ['ERR2504225', 'ERR2504226', 'ERR2504227', 'ERR2504228'],
    'KHDRBS1': ['ERR2504229', 'ERR2504230', 'ERR2504231', 'ERR2504232']
}

bp_complement = {
    'A': 'U',
    'C': 'G',
    'G': 'C',
    'U': 'A'
}

rbp_limit = 500000

train_test_split = 0.2


def create_htr_selex_dataset_multiclass(all_rbps, output_data_path):
    out_train_file = open(os.path.join(output_data_path, 'train.fa'), 'w')
    out_test_file = open(os.path.join(output_data_path, 'test.fa'), 'w')

    for class_label, rbp_name in enumerate(all_rbps):
        rbp_seq_ids = []
        rbp_seq = []
        rbp_dir = rbp_folders[rbp_name]
        for dir in rbp_dir:
            fastq_file_path = glob.glob(os.path.join(fastq_path, dir, '*fastq.gz'))[0]
            subprocess.call(
                'cutadapt %s -q 20 -o tmp.fa --fasta --report minimal' % (fastq_file_path),
                shell=True)
            with open('tmp.fa', 'r') as file:
                for line in file:
                    if line.startswith('>'):
                        rbp_seq_ids.append(line.rstrip())
                    else:
                        if 'N' in line:
                            rbp_seq_ids.pop()
                            continue
                        # taking the reverse complement
                        rbp_seq.append(
                            ''.join(map(lambda x: bp_complement[x], reversed(line.rstrip().replace('T', 'U')))))

        rbp_seq, rbp_seq_indices = np.unique(rbp_seq, return_index=True)
        rbp_seq_ids = np.array(rbp_seq_ids)[rbp_seq_indices]

        # check length
        all_length = np.array([len(seq) for seq in rbp_seq])
        rbp_len_indices = np.where((all_length == 40) | (all_length == 26))
        rbp_seq = rbp_seq[rbp_len_indices]
        rbp_seq_ids = rbp_seq_ids[rbp_len_indices]

        if len(rbp_seq) > rbp_limit:
            sampled_idx = np.random.permutation(len(rbp_seq))[:rbp_limit]
            rbp_seq = np.array(rbp_seq)[sampled_idx]
            rbp_seq_ids = rbp_seq_ids[sampled_idx]

        test_seq = rbp_seq[:int(len(rbp_seq) * train_test_split)]
        test_ids = rbp_seq_ids[:int(len(rbp_seq_ids) * train_test_split)]
        train_seq = rbp_seq[int(len(rbp_seq) * train_test_split):]
        train_ids = rbp_seq_ids[int(len(rbp_seq_ids) * train_test_split):]
        for seq_id, seq in zip(test_ids, test_seq):
            out_test_file.write('%s label:%d\n%s\n' % (seq_id, class_label, seq))
        for seq_id, seq in zip(train_ids, train_seq):
            out_train_file.write('%s label:%d\n%s\n' % (seq_id, class_label, seq))

def create_htr_selex_dataset_multilabel(all_rbps, output_data_path):
    all_seq_dict = {}
    nb_dups = 0

    for class_label, rbp_name in enumerate(all_rbps):
        class_label = str(class_label)
        rbp_seq, rbp_seq_id, rbp_seq_cycle = [], [], []
        rbp_dir = rbp_folders[rbp_name]
        for cycle, dir in enumerate(rbp_dir):
            fastq_file_path = glob.glob(os.path.join(fastq_path, dir, '*fastq.gz'))[0]
            subprocess.call(
                'cutadapt %s -q 20 -o tmp.fa --fasta --report minimal' % (fastq_file_path),
                shell=True)
            with open('tmp.fa', 'r') as file:
                for line in file:
                    if line.startswith('>'):
                        rbp_seq_id.append(line.rstrip())
                    else:
                        if 'N' in line:
                            rbp_seq_id.pop()
                            continue
                        # taking the reverse complement
                        rbp_seq.append(
                            ''.join(map(lambda x: bp_complement[x], reversed(line.rstrip().replace('T', 'U')))))
                        rbp_seq_cycle.append(cycle+1)

        # try to retain more sequences at the last cycle
        rbp_seq = list(reversed(rbp_seq))
        rbp_seq_id = list(reversed(rbp_seq_id))
        rbp_seq_cycle = list(reversed(rbp_seq_cycle))

        # remove duplicates
        rbp_seq, rbp_seq_idx = np.unique(rbp_seq, return_index=True)
        rbp_seq_id = np.array(rbp_seq_id)[rbp_seq_idx]
        rbp_seq_cycle = np.array(rbp_seq_cycle)[rbp_seq_idx]

        # check length
        all_length = np.array([len(seq) for seq in rbp_seq])
        rbp_len_idx = np.where((all_length == 40) | (all_length == 26))
        rbp_seq = rbp_seq[rbp_len_idx]
        rbp_seq_id = rbp_seq_id[rbp_len_idx]
        rbp_seq_cycle = rbp_seq_cycle[rbp_len_idx]

        if len(rbp_seq) > rbp_limit:
            probs = rbp_seq_cycle / np.sum(rbp_seq_cycle)
            # fetch idx first
            sampled_idx = np.random.choice(list(range(len(rbp_seq))), size=rbp_limit, replace=False, p=probs)

            rbp_seq = np.array(rbp_seq)[sampled_idx]
            rbp_seq_id = rbp_seq_id[sampled_idx]

        for seq, seq_id in zip(rbp_seq, rbp_seq_id):
            if seq not in all_seq_dict:
                all_seq_dict[seq] = (seq_id, [class_label])
            else:
                nb_dups += 1
                all_seq_dict[seq][1].append(class_label)

    print('number of duplicates', nb_dups)
    all_seq, all_ids, all_labels = [], [], []
    for seq, (id, labels) in all_seq_dict.items():
        all_seq.append(seq)
        all_ids.append(id)
        all_labels.append(','.join(labels))

    final_perm = np.random.permutation(len(all_seq))

    with open(os.path.join(output_data_path, 'test.fa'), 'w') as file:
        for test_idx in final_perm[:int(len(all_seq) * train_test_split)]:
            file.write('%s label:%s\n%s\n' % (all_ids[test_idx], all_labels[test_idx], all_seq[test_idx]))

    with open(os.path.join(output_data_path, 'train.fa'), 'w') as file:
        for train_idx in final_perm[int(len(all_seq) * train_test_split):]:
            file.write('%s label:%s\n%s\n' % (all_ids[train_idx], all_labels[train_idx], all_seq[train_idx]))


if __name__ == "__main__":
    output_folder = 'data/htr_selex_10rbps_multilabel/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    create_htr_selex_dataset_multilabel(
        ['BOLL', 'CARHSP1', 'CELF1', 'CSDA', 'CSTF2T',
         'DAZ1', 'ELAVL3', 'ESRP1', 'HEXIM1', 'IGF2BP1'], output_folder)
