import os
import random

from supervised_encoder_models.task_dataloader import create_rnacompete_s_dataset, \
    rnacompete_s_all_rbps, rnacompete_s_datapath, rnacompete_s_pool_datapath

global_seq_limit = 500000
# strength_thres = {
#     'HuR': 4,
#     'PTB': 3,
#     'QKI': 4,
#     'Vts1': 3,
#     'RBMY': 3,
#     'SF2': 4,
#     'SLBP': 4,
# }

for rbp_name in rnacompete_s_all_rbps:
    pos_datapath_filled = rnacompete_s_datapath.format('%d_%s' % (rnacompete_s_all_rbps.index(rbp_name) + 1, rbp_name))

    pos_seq, neg_seq = create_rnacompete_s_dataset(
        pos_datapath_filled, rnacompete_s_pool_datapath
        , global_seq_limit)

    dataset_dir = 'data/RNAcompete_S/curated_dataset'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    train_test_ratio = 0.2
    size_pos = len(pos_seq)
    size_neg = len(neg_seq)
    random.shuffle(pos_seq)
    random.shuffle(neg_seq)

    test_seq = pos_seq[:int(size_pos * train_test_ratio)] + neg_seq[:int(size_neg * train_test_ratio)]
    test_targets = [1] * int(size_pos * train_test_ratio) + [0] * int(size_neg * train_test_ratio)
    train_seq = pos_seq[int(size_pos * train_test_ratio):] + neg_seq[int(size_neg * train_test_ratio):]
    train_targets = [1] * int(size_pos * (1 - train_test_ratio)) + [0] * int(size_neg * (1 - train_test_ratio))

    with open(os.path.join(dataset_dir, '%s_test.fa' % (rbp_name)), 'w') as file:
        for seq, label in zip(test_seq, test_targets):
            file.write('> label:%d\n%s\n' % (label, seq))

    with open(os.path.join(dataset_dir, '%s_train.fa' % (rbp_name)), 'w') as file:
        for seq, label in zip(train_seq, train_targets):
            file.write('> label:%d\n%s\n' % (label, seq))
