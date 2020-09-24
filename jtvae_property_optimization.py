import torch
import numpy as np
import RNA
import argparse
import os

from lib.tree_decomp import RNAJunctionTree
from supervised_encoder_models.task_dataloader import TaskFolder, \
    rnacompete_s_all_rbps, read_curated_rnacompete_s_dataset

# semi-supervised VAE model
from supervised_encoder_models.supervised_vae_model import SUPERVISED_VAE_Model
# fully-supervised encoders as oracle
from supervised_encoder_models.supervised_encoder_model import FULL_ENC_Model

from jtvae_models.GraphEncoder import GraphEncoder as jtvae_GraphEncoder
from jtvae_models.TreeEncoder import TreeEncoder
from jtvae_models.BranchedTreeEncoder import BranchedTreeEncoder


def get_regressor_input(batch_seq, batch_struct, tree_enc_type='jtvae'):
    all_trees = []
    for seq, struct in zip(batch_seq, batch_struct):
        tree = RNAJunctionTree(seq, struct)
        all_trees.append(tree)

    graph_encoder_input = jtvae_GraphEncoder.prepare_batch_data(
        [(tree.rna_seq, tree.rna_struct) for tree in all_trees])
    if tree_enc_type == 'jtvae':
        tree_encoder_input = TreeEncoder.prepare_batch_data(all_trees)
    elif tree_enc_type == 'jtvae_branched':
        tree_encoder_input = BranchedTreeEncoder.prepare_batch_data(all_trees)

    return (all_trees, graph_encoder_input, tree_encoder_input)


def optimize_rna_with_fixed_iters(batch_seq, batch_struct, nb_steps=100):
    # initial sequence and structure from the test set
    batch_input = get_regressor_input(batch_seq, batch_struct, tree_enc_type='jtvae')
    _, graph_encoder_input, tree_encoder_input = batch_input
    init_graph_vectors, init_tree_vectors = vae_model.vae.encode(graph_encoder_input, tree_encoder_input)
    z_vec = init_z_vec = torch.cat([vae_model.vae.g_mean(init_graph_vectors), vae_model.vae.t_mean(init_tree_vectors)],
                                   dim=-1)
    batch_size = len(batch_seq)
    decoded_rna_trajectory = []
    emb_pred_trajectory = []
    regressor_pred_trajectory = []
    for _ in range(batch_size):
        decoded_rna_trajectory.append([])
        emb_pred_trajectory.append([])
        regressor_pred_trajectory.append([])
    batch_indices = list(range(len(batch_seq)))

    for step in range(nb_steps):
        if len(batch_indices) == 0:
            break
        vae_model.zero_grad()
        batch_z_vec = z_vec.index_select(0, torch.as_tensor(batch_indices).to(device))
        # embedding prediction
        emb_pred = vae_model.classifier_output(torch.relu(vae_model.classifier_nonlinear(batch_z_vec)))
        np_emb_pred = emb_pred.cpu().detach().numpy()[:, 0]
        for i, idx in enumerate(batch_indices):
            emb_pred_trajectory[idx].append(np_emb_pred[i])

        # reconstruct RNA molecule
        with torch.no_grad():
            all_rna_seq, all_nodes, all_successful = vae_model.vae.decoder.decode(
                batch_z_vec[:, 64:], batch_z_vec[:, :64],
                prob_decode=False, enforce_topo_prior=True, enforce_hpn_prior=True, enforce_dec_prior=True)

        batch_decoded_rna, batch_decoded_struct, batch_indices_to_remove = [], [], []
        for idx, rna_seq, nodes, successful in zip(batch_indices, all_rna_seq, all_nodes, all_successful):
            if successful is True:
                tree = RNAJunctionTree(rna_seq, None, nodes=nodes)
                if tree.is_valid is True:
                    rna_struct = tree.rna_struct
                    fd = tree.fe_deviation
                    decoded_rna_trajectory[idx].append((rna_seq, rna_struct, fd))
                    batch_decoded_rna.append(rna_seq)
                    batch_decoded_struct.append(rna_struct)
                    continue
                else:
                    # ending due to decoding error
                    batch_indices_to_remove.append(idx)
            else:
                # ending due to decoding error
                batch_indices_to_remove.append(idx)

        for idx in batch_indices_to_remove:
            batch_indices.remove(idx)

        regressor_input = get_regressor_input(batch_decoded_rna, batch_decoded_struct, tree_enc_type='jtvae')

        # full regressor prediction
        regressor_pred = regressor_model.predict(regressor_input)[:, 0]
        for i, idx in enumerate(batch_indices):
            try:
                regressor_pred_trajectory[idx].append(regressor_pred[i])
            except IndexError as e:
                import pdb
                pdb.set_trace()

        am_loss = -torch.sum(emb_pred)
        z_vec.retain_grad()
        am_loss.backward(retain_graph=True)
        input_grad = z_vec.grad.detach().clone()
        z_vec.grad.data.zero_()
        z_vec -= input_grad * 1e-1

    return decoded_rna_trajectory, emb_pred_trajectory, regressor_pred_trajectory


parser = argparse.ArgumentParser()
parser.add_argument('--rbp_name', type=str, default='PTB')

rbp_vae_path = {
    'HuR': 'output/jtvae-rnacompeteS-mb-5e-4/20200912-034714-HuR-mb-5e-4-sb-5e-4-fixed-10epochs/model.epoch-10',
    'PTB': 'output/jtvae-rnacompeteS-mb-5e-4/20200912-042212-PTB-mb-5e-4-sb-5e-4-fixed-10epochs/model.epoch-10',
    'QKI': 'output/jtvae-rnacompeteS-mb-5e-4/20200912-060527-QKI-mb-5e-4-sb-5e-4-fixed-10epochs/model.epoch-10',
    'Vts1': 'output/jtvae-rnacompeteS-mb-5e-4/20200912-061521-Vts1-mb-5e-4-sb-5e-4-fixed-10epochs/model.epoch-10',
    'RBMY': 'output/jtvae-rnacompeteS-mb-5e-4/20200912-062223-RBMY-mb-5e-4-sb-5e-4-fixed-10epochs/model.epoch-10',
    'SF2': 'output/jtvae-rnacompeteS-mb-5e-4/20200912-063408-SF2-mb-5e-4-sb-5e-4-fixed-10epochs/model.epoch-10',
    'SLBP': 'output/jtvae-rnacompeteS-mb-5e-4/20200912-064624-SLBP-mb-5e-4-sb-5e-4-fixed-10epochs/model.epoch-10'
}

rbp_regressor_path = {
    'HuR': 'full-rnacompete-S-regressor/jtvae_1/HuR/model.epoch-4',
    'PTB': 'full-rnacompete-S-regressor/jtvae_1/PTB/model.epoch-6',
    'QKI': 'full-rnacompete-S-regressor/jtvae_1/QKI/model.epoch-7',
    'Vts1': 'full-rnacompete-S-regressor/jtvae_1/Vts1/model.epoch-7',
    'RBMY': 'full-rnacompete-S-regressor/jtvae_1/RBMY/model.epoch-9',
    'SF2': 'full-rnacompete-S-regressor/jtvae_1/SF2/model.epoch-5',
    'SLBP': 'full-rnacompete-S-regressor/jtvae_1/SLBP/model.epoch-6'
}

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    rbp_name = args.rbp_name

    datapath_test = 'data/RNAcompete_S/curated_dataset/' + rbp_name + '_test.fa'
    test_pos, test_neg = read_curated_rnacompete_s_dataset(datapath_test)
    test_seq = test_pos + test_neg
    test_targets = [1] * len(test_pos) + [0] * len(test_neg)
    test_targets = np.array(test_targets)[:, None]

    device = torch.device('cuda:0')
    vae_model = SUPERVISED_VAE_Model(128, 512, 1, device=device,
                                     vae_type='jtvae', loss_type='binary_ce').to(device)
    vae_model.eval()
    regressor_model = FULL_ENC_Model(128, 256, 1, device=device,
                                     vae_type='jtvae', loss_type='binary_ce').to(device)
    regressor_model.eval()

    # local weights
    vae_model.load_state_dict(torch.load(rbp_vae_path[rbp_name], map_location=device)['model_weights'])
    regressor_model.load_state_dict(torch.load(rbp_regressor_path[rbp_name], map_location=device)['model_weights'])

    batch_size = 128
    test_loader = TaskFolder(test_seq, test_targets, batch_size, shuffle=False,
                             preprocess_type='jtvae', num_workers=8)

    # sub sampling RNAs
    all_emb_preds, all_su_preds = [], []
    with torch.no_grad():
        for batch_input, batch_label in test_loader:
            # compute various metrics
            ret_dict = vae_model(batch_input, batch_label, pass_decoder=False)
            all_su_preds.append(ret_dict['supervised_preds'])
            all_emb_preds.append(regressor_model.predict(batch_input))

    all_emb_preds = np.concatenate(all_emb_preds, axis=0)[:, 0]
    all_su_preds = np.concatenate(all_su_preds, axis=0)[:, 0]

    # all_idx = np.where((test_targets[:, 0] == (all_emb_preds > 0.5).astype(np.int)) & (
    #         test_targets[:, 0] == (all_su_preds > 0.5).astype(np.int)) & (test_targets[:, 0] == 0))[0]
    all_idx = np.where((test_targets[:, 0] == (all_emb_preds > 0.5).astype(np.int)) & (
            test_targets[:, 0] == (all_su_preds > 0.5).astype(np.int)))[0]
    all_idx = np.random.choice(all_idx, size=min(10000, len(all_idx)), replace=False)
    subset_test_seq = list(np.array(test_seq)[all_idx])
    subset_test_targets = test_targets[all_idx][:, 0]
    print('number of RNA being optimized:', len(subset_test_seq))

    nb_success = 0
    improvement = []

    vae_model.train()
    all_decoded_rna_t, all_emb_pred_t, all_reg_pred_t = [], [], []
    for i in range(0, len(subset_test_seq), batch_size):
        batch_seq = subset_test_seq[i: i + batch_size]
        batch_targets = subset_test_targets[i: i + batch_size]
        batch_struct = []
        for seq in batch_seq:
            batch_struct.append(RNA.fold(seq)[0])

        decoded_rna_trajectory, emb_pred_trajectory, regressor_pred_trajectory = \
            optimize_rna_with_fixed_iters(batch_seq, batch_struct, 200)
        all_decoded_rna_t.extend(decoded_rna_trajectory)
        all_emb_pred_t.extend(emb_pred_trajectory)
        all_reg_pred_t.extend(regressor_pred_trajectory)
        original_regressor_preds = [trajectory[0] for trajectory in regressor_pred_trajectory]
        final_regressor_preds = [trajectory[-1] for trajectory in regressor_pred_trajectory]
        nb_success += np.sum(np.array(final_regressor_preds) > np.array(original_regressor_preds))
        improvement.extend(np.array(final_regressor_preds) - np.array(original_regressor_preds))

    print('success rate: %d/%d' % (nb_success, len(subset_test_seq)))
    print('improvement rate: %.3f/%.3f' % (float(np.mean(improvement)), float(np.std(improvement))))

    # save results
    save_path = os.path.join(os.path.dirname(rbp_vae_path[rbp_name]), 'optimization')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, 'all_decoded_rna_t_200_pos_neg'), all_decoded_rna_t)
    np.save(os.path.join(save_path, 'all_emb_pred_t_200_pos_neg'), all_emb_pred_t)
    np.save(os.path.join(save_path, 'all_reg_pred_t_200_pos_neg'), all_reg_pred_t)
