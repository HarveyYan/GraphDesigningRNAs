import torch
from torch import nn
import numpy as np

from baseline_models.FlowLSTMVAE import LSTMVAE
from baseline_models.GraphLSTMVAE import GraphLSTMVAE
from jtvae_models.VAE import JunctionTreeVAE


class SUPERVISED_VAE_Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super(SUPERVISED_VAE_Model, self).__init__()
        self.input_size = input_size  # latent dim of vae
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        self.vae_type = kwargs.get('vae_type', 'lstm')
        assert self.vae_type in ['lstm', 'graph_lstm', 'jtvae', 'jtvae_branched']

        if self.vae_type == 'lstm':
            self.vae = LSTMVAE(
                512, 128, 2, device=self.device, use_attention=True,
                use_flow_prior=True, use_aux_regressor=False).to(self.device)
        elif self.vae_type == 'graph_lstm':
            self.vae = GraphLSTMVAE(
                512, 128, 10, device=self.device, use_attention=False,
                use_flow_prior=True, use_aux_regressor=False).to(self.device)
        elif self.vae_type == 'jtvae':
            self.vae = JunctionTreeVAE(
                512, 64, 5, 10, decode_nuc_with_lstm=True, tree_encoder_arch='baseline',
                use_flow_prior=True, device=self.device).to(self.device)
        elif self.vae_type == 'jtvae_branched':
            self.vae = JunctionTreeVAE(
                512, 64, 5, 10, decode_nuc_with_lstm=True, tree_encoder_arch='branched',
                decoder_version='v1', use_flow_prior=True, device=self.device).to(self.device)

        self.loss_type = kwargs.get('loss_type', 'mse')
        assert self.loss_type in ['mse', 'binary_ce', 'ce']

        if self.hidden_size is not None:
            self.classifier_nonlinear = nn.Linear(input_size, self.hidden_size)
            self.classifier_output = nn.Linear(self.hidden_size, output_size)
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.classifier_output = nn.Linear(self.input_size, output_size)

        if self.loss_type == 'mse':
            self.loss = nn.MSELoss(reduction="none")
        elif self.loss_type == 'binary_ce':
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch_input, batch_label, pass_decoder=True):
        batch_size = len(batch_label)

        ret_dict = {}
        if self.vae_type == 'lstm':
            batch_joint_encodings, batch_seq_label = batch_input
            latent_vec = self.vae.encode(batch_joint_encodings)
            z_vec = self.vae.mean(latent_vec)

            if pass_decoder:
                latent_vec, (entropy, log_pz) = self.vae.rsample(latent_vec, nsamples=1)
                latent_vec = latent_vec[:, 0, :]  # squeeze
                ret_dict = self.vae.decoder(batch_joint_encodings, latent_vec, batch_seq_label)
                ret_dict['entropy_loss'] = -entropy.mean()
                ret_dict['prior_loss'] = -log_pz.mean()
        ### todo, plug vaes
        elif self.vae_type == 'graph_lstm':
            graph_encoder_input, batch_joint_encoding, batch_seq_label = batch_input
            latent_vec = self.vae.encode(graph_encoder_input)
            z_vec = self.vae.mean(latent_vec)

            if pass_decoder:
                latent_vec, (entropy, log_pz) = self.vae.rsample(latent_vec, nsamples=1)
                latent_vec = latent_vec[:, 0, :]  # squeeze
                ret_dict = self.vae.decoder(batch_joint_encoding, latent_vec, batch_seq_label)
                ret_dict['entropy_loss'] = -entropy.mean()
                ret_dict['prior_loss'] = -log_pz.mean()

        elif self.vae_type == 'jtvae' or self.vae_type == 'jtvae_branched':
            tree_batch, graph_encoder_input, tree_encoder_input = batch_input
            graph_vectors, tree_vectors = self.vae.encode(graph_encoder_input, tree_encoder_input)

            z_vec = torch.cat([self.vae.g_mean(graph_vectors),
                               self.vae.t_mean(tree_vectors)], dim=-1)

            if pass_decoder:
                (_, graph_z_vecs, tree_z_vecs), (entropy, log_pz) = self.vae.rsample(graph_vectors, tree_vectors)
                graph_z_vecs = graph_z_vecs[:, 0, :]
                tree_z_vecs = tree_z_vecs[:, 0, :]
                ret_dict = self.vae.decoder(tree_batch, tree_z_vecs, graph_z_vecs)
                ret_dict['entropy_loss'] = -entropy.mean()
                ret_dict['prior_loss'] = -log_pz.mean()


        if self.loss_type == 'mse' or self.loss_type == 'binary_ce':
            batch_label = torch.as_tensor(batch_label.astype(np.float32)).to(self.device)
        else:
            batch_label = torch.as_tensor(batch_label.astype(np.long)).to(self.device)

        if self.hidden_size is not None:
            intermediate = torch.relu(self.classifier_nonlinear(z_vec))
            preds = self.classifier_output(self.dropout(intermediate))
        else:
            preds = self.classifier_output(batch_input)

        loss = self.loss(preds, batch_label)

        if self.loss_type == 'mse':
            preds = preds.cpu().detach().numpy()
        elif self.loss_type == 'binary_ce':
            preds = torch.sigmoid(preds).cpu().detach().numpy()
        else:
            preds = torch.softmax(preds, dim=-1).cpu().detach().numpy()

        ret_dict['supervised_loss'] = torch.sum(loss)
        ret_dict['nb_supervised_preds'] = batch_size
        ret_dict['supervised_preds'] = preds

        return ret_dict
