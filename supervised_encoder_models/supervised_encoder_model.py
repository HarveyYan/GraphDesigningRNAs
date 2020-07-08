import torch
from torch import nn
import numpy as np

from baseline_models.FlowLSTMVAE import LSTMVAE
from baseline_models.GraphLSTMVAE import GraphLSTMVAE
from jtvae_models.VAE import JunctionTreeVAE
from baseline_models.SimpleSeqonlyVAE import SimpleSeqOnlyVAE


class FULL_ENC_Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super(FULL_ENC_Model, self).__init__()
        self.input_size = input_size  # latent dim of vae
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        self.vae_type = kwargs.get('vae_type', 'lstm')
        assert self.vae_type in ['lstm', 'lstm_seqonly', 'graph_lstm', 'jtvae', 'jtvae_branched']

        if self.vae_type == 'lstm':
            self.vae = LSTMVAE(
                512, 128, 2, device=self.device, use_attention=True,
                use_flow_prior=False, use_aux_regressor=False).to(self.device)
            del self.vae.var
            del self.vae.decoder
        elif self.vae_type == 'lstm_seqonly':
            self.vae = SimpleSeqOnlyVAE(
                512, 128, 2, use_attention=True,
                device=self.device).to(self.device)
        elif self.vae_type == 'graph_lstm':
            self.vae = GraphLSTMVAE(
                512, 128, 10, device=self.device, use_attention=False,
                use_flow_prior=False, use_aux_regressor=False).to(self.device)
            del self.vae.var
            del self.vae.decoder
        elif self.vae_type == 'jtvae':
            self.vae = JunctionTreeVAE(
                512, 64, 5, 10, decode_nuc_with_lstm=True, tree_encoder_arch='baseline',
                use_flow_prior=False, device=self.device).to(self.device)
            del self.vae.g_var
            del self.vae.t_var
            del self.vae.decoder
        elif self.vae_type == 'jtvae_branched':
            self.vae = JunctionTreeVAE(
                512, 64, 5, 10, decode_nuc_with_lstm=True, tree_encoder_arch='branched',
                decoder_version='v1', use_flow_prior=False, device=self.device).to(self.device)
            del self.vae.g_var
            del self.vae.t_var
            del self.vae.decoder

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
            self.loss = nn.BCELoss(reduction="none")
        else:
            self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch_input, batch_label):
        batch_size = len(batch_label)

        if self.vae_type == 'lstm':
            batch_joint_encodings, _ = batch_input
            latent_vec = self.vae.encode(batch_joint_encodings)
            z_vec = self.vae.mean(latent_vec)
        elif self.vae_type == 'lstm_seqonly':
            batch_encodings, _ = batch_input
            latent_vec = self.vae.encode(batch_encodings)
            z_vec = self.vae.mean(latent_vec)
        elif self.vae_type == 'graph_lstm':
            graph_encoder_input, _, _ = batch_input
            latent_vec = self.vae.encode(graph_encoder_input)
            z_vec = self.vae.mean(latent_vec)
        elif self.vae_type == 'jtvae' or self.vae_type == 'jtvae_branched':
            _, graph_encoder_input, tree_encoder_input = batch_input
            graph_vectors, tree_vectors = self.vae.encode(graph_encoder_input, tree_encoder_input)
            z_vec = torch.cat([self.vae.g_mean(graph_vectors),
                               self.vae.t_mean(tree_vectors)], dim=-1)

        if self.loss_type == 'mse':
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

        ret_dict = {
            'loss': torch.sum(loss),
            'nb_preds': batch_size,
            'preds': preds
        }

        return ret_dict
