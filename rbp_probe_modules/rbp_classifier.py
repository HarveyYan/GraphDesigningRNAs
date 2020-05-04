import torch
from torch import nn
import numpy as np


class RBP_EMB_Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, pretrain_vae, **kwargs):
        super(RBP_EMB_Classifier, self).__init__()
        self.input_size = input_size  # latent dim of vae
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.vae_type = kwargs.get('vae_type', 'lstm')
        assert self.vae_type in ['lstm', 'graph_lstm', 'jtvae']
        self.pretrained_vae = pretrain_vae

        self.classifier_nonlinear = nn.Linear(input_size, hidden_size)
        self.classifier_output = nn.Linear(hidden_size, output_size)

        '''beware multi-task learning setup'''
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, batch_input, batch_label):
        with torch.no_grad():
            if self.vae_type == 'lstm':
                latent_vec = self.pretrained_vae.encode(batch_input)
                z_vec = self.pretrained_vae.mean(latent_vec)
            elif self.vae_type == 'graph_lstm':
                latent_vec = self.pretrained_vae.encode(*batch_input)
                z_vec = self.pretrained_vae.mean(latent_vec)
            elif self.vae_type == 'jtvae':
                graph_vectors, tree_vectors = self.pretrained_vae.encode(*batch_input)
                z_vec = torch.cat([self.pretrained_vae.g_mean(graph_vectors),
                                   self.pretrained_vae.t_mean(tree_vectors)], dim=-1)

        batch_size = len(batch_input)
        batch_label = torch.as_tensor(batch_label.astype(np.float32)).to(self.device)
        preds = self.classifier_output(torch.relu(self.classifier_nonlinear(z_vec)))

        loss = self.loss(preds, batch_label)
        preds = torch.sigmoid(preds).cpu().detach().numpy()

        ret_dict = {
            'loss': torch.sum(loss),
            'nb_preds': batch_size * self.output_size,
            'preds': preds
        }

        return ret_dict
