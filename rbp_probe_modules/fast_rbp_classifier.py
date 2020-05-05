import torch
from torch import nn
import numpy as np

class RBP_EMB_Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super(RBP_EMB_Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.vae_type = kwargs.get('vae_type', 'lstm')
        assert self.vae_type in ['lstm', 'graph_lstm', 'jtvae']

        self.classifier_nonlinear = nn.Linear(input_size, hidden_size)
        self.classifier_output = nn.Linear(hidden_size, output_size)

        '''beware multi-task learning setup'''
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, batch_input, batch_label):
        batch_size = len(batch_input)
        batch_input = batch_input.to(self.device)
        batch_label = torch.as_tensor(batch_label.astype(np.float32)).to(self.device)
        preds = self.classifier_output(torch.relu(self.classifier_nonlinear(batch_input)))

        loss = self.loss(preds, batch_label)
        preds = torch.sigmoid(preds).cpu().detach().numpy()

        ret_dict = {
            'loss': torch.sum(loss),
            'nb_preds': batch_size * self.output_size,
            'preds': preds
        }

        return ret_dict