import torch
from torch import nn
import numpy as np

class ncRNA_EMB_Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super(ncRNA_EMB_Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        if self.hidden_size is not None:
            self.classifier_nonlinear = nn.Linear(input_size, self.hidden_size)
            self.classifier_output = nn.Linear(self.hidden_size, output_size)
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.classifier_output = nn.Linear(self.input_size, output_size)

        '''beware multi-task learning setup'''
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch_input, batch_label):
        batch_size = len(batch_input)
        batch_input = batch_input.to(self.device)
        batch_label = torch.as_tensor(batch_label.astype(np.long)).to(self.device)
        if self.hidden_size is not None:
            intermediate = torch.relu(self.classifier_nonlinear(batch_input))
            preds = self.classifier_output(self.dropout(intermediate))
        else:
            preds = self.classifier_output(batch_input)

        loss = self.loss(preds, batch_label)
        preds = torch.softmax(preds, dim=-1).cpu().detach().numpy()

        ret_dict = {
            'loss': torch.sum(loss),
            'nb_preds': batch_size,
            'preds': preds
        }

        return ret_dict