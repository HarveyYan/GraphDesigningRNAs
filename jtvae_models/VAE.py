import torch
import torch.nn as nn

from jtvae_models.GraphEncoder import GraphEncoder
from jtvae_models.TreeEncoder import TreeEncoder
from jtvae_models.Decoder import UnifiedDecoder

class JunctionTreeVAE(nn.Module):

    def __init__(self, hidden_dim, latent_dim, depthG, depthT, **kwargs):
        super(JunctionTreeVAE, self).__init__()
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.depthG = depthG
        self.depthT = depthT

        self.g_encoder = GraphEncoder(self.hidden_dim, self.depthG, **kwargs)
        self.g_mean = nn.Linear(hidden_dim, latent_dim)
        self.g_var = nn.Linear(hidden_dim, latent_dim)

        self.t_encoder = TreeEncoder(self.hidden_dim, self.depthT, **kwargs)
        self.t_mean = nn.Linear(hidden_dim, latent_dim)
        self.t_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = UnifiedDecoder(hidden_dim, latent_dim, **kwargs)
        # self.g_decoder = GraphDecoder(hidden_dim, latent_dim)


    def encode(self, g_encoder_input, t_encoder_input):
        nuc_embedding, graph_vectors = self.g_encoder(*g_encoder_input)
        enc_tree_messages, tree_vectors = self.t_encoder(nuc_embedding, *t_encoder_input)
        return graph_vectors, tree_vectors, enc_tree_messages

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))  # Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).to(self.device)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def forward(self, input, beta):
        tree_batch, graph_encoder_input, tree_encoder_input = input
        graph_vectors, tree_vectors, enc_tree_messages = self.encode(graph_encoder_input, tree_encoder_input)

        graph_latent_vec, graph_kl_loss = self.rsample(graph_vectors, self.g_mean, self.g_var)
        tree_latent_vec, tree_kl_loss = self.rsample(tree_vectors, self.t_mean, self.t_var)

        all_kl_loss = graph_kl_loss + tree_kl_loss

        # when training the decoders it is imperative to use teacher forcing,
        # i.e. feeding the ground truth topology and subgraph conformation
        all_loss, all_acc, tree_messages, tree_traces = \
            self.decoder(tree_batch, tree_latent_vec, graph_latent_vec)

        return sum(all_loss) + beta * all_kl_loss, all_kl_loss, all_acc


if __name__ == "__main__":
    test_model = JunctionTreeVAE(30, 30, 10, 20)
    from lib.data_utils import JunctionTreeFolder

    loader = JunctionTreeFolder('../data/rna_jt', 32, num_workers=0)
    for batch in loader:
        tree_batch, graph_encoder_input, tree_encoder_input = batch
        all_loss, kl_loss, all_acc = test_model(batch, 1.)
        print(all_acc)
        print('=' * 30)