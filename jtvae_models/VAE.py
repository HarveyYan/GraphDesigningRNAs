import torch
import torch.nn as nn
import numpy as np
import copy

from jtvae_models.GraphEncoder import GraphEncoder
from jtvae_models.TreeEncoder import TreeEncoder
from jtvae_models.ParallelAltDecoder import UnifiedDecoder
from jtvae_models.OrderedTreeEncoder import OrderedTreeEncoder

from cnf_models.flow import get_latent_cnf
from lib.nn_utils import log_sum_exp


class JunctionTreeVAE(nn.Module):

    def __init__(self, hidden_dim, latent_dim, depthG, depthT, **kwargs):
        super(JunctionTreeVAE, self).__init__()
        self.device = kwargs.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.depthG = depthG
        self.depthT = depthT
        self.tree_encoder_arch = kwargs.get('tree_encoder_arch', 'baseline')
        assert self.tree_encoder_arch in ['baseline', 'ordnuc'], 'selected tree encoder arch \'%s\' is not unknown' % (
            self.tree_encoder_arch)
        self.use_flow_prior = kwargs.get('use_flow_prior', True)

        self.g_encoder = GraphEncoder(self.hidden_dim, self.depthG, **kwargs)
        self.g_mean = nn.Linear(hidden_dim, latent_dim)
        self.g_var = nn.Linear(hidden_dim, latent_dim)

        if self.tree_encoder_arch == 'baseline':  # unordered segment, baseline
            self.t_encoder = TreeEncoder(self.hidden_dim, self.depthT, **kwargs)
        elif self.tree_encoder_arch == 'ordnuc':
            self.t_encoder = OrderedTreeEncoder(self.hidden_dim, self.depthT, **kwargs)
        self.t_mean = nn.Linear(hidden_dim, latent_dim)
        self.t_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = UnifiedDecoder(hidden_dim, latent_dim, **kwargs)

        if self.use_flow_prior:
            self.flow_args = {
                'latent_dims': "256-256",
                'latent_num_blocks': 1,
                'zdim': latent_dim * 2,  # because we have two latent variables
                'layer_type': 'concatsquash',
                'nonlinearity': 'tanh',
                'time_length': 0.5,
                'train_T': True,
                'solver': 'dopri5',
                'use_adjoint': True,
                'atol': 1e-5,
                'rtol': 1e-5,
                'batch_norm': False,
                'bn_lag': 0,
                'sync_bn': False,
                'device': self.device
            }
            self.latent_cnf = get_latent_cnf(self.flow_args)

    def encode(self, g_encoder_input, t_encoder_input):
        nuc_embedding, graph_vectors = self.g_encoder(*g_encoder_input)
        tree_vectors = self.t_encoder(nuc_embedding, *t_encoder_input)
        return graph_vectors, tree_vectors

    def rsample(self, graph_latent_vec, tree_latent_vec, nsamples=1):
        batch_size = graph_latent_vec.size(0)

        graph_z_mean = self.g_mean(graph_latent_vec)
        graph_z_log_var = -torch.abs(self.g_var(graph_latent_vec))  # Following Mueller et al.
        tree_z_mean = self.t_mean(tree_latent_vec)
        tree_z_log_var = -torch.abs(self.t_var(tree_latent_vec))  # Following Mueller et al.

        z_mean = torch.cat([graph_z_mean, tree_z_mean], dim=-1)
        z_log_var = torch.cat([graph_z_log_var, tree_z_log_var], dim=-1)

        entropy = self.gaussian_entropy(z_log_var)  # batch_size,
        z_vecs = self.reparameterize(z_mean, z_log_var, nsamples).reshape(batch_size * nsamples, self.latent_dim * 2)

        if self.use_flow_prior:
            w, delta_log_pw = self.latent_cnf(z_vecs, None, torch.zeros(batch_size * nsamples, 1).to(z_vecs))
            log_pw = self.standard_normal_logprob(w).reshape(batch_size, nsamples, 1)
            delta_log_pw = delta_log_pw.reshape(batch_size, nsamples, 1)
            log_pz = log_pw - delta_log_pw
        else:
            log_pz = self.standard_normal_logprob(z_vecs).reshape(batch_size, nsamples, 1)

        z_vecs = z_vecs.reshape(batch_size, nsamples, self.latent_dim * 2)
        graph_z_vecs = z_vecs[:, :, :self.latent_dim]
        tree_z_vecs = z_vecs[:, :, self.latent_dim:]

        return (z_vecs, graph_z_vecs, tree_z_vecs), (entropy, log_pz)

    def reparameterize(self, mean, logvar, nsamples=1):
        batch_size, nz = mean.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mean.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_().to(self.device)

        return mu_expd + torch.mul(eps, std_expd)

    def standard_normal_logprob(self, z):
        dim = z.size(-1)
        log_z = -0.5 * dim * np.log(2 * np.pi)
        return log_z - 0.5 * torch.sum(z.pow(2), dim=-1, keepdim=True)

    def gaussian_entropy(self, logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def calc_mi(self, input_batch, graph_latent_vec=None, tree_latent_vec=None):
        """Approximate the mutual information between x and z under the approximate posterior
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
        Returns: Float
        """
        # [x_batch, nz]
        if graph_latent_vec is None:
            graph_latent_vec, tree_latent_vec = self.encode(*input_batch)
        g_z_mean = self.g_mean(graph_latent_vec)
        g_z_log_var = -torch.abs(self.g_var(graph_latent_vec))
        t_z_mean = self.t_mean(tree_latent_vec)
        t_z_log_var = -torch.abs(self.t_var(tree_latent_vec))
        z_mean = torch.cat([g_z_mean, t_z_mean], dim=-1)
        z_log_var = torch.cat([g_z_log_var, t_z_log_var], dim=-1)

        x_batch, nz = z_mean.size()
        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * np.log(2 * np.pi) - 0.5 * (1 + z_log_var).sum(-1)).mean()
        # [z_batch, 1, nz]
        z_samples = self.reparameterize(z_mean, z_log_var, nsamples=1)
        # [1, x_batch, nz]
        mu, logvar = z_mean.unsqueeze(0), z_log_var.unsqueeze(0)
        var = logvar.exp()
        # (z_batch, x_batch, nz)
        dev = z_samples - mu
        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * np.log(2 * np.pi) + logvar.sum(-1))
        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - np.log(x_batch)
        return (neg_entropy - log_qz.mean(-1)).item()

    def eval_inference_dist(self, input_batch, z_vec, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z_vec.size(2)

        if not param:
            graph_vectors, tree_vectors = self.encode(*input_batch)
            g_mu, g_logvar = self.g_mean(graph_vectors), -torch.abs(self.g_var(graph_vectors))
            t_mu, t_logvar = self.t_mean(tree_vectors), -torch.abs(self.t_var(tree_vectors))
            mu = torch.cat([g_mu, t_mu], dim=-1)
            logvar = torch.cat([g_logvar, t_logvar], dim=-1)
        else:
            mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z_vec - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * np.log(2 * np.pi) + logvar.sum(-1))

        return log_density

    def nll_iw(self, input_batch, nsamples, ns=100, graph_vectors=None, tree_vectors=None):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10

        tree_batch, graph_encoder_input, tree_encoder_input = input_batch

        tmp = []
        batch_size = len(tree_batch)
        if graph_vectors is None:
            graph_vectors, tree_vectors = self.encode(graph_encoder_input, tree_encoder_input)
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]
            (z_vec, graph_z_vec, tree_z_vec), (entropy, log_pz) = self.rsample(graph_vectors, tree_vectors, ns)

            # [batch, ns], log p(x,z)
            graph_z_vec_reshaped = graph_z_vec.reshape(batch_size * ns, self.latent_dim)
            tree_z_vec_reshaped = tree_z_vec.reshape(batch_size * ns, self.latent_dim)
            rep_tree_batch = []
            for rna in tree_batch:
                for _ in range(ns):
                    rep_tree_batch.append(copy.deepcopy(rna))
            ret_dict = self.decoder(rep_tree_batch, tree_z_vec_reshaped, graph_z_vec_reshaped)
            recon_log_prob = - ret_dict['batch_nuc_pred_loss'].reshape(batch_size, ns) - \
                             ret_dict['batch_hpn_pred_loss'].reshape(batch_size, ns) - \
                             ret_dict['batch_stop_pred_loss'].reshape(batch_size, ns)
            log_comp_ll = log_pz[:, :, 0] + recon_log_prob

            # log q(z|x)
            log_infer_ll = self.eval_inference_dist(
                input_batch, z_vec,
                param=(torch.cat([self.g_mean(graph_vectors), self.t_mean(tree_vectors)], dim=-1),
                       -torch.abs(torch.cat([self.g_var(graph_vectors), self.t_var(tree_vectors)], dim=-1))))

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - np.log(nsamples)

        return -ll_iw

    def forward(self, input):
        tree_batch, graph_encoder_input, tree_encoder_input = input
        graph_vectors, tree_vectors = self.encode(graph_encoder_input, tree_encoder_input)

        (_, graph_z_vecs, tree_z_vecs), (entropy, log_pz) = self.rsample(graph_vectors, tree_vectors)
        graph_z_vecs = graph_z_vecs[:, 0, :]
        tree_z_vecs = tree_z_vecs[:, 0, :]

        # when training the decoders it is imperative to use teacher forcing,
        # i.e. feeding the ground truth topology and subgraph conformation
        ret_dict = self.decoder(tree_batch, tree_z_vecs, graph_z_vecs)

        ret_dict['entropy_loss'] = -entropy.mean()
        ret_dict['prior_loss'] = -log_pz.mean()

        return ret_dict


if __name__ == "__main__":
    test_model = JunctionTreeVAE(30, 30, 10, 20)
    from lib.data_utils import JunctionTreeFolder

    loader = JunctionTreeFolder('../data/rna_jt', 32, num_workers=0)
    for batch in loader:
        tree_batch, graph_encoder_input, tree_encoder_input = batch
        all_loss, kl_loss, all_acc = test_model(batch, 1.)
        print(all_acc)
        print('=' * 30)
