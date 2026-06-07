r"""
DisenCDR
################################################
Reference:
    Caiyuan Zheng et al. "DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation." in SIGIR 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.utils import InputType

from .gnn_modules.gcn import build_domain_adjacency, kld_gauss
from .gnn_modules.single_vbge import SingleVBGE
from .gnn_modules.cross_vbge import CrossVBGE


class DisenCDR(CrossDomainRecommender):
    r"""DisenCDR disentangles domain-shared and domain-specific user representations
    via variational graph neural networks for intra-domain cross-domain recommendation.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DisenCDR, self).__init__(config, dataset)

        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        self.device = config['device']
        dim = config['embedding_size']
        hidden = config['hidden_size'] if 'hidden_size' in config else dim
        n_layers = config['n_layers'] if 'n_layers' in config else 2
        dropout = config['dropout'] if 'dropout' in config else 0.3
        leakey = config['leakey'] if 'leakey' in config else 0.1
        beta = config['beta'] if 'beta' in config else 0.9
        rate = config['rate'] if 'rate' in config else 0.5

        self.beta = beta
        self.warmup_epochs = config['warmup_epochs'] if config['warmup_epochs'] is not None else 10
        self.train_epoch = 0

        opt = {
            "feature_dim": dim,
            "hidden_dim": hidden,
            "GNN": n_layers,
            "dropout": dropout,
            "leakey": leakey,
            "rate": rate,
            "cuda": config['device'].type == 'cuda',
        }

        # GNN modules
        self.source_specific_GNN = SingleVBGE(opt)
        self.source_share_GNN = SingleVBGE(opt)
        self.target_specific_GNN = SingleVBGE(opt)
        self.target_share_GNN = SingleVBGE(opt)
        self.share_GNN = CrossVBGE(opt)

        # Users share a global ID space; each GNN keeps its own local item space.
        N_u = self.total_num_users
        self.source_user_emb = nn.Embedding(N_u, dim)
        self.target_user_emb = nn.Embedding(N_u, dim)
        self.source_item_emb = nn.Embedding(self.source_num_items, dim)
        self.target_item_emb = nn.Embedding(self.target_num_items, dim)
        self.source_user_emb_share = nn.Embedding(N_u, dim)
        self.target_user_emb_share = nn.Embedding(N_u, dim)

        self.share_mean = nn.Linear(dim * 2, dim)
        self.share_sigma = nn.Linear(dim * 2, dim)

        # Build sparse UV / VU adjacency matrices
        self.source_UV, self.source_VU = build_domain_adjacency(
            dataset, 'source', N_u, self.source_num_items,
            self.target_num_items, self.overlapped_num_items,
        )
        self.target_UV, self.target_VU = build_domain_adjacency(
            dataset, 'target', N_u, self.source_num_items,
            self.target_num_items, self.overlapped_num_items,
        )
        self.source_UV = self.source_UV.to(self.device)
        self.source_VU = self.source_VU.to(self.device)
        self.target_UV = self.target_UV.to(self.device)
        self.target_VU = self.target_VU.to(self.device)

        # User/item index tensors
        self._u_idx = torch.arange(N_u, device=self.device)
        self._si_idx = torch.arange(self.source_num_items, device=self.device)
        self._ti_idx = torch.arange(self.target_num_items, device=self.device)

        # Cache for evaluation
        self._target_user_cache = None
        self._target_item_cache = None
        self.other_parameter_name = ['_target_user_cache', '_target_item_cache']

        nn.init.xavier_normal_(self.source_user_emb.weight)
        nn.init.xavier_normal_(self.target_user_emb.weight)
        nn.init.xavier_normal_(self.source_item_emb.weight)
        nn.init.xavier_normal_(self.target_item_emb.weight)
        nn.init.xavier_normal_(self.source_user_emb_share.weight)
        nn.init.xavier_normal_(self.target_user_emb_share.weight)

    def _reparameters(self, mean, logstd):
        logstd = torch.clamp(logstd, -10, 10)
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        if self.training:
            noise = torch.randn(mean.size(0), mean.size(1), device=mean.device)
            sampled_z = noise * sigma + mean
        else:
            sampled_z = mean
        kld = kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, (1 - self.beta) * kld

    def _forward(self):
        src_u = self.source_user_emb(self._u_idx)
        tgt_u = self.target_user_emb(self._u_idx)
        src_i = self.source_item_emb(self._si_idx)
        tgt_i = self.target_item_emb(self._ti_idx)
        src_u_share = self.source_user_emb_share(self._u_idx)
        tgt_u_share = self.target_user_emb_share(self._u_idx)

        src_sp_u, src_sp_i = self.source_specific_GNN(src_u, src_i, self.source_UV, self.source_VU)
        tgt_sp_u, tgt_sp_i = self.target_specific_GNN(tgt_u, tgt_i, self.target_UV, self.target_VU)

        src_mean, src_sigma = self.source_share_GNN.forward_user_share(src_u, self.source_UV, self.source_VU)
        tgt_mean, tgt_sigma = self.target_share_GNN.forward_user_share(tgt_u, self.target_UV, self.target_VU)

        mean, sigma = self.share_GNN(src_u_share, tgt_u_share,
                                     self.source_UV, self.source_VU,
                                     self.target_UV, self.target_VU)

        user_share, share_kld = self._reparameters(mean, sigma)

        src_share_kld = kld_gauss(mean, sigma, src_mean, src_sigma)
        tgt_share_kld = kld_gauss(mean, sigma, tgt_mean, tgt_sigma)
        self.kld_loss = share_kld + self.beta * src_share_kld + self.beta * tgt_share_kld

        src_learn_u = user_share + src_sp_u
        tgt_learn_u = user_share + tgt_sp_u

        return src_learn_u, src_sp_i, tgt_learn_u, tgt_sp_i

    def _warmup_forward(self):
        src_u = self.source_user_emb(self._u_idx)
        tgt_u = self.target_user_emb(self._u_idx)
        src_i = self.source_item_emb(self._si_idx)
        tgt_i = self.target_item_emb(self._ti_idx)

        src_sp_u, src_sp_i = self.source_specific_GNN(src_u, src_i, self.source_UV, self.source_VU)
        tgt_sp_u, tgt_sp_i = self.target_specific_GNN(tgt_u, tgt_i, self.target_UV, self.target_VU)
        self.kld_loss = 0
        return src_sp_u, src_sp_i, tgt_sp_u, tgt_sp_i

    def set_train_epoch(self, epoch_idx):
        self.train_epoch = epoch_idx

    def _source_item_to_local(self, item):
        return torch.where(
            item < self.overlapped_num_items,
            item,
            item - (self.target_num_items - self.overlapped_num_items),
        )

    def calculate_loss(self, interaction):
        self._target_user_cache = None
        self._target_item_cache = None

        src_u_idx = interaction[self.SOURCE_USER_ID]
        src_i_idx = self._source_item_to_local(interaction[self.SOURCE_ITEM_ID])
        src_label = interaction[self.SOURCE_LABEL]
        tgt_u_idx = interaction[self.TARGET_USER_ID]
        tgt_i_idx = interaction[self.TARGET_ITEM_ID]
        tgt_label = interaction[self.TARGET_LABEL]

        if self.train_epoch < self.warmup_epochs:
            src_u, src_i, tgt_u, tgt_i = self._warmup_forward()
        else:
            src_u, src_i, tgt_u, tgt_i = self._forward()

        src_user_f = src_u[src_u_idx]
        src_item_f = src_i[src_i_idx]
        tgt_user_f = tgt_u[tgt_u_idx]
        tgt_item_f = tgt_i[tgt_i_idx]

        src_score = (src_user_f * src_item_f).sum(dim=-1)
        tgt_score = (tgt_user_f * tgt_item_f).sum(dim=-1)

        bce = nn.BCEWithLogitsLoss()
        specific_kld = (
            self.source_specific_GNN.encoder[-1].kld_loss
            + self.target_specific_GNN.encoder[-1].kld_loss
        )
        loss = (2 * bce(src_score, src_label.float()) +
                2 * bce(tgt_score, tgt_label.float()) +
                specific_kld + self.kld_loss)

        return loss

    def _get_eval_embeddings(self):
        if self._target_user_cache is None:
            with torch.no_grad():
                _, _, tgt_u, tgt_i = self._forward()
            self._target_user_cache = tgt_u
            self._target_item_cache = tgt_i
        return self._target_user_cache, self._target_item_cache

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        tgt_u, tgt_i = self._get_eval_embeddings()
        return (tgt_u[user] * tgt_i[item]).sum(dim=-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        tgt_u, tgt_i = self._get_eval_embeddings()
        user_e = tgt_u[user]
        all_item_e = tgt_i[:self.target_num_items]
        return torch.matmul(user_e, all_item_e.t()).view(-1)
