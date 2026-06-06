r"""
DRLCDR
################################################
Reference:
    Disentangled Representation Learning for Cross-Domain Recommendation with Conditional Variational Graph Networks.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.utils import InputType

from .gnn_modules.gcn import normalize, sparse_mx_to_torch_sparse_tensor, kld_gauss
from .gnn_modules.single_vbge import SingleVBGE
from .gnn_modules.conditional_vbge import ConditionalVBGE


class DRLCDR(CrossDomainRecommender):
    r"""DRLCDR uses conditional variational GNNs to disentangle and transfer
    cross-domain user representations for intra-domain recommendation.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DRLCDR, self).__init__(config, dataset)

        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        self.device = config['device']
        dim = config['embedding_size']
        hidden = config['hidden_size'] if 'hidden_size' in config else dim
        n_layers = config['n_layers'] if 'n_layers' in config else 2
        dropout = config['dropout'] if 'dropout' in config else 0.3
        leakey = config['leakey'] if 'leakey' in config else 0.1
        condi_weight = config['condi_weight'] if 'condi_weight' in config else 0.3
        condi_non_weight = config['condi_non_weight'] if 'condi_non_weight' in config else 10.0
        condi_condi_weight = config['condi_condi_weight'] if 'condi_condi_weight' in config else 10.0

        self.condi_non_weight = condi_non_weight
        self.condi_condi_weight = condi_condi_weight

        opt = {
            "feature_dim": dim,
            "hidden_dim": hidden,
            "GNN": n_layers,
            "dropout": dropout,
            "leakey": leakey,
            "condi_weight": condi_weight,
            "isConditional": config['isConditional'] if 'isConditional' in config else True,
            "isCondi_norm": config['isCondi_norm'] if 'isCondi_norm' in config else False,
            "cuda": config['device'].type == 'cuda',
        }

        # GNN modules
        self.source_specific_GNN = SingleVBGE(opt)
        self.source_sp_GNN = SingleVBGE(opt)
        self.target_specific_GNN = SingleVBGE(opt)
        self.target_sp_GNN = SingleVBGE(opt)
        self.conditional_GNN = ConditionalVBGE(opt)

        # Embeddings
        N_u = self.total_num_users
        N_i = self.total_num_items
        self.source_user_emb = nn.Embedding(N_u, dim)
        self.target_user_emb = nn.Embedding(N_u, dim)
        self.source_item_emb = nn.Embedding(N_i, dim)
        self.target_item_emb = nn.Embedding(N_i, dim)
        self.source_user_emb_share = nn.Embedding(N_u, dim)
        self.target_user_emb_share = nn.Embedding(N_u, dim)

        # Build sparse UV / VU matrices
        src_mat = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        tgt_mat = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)

        self.source_UV = sparse_mx_to_torch_sparse_tensor(
            normalize(sp.coo_matrix(src_mat, shape=(N_u, N_i)))).to(self.device)
        self.source_VU = sparse_mx_to_torch_sparse_tensor(
            normalize(sp.coo_matrix(src_mat.T, shape=(N_i, N_u)))).to(self.device)
        self.target_UV = sparse_mx_to_torch_sparse_tensor(
            normalize(sp.coo_matrix(tgt_mat, shape=(N_u, N_i)))).to(self.device)
        self.target_VU = sparse_mx_to_torch_sparse_tensor(
            normalize(sp.coo_matrix(tgt_mat.T, shape=(N_i, N_u)))).to(self.device)

        self._u_idx = torch.arange(N_u, device=self.device)
        self._si_idx = torch.arange(N_i, device=self.device)

        self._target_user_cache = None
        self._target_item_cache = None
        self.other_parameter_name = ['_target_user_cache', '_target_item_cache']

        nn.init.xavier_normal_(self.source_user_emb.weight)
        nn.init.xavier_normal_(self.target_user_emb.weight)
        nn.init.xavier_normal_(self.source_item_emb.weight)
        nn.init.xavier_normal_(self.target_item_emb.weight)
        nn.init.xavier_normal_(self.source_user_emb_share.weight)
        nn.init.xavier_normal_(self.target_user_emb_share.weight)

    def _reparameters(self, mean, logstd, condi_weight):
        logstd = torch.clamp(logstd, -10, 10)
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        if self.training:
            noise = torch.randn(mean.size(0), mean.size(1), device=mean.device)
            sampled_z = noise * sigma + mean
        else:
            sampled_z = mean
        kld = kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, condi_weight * kld

    def _forward(self):
        src_u = self.source_user_emb(self._u_idx)
        tgt_u = self.target_user_emb(self._u_idx)
        src_i = self.source_item_emb(self._si_idx)
        tgt_i = self.target_item_emb(self._si_idx)
        src_u_share = self.source_user_emb_share(self._u_idx)
        tgt_u_share = self.target_user_emb_share(self._u_idx)

        src_sp_u, src_sp_i = self.source_specific_GNN(src_u, src_i, self.source_UV, self.source_VU)
        tgt_sp_u, tgt_sp_i = self.target_specific_GNN(tgt_u, tgt_i, self.target_UV, self.target_VU)

        src_mean, src_sigma = self.source_sp_GNN.forward_user_share(src_u, self.source_UV, self.source_VU)
        tgt_mean, tgt_sigma = self.target_sp_GNN.forward_user_share(tgt_u, self.target_UV, self.target_VU)

        cs_mean, cs_sigma, ct_mean, ct_sigma = self.conditional_GNN(
            src_u_share, tgt_u_share,
            self.source_UV, self.source_VU,
            self.target_UV, self.target_VU,
            src_sp_u, tgt_sp_u
        )

        condi_src_u, condi_src_kld = self._reparameters(cs_mean, cs_sigma, 1.0)
        condi_tgt_u, condi_tgt_kld = self._reparameters(ct_mean, ct_sigma, 1.0)

        src_condi_kld = kld_gauss(cs_mean, cs_sigma, src_mean, src_sigma)
        tgt_condi_kld = kld_gauss(ct_mean, ct_sigma, tgt_mean, tgt_sigma)
        src_tgt_kld = -kld_gauss(cs_mean, cs_sigma, ct_mean, ct_sigma)

        self.kld_loss = (condi_src_kld + condi_tgt_kld +
                         self.condi_non_weight * src_condi_kld +
                         self.condi_non_weight * tgt_condi_kld -
                         self.condi_condi_weight * src_tgt_kld)

        src_learn_u = condi_src_u + src_sp_u
        tgt_learn_u = condi_tgt_u + tgt_sp_u

        return src_learn_u, src_sp_i, tgt_learn_u, tgt_sp_i

    def calculate_loss(self, interaction):
        self._target_user_cache = None
        self._target_item_cache = None

        src_u_idx = interaction[self.SOURCE_USER_ID]
        src_i_idx = interaction[self.SOURCE_ITEM_ID]
        src_label = interaction[self.SOURCE_LABEL]
        tgt_u_idx = interaction[self.TARGET_USER_ID]
        tgt_i_idx = interaction[self.TARGET_ITEM_ID]
        tgt_label = interaction[self.TARGET_LABEL]

        src_u, src_i, tgt_u, tgt_i = self._forward()

        src_score = (src_u[src_u_idx] * src_i[src_i_idx]).sum(dim=-1)
        tgt_score = (tgt_u[tgt_u_idx] * tgt_i[tgt_i_idx]).sum(dim=-1)

        bce = nn.BCEWithLogitsLoss()
        loss = (bce(src_score, src_label.float()) +
                bce(tgt_score, tgt_label.float()) +
                self.kld_loss)
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
