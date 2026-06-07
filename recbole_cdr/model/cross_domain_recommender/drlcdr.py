r"""
DRLCDR
################################################
Reference:
    Disentangled Representation Learning for Cross-Domain Recommendation with Conditional Variational Graph Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.utils import InputType

from .gnn_modules.gcn import build_domain_adjacency, kld_gauss
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
        self.condi_weight = condi_weight
        self.warmup_epochs = config['warmup_epochs'] if config['warmup_epochs'] is not None else 10
        self.train_epoch = 0

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
        self.source_user_emb = nn.Embedding(N_u, dim)
        self.target_user_emb = nn.Embedding(N_u, dim)
        self.source_item_emb = nn.Embedding(self.source_num_items, dim)
        self.target_item_emb = nn.Embedding(self.target_num_items, dim)
        self.source_user_emb_share = nn.Embedding(N_u, dim)
        self.target_user_emb_share = nn.Embedding(N_u, dim)

        # Build sparse UV / VU matrices
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

        self._u_idx = torch.arange(N_u, device=self.device)
        self._si_idx = torch.arange(self.source_num_items, device=self.device)
        self._ti_idx = torch.arange(self.target_num_items, device=self.device)

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
        tgt_i = self.target_item_emb(self._ti_idx)
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

        condi_src_u, condi_src_kld = self._reparameters(cs_mean, cs_sigma, self.condi_weight)
        condi_tgt_u, condi_tgt_kld = self._reparameters(ct_mean, ct_sigma, self.condi_weight)

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

        src_score = (src_u[src_u_idx] * src_i[src_i_idx]).sum(dim=-1)
        tgt_score = (tgt_u[tgt_u_idx] * tgt_i[tgt_i_idx]).sum(dim=-1)

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
