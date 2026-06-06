# -*- coding: utf-8 -*-

r"""
UniCDR
################################################
Reference:
    Jiangxia Cao et al. "Towards Universal Cross-Domain Recommendation." in WSDM 2023.

Port notes:
    This port supports the ``dual-user-intra`` task: two domains with a shared user embedding
    for overlapping users.  User history sequences are pre-built from the dataset's interaction
    data at init time so that no custom data pipeline is required.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class BehaviorAggregator(nn.Module):
    def __init__(self, embedding_dim, aggregator, lambda_a, dropout_rate):
        super().__init__()
        self.aggregator = aggregator
        self.lambda_a = lambda_a

        self.W_agg = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if aggregator == "user_attention":
            self.W_att = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
            )
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        else:
            self.dropout = None

    def forward(self, id_emb, sequence_emb, score=None):
        # id_emb: [B, D], sequence_emb: [B, L, D], score: [B, L] (optional)
        if self.aggregator == "mean":
            out = self._mean_pooling(sequence_emb)
        elif self.aggregator == "user_attention":
            out = self._user_attention_pooling(id_emb, sequence_emb)
        elif self.aggregator == "item_similarity":
            out = self._item_similarity_pooling(sequence_emb, score)
        else:
            out = id_emb
        return self.lambda_a * id_emb + (1 - self.lambda_a) * out

    def _user_attention_pooling(self, id_emb, sequence_emb):
        key = self.W_att(sequence_emb)                                    # [B, L, D]
        mask = sequence_emb.sum(dim=-1) == 0                              # [B, L]
        attention = torch.bmm(key, id_emb.unsqueeze(-1)).squeeze(-1)      # [B, L]
        attention = self._masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        out = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)  # [B, D]
        return self.W_agg(out)

    def _mean_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1e-12)
        return self.W_agg(mean)

    def _item_similarity_pooling(self, sequence_emb, score):
        if score.dim() != 2:
            score = score.view(score.size(0), -1)
        score = F.softmax(score, dim=-1).unsqueeze(-1)
        out = (score * sequence_emb).sum(dim=1)
        return self.W_agg(out)

    def _masked_softmax(self, X, mask):
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1e-12)


class UniCDR(CrossDomainRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        dim = config['latent_dim']
        aggregator = config['aggregator']
        lambda_a = config['lambda_a']
        dropout = config['dropout']
        self.lambda_loss = config['lambda_loss']
        self.maxlen = config['maxlen']

        # Domain-specific embeddings (item 0 = padding)
        self.source_user_emb = nn.Embedding(self.total_num_users, dim)
        self.target_user_emb = nn.Embedding(self.target_num_users, dim)
        self.source_item_emb = nn.Embedding(self.total_num_items, dim, padding_idx=0)
        self.target_item_emb = nn.Embedding(self.target_num_items, dim, padding_idx=0)

        # Shared user embedding: covers both global source IDs and local target IDs
        # (overlapping users have IDs 0..overlapped_num_users-1 in both spaces)
        self.share_user_emb = nn.Embedding(self.total_num_users, dim)

        # Behavior aggregators: source-specific, target-specific, global (shared)
        self.source_agg = BehaviorAggregator(dim, aggregator, lambda_a, dropout)
        self.target_agg = BehaviorAggregator(dim, aggregator, lambda_a, dropout)
        self.global_agg = BehaviorAggregator(dim, aggregator, lambda_a, dropout)

        # Discriminators for MI maximization between specific and shared representations
        self.source_dis = nn.Bilinear(dim, dim, 1)
        self.target_dis = nn.Bilinear(dim, dim, 1)

        self.criterion = nn.BCEWithLogitsLoss()

        # Pre-build user history caches from dataset interaction records.
        # Source history is indexed by global user IDs (total_num_users).
        # Target history is indexed by local target user IDs (target_num_users).
        src_hist, _, src_len = dataset.source_domain_dataset.get_history_matrix(
            self.total_num_users, self.total_num_items, 'user'
        )
        tgt_hist, _, tgt_len = dataset.target_domain_dataset.get_history_matrix(
            self.target_num_users, self.target_num_items, 'user'
        )
        self.register_buffer('source_hist', src_hist)   # [total_num_users, max_src_len]
        self.register_buffer('source_hlen', src_len)    # [total_num_users]
        self.register_buffer('target_hist', tgt_hist)   # [target_num_users, max_tgt_len]
        self.register_buffer('target_hlen', tgt_len)    # [target_num_users]

        self.apply(xavier_normal_initialization)
        self._critic_loss = 0.0

    def set_phase(self, phase):
        self.phase = phase

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ctx_emb(self, user_ids, hist_mat, item_emb):
        """Return context item embeddings [B, maxlen, D], vectorized."""
        take = min(hist_mat.size(1), self.maxlen)
        ctx = hist_mat[user_ids, :take]                              # [B, take]
        if take < self.maxlen:
            pad = ctx.new_zeros(ctx.size(0), self.maxlen - take)
            ctx = torch.cat([ctx, pad], dim=1)                       # [B, maxlen]
        return item_emb(ctx)                                         # [B, maxlen, D]

    def _global_ctx_emb(self, user_ids, domain):
        """Cross-domain context: concatenate source + target history embeddings [B, 2*maxlen, D]."""
        if domain == 'source':
            # For non-overlapping source users (id >= target_num_users), target history is empty.
            valid = (user_ids < self.target_num_users)
            clamped = user_ids.clamp(max=self.target_num_users - 1)
            src_emb = self._ctx_emb(user_ids, self.source_hist, self.source_item_emb)
            tgt_emb = self._ctx_emb(clamped, self.target_hist, self.target_item_emb)
            tgt_emb = tgt_emb * valid.float().unsqueeze(1).unsqueeze(2)
        else:
            # For non-overlapping target users (id >= total_num_users after clamp = 0), src is empty.
            valid = (user_ids < self.total_num_users)
            clamped = user_ids.clamp(max=self.total_num_users - 1)
            tgt_emb = self._ctx_emb(user_ids, self.target_hist, self.target_item_emb)
            src_emb = self._ctx_emb(clamped, self.source_hist, self.source_item_emb)
            src_emb = src_emb * valid.float().unsqueeze(1).unsqueeze(2)
        return torch.cat([src_emb, tgt_emb], dim=1)                 # [B, 2*maxlen, D]

    def _forward_user(self, domain, user_ids):
        """
        Compute user representation = domain-specific + shared global.
        domain: 'source' (global user IDs) or 'target' (local target user IDs).
        Returns user embeddings [B, D] and updates self._critic_loss.
        """
        if domain == 'source':
            id_emb = self.source_user_emb(user_ids)
            ctx_emb = self._ctx_emb(user_ids, self.source_hist, self.source_item_emb)
            specific = self.source_agg(id_emb, ctx_emb)
            dis = self.source_dis
            global_id_emb = self.share_user_emb(user_ids)
        else:
            id_emb = self.target_user_emb(user_ids)
            ctx_emb = self._ctx_emb(user_ids, self.target_hist, self.target_item_emb)
            specific = self.target_agg(id_emb, ctx_emb)
            dis = self.target_dis
            # Overlapping target users (id < overlapped_num_users) share global ID space.
            global_id_emb = self.share_user_emb(user_ids.clamp(max=self.total_num_users - 1))

        global_ctx = self._global_ctx_emb(user_ids, domain)         # [B, 2*maxlen, D]
        shared = self.global_agg(global_id_emb, global_ctx)

        # MI maximization: discriminator tries to distinguish specific↔shared (pos) from
        # shuffled↔shared (neg).
        if self.training:
            B = user_ids.size(0)
            if B > 1:
                shift = torch.randint(1, B, (1,), device=user_ids.device).item()
                neg_idx = (torch.arange(B, device=user_ids.device) + shift) % B
                pos = dis(specific, shared).view(-1)
                neg = dis(specific[neg_idx], shared).view(-1)
                pos_lbl = torch.ones_like(pos)
                neg_lbl = torch.zeros_like(neg)
                self._critic_loss = self._critic_loss + self.criterion(pos, pos_lbl) + self.criterion(neg, neg_lbl)

        return specific + shared

    # ------------------------------------------------------------------
    # RecBole interface
    # ------------------------------------------------------------------

    def calculate_loss(self, interaction):
        self._critic_loss = torch.tensor(0.0, device=self.device)

        loss_rec = torch.tensor(0.0, device=self.device)

        if self.phase in ('SOURCE', 'BOTH'):
            user = interaction[self.SOURCE_USER_ID]
            pos_item = interaction[self.SOURCE_ITEM_ID]
            neg_item = interaction[self.SOURCE_NEG_ITEM_ID]

            user_e = self._forward_user('source', user)
            pos_e = self.source_item_emb(pos_item)
            neg_e = self.source_item_emb(neg_item)

            pos_score = (user_e * pos_e).sum(-1)
            neg_score = (user_e * neg_e).sum(-1)
            loss_rec = loss_rec + self.criterion(pos_score, torch.ones_like(pos_score)) + \
                                  self.criterion(neg_score, torch.zeros_like(neg_score))

        if self.phase in ('TARGET', 'BOTH'):
            user = interaction[self.TARGET_USER_ID]
            pos_item = interaction[self.TARGET_ITEM_ID]
            neg_item = interaction[self.TARGET_NEG_ITEM_ID]

            user_e = self._forward_user('target', user)
            pos_e = self.target_item_emb(pos_item)
            neg_e = self.target_item_emb(neg_item)

            pos_score = (user_e * pos_e).sum(-1)
            neg_score = (user_e * neg_e).sum(-1)
            loss_rec = loss_rec + self.criterion(pos_score, torch.ones_like(pos_score)) + \
                                  self.criterion(neg_score, torch.zeros_like(neg_score))

        lam = self.lambda_loss
        return lam * loss_rec + (1 - lam) * self._critic_loss

    def predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            item = interaction[self.SOURCE_ITEM_ID]
            user_e = self._forward_user('source', user)
            item_e = self.source_item_emb(item)
        else:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            user_e = self._forward_user('target', user)
            item_e = self.target_item_emb(item)
        return torch.sigmoid((user_e * item_e).sum(-1))

    def full_sort_predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            user_e = self._forward_user('source', user)
            # Global item layout: [0..overlap) | [overlap..target_num) | [target_num..total_num)
            # Source items = overlapping + source-only
            aie = self.source_item_emb.weight
            all_item_e = torch.cat([
                aie[:self.overlapped_num_items],    # overlapping items
                aie[self.target_num_items:],         # source-only items
            ], dim=0)                                # → [source_num_items, dim]
        else:
            user = interaction[self.TARGET_USER_ID]
            user_e = self._forward_user('target', user)
            # Target items = overlapping + target-only = [0..target_num_items)
            all_item_e = self.target_item_emb.weight  # [target_num_items, dim]
        score = torch.matmul(user_e, all_item_e.T)
        return torch.sigmoid(score).view(-1)
