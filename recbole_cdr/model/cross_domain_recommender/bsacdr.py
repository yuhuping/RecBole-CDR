# -*- coding: utf-8 -*-

r"""Bounded Safe Adaptive Cross-Domain Recommendation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.cross_domain_recommender.sacdr import _build_activity
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class BSACDR(CrossDomainRecommender):
    r"""Transfer unit-sphere private preferences under margin supervision."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.private_size = config['private_size']
        self.gate_hidden_size = config['gate_hidden_size']
        self.local_scale = config['local_scale']
        self.transfer_scale = config['transfer_scale']
        self.source_loss_weight = config['source_loss_weight']
        self.anchor_loss_weight = config['anchor_loss_weight']
        self.local_loss_weight = config['local_loss_weight']
        self.safety_weight = config['safety_weight']
        self.gate_supervision_weight = config[
            'gate_supervision_weight'
        ]
        self.gate_sparsity_weight = config['gate_sparsity_weight']
        self.embedding_reg_weight = config['embedding_reg_weight']
        self.gate_initial_bias = config['gate_initial_bias']

        self.shared_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.private_user_embedding = nn.ModuleList([
            nn.Embedding(self.total_num_users, self.private_size),
            nn.Embedding(self.total_num_users, self.private_size),
        ])
        self.item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.local_projection = nn.ModuleList([
            nn.Linear(self.private_size, self.embedding_size, bias=False),
            nn.Linear(self.private_size, self.embedding_size, bias=False),
        ])
        self.cross_projection = nn.ModuleList([
            nn.Linear(self.private_size, self.embedding_size, bias=False),
            nn.Linear(self.private_size, self.embedding_size, bias=False),
        ])
        self.reliability_gate = nn.ModuleList([
            self._make_gate(),
            self._make_gate(),
        ])
        self.bpr_loss = BPRLoss()
        self.register_buffer('user_activity', _build_activity(dataset))

        self.apply(xavier_normal_initialization)
        for gate in self.reliability_gate:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, self.gate_initial_bias)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def _make_gate(self):
        return nn.Sequential(
            nn.Linear(6, self.gate_hidden_size),
            nn.GELU(),
            nn.Linear(self.gate_hidden_size, 1),
        )

    @staticmethod
    def _bounded_orthogonal(raw, anchor):
        coefficient = (
            raw * anchor
        ).sum(dim=1, keepdim=True) / anchor.pow(2).sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        orthogonal = raw - coefficient * anchor
        anchor_norm = anchor.norm(dim=1, keepdim=True).detach()
        return F.normalize(orthogonal, dim=1) * anchor_norm

    def _user_state(self, users, domain):
        other_domain = 1 - domain
        shared = self.shared_user_embedding(users)
        local = self.local_projection[domain](
            self.private_user_embedding[domain](users)
        )
        cross = self.cross_projection[domain](
            self.private_user_embedding[other_domain](users).detach()
        )
        local = self._bounded_orthogonal(local, shared)
        cross = self._bounded_orthogonal(cross, shared)

        agreement = F.cosine_similarity(
            local, cross, dim=1, eps=1e-8
        ).unsqueeze(1)
        local_activity = self.user_activity[users, domain:domain + 1]
        other_activity = self.user_activity[
            users, other_domain:other_domain + 1
        ]
        features = torch.cat((
            agreement,
            local_activity,
            other_activity,
            other_activity - local_activity,
            local_activity * other_activity,
            torch.abs(other_activity - local_activity),
        ), dim=1)
        gate = torch.sigmoid(
            self.reliability_gate[domain](features)
        )
        anchor = shared
        local_condition = anchor + self.local_scale * local
        condition = (
            local_condition + self.transfer_scale * gate * cross
        )
        return anchor, local, cross, gate, local_condition, condition

    @staticmethod
    def _score(condition, item_embedding):
        return (condition * item_embedding).sum(dim=1)

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        (
            anchor,
            local,
            cross,
            gate,
            local_condition,
            condition,
        ) = self._user_state(users, domain)
        positive_embedding = self.item_embedding(positives)
        negative_embedding = self.item_embedding(negatives)
        item_difference = positive_embedding - negative_embedding

        anchor_margin = (
            anchor * item_difference
        ).sum(dim=1)
        local_margin = (
            local_condition * item_difference
        ).sum(dim=1)
        full_margin = (
            condition * item_difference
        ).sum(dim=1)
        cross_utility = (
            cross.detach() * item_difference.detach()
        ).sum(dim=1, keepdim=True)
        reliability_label = cross_utility.gt(0).float()

        ranking = F.softplus(-full_margin).mean()
        anchor_ranking = F.softplus(-anchor_margin).mean()
        local_ranking = F.softplus(-local_margin).mean()
        safety = F.relu(
            anchor_margin.detach() - full_margin
        ).mean()
        gate_supervision = F.binary_cross_entropy(
            gate, reliability_label
        )
        regularization = (
            anchor.pow(2).mean()
            + positive_embedding.pow(2).mean()
            + negative_embedding.pow(2).mean()
        )
        return (
            ranking
            + self.anchor_loss_weight * anchor_ranking
            + self.local_loss_weight * local_ranking
            + self.safety_weight * safety
            + self.gate_supervision_weight * gate_supervision
            + self.gate_sparsity_weight * gate.mean()
            + self.embedding_reg_weight * regularization
        )

    def calculate_loss(self, interaction):
        source_loss = self._domain_loss(interaction, 0)
        target_loss = self._domain_loss(interaction, 1)
        return (
            self.source_loss_weight * source_loss
            + (1 - self.source_loss_weight) * target_loss
        )

    def predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        condition = self._user_state(users, 1)[-1]
        return self._score(condition, self.item_embedding(items))

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        condition = self._user_state(users, 1)[-1]
        items = self.item_embedding.weight[:self.target_num_items]
        return torch.matmul(condition, items.t()).reshape(-1)
