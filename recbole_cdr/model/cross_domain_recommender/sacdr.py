# -*- coding: utf-8 -*-

r"""Safe Adaptive Cross-Domain Recommendation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


def _training_interactions(domain_dataset):
    interactions = domain_dataset.inter_feat
    file_sizes = getattr(domain_dataset, 'file_size_list', None)
    if file_sizes:
        interactions = interactions[:file_sizes[0]]
    return interactions


def _build_activity(dataset):
    activities = []
    for domain_dataset in (
        dataset.source_domain_dataset,
        dataset.target_domain_dataset,
    ):
        interactions = _training_interactions(domain_dataset)
        users = interactions[domain_dataset.uid_field]
        counts = torch.bincount(
            users, minlength=dataset.num_total_user
        ).float()
        activities.append(torch.log1p(counts))
    activity = torch.stack(activities, dim=1)
    return activity / activity.max(dim=0, keepdim=True).values.clamp_min(1)


class SACDR(CrossDomainRecommender):
    r"""Decompose user preference and transfer only reliable residuals.

    The shared embedding is the stable ranking anchor. Domain-private
    residuals are projected away from that anchor, while a directional gate
    uses activity and preference agreement to suppress negative transfer.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.private_size = config['private_size']
        self.gate_hidden_size = config['gate_hidden_size']
        self.local_scale = config['local_scale']
        self.transfer_scale = config['transfer_scale']
        self.source_loss_weight = config['source_loss_weight']
        self.local_loss_weight = config['local_loss_weight']
        self.safety_weight = config['safety_weight']
        self.alignment_weight = config['alignment_weight']
        self.orthogonal_weight = config['orthogonal_weight']
        self.gate_weight = config['gate_weight']
        self.embedding_reg_weight = config['embedding_reg_weight']
        self.gate_initial_bias = config['gate_initial_bias']
        self.backbone_warmup_epochs = config[
            'backbone_warmup_epochs'
        ]

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
        self.register_buffer(
            'train_epoch_value', torch.zeros((), dtype=torch.long)
        )
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

    def set_train_epoch(self, epoch):
        self.train_epoch_value.fill_(epoch)

    def _residual_active(self):
        return self.train_epoch_value.item() >= self.backbone_warmup_epochs

    @staticmethod
    def _orthogonal_residual(residual, anchor):
        coefficient = (
            residual * anchor
        ).sum(dim=1, keepdim=True) / anchor.pow(2).sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        return residual - coefficient * anchor

    def _user_state(self, users, domain):
        other_domain = 1 - domain
        shared = self.shared_user_embedding(users)
        if not self._residual_active():
            zeros = torch.zeros_like(shared)
            gate = torch.zeros(
                shared.size(0), 1, device=shared.device
            )
            return shared, zeros, zeros, gate, shared, shared

        shared = shared.detach()
        local = self.local_projection[domain](
            self.private_user_embedding[domain](users)
        )
        cross = self.cross_projection[domain](
            self.private_user_embedding[other_domain](users).detach()
        )
        local = self._orthogonal_residual(local, shared)
        cross = self._orthogonal_residual(cross, shared)

        agreement = F.cosine_similarity(
            local, cross, dim=1, eps=1e-8
        ).unsqueeze(1)
        local_norm = local.norm(dim=1, keepdim=True) / math.sqrt(
            self.embedding_size
        )
        cross_norm = cross.norm(dim=1, keepdim=True) / math.sqrt(
            self.embedding_size
        )
        local_activity = self.user_activity[users, domain:domain + 1]
        other_activity = self.user_activity[
            users, other_domain:other_domain + 1
        ]
        gate_features = torch.cat((
            agreement,
            local_norm,
            cross_norm,
            local_activity,
            other_activity,
            other_activity - local_activity,
        ), dim=1)
        gate = torch.sigmoid(
            self.reliability_gate[domain](gate_features)
        )
        local_condition = shared + self.local_scale * local
        full_condition = (
            local_condition + self.transfer_scale * gate * cross
        )
        return shared, local, cross, gate, local_condition, full_condition

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

        if not self._residual_active():
            shared = self.shared_user_embedding(users)
            positive_embedding = self.item_embedding(positives)
            negative_embedding = self.item_embedding(negatives)
            return (
                self.bpr_loss(
                    self._score(shared, positive_embedding),
                    self._score(shared, negative_embedding),
                )
                + self.embedding_reg_weight * (
                    shared.pow(2).mean()
                    + positive_embedding.pow(2).mean()
                    + negative_embedding.pow(2).mean()
                )
            )

        (
            shared,
            local,
            cross,
            gate,
            local_condition,
            full_condition,
        ) = self._user_state(users, domain)
        positive_embedding = self.item_embedding(positives).detach()
        negative_embedding = self.item_embedding(negatives).detach()

        positive_score = self._score(
            full_condition, positive_embedding
        )
        negative_score = self._score(
            full_condition, negative_embedding
        )
        local_positive = self._score(
            local_condition, positive_embedding
        )
        local_negative = self._score(
            local_condition, negative_embedding
        )
        full_margin = positive_score - negative_score
        local_margin = local_positive - local_negative

        ranking = self.bpr_loss(positive_score, negative_score)
        local_ranking = self.bpr_loss(local_positive, local_negative)
        safety = F.relu(local_margin.detach() - full_margin).mean()
        alignment = (
            1 - F.cosine_similarity(
                local.detach(), cross, dim=1, eps=1e-8
            )
        ).mean()
        orthogonal = (
            F.cosine_similarity(
                shared, local, dim=1, eps=1e-8
            ).pow(2).mean()
            + F.cosine_similarity(
                shared, cross, dim=1, eps=1e-8
            ).pow(2).mean()
        )
        regularization = (
            shared.pow(2).mean()
            + local.pow(2).mean()
            + cross.pow(2).mean()
            + positive_embedding.pow(2).mean()
            + negative_embedding.pow(2).mean()
        )
        return (
            ranking
            + self.local_loss_weight * local_ranking
            + self.safety_weight * safety
            + self.alignment_weight * alignment
            + self.orthogonal_weight * orthogonal
            + self.gate_weight * gate.mean()
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
