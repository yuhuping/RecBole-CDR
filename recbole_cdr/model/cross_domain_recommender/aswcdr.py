# -*- coding: utf-8 -*-

r"""Adaptive Source-Weighting Cross-Domain Recommendation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType

from recbole_cdr.model.cross_domain_recommender.sacdr import _build_activity
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class ASWCDR(CrossDomainRecommender):
    r"""Allocate source supervision by target scarcity and training progress."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.source_loss_weight = config['source_loss_weight']
        self.adaptive_source_weighting = config[
            'adaptive_source_weighting'
        ]
        self.weighting_mode = config['weighting_mode'] or 'joint'
        self.use_source_curriculum = config['use_source_curriculum']
        self.scarcity_power = config['scarcity_power']
        self.minimum_user_weight = config['minimum_user_weight']
        self.maximum_user_weight = config['maximum_user_weight']
        self.minimum_source_scale = config['minimum_source_scale']
        self.curriculum_epochs = config['curriculum_epochs']
        self.embedding_reg_weight = config['embedding_reg_weight']
        self.current_epoch = 0
        self.phase = 'BOTH'

        self.user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.register_buffer('user_activity', _build_activity(dataset))
        source_evidence = self.user_activity[:, 0].clamp_min(1e-4)
        target_scarcity = (
            1 - self.user_activity[:, 1]
        ).clamp_min(1e-4).pow(self.scarcity_power)
        if self.weighting_mode == 'scarcity':
            reliability = target_scarcity
        elif self.weighting_mode == 'evidence':
            reliability = source_evidence
        elif self.weighting_mode == 'joint':
            reliability = source_evidence * target_scarcity
        else:
            raise ValueError(
                'Unsupported ASWCDR weighting_mode: %s'
                % self.weighting_mode
            )
        reliability = reliability / reliability[
            1:self.overlapped_num_users
        ].mean().clamp_min(1e-8)
        self.register_buffer(
            'source_user_weight',
            reliability.clamp(
                self.minimum_user_weight,
                self.maximum_user_weight,
            ),
        )

        self.apply(xavier_normal_initialization)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def set_train_epoch(self, epoch):
        self.current_epoch = epoch

    def set_phase(self, phase):
        self.phase = phase

    def _curriculum_scale(self):
        if not self.use_source_curriculum:
            return 1.0
        progress = min(
            self.current_epoch / max(self.curriculum_epochs - 1, 1),
            1.0,
        )
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (
            self.minimum_source_scale
            + (1 - self.minimum_source_scale) * cosine
        )

    def _pairwise_loss(self, users, positives, negatives, weights=None):
        user_embedding = self.user_embedding(users)
        item_difference = (
            self.item_embedding(positives)
            - self.item_embedding(negatives)
        )
        losses = F.softplus(
            -(user_embedding * item_difference).sum(dim=1)
        )
        if weights is None:
            return losses.mean()
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    def calculate_loss(self, interaction):
        target_users = interaction[self.TARGET_USER_ID]
        target_positive = interaction[self.TARGET_ITEM_ID]
        target_negative = interaction[self.TARGET_NEG_ITEM_ID]
        target_loss = self._pairwise_loss(
            target_users,
            target_positive,
            target_negative,
        )
        if self.phase == 'TARGET':
            return target_loss + self.embedding_reg_weight * (
                self.user_embedding(target_users).pow(2).mean()
                + self.item_embedding(target_positive).pow(2).mean()
            )

        source_users = interaction[self.SOURCE_USER_ID]
        source_positive = interaction[self.SOURCE_ITEM_ID]
        source_negative = interaction[self.SOURCE_NEG_ITEM_ID]

        source_weights = None
        if self.adaptive_source_weighting:
            source_weights = self.source_user_weight[source_users]
        source_loss = self._pairwise_loss(
            source_users,
            source_positive,
            source_negative,
            source_weights,
        )
        effective_source_weight = (
            self.source_loss_weight * self._curriculum_scale()
        )
        regularization = (
            self.user_embedding(source_users).pow(2).mean()
            + self.user_embedding(target_users).pow(2).mean()
            + self.item_embedding(source_positive).pow(2).mean()
            + self.item_embedding(target_positive).pow(2).mean()
        )
        return (
            effective_source_weight * source_loss
            + (1 - effective_source_weight) * target_loss
            + self.embedding_reg_weight * regularization
        )

    def predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        return (
            self.user_embedding(users) * self.item_embedding(items)
        ).sum(dim=1)

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = self.item_embedding.weight[:self.target_num_items]
        return torch.matmul(
            self.user_embedding(users), items.t()
        ).reshape(-1)
