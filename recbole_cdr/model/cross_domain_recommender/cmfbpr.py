# -*- coding: utf-8 -*-

r"""CMF with pairwise BPR training.

This keeps CMF's shared user/item factorization architecture and replaces the
official pointwise BCE objective with the BPR setup used by the direct models.
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class CMFBPR(CrossDomainRecommender):
    r"""Collective matrix factorization trained with BPR."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = config["embedding_size"]
        self.source_loss_weight = config["source_loss_weight"]
        if self.source_loss_weight is None:
            self.source_loss_weight = config["source_domain"].get(
                "loss_weight", 0.2
            )
        self.user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.bpr_loss = BPRLoss()
        self.apply(xavier_normal_initialization)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    @staticmethod
    def _score(user_embedding, item_embedding):
        return (user_embedding * item_embedding).sum(dim=1)

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        user_embedding = self.user_embedding(users)
        positive_score = self._score(
            user_embedding, self.item_embedding(positives)
        )
        negative_score = self._score(
            user_embedding, self.item_embedding(negatives)
        )
        return self.bpr_loss(positive_score, negative_score)

    def calculate_loss(self, interaction):
        source_loss = self._domain_loss(interaction, 0)
        target_loss = self._domain_loss(interaction, 1)
        return (
            self.source_loss_weight * source_loss
            + (1.0 - self.source_loss_weight) * target_loss
        )

    def predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        return self._score(
            self.user_embedding(users), self.item_embedding(items)
        )

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        target_items = self.item_embedding.weight[: self.target_num_items]
        return torch.matmul(self.user_embedding(users), target_items.t()).view(
            -1
        )
