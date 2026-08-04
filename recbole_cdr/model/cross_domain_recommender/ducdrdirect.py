# -*- coding: utf-8 -*-

r"""Direct user-item matching ablation for DUCDR."""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class DUCDRDirect(CrossDomainRecommender):
    r"""Remove diffusion and score candidate items with user-item dot product."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = config["embedding_size"]
        self.source_loss_weight = config["source_loss_weight"]
        if self.source_loss_weight is None:
            self.source_loss_weight = config["source_domain"].get(
                "loss_weight", 0.2
            )
        self.user_emb = nn.Embedding(self.total_num_users, self.embedding_size)
        self.item_emb = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.bpr_loss = BPRLoss()
        self.apply(xavier_normal_initialization)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

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
        user_embedding = self.user_emb(users)
        return self.bpr_loss(
            self._score(user_embedding, self.item_emb(positives)),
            self._score(user_embedding, self.item_emb(negatives)),
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
        return self._score(self.user_emb(users), self.item_emb(items))

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        target_items = self.item_emb.weight[:self.target_num_items]
        return torch.matmul(self.user_emb(users), target_items.t()).view(-1)
