# -*- coding: utf-8 -*-

r"""Gradient-direction weighted direct user-item matching for DUCDR."""

import torch
import torch.nn.functional as F

from recbole_cdr.model.cross_domain_recommender.ducdrdirect import (
    DUCDRDirect,
)


class GDUCDRDirect(DUCDRDirect):
    r"""DUCDRDirect with target-domain gradient-direction sample weighting.

    The scoring and inference path is identical to DUCDRDirect. During target
    domain training, samples whose pairwise update direction deviates from the
    moving population direction receive a smaller shared loss weight.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.direction_tau = config["direction_tau"]
        self.direction_temperature = config["direction_temperature"]
        self.direction_min_weight = config["direction_min_weight"]
        self.direction_ema = config["direction_ema"]
        self.register_buffer(
            "target_direction_center",
            torch.zeros(self.embedding_size),
        )
        self.register_buffer(
            "direction_center_initialized",
            torch.zeros(1, dtype=torch.bool),
        )

    def _pair_direction(self, positive_embedding, negative_embedding):
        return F.normalize(positive_embedding - negative_embedding, dim=1)

    @torch.no_grad()
    def _update_direction_center(self, directions):
        batch_center = F.normalize(directions.mean(dim=0), dim=0)
        if not bool(self.direction_center_initialized.item()):
            self.target_direction_center.copy_(batch_center)
            self.direction_center_initialized.fill_(True)
            return
        updated = (
            self.direction_ema * self.target_direction_center
            + (1.0 - self.direction_ema) * batch_center
        )
        self.target_direction_center.copy_(F.normalize(updated, dim=0))

    def _target_direction_weight(self, positive_embedding, negative_embedding):
        directions = self._pair_direction(
            positive_embedding.detach(), negative_embedding.detach()
        )
        if not bool(self.direction_center_initialized.item()):
            self._update_direction_center(directions)
            return torch.ones(
                positive_embedding.size(0), device=positive_embedding.device
            )

        alignment = torch.matmul(directions, self.target_direction_center)
        weight = torch.sigmoid(
            (alignment - self.direction_tau) / self.direction_temperature
        )
        weight = self.direction_min_weight + (
            1.0 - self.direction_min_weight
        ) * weight
        self._update_direction_center(directions)
        return weight.detach()

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
        positive_embedding = self.item_emb(positives)
        negative_embedding = self.item_emb(negatives)
        positive_score = self._score(user_embedding, positive_embedding)
        negative_score = self._score(user_embedding, negative_embedding)
        loss = -F.logsigmoid(positive_score - negative_score)
        if domain == 1:
            loss = loss * self._target_direction_weight(
                positive_embedding, negative_embedding
            )
        return loss.mean()
