# -*- coding: utf-8 -*-

r"""Gradient-direction weighted DUCDR.

This variant keeps DUCDR's history-free inference path and only reweights
target-domain training samples whose pairwise update direction deviates from
the moving population direction.
"""

import torch
import torch.nn.functional as F

from recbole_cdr.model.cross_domain_recommender.ducdr import DUCDR


class GDUCDR(DUCDR):
    r"""DUCDR with target-domain gradient-direction sample routing."""

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

    def _diffusion_training_loss(self, item_embedding, condition, weight=None):
        timesteps = torch.randint(
            0, self.timesteps, (item_embedding.size(0),),
            device=item_embedding.device,
        )
        noise = torch.randn_like(item_embedding)
        noisy = (
            self.sqrt_alpha_cumprod[timesteps].view(-1, 1) * item_embedding
            + self.sqrt_one_minus_alpha_cumprod[timesteps].view(-1, 1) * noise
        )
        prediction = self._denoise_conditioned(
            noisy, self._apply_condition_dropout(condition), timesteps
        )
        loss = F.mse_loss(prediction, item_embedding, reduction="none").mean(
            dim=1
        )
        if weight is not None:
            loss = loss * weight
        return loss.mean()

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        condition = self._user_representation(
            users, domain, positive=positives
        )
        positive_embedding = self.item_emb(positives)
        negative_embedding = self.item_emb(negatives)
        positive_score = (condition * positive_embedding).sum(dim=1)
        negative_score = (condition * negative_embedding).sum(dim=1)

        weight = None
        if domain == 1:
            weight = self._target_direction_weight(
                positive_embedding, negative_embedding
            )

        if self.loss_name == "bpr":
            recommendation_loss = -F.logsigmoid(
                positive_score - negative_score
            )
            if weight is not None:
                recommendation_loss = recommendation_loss * weight
            recommendation_loss = recommendation_loss.mean()
        elif self.loss_name == "bce":
            recommendation_loss = (
                F.binary_cross_entropy_with_logits(
                    positive_score,
                    torch.ones_like(positive_score),
                    reduction="none",
                )
                + F.binary_cross_entropy_with_logits(
                    negative_score,
                    torch.zeros_like(negative_score),
                    reduction="none",
                )
            )
            if weight is not None:
                recommendation_loss = recommendation_loss * weight
            recommendation_loss = recommendation_loss.mean()
        else:
            recommendation_loss = F.mse_loss(
                condition, positive_embedding, reduction="none"
            ).mean(dim=1)
            if weight is not None:
                recommendation_loss = recommendation_loss * weight
            recommendation_loss = recommendation_loss.mean()

        return recommendation_loss + self._diffusion_training_loss(
            positive_embedding, condition, weight
        )
