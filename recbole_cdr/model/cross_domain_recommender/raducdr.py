# -*- coding: utf-8 -*-

r"""Rank-aligned direct user-conditioned diffusion for CDR."""

import torch
import torch.nn.functional as F

from recbole_cdr.model.cross_domain_recommender.cdcdr import _extract
from recbole_cdr.model.cross_domain_recommender.ducdr import DUCDR


class RADUCDR(DUCDR):
    r"""DUCDR with an extra BPR loss on denoised item-space vectors.

    DUCDR trains BPR on the direct user condition but ranks candidates with the
    diffusion-generated item-like vector at inference. This variant aligns that
    denoised vector with the ranking objective during training.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.diffusion_rank_weight = config["diffusion_rank_weight"]

    def _diffusion_training_loss(
        self, item_embedding, condition, negative_embedding=None
    ):
        timesteps = torch.randint(
            0, self.timesteps, (item_embedding.size(0),),
            device=item_embedding.device,
        )
        noise = torch.randn_like(item_embedding)
        noisy = (
            _extract(self.sqrt_alpha_cumprod, timesteps, item_embedding.shape)
            * item_embedding
            + _extract(
                self.sqrt_one_minus_alpha_cumprod,
                timesteps,
                item_embedding.shape,
            )
            * noise
        )
        prediction = self._denoise_conditioned(
            noisy, self._apply_condition_dropout(condition), timesteps
        )
        mse_loss = F.mse_loss(prediction, item_embedding)
        if negative_embedding is None or self.diffusion_rank_weight <= 0:
            return mse_loss

        positive_score = (prediction * item_embedding).sum(dim=1)
        negative_score = (prediction * negative_embedding).sum(dim=1)
        rank_loss = self.bpr_loss(positive_score, negative_score)
        return mse_loss + self.diffusion_rank_weight * rank_loss

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

        if self.loss_name == "bpr":
            recommendation_loss = self.bpr_loss(
                positive_score, negative_score
            )
        elif self.loss_name == "bce":
            recommendation_loss = (
                self.bce_loss(positive_score, torch.ones_like(positive_score))
                + self.bce_loss(
                    negative_score, torch.zeros_like(negative_score)
                )
            )
        else:
            recommendation_loss = self.mse_loss(condition, positive_embedding)

        return recommendation_loss + self._diffusion_training_loss(
            positive_embedding, condition, negative_embedding
        )
