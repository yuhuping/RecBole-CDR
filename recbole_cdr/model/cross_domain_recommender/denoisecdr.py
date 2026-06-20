# -*- coding: utf-8 -*-

r"""One-step denoising item-space matching for cross-domain recommendation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization

from recbole_cdr.model.cross_domain_recommender.cdcdr import CDCDR


class DenoiseCDR(CDCDR):
    r"""Replace multi-step diffusion with one-step denoising generation.

    The model keeps the effective CD-CDR surface: direct user condition,
    pairwise ranking, and item-space candidate matching. Instead of reverse
    diffusion, a denoising MLP learns to map a perturbed user condition to the
    positive item embedding.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.source_loss_weight is None:
            self.source_loss_weight = config["source_domain"].get(
                "loss_weight", 0.2
            )
        self.noise_std = config["noise_std"]
        self.denoise_loss_weight = config["denoise_loss_weight"]
        self.generator = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
        )
        self.generator.apply(xavier_normal_initialization)

    def _user_representation(self, users, domain, positive=None):
        return self.user_emb(users)

    def _diffusion_training_loss(self, item_embedding, condition):
        noisy_condition = condition + self.noise_std * torch.randn_like(
            condition
        )
        prediction = self.generator(
            torch.cat((noisy_condition, condition), dim=1)
        )
        return self.denoise_loss_weight * F.mse_loss(
            prediction, item_embedding
        )

    def _sample_item_embedding(self, condition):
        return self.generator(torch.cat((condition, condition), dim=1))
