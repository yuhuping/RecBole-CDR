# -*- coding: utf-8 -*-

r"""DUCDR-Lite: history-free direct user-conditioned diffusion."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.cross_domain_recommender.cdcdr import (
    SinusoidalPositionEmbeddings,
    _cosine_beta_schedule,
    _exp_beta_schedule,
    _extract,
    _linear_beta_schedule,
    _sqrt_beta_schedule,
)
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class DUCDRLite(CrossDomainRecommender):
    r"""Remove CD-CDR's inactive history branch and keep item-space diffusion."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = config["embedding_size"]
        self.loss_name = config["loss_n"]
        self.timesteps = config["timestep"]
        self.guidance_weight = config["uncon_w"]
        self.unconditional_probability = config["uncon_p"]
        self.ddim_stride = config["ddim_stride"]
        self.source_loss_weight = config["source_loss_weight"]
        if self.source_loss_weight is None:
            self.source_loss_weight = config["source_domain"].get(
                "loss_weight", 0.2
            )

        self.user_emb = nn.Embedding(self.total_num_users, self.embedding_size)
        self.item_emb = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.none_embedding = nn.Embedding(1, self.embedding_size)
        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, self.embedding_size),
            nn.GELU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        if config["diffuser_type"] == "mlp1":
            self.diffusion_mlp = nn.Linear(
                self.embedding_size * 3, self.embedding_size
            )
        elif config["diffuser_type"] == "mlp2":
            self.diffusion_mlp = nn.Sequential(
                nn.Linear(self.embedding_size * 3, self.embedding_size * 2),
                nn.GELU(),
                nn.Linear(self.embedding_size * 2, self.embedding_size),
            )
        else:
            raise ValueError(
                "Unsupported diffuser_type: %s" % config["diffuser_type"]
            )

        self.bpr_loss = BPRLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        beta_schedule = config["beta_sche"]
        if beta_schedule == "linear":
            betas = _linear_beta_schedule(self.timesteps)
        elif beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.timesteps)
        elif beta_schedule == "exp":
            betas = _exp_beta_schedule(self.timesteps)
        elif beta_schedule == "sqrt":
            betas = _sqrt_beta_schedule(self.timesteps)
        else:
            raise ValueError("Unsupported beta schedule: %s" % beta_schedule)
        self._register_diffusion_schedule(betas)

        self.apply(xavier_normal_initialization)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def _register_diffusion_schedule(self, betas):
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            torch.sqrt(1.0 - alpha_cumprod),
        )

        ddim_steps = list(range(0, self.timesteps, self.ddim_stride))
        if ddim_steps[-1] != self.timesteps - 1:
            ddim_steps.append(self.timesteps - 1)
        ddim_steps = torch.tensor(ddim_steps, dtype=torch.long)
        ddim_alpha = alpha_cumprod[ddim_steps]
        ddim_alpha_prev = F.pad(ddim_alpha[:-1], (1, 0), value=1.0)
        reciprocal_noise = torch.sqrt(1.0 / ddim_alpha - 1)
        coefficient_1 = (
            torch.sqrt(ddim_alpha_prev)
            - torch.sqrt(1.0 - ddim_alpha_prev) / reciprocal_noise
        )
        coefficient_1[0] = 1.0
        coefficient_2 = (
            torch.sqrt(1.0 - ddim_alpha_prev)
            / torch.sqrt(1.0 - ddim_alpha)
        )
        coefficient_2[0] = 0.0
        self.register_buffer("ddim_steps", ddim_steps)
        self.register_buffer("ddim_coefficient_1", coefficient_1)
        self.register_buffer("ddim_coefficient_2", coefficient_2)

    def _user_representation(self, users, domain, positive=None):
        return self.user_emb(users)

    def _denoise_conditioned(self, noisy_embedding, condition, timesteps):
        time_embedding = self.step_mlp(timesteps)
        return self.diffusion_mlp(
            torch.cat((noisy_embedding, condition, time_embedding), dim=1)
        )

    def _denoise_unconditioned(self, noisy_embedding, timesteps):
        condition = self.none_embedding.weight.expand(
            noisy_embedding.size(0), -1
        )
        return self._denoise_conditioned(noisy_embedding, condition, timesteps)

    def _apply_condition_dropout(self, condition):
        keep = (
            torch.rand(
                condition.size(0), 1, device=condition.device
            ) >= self.unconditional_probability
        ).float()
        unconditional = self.none_embedding.weight.expand_as(condition)
        return keep * condition + (1 - keep) * unconditional

    def _diffusion_training_loss(self, item_embedding, condition):
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
        return F.mse_loss(prediction, item_embedding)

    @torch.no_grad()
    def _sample_item_embedding(self, condition):
        sample = torch.randn_like(condition)
        for index in reversed(range(self.ddim_steps.size(0))):
            timesteps = self.ddim_steps[index].expand(condition.size(0))
            conditioned = self._denoise_conditioned(
                sample, condition, timesteps
            )
            unconditioned = self._denoise_unconditioned(sample, timesteps)
            predicted_start = (
                (1 + self.guidance_weight) * conditioned
                - self.guidance_weight * unconditioned
            )
            sample = (
                self.ddim_coefficient_1[index] * predicted_start
                + self.ddim_coefficient_2[index] * sample
            )
        return sample

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        condition = self._user_representation(users, domain, positives)
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
            positive_embedding, condition
        )

    def calculate_loss(self, interaction):
        source_loss = self._domain_loss(interaction, 0)
        target_loss = self._domain_loss(interaction, 1)
        return (
            self.source_loss_weight * source_loss
            + (1 - self.source_loss_weight) * target_loss
        )

    def _target_samples(self, users):
        target_condition = self._user_representation(users, 1)
        return self._sample_item_embedding(target_condition)

    def predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        unique_users, inverse = torch.unique(
            users, sorted=False, return_inverse=True
        )
        generated = self._target_samples(unique_users)
        return (generated[inverse] * self.item_emb(items)).sum(dim=1)

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        generated = self._target_samples(users)
        target_items = self.item_emb.weight[:self.target_num_items]
        return torch.matmul(generated, target_items.t()).view(-1)
