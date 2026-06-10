# -*- coding: utf-8 -*-

r"""Reliability-aware Guided Diffusion for cross-domain recommendation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization

from recbole_cdr.model.cross_domain_recommender.cdcdr import (
    CDCDR,
    _extract,
    _remove_positive,
)


class RGDCDR(CDCDR):
    r"""Use user- and timestep-specific gates to control cross-domain transfer.

    Unlike the released CD-CDR implementation, every prediction is conditioned
    on histories from both domains. A learned reliability score determines how
    much of the auxiliary-domain preference enters each diffusion step.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        reliability_hidden = config['reliability_hidden']
        self.diffusion_loss_weight = config['diffusion_loss_weight']
        self.negative_transfer_weight = config['negative_transfer_weight']
        self.gate_regularization = config['gate_regularization']
        self.transfer_scale = config['transfer_scale']
        self.guidance_floor = config['guidance_floor']

        feature_size = self.embedding_size * 4 + 2
        self.reliability_mlp = nn.Sequential(
            nn.Linear(feature_size, reliability_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(reliability_hidden, 2),
        )
        self.transfer_maps = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        self.timestep_gate = nn.Sequential(
            nn.Linear(2, reliability_hidden),
            nn.GELU(),
            nn.Linear(reliability_hidden, 1),
        )
        self.condition_norm = nn.ModuleList([
            nn.LayerNorm(self.embedding_size),
            nn.LayerNorm(self.embedding_size),
        ])

        self.reliability_mlp.apply(xavier_normal_initialization)
        self.transfer_maps.apply(xavier_normal_initialization)
        self.timestep_gate.apply(xavier_normal_initialization)

    def _domain_view(self, users, domain, positive=None):
        history = self.history_item_id[domain][users]
        history_len = self.history_item_len[domain][users]
        if positive is not None:
            history, history_len = _remove_positive(
                positive, history, history_len
            )
        user_embedding = self.user_emb(users)
        fused = self._aggregate_history(
            user_embedding, self.item_emb(history), history_len
        )
        history_scale = max(1 - self.gamma, 1e-6)
        history_preference = (
            fused - self.gamma * user_embedding
        ) / history_scale
        return fused, history_preference, history_len

    def _dual_domain_state(self, users, domain, positive=None):
        source, source_preference, source_len = self._domain_view(
            users, 0, positive=positive if domain == 0 else None
        )
        target, target_preference, target_len = self._domain_view(
            users, 1, positive=positive if domain == 1 else None
        )

        features = torch.cat((
            source_preference,
            target_preference,
            torch.abs(source_preference - target_preference),
            source_preference * target_preference,
            torch.log1p(source_len.float()).unsqueeze(1),
            torch.log1p(target_len.float()).unsqueeze(1),
        ), dim=1)
        reliability = torch.sigmoid(self.reliability_mlp(features)[:, domain:domain + 1])

        local = source if domain == 0 else target
        auxiliary = target_preference if domain == 0 else source_preference
        transfer = self.transfer_maps[domain](auxiliary)
        return local, transfer, reliability

    def _dynamic_condition(self, local, transfer, reliability, timesteps, domain):
        progress = timesteps.float() / max(self.timesteps - 1, 1)
        time_features = torch.stack((
            progress,
            torch.sin(math.pi * progress),
        ), dim=1)
        reliability_logit = torch.logit(reliability.clamp(1e-4, 1 - 1e-4))
        gate = torch.sigmoid(reliability_logit + self.timestep_gate(time_features))
        condition = local + self.transfer_scale * gate * transfer
        return self.condition_norm[domain](condition), gate

    def _diffusion_training_loss(self, item_embedding, state, domain):
        local, transfer, reliability = state
        timesteps = torch.randint(
            0,
            self.timesteps,
            (item_embedding.size(0),),
            device=item_embedding.device,
        )
        condition, gate = self._dynamic_condition(
            local, transfer, reliability, timesteps, domain
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
        return F.mse_loss(prediction, item_embedding), gate

    def _recommendation_loss(self, positive_score, negative_score):
        if self.loss_name == 'bpr':
            return self.bpr_loss(positive_score, negative_score)
        if self.loss_name == 'bce':
            return (
                self.bce_loss(positive_score, torch.ones_like(positive_score))
                + self.bce_loss(negative_score, torch.zeros_like(negative_score))
            )
        raise ValueError('RGDCDR supports bpr or bce recommendation loss.')

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        state = self._dual_domain_state(users, domain, positive=positives)
        local, transfer, reliability = state
        zero_steps = torch.zeros_like(users)
        condition, ranking_gate = self._dynamic_condition(
            local, transfer, reliability, zero_steps, domain
        )

        positive_embedding = self.item_emb(positives)
        negative_embedding = self.item_emb(negatives)
        positive_score = (condition * positive_embedding).sum(dim=1)
        negative_score = (condition * negative_embedding).sum(dim=1)
        recommendation_loss = self._recommendation_loss(
            positive_score, negative_score
        )

        local_condition = self.condition_norm[domain](local)
        local_positive = (local_condition * positive_embedding).sum(dim=1)
        local_negative = (local_condition * negative_embedding).sum(dim=1)
        local_margin = local_positive - local_negative
        cross_margin = positive_score - negative_score
        negative_transfer_loss = F.relu(
            local_margin.detach() - cross_margin
        ).mean()

        diffusion_loss, diffusion_gate = self._diffusion_training_loss(
            positive_embedding, state, domain
        )
        gate_loss = 0.5 * (
            ranking_gate.mean() + diffusion_gate.mean()
        )
        return (
            recommendation_loss
            + self.diffusion_loss_weight * diffusion_loss
            + self.negative_transfer_weight * negative_transfer_loss
            + self.gate_regularization * gate_loss
        )

    @torch.no_grad()
    def _sample_item_embedding(self, state, domain):
        local, transfer, reliability = state
        sample = torch.randn_like(local)
        for index in reversed(range(self.ddim_steps.size(0))):
            timesteps = self.ddim_steps[index].expand(local.size(0))
            condition, gate = self._dynamic_condition(
                local, transfer, reliability, timesteps, domain
            )
            conditioned = self._denoise_conditioned(
                sample, condition, timesteps
            )
            unconditioned = self._denoise_unconditioned(sample, timesteps)
            guidance = (
                self.guidance_floor
                + (self.guidance_weight - self.guidance_floor) * gate
            )
            predicted_start = (
                (1 + guidance) * conditioned - guidance * unconditioned
            )
            sample = (
                self.ddim_coefficient_1[index] * predicted_start
                + self.ddim_coefficient_2[index] * sample
            )
        return sample

    def _target_samples(self, users):
        state = self._dual_domain_state(users, 1)
        return self._sample_item_embedding(state, 1)
