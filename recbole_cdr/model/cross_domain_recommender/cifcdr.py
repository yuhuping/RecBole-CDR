# -*- coding: utf-8 -*-

r"""Conditional information-flow CDR on a CD-CDR item-space backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.cross_domain_recommender.cdcdr import (
    CDCDR,
    _remove_positive,
)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, weight):
        ctx.weight = weight
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


def _grad_reverse(tensor, weight):
    return _GradientReverse.apply(tensor, weight)


class CIFCDR(CDCDR):
    r"""Filter transferable source signal before diffusion item matching.

    The CD-CDR user-conditioned item generator remains the ranking backbone.
    Auxiliary-domain preference is decomposed into transferable and noisy
    components. Only the target-aware transferable component enters the
    diffusion condition; the noisy component is adversarially prevented from
    carrying target-ranking signal.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.source_loss_weight is None:
            self.source_loss_weight = config["source_domain"].get(
                "loss_weight", 0.2
            )
        self.transfer_scale = config["transfer_scale"]
        self.transfer_loss_weight = config["transfer_loss_weight"]
        self.noise_adv_weight = config["noise_adv_weight"]
        self.decorrelation_weight = config["decorrelation_weight"]
        self.safety_weight = config["safety_weight"]
        self.gate_regularization = config["gate_regularization"]
        self.grl_lambda = config["grl_lambda"]

        self.transfer_maps = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        self.noise_maps = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        gate_input_size = self.embedding_size * 4 + 2
        self.transfer_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gate_input_size, self.embedding_size),
                nn.GELU(),
                nn.Dropout(self.dropout_probability),
                nn.Linear(self.embedding_size, 1),
            ),
            nn.Sequential(
                nn.Linear(gate_input_size, self.embedding_size),
                nn.GELU(),
                nn.Dropout(self.dropout_probability),
                nn.Linear(self.embedding_size, 1),
            ),
        ])
        self.noise_adversary = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        self.condition_norm = nn.ModuleList([
            nn.LayerNorm(self.embedding_size),
            nn.LayerNorm(self.embedding_size),
        ])
        for module in (
            list(self.transfer_maps)
            + list(self.noise_maps)
            + list(self.transfer_gate)
            + list(self.noise_adversary)
            + list(self.condition_norm)
        ):
            module.apply(self._init_added_module)
        for gate in self.transfer_gate:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, config["gate_initial_bias"])

    @staticmethod
    def _init_added_module(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _orthogonalize(vector, anchor):
        coefficient = (vector * anchor).sum(dim=1, keepdim=True)
        coefficient = coefficient / anchor.pow(2).sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        return vector - coefficient * anchor

    def _domain_view(self, users, domain, positive=None):
        history = self.history_item_id[domain][users]
        history_len = self.history_item_len[domain][users]
        if positive is not None:
            history, history_len = _remove_positive(
                positive, history, history_len
            )
        user_embedding = self.user_emb(users)
        condition = self._aggregate_history(
            user_embedding,
            self.item_emb(history),
            history_len,
        )
        return condition, history_len

    def _filtered_state(self, users, domain, positive=None):
        other_domain = 1 - domain
        local, local_len = self._domain_view(
            users, domain, positive=positive
        )
        auxiliary, auxiliary_len = self._domain_view(users, other_domain)
        raw_transfer = self.transfer_maps[domain](auxiliary)
        transfer = self._orthogonalize(raw_transfer, local)
        noise = self.noise_maps[domain](auxiliary)
        noise = self._orthogonalize(noise, transfer)

        gate_features = torch.cat((
            local,
            transfer,
            torch.abs(local - transfer),
            local * transfer,
            torch.log1p(local_len.float()).unsqueeze(1),
            torch.log1p(auxiliary_len.float()).unsqueeze(1),
        ), dim=1)
        gate = torch.sigmoid(self.transfer_gate[domain](gate_features))
        condition = local + self.transfer_scale * gate * transfer
        return local, transfer, noise, gate, condition

    def _recommendation_loss(self, positive_score, negative_score):
        if self.loss_name == "bpr":
            return self.bpr_loss(positive_score, negative_score)
        if self.loss_name == "bce":
            return (
                self.bce_loss(positive_score, torch.ones_like(positive_score))
                + self.bce_loss(
                    negative_score, torch.zeros_like(negative_score)
                )
            )
        raise ValueError("CIFCDR supports bpr or bce recommendation loss.")

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        local, transfer, noise, gate, condition = self._filtered_state(
            users, domain, positives
        )
        positive_embedding = self.item_emb(positives)
        negative_embedding = self.item_emb(negatives)
        positive_score = (condition * positive_embedding).sum(dim=1)
        negative_score = (condition * negative_embedding).sum(dim=1)
        recommendation_loss = self._recommendation_loss(
            positive_score, negative_score
        )

        local_score_pos = (local * positive_embedding).sum(dim=1)
        local_score_neg = (local * negative_embedding).sum(dim=1)
        safety = F.relu(
            (local_score_pos - local_score_neg).detach()
            - (positive_score - negative_score)
        ).mean()

        transfer_loss = self.bpr_loss(
            (transfer * positive_embedding).sum(dim=1),
            (transfer * negative_embedding).sum(dim=1),
        )
        reversed_noise = _grad_reverse(noise, self.grl_lambda)
        adversarial = self.noise_adversary[domain](reversed_noise)
        noise_loss = self.bpr_loss(
            (adversarial * positive_embedding.detach()).sum(dim=1),
            (adversarial * negative_embedding.detach()).sum(dim=1),
        )
        decorrelation = (
            F.cosine_similarity(
                transfer, noise, dim=1, eps=1e-8
            ).pow(2).mean()
            + F.cosine_similarity(
                local, noise, dim=1, eps=1e-8
            ).pow(2).mean()
        )
        return (
            recommendation_loss
            + self._diffusion_training_loss(positive_embedding, condition)
            + self.transfer_loss_weight * transfer_loss
            + self.noise_adv_weight * noise_loss
            + self.decorrelation_weight * decorrelation
            + self.safety_weight * safety
            + self.gate_regularization * gate.mean()
        )

    def _target_samples(self, users):
        condition = self._filtered_state(users, 1)[-1]
        return self._sample_item_embedding(condition)
