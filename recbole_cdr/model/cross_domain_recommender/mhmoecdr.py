# -*- coding: utf-8 -*-

r"""Multi-head item-space MoE for cross-domain recommendation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.cross_domain_recommender.cdcdr import (
    CDCDR,
    _remove_positive,
)


class MHMOECDR(CDCDR):
    r"""Candidate-level subspace routing on top of CD-CDR.

    The backbone keeps CD-CDR's user-conditioned diffusion item generator.
    Experts are item-space conditions, not pretrained model votes: target-only,
    filtered transfer, denoised preference, transfer-only, and uncertainty
    residual. Candidate-aware routing mixes them per head.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.source_loss_weight is None:
            self.source_loss_weight = config["source_domain"].get(
                "loss_weight", 0.2
            )
        self.num_heads = config["num_heads"]
        if self.embedding_size % self.num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads.")
        self.head_size = self.embedding_size // self.num_heads
        self.num_experts = 5
        self.transfer_scale = config["transfer_scale"]
        self.expert_loss_weight = config["expert_loss_weight"]
        self.denoise_loss_weight = config["denoise_loss_weight"]
        self.safety_weight = config["safety_weight"]
        self.diversity_weight = config["diversity_weight"]
        self.entropy_weight = config["entropy_weight"]
        self.gate_regularization = config["gate_regularization"]
        self.inference_sample_experts = config["inference_sample_experts"]

        self.transfer_maps = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        self.denoise_maps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size * 3, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, self.embedding_size),
            ),
            nn.Sequential(
                nn.Linear(self.embedding_size * 3, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, self.embedding_size),
            ),
        ])
        self.uncertainty_maps = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Linear(self.embedding_size, self.embedding_size),
        ])
        gate_feature_size = self.head_size * 2 + self.num_experts + 3
        self.router = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gate_feature_size, self.embedding_size),
                nn.GELU(),
                nn.Dropout(self.dropout_probability),
                nn.Linear(self.embedding_size, self.num_experts),
            ),
            nn.Sequential(
                nn.Linear(gate_feature_size, self.embedding_size),
                nn.GELU(),
                nn.Dropout(self.dropout_probability),
                nn.Linear(self.embedding_size, self.num_experts),
            ),
        ])
        self.transfer_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size * 4 + 2, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, 1),
            ),
            nn.Sequential(
                nn.Linear(self.embedding_size * 4 + 2, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, 1),
            ),
        ])
        self.condition_norm = nn.ModuleList([
            nn.LayerNorm(self.embedding_size),
            nn.LayerNorm(self.embedding_size),
        ])
        for module in (
            list(self.transfer_maps)
            + list(self.denoise_maps)
            + list(self.uncertainty_maps)
            + list(self.router)
            + list(self.transfer_gate)
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
        condition = self._aggregate_history(
            self.user_emb(users),
            self.item_emb(history),
            history_len,
        )
        return condition, history_len

    def _condition_paths(self, users, domain, positive=None):
        other_domain = 1 - domain
        local, local_len = self._domain_view(
            users, domain, positive=positive
        )
        auxiliary, auxiliary_len = self._domain_view(users, other_domain)
        transfer = self._orthogonalize(
            self.transfer_maps[domain](auxiliary), local
        )
        gate_features = torch.cat((
            local,
            transfer,
            torch.abs(local - transfer),
            local * transfer,
            torch.log1p(local_len.float()).unsqueeze(1),
            torch.log1p(auxiliary_len.float()).unsqueeze(1),
        ), dim=1)
        gate = torch.sigmoid(self.transfer_gate[domain](gate_features))
        filtered = local + self.transfer_scale * gate * transfer
        denoised = filtered + 0.1 * self.denoise_maps[domain](
            torch.cat((local, transfer, filtered), dim=1)
        )
        uncertainty = local + 0.05 * gate * self.uncertainty_maps[domain](
            torch.abs(local - transfer)
        )
        transfer_only = local + self.transfer_scale * gate * transfer
        paths = torch.stack((
            local,
            filtered,
            denoised,
            transfer_only,
            uncertainty,
        ), dim=1)
        return paths, gate

    def _split_heads(self, tensor):
        return tensor.view(tensor.size(0), self.num_heads, self.head_size)

    def _expert_scores(self, paths, items):
        item_embedding = self.item_emb(items)
        return (paths * item_embedding.unsqueeze(1)).sum(dim=2)

    def _route_scores(self, users, items, domain, positive=None,
                      generated_paths=None):
        paths, gate = self._condition_paths(users, domain, positive)
        score_paths = generated_paths if generated_paths is not None else paths
        expert_scores = self._expert_scores(score_paths, items)
        item_heads = self._split_heads(self.item_emb(items))
        filtered_heads = self._split_heads(paths[:, 1])
        scalar = torch.cat((
            expert_scores,
            gate,
            expert_scores.std(dim=1, keepdim=True, unbiased=False),
            expert_scores[:, 1:2] - expert_scores[:, 0:1],
        ), dim=1)
        router_input = torch.cat((
            filtered_heads,
            item_heads,
            scalar.unsqueeze(1).expand(-1, self.num_heads, -1),
        ), dim=2)
        weights = F.softmax(
            self.router[domain](
                router_input.reshape(-1, router_input.size(-1))
            ).view(users.size(0), self.num_heads, self.num_experts),
            dim=2,
        )
        path_heads = score_paths.view(
            users.size(0), self.num_experts, self.num_heads, self.head_size
        ).permute(0, 2, 1, 3)
        head_scores = (path_heads * item_heads.unsqueeze(2)).sum(dim=3)
        routed = (weights * head_scores).sum(dim=2).sum(dim=1)
        return routed, expert_scores, weights, paths, gate

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
        raise ValueError("MHMOECDR supports bpr or bce recommendation loss.")

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        positive_score, positive_experts, weights, paths, gate = (
            self._route_scores(users, positives, domain, positives)
        )
        negative_score, negative_experts, _, _, _ = self._route_scores(
            users, negatives, domain, positives
        )
        recommendation_loss = self._recommendation_loss(
            positive_score, negative_score
        )
        expert_loss = torch.stack([
            self.bpr_loss(positive_experts[:, i], negative_experts[:, i])
            for i in range(self.num_experts)
        ]).mean()
        anchor_positive = positive_experts[:, 1]
        anchor_negative = negative_experts[:, 1]
        local_margin = positive_experts[:, 0] - negative_experts[:, 0]
        routed_margin = positive_score - negative_score
        safety = F.relu(local_margin.detach() - routed_margin).mean()
        denoise_loss = F.mse_loss(paths[:, 2], paths[:, 1].detach())
        diversity = self._diversity_loss(paths)
        entropy = -(
            weights * weights.clamp_min(1e-8).log()
        ).sum(dim=2).mean()
        return (
            recommendation_loss
            + self._diffusion_training_loss(
                self.item_emb(positives), paths[:, 1]
            )
            + self.expert_loss_weight * expert_loss
            + self.denoise_loss_weight * denoise_loss
            + self.safety_weight * safety
            + self.diversity_weight * diversity
            - self.entropy_weight * entropy
            + self.gate_regularization * gate.mean()
            + 0.1 * self.bpr_loss(anchor_positive, anchor_negative)
        )

    def _diversity_loss(self, paths):
        normalized = F.normalize(paths, dim=2, eps=1e-8)
        correlation = torch.bmm(normalized, normalized.transpose(1, 2))
        identity = torch.eye(self.num_experts, device=paths.device)
        return (correlation - identity).pow(2).mean()

    @torch.no_grad()
    def _generated_score_paths(self, users, paths):
        if self.inference_sample_experts:
            generated = [
                self._sample_item_embedding(paths[:, i])
                for i in range(self.num_experts)
            ]
            return torch.stack(generated, dim=1)
        score_paths = paths.clone()
        score_paths[:, 1] = self._sample_item_embedding(paths[:, 1])
        return score_paths

    def predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        unique_users, inverse = torch.unique(
            users, sorted=False, return_inverse=True
        )
        paths, _ = self._condition_paths(unique_users, 1)
        generated_paths = self._generated_score_paths(unique_users, paths)
        return self._route_scores(
            users,
            items,
            1,
            generated_paths=generated_paths[inverse],
        )[0]

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        target_items = torch.arange(
            self.target_num_items, device=users.device
        )
        outputs = []
        for user in users:
            item_users = user.expand(target_items.size(0))
            paths, _ = self._condition_paths(user.unsqueeze(0), 1)
            generated_paths = self._generated_score_paths(
                user.unsqueeze(0), paths
            ).expand(target_items.size(0), -1, -1)
            outputs.append(
                self._route_scores(
                    item_users,
                    target_items,
                    1,
                    generated_paths=generated_paths,
                )[0]
            )
        return torch.stack(outputs).reshape(-1)
