# -*- coding: utf-8 -*-

r"""MEXCDR: jointly trained multi-expert cross-domain recommendation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


def _training_interactions(domain_dataset):
    interactions = domain_dataset.inter_feat
    file_sizes = getattr(domain_dataset, "file_size_list", None)
    if file_sizes:
        interactions = interactions[:file_sizes[0]]
    return interactions


def _build_user_context(dataset, history_len):
    histories = []
    lengths = []
    activities = []
    for domain_dataset in (
        dataset.source_domain_dataset,
        dataset.target_domain_dataset,
    ):
        interactions = _training_interactions(domain_dataset)
        users = interactions[domain_dataset.uid_field].cpu()
        items = interactions[domain_dataset.iid_field].cpu()

        per_user = [[] for _ in range(dataset.num_total_user)]
        for user, item in zip(users.tolist(), items.tolist()):
            per_user[user].append(item)

        history = torch.zeros(
            dataset.num_total_user, history_len, dtype=torch.long
        )
        length = torch.zeros(dataset.num_total_user, dtype=torch.long)
        for user, item_list in enumerate(per_user):
            selected = item_list[-history_len:]
            if selected:
                history[user, :len(selected)] = torch.tensor(selected)
                length[user] = len(selected)

        counts = torch.bincount(
            users, minlength=dataset.num_total_user
        ).float()
        histories.append(history)
        lengths.append(length)
        activities.append(torch.log1p(counts))

    activity = torch.stack(activities, dim=1)
    activity = activity / activity.max(
        dim=0, keepdim=True
    ).values.clamp_min(1)
    return (
        torch.stack(histories),
        torch.stack(lengths),
        activity,
    )


class MEXCDR(CrossDomainRecommender):
    r"""Train four complementary experts jointly from random initialization.

    Experts capture shared collaborative preference, domain-local preference,
    reliability-filtered transfer, and dual-domain history attention. Each
    expert receives direct pairwise supervision, while a learned mixture and
    second-largest consensus term optimize the final ranking.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.shared_size = config["shared_size"]
        self.expert_size = config["expert_size"]
        self.history_len = config["history_len"]
        self.source_loss_weight = config["source_loss_weight"]
        self.expert_loss_weight = config["expert_loss_weight"]
        self.safety_weight = config["safety_weight"]
        self.diversity_weight = config["diversity_weight"]
        self.reg_weight = config["reg_weight"]
        self.consensus_weight = config["consensus_weight"]
        self.gate_hidden_size = config["gate_hidden_size"]

        self.shared_user = nn.Embedding(
            self.total_num_users, self.shared_size
        )
        self.shared_item = nn.Embedding(
            self.total_num_items, self.shared_size, padding_idx=0
        )

        self.local_user = nn.ModuleList([
            nn.Embedding(self.total_num_users, self.expert_size),
            nn.Embedding(self.total_num_users, self.expert_size),
        ])
        self.local_item = nn.ModuleList([
            nn.Embedding(
                self.total_num_items, self.expert_size, padding_idx=0
            ),
            nn.Embedding(
                self.total_num_items, self.expert_size, padding_idx=0
            ),
        ])
        self.transfer_projection = nn.ModuleList([
            nn.Linear(self.expert_size, self.expert_size, bias=False),
            nn.Linear(self.expert_size, self.expert_size, bias=False),
        ])
        self.transfer_gate = nn.ModuleList([
            self._make_gate(),
            self._make_gate(),
        ])
        self.history_projection = nn.ModuleList([
            nn.Linear(self.expert_size, self.expert_size, bias=False),
            nn.Linear(self.expert_size, self.expert_size, bias=False),
        ])
        self.history_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.expert_size * 3 + 2, self.gate_hidden_size),
                nn.GELU(),
                nn.Linear(self.gate_hidden_size, 1),
            ),
            nn.Sequential(
                nn.Linear(self.expert_size * 3 + 2, self.gate_hidden_size),
                nn.GELU(),
                nn.Linear(self.gate_hidden_size, 1),
            ),
        ])

        self.expert_logits = nn.Parameter(torch.zeros(2, 4))
        self.bpr_loss = BPRLoss()

        history, history_len, activity = _build_user_context(
            dataset, self.history_len
        )
        self.register_buffer("user_history", history)
        self.register_buffer("user_history_len", history_len)
        self.register_buffer("user_activity", activity)

        self.apply(xavier_normal_initialization)
        for gate in self.transfer_gate:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, -1.0)
        for gate in self.history_gate:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, -1.0)
        with torch.no_grad():
            self.shared_item.weight[0].zero_()
            for embedding in self.local_item:
                embedding.weight[0].zero_()

    def _make_gate(self):
        return nn.Sequential(
            nn.Linear(5, self.gate_hidden_size),
            nn.GELU(),
            nn.Linear(self.gate_hidden_size, 1),
        )

    @staticmethod
    def _dot(user_embedding, item_embedding):
        return (user_embedding * item_embedding).sum(dim=1)

    def _history_attention(
        self, users, query, history_domain, output_domain, positive=None
    ):
        history = self.user_history[history_domain, users]
        mask = history.ne(0)
        if positive is not None and history_domain == output_domain:
            mask = mask & history.ne(positive.unsqueeze(1))
        history_embedding = self.local_item[history_domain](history)
        if history_domain != output_domain:
            history_embedding = self.history_projection[output_domain](
                history_embedding
            )

        attention = (
            history_embedding * query.unsqueeze(1)
        ).sum(dim=2) / math.sqrt(self.expert_size)
        attention = attention.masked_fill(~mask, -1e9)
        weights = F.softmax(attention, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        return (weights.unsqueeze(2) * history_embedding).sum(dim=1)

    def _expert_states(self, users, domain, positive=None):
        other_domain = 1 - domain
        shared = self.shared_user(users)
        local = self.local_user[domain](users)
        other = self.local_user[other_domain](users)
        transferred = self.transfer_projection[domain](other)

        agreement = F.cosine_similarity(
            local, transferred, dim=1, eps=1e-8
        ).unsqueeze(1)
        local_norm = local.norm(dim=1, keepdim=True) / math.sqrt(
            self.expert_size
        )
        transfer_norm = transferred.norm(
            dim=1, keepdim=True
        ) / math.sqrt(self.expert_size)
        local_activity = self.user_activity[
            users, domain:domain + 1
        ]
        other_activity = self.user_activity[
            users, other_domain:other_domain + 1
        ]
        transfer_features = torch.cat((
            agreement,
            local_norm,
            transfer_norm,
            local_activity,
            other_activity,
        ), dim=1)
        transfer_gate = torch.sigmoid(
            self.transfer_gate[domain](transfer_features)
        )
        transfer = transfer_gate * transferred

        local_history = self._history_attention(
            users, local, domain, domain, positive
        )
        cross_history = self._history_attention(
            users, local, other_domain, domain
        )
        history_features = torch.cat((
            local,
            local_history,
            cross_history,
            local_activity,
            other_activity,
        ), dim=1)
        cross_history_gate = torch.sigmoid(
            self.history_gate[domain](history_features)
        )
        history = (
            local + local_history
            + cross_history_gate * cross_history
        )
        return shared, local, transfer, history

    def _expert_scores(self, users, items, domain, positive=None):
        shared, local, transfer, history = self._expert_states(
            users, domain, positive
        )
        shared_item = self.shared_item(items)
        local_item = self.local_item[domain](items)
        return torch.stack((
            self._dot(shared, shared_item),
            self._dot(local, local_item),
            self._dot(transfer, local_item),
            self._dot(history, local_item),
        ), dim=1)

    @staticmethod
    def _batch_standardize(positive_scores, negative_scores):
        joined = torch.cat((positive_scores, negative_scores), dim=0)
        mean = joined.mean(dim=0, keepdim=True)
        std = joined.std(
            dim=0, unbiased=False, keepdim=True
        ).clamp_min(1e-6)
        return (
            (positive_scores - mean) / std,
            (negative_scores - mean) / std,
        )

    def _fused_scores(self, standardized_scores, domain):
        weights = F.softmax(self.expert_logits[domain], dim=0)
        fused = (standardized_scores * weights).sum(dim=1)
        consensus = standardized_scores.topk(
            2, dim=1
        ).values[:, -1]
        return fused + self.consensus_weight * consensus

    @staticmethod
    def _diversity_loss(margins):
        if margins.size(0) < 2:
            return margins.new_zeros(())
        centered = margins - margins.mean(dim=0, keepdim=True)
        normalized = centered / centered.pow(2).mean(
            dim=0, keepdim=True
        ).sqrt().clamp_min(1e-6)
        correlation = normalized.t().matmul(normalized) / margins.size(0)
        identity = torch.eye(
            correlation.size(0), device=correlation.device
        )
        return ((correlation - identity) ** 2).mean()

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        positive_scores = self._expert_scores(
            users, positives, domain, positives
        )
        negative_scores = self._expert_scores(
            users, negatives, domain, positives
        )
        standardized_positive, standardized_negative = (
            self._batch_standardize(positive_scores, negative_scores)
        )
        fused_positive = self._fused_scores(
            standardized_positive, domain
        )
        fused_negative = self._fused_scores(
            standardized_negative, domain
        )

        expert_loss = torch.stack([
            self.bpr_loss(
                positive_scores[:, expert],
                negative_scores[:, expert],
            )
            for expert in range(positive_scores.size(1))
        ]).mean()
        fused_loss = self.bpr_loss(fused_positive, fused_negative)
        shared_margin = (
            standardized_positive[:, 0]
            - standardized_negative[:, 0]
        )
        fused_margin = fused_positive - fused_negative
        safety = F.relu(
            shared_margin.detach() - fused_margin
        ).mean()
        diversity = self._diversity_loss(
            positive_scores - negative_scores
        )

        regularization = (
            self.shared_user(users).pow(2).mean()
            + self.shared_item(positives).pow(2).mean()
            + self.shared_item(negatives).pow(2).mean()
            + self.local_user[domain](users).pow(2).mean()
            + self.local_item[domain](positives).pow(2).mean()
            + self.local_item[domain](negatives).pow(2).mean()
        )
        return (
            fused_loss
            + self.expert_loss_weight * expert_loss
            + self.safety_weight * safety
            + self.diversity_weight * diversity
            + self.reg_weight * regularization
        )

    def calculate_loss(self, interaction):
        source_loss = self._domain_loss(interaction, 0)
        target_loss = self._domain_loss(interaction, 1)
        return (
            self.source_loss_weight * source_loss
            + (1 - self.source_loss_weight) * target_loss
        )

    def _candidate_fusion(self, users, expert_scores, domain):
        output = torch.empty_like(expert_scores[:, 0])
        weights = F.softmax(self.expert_logits[domain], dim=0)
        for user in torch.unique(users):
            mask = users.eq(user)
            scores = expert_scores[mask]
            standardized = (
                scores - scores.mean(dim=0, keepdim=True)
            ) / scores.std(
                dim=0, unbiased=False, keepdim=True
            ).clamp_min(1e-8)
            fused = (standardized * weights).sum(dim=1)
            consensus = standardized.topk(
                2, dim=1
            ).values[:, -1]
            consensus = (
                consensus - consensus.mean()
            ) / consensus.std(unbiased=False).clamp_min(1e-8)
            output[mask] = (
                fused + self.consensus_weight * consensus
            )
        return output

    def predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        scores = self._expert_scores(users, items, 1)
        return self._candidate_fusion(users, scores, 1)

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        target_items = torch.arange(
            self.target_num_items, device=users.device
        )
        outputs = []
        for user in users:
            item_users = user.expand(target_items.size(0))
            scores = self._expert_scores(item_users, target_items, 1)
            outputs.append(
                self._candidate_fusion(item_users, scores, 1)
            )
        return torch.stack(outputs).reshape(-1)
