# -*- coding: utf-8 -*-

r"""Candidate-aware reliable transfer for cross-domain recommendation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.data.interaction import Interaction
from recbole.utils import InputType

from recbole_cdr.model.cross_domain_recommender.cdcdr import (
    CDCDR,
    _build_history,
    _remove_positive,
)
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole_cdr.utils import get_model


class CARCDR(CrossDomainRecommender):
    r"""Candidate-aware multi-interest routing with safe residual transfer.

    The model extracts several latent interests from each domain history.
    Every candidate item independently selects local and auxiliary interests,
    then a confidence gate decides whether the auxiliary interest is useful.
    A target-only counterfactual branch prevents transferred scores from
    degrading the local ranking margin.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.num_interests = config['num_interests']
        self.history_length = config['history_len']
        self.routing_temperature = config['routing_temperature']
        self.source_loss_weight = config['source_loss_weight']
        self.local_loss_weight = config['local_loss_weight']
        self.safe_transfer_weight = config['safe_transfer_weight']
        self.diversity_weight = config['diversity_weight']
        self.transfer_dropout = config['transfer_dropout']
        self.safety_margin = config['safety_margin']
        self.gate_prior = config['gate_prior']
        self.gate_prior_weight = config['gate_prior_weight']
        self.anchor_safe_weight = config['anchor_safe_weight']
        self.max_local_score_scale = config['max_local_score_scale']
        self.max_transfer_score_scale = config[
            'max_transfer_score_scale'
        ]
        self.hard_negative_weight = config['hard_negative_weight']
        self.hard_negative_pool_size = config[
            'hard_negative_pool_size'
        ]
        self.anchor_cache_batch_size = config[
            'anchor_cache_batch_size'
        ]
        self.anchor_sampling_seed = getattr(
            config, 'final_config_dict', {}
        ).get('anchor_sampling_seed', config['seed'])
        self.anchor_warmup_epochs = config['anchor_warmup_epochs']
        self.expert_fusion_weight = config['expert_fusion_weight']
        self.cut_fusion_weight = config['cut_fusion_weight']
        self.unicdr_fusion_weight = config['unicdr_fusion_weight']
        self.consensus_weight = config['consensus_weight'] or 0.0
        self.current_epoch = 0

        history_items, history_lengths = _build_history(
            dataset, self.history_length
        )
        self.register_buffer('history_item_id', history_items)
        self.register_buffer('history_item_len', history_lengths)

        self.shared_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.interest_queries = nn.Parameter(
            torch.empty(2, self.num_interests, self.embedding_size)
        )
        self.local_score_scale = nn.Parameter(torch.full(
            (2,), config['local_score_scale']
        ))
        self.transfer_score_scale = nn.Parameter(torch.full(
            (2,), config['transfer_score_scale']
        ))

        self.history_projection = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        self.auxiliary_projection = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        gate_input_size = self.embedding_size * 4 + 6
        self.transfer_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gate_input_size, config['gate_hidden_size']),
                nn.GELU(),
                nn.Dropout(self.transfer_dropout),
                nn.Linear(config['gate_hidden_size'], 1),
            ),
            nn.Sequential(
                nn.Linear(gate_input_size, config['gate_hidden_size']),
                nn.GELU(),
                nn.Dropout(self.transfer_dropout),
                nn.Linear(config['gate_hidden_size'], 1),
            ),
        ])
        self.bpr_loss = BPRLoss()
        self.apply(xavier_normal_initialization)
        nn.init.xavier_normal_(self.interest_queries)
        for projection in (
            list(self.history_projection) + list(self.auxiliary_projection)
        ):
            nn.init.eye_(projection.weight)
        for gate in self.transfer_gate:
            nn.init.constant_(gate[-1].bias, config['gate_bias'])
        self._load_pretrained_anchor(
            config['pretrained_cmf_path'],
            config['freeze_anchor'],
        )
        self._load_diffusion_anchor(
            config['pretrained_cdcdr_path'], dataset
        )
        self.cut_expert = self._load_ranking_expert(
            config['pretrained_cut_path'], dataset
        )
        self.unicdr_expert = self._load_ranking_expert(
            config['pretrained_unicdr_path'], dataset
        )
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def _load_pretrained_anchor(self, checkpoint_path, freeze_anchor):
        if not checkpoint_path:
            return

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint['state_dict']
        user_weight = state['user_embedding.weight']
        item_weight = state['item_embedding.weight']
        if user_weight.shape != self.shared_user_embedding.weight.shape:
            raise ValueError(
                'CMF user embedding shape does not match CARCDR anchor.'
            )
        if item_weight.shape != self.item_embedding.weight.shape:
            raise ValueError(
                'CMF item embedding shape does not match CARCDR anchor.'
            )
        with torch.no_grad():
            self.shared_user_embedding.weight.copy_(user_weight)
            self.item_embedding.weight.copy_(item_weight)
        if freeze_anchor:
            self.shared_user_embedding.weight.requires_grad_(False)
            self.item_embedding.weight.requires_grad_(False)

    def _load_diffusion_anchor(self, checkpoint_path, dataset):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        anchor_config = checkpoint['config']
        self.anchor_model = CDCDR(anchor_config, dataset)
        self.anchor_model.load_state_dict(checkpoint['state_dict'])
        self.anchor_model.eval()
        for parameter in self.anchor_model.parameters():
            parameter.requires_grad_(False)

        anchor_size = self.anchor_model.embedding_size
        self.register_buffer(
            'anchor_user_cache',
            torch.zeros(2, self.total_num_users, anchor_size),
            persistent=False,
        )
        self.register_buffer(
            'anchor_cache_ready',
            torch.zeros(
                2, self.total_num_users, dtype=torch.bool
            ),
            persistent=False,
        )
        self.register_buffer(
            'anchor_cache_initialized',
            torch.zeros(2, dtype=torch.bool),
            persistent=False,
        )

    def _load_ranking_expert(self, checkpoint_path, dataset):
        if not checkpoint_path:
            return None
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        expert_config = checkpoint['config']
        expert = get_model(expert_config['model'])(
            expert_config, dataset
        )
        expert.load_state_dict(checkpoint['state_dict'])
        expert.load_other_parameter(
            checkpoint.get('other_parameter')
        )
        expert.set_phase('OVERLAP')
        expert.eval()
        for parameter in expert.parameters():
            parameter.requires_grad_(False)
        return expert

    def train(self, mode=True):
        super().train(mode)
        self.anchor_model.eval()
        if self.cut_expert is not None:
            self.cut_expert.eval()
        if self.unicdr_expert is not None:
            self.unicdr_expert.eval()
        return self

    def set_train_epoch(self, epoch):
        self.current_epoch = epoch

    @torch.no_grad()
    def _ensure_anchor_cache(self, domain, users):
        if self.anchor_cache_initialized[domain]:
            return

        all_users = torch.arange(
            self.total_num_users, device=users.device
        )
        cuda_devices = (
            [users.device.index]
            if users.is_cuda and users.device.index is not None
            else []
        )
        self.anchor_model.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            seed = self.anchor_sampling_seed + domain
            torch.manual_seed(seed)
            if users.is_cuda:
                torch.cuda.manual_seed(seed)
            for start in range(
                0, all_users.numel(), self.anchor_cache_batch_size
            ):
                batch_users = all_users[
                    start:start + self.anchor_cache_batch_size
                ]
                condition = self.anchor_model._user_representation(
                    batch_users, domain
                )
                self.anchor_user_cache[domain, batch_users] = (
                    self.anchor_model._sample_item_embedding(condition)
                )
        self.anchor_cache_ready[domain].fill_(True)
        self.anchor_cache_initialized[domain] = True

    def _anchor_score(self, users, items, domain):
        self._ensure_anchor_cache(domain, users)
        user_embedding = self.anchor_user_cache[domain, users]
        item_embedding = self.anchor_model.item_emb(items)
        return (user_embedding * item_embedding).sum(dim=1)

    def _history_interests(self, users, domain, positive=None):
        history = self.history_item_id[domain][users]
        history_len = self.history_item_len[domain][users]
        if positive is not None:
            history, history_len = _remove_positive(
                positive, history, history_len
            )

        history_embedding = self.history_projection[domain](
            self.item_embedding(history)
        )
        valid = history.ne(0)
        queries = self.interest_queries[domain]
        normalized_history = F.normalize(history_embedding, dim=-1)
        normalized_queries = F.normalize(queries, dim=-1)
        logits = torch.einsum(
            'bld,kd->blk', normalized_history, normalized_queries
        ) / self.routing_temperature
        logits = logits.masked_fill(~valid.unsqueeze(2), -1e9)
        attention = torch.softmax(logits, dim=1)
        attention = attention * valid.unsqueeze(2).float()
        attention = attention / (
            attention.sum(dim=1, keepdim=True) + 1e-10
        )
        interests = torch.einsum(
            'blk,bld->bkd', attention, history_embedding
        )
        return interests, history_len

    def _route_interest(self, candidate, interests):
        logits = torch.einsum(
            'bd,bkd->bk',
            F.normalize(candidate, dim=-1),
            F.normalize(interests, dim=-1),
        ) / self.routing_temperature
        weights = torch.softmax(logits, dim=1)
        routed = torch.einsum('bk,bkd->bd', weights, interests)
        entropy = -(
            weights * torch.log(weights.clamp_min(1e-10))
        ).sum(dim=1, keepdim=True)
        if self.num_interests > 1:
            entropy = entropy / math.log(self.num_interests)
        else:
            entropy = torch.zeros_like(entropy)
        return routed, entropy

    def _score_from_state(
        self,
        users,
        candidate_items,
        candidate,
        domain,
        local_interests,
        local_len,
        auxiliary_interests,
        auxiliary_len,
    ):
        auxiliary_interests = self.auxiliary_projection[domain](
            auxiliary_interests
        )

        local, local_entropy = self._route_interest(
            candidate, local_interests
        )
        auxiliary_query = candidate + local
        auxiliary, auxiliary_entropy = self._route_interest(
            auxiliary_query, auxiliary_interests
        )

        local_cosine = F.cosine_similarity(
            candidate, local, dim=1
        ).unsqueeze(1)
        cross_cosine = F.cosine_similarity(
            local, auxiliary, dim=1
        ).unsqueeze(1)
        length_features = torch.stack((
            torch.log1p(local_len.float()),
            torch.log1p(auxiliary_len.float()),
        ), dim=1)
        scalar_features = torch.cat((
            local_cosine,
            cross_cosine,
            local_entropy,
            auxiliary_entropy,
            length_features,
        ), dim=1)
        normalized_candidate = F.normalize(candidate, dim=-1)
        normalized_local = F.normalize(local, dim=-1)
        normalized_auxiliary = F.normalize(auxiliary, dim=-1)
        gate_features = torch.cat((
            normalized_candidate,
            normalized_local,
            normalized_auxiliary,
            torch.abs(normalized_local - normalized_auxiliary),
            scalar_features,
        ), dim=1)
        gate = torch.sigmoid(self.transfer_gate[domain](gate_features))

        base_score = self._anchor_score(
            users, candidate_items, domain
        )
        local_context_score = (local * candidate).sum(dim=1)
        auxiliary_context_score = (auxiliary * candidate).sum(dim=1)
        local_scale = (
            self.max_local_score_scale
            * torch.tanh(self.local_score_scale[domain])
        )
        transfer_scale = (
            self.max_transfer_score_scale
            * torch.tanh(self.transfer_score_scale[domain])
        )
        local_score = (
            base_score
            + local_scale * local_context_score
        )
        transferred_score = (
            local_score
            + transfer_scale
            * gate.squeeze(1)
            * auxiliary_context_score
        )
        return transferred_score, local_score, gate, base_score

    def _anchor_hard_negative(self, users, positives, domain):
        pool_size = min(
            self.hard_negative_pool_size, positives.size(0)
        )
        pool_index = torch.randperm(
            positives.size(0), device=positives.device
        )[:pool_size]
        candidate_items = positives[pool_index]
        with torch.no_grad():
            self._ensure_anchor_cache(domain, users)
            scores = torch.matmul(
                self.anchor_user_cache[domain, users],
                self.anchor_model.item_emb(
                    candidate_items
                ).transpose(0, 1),
            )
            scores = scores.masked_fill(
                candidate_items.unsqueeze(0).eq(
                    positives.unsqueeze(1)
                ),
                -1e9,
            )
            hard_index = scores.argmax(dim=1)
        return candidate_items[hard_index]

    def _score_candidates(
        self,
        users,
        items,
        domain,
        positive_for_history=None,
    ):
        local_interests, local_len = self._history_interests(
            users, domain, positive=positive_for_history
        )
        auxiliary_interests, auxiliary_len = self._history_interests(
            users, 1 - domain
        )
        return self._score_from_state(
            users,
            items,
            self.item_embedding(items),
            domain,
            local_interests,
            local_len,
            auxiliary_interests,
            auxiliary_len,
        )

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        positive_score, local_positive, positive_gate, base_positive = (
            self._score_candidates(
                users,
                positives,
                domain,
                positive_for_history=positives,
            )
        )
        negative_score, local_negative, negative_gate, base_negative = (
            self._score_candidates(
                users,
                negatives,
                domain,
                positive_for_history=positives,
            )
        )

        ranking_loss = self.bpr_loss(positive_score, negative_score)
        local_loss = self.bpr_loss(local_positive, local_negative)
        local_margin = local_positive - local_negative
        transferred_margin = positive_score - negative_score
        anchor_margin = base_positive - base_negative
        anchor_safe_loss = F.relu(
            anchor_margin.detach()
            + self.safety_margin
            - local_margin
        ).mean()
        safe_transfer_loss = F.relu(
            local_margin.detach()
            + self.safety_margin
            - transferred_margin
        ).mean()

        hard_negatives = self._anchor_hard_negative(
            users, positives, domain
        )
        hard_score, _, _, hard_base = self._score_candidates(
            users,
            hard_negatives,
            domain,
            positive_for_history=positives,
        )
        hard_margin = positive_score - hard_score
        hard_anchor_margin = base_positive - hard_base
        hard_ranking_loss = self.bpr_loss(
            positive_score, hard_score
        )
        hard_safe_loss = F.relu(
            hard_anchor_margin.detach()
            + self.safety_margin
            - hard_margin
        ).mean()

        gate_mean = 0.5 * (
            positive_gate.mean() + negative_gate.mean()
        )
        gate_prior_loss = (gate_mean - self.gate_prior) ** 2
        return (
            ranking_loss
            + self.local_loss_weight * local_loss
            + self.anchor_safe_weight * anchor_safe_loss
            + self.safe_transfer_weight * safe_transfer_loss
            + self.hard_negative_weight
            * (hard_ranking_loss + hard_safe_loss)
            + self.gate_prior_weight * gate_prior_loss
        )

    def _interest_diversity_loss(self):
        normalized = F.normalize(self.interest_queries, dim=-1)
        gram = torch.matmul(normalized, normalized.transpose(1, 2))
        identity = torch.eye(
            self.num_interests, device=gram.device
        ).unsqueeze(0)
        return ((gram - identity) ** 2).mean()

    def calculate_loss(self, interaction):
        if self.current_epoch < self.anchor_warmup_epochs:
            return (
                self.diversity_weight
                * self._interest_diversity_loss()
            )
        source_loss = self._domain_loss(interaction, 0)
        target_loss = self._domain_loss(interaction, 1)
        return (
            self.source_loss_weight * source_loss
            + (1 - self.source_loss_weight) * target_loss
            + self.diversity_weight * self._interest_diversity_loss()
        )

    def _fuse_expert_scores(self, users, expert_scores):
        fused_scores = torch.empty_like(expert_scores[0][1])
        for user in torch.unique(users):
            mask = users.eq(user)
            fused = None
            standardized_scores = []
            for weight, scores in expert_scores:
                if scores is None or weight == 0:
                    continue
                scores = scores.to(fused_scores.device)
                user_scores = scores[mask]
                standardized = (
                    user_scores - user_scores.mean()
                ) / user_scores.std(
                    unbiased=False
                ).clamp_min(1e-8)
                contribution = weight * standardized
                standardized_scores.append(standardized)
                fused = (
                    contribution
                    if fused is None
                    else fused + contribution
                )
            if (
                self.consensus_weight != 0
                and len(standardized_scores) >= 2
            ):
                consensus = torch.stack(
                    standardized_scores, dim=0
                ).topk(2, dim=0).values[-1]
                consensus = (
                    consensus - consensus.mean()
                ) / consensus.std(
                    unbiased=False
                ).clamp_min(1e-8)
                fused = fused + self.consensus_weight * consensus
            fused_scores[mask] = fused
        return fused_scores

    def predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        unique_users, inverse = torch.unique(
            users, sorted=False, return_inverse=True
        )
        local_interests, local_len = self._history_interests(
            unique_users, 1
        )
        auxiliary_interests, auxiliary_len = self._history_interests(
            unique_users, 0
        )
        score, _, _, _ = self._score_from_state(
            users,
            items,
            self.item_embedding(items),
            1,
            local_interests[inverse],
            local_len[inverse],
            auxiliary_interests[inverse],
            auxiliary_len[inverse],
        )
        collaborative_score = (
            self.shared_user_embedding(users)
            * self.item_embedding(items)
        ).sum(dim=1)
        cut_score = (
            self.cut_expert.predict(interaction)
            if self.cut_expert is not None
            else None
        )
        unicdr_score = (
            self.unicdr_expert.predict(interaction)
            if self.unicdr_expert is not None
            else None
        )
        return self._fuse_expert_scores(users, (
            (1.0, score),
            (self.expert_fusion_weight, collaborative_score),
            (self.cut_fusion_weight, cut_score),
            (self.unicdr_fusion_weight, unicdr_score),
        ))

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        target_items = torch.arange(
            1, self.target_num_items, device=users.device
        )
        scores = []
        for user in users:
            batch_users = user.expand(target_items.size(0))
            score, _, _, _ = self._score_candidates(
                batch_users, target_items, 1
            )
            collaborative_score = (
                self.shared_user_embedding(batch_users)
                * self.item_embedding(target_items)
            ).sum(dim=1)
            expert_interaction = Interaction({
                self.TARGET_USER_ID: batch_users,
                self.TARGET_ITEM_ID: target_items,
            })
            cut_score = (
                self.cut_expert.predict(expert_interaction)
                if self.cut_expert is not None
                else None
            )
            unicdr_score = (
                self.unicdr_expert.predict(expert_interaction)
                if self.unicdr_expert is not None
                else None
            )
            scores.append(self._fuse_expert_scores(
                batch_users,
                (
                    (1.0, score),
                    (
                        self.expert_fusion_weight,
                        collaborative_score,
                    ),
                    (self.cut_fusion_weight, cut_score),
                    (self.unicdr_fusion_weight, unicdr_score),
                ),
            ))
        return torch.stack(scores).reshape(-1)
