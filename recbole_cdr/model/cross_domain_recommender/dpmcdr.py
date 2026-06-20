# -*- coding: utf-8 -*-

r"""Distributional Prototype Mixture for cross-domain recommendation."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


def _training_interactions(domain_dataset):
    inter_feat = domain_dataset.inter_feat
    file_sizes = getattr(domain_dataset, 'file_size_list', None)
    if file_sizes:
        inter_feat = inter_feat[:file_sizes[0]]
    return inter_feat


def _build_training_history(dataset, history_len):
    histories = []
    lengths = []
    user_num = dataset.num_total_user
    for domain_dataset in (
        dataset.source_domain_dataset,
        dataset.target_domain_dataset,
    ):
        inter_feat = _training_interactions(domain_dataset)
        users = inter_feat[domain_dataset.uid_field].numpy()
        items = inter_feat[domain_dataset.iid_field].numpy()
        timestamps = inter_feat[domain_dataset.time_field].numpy()
        order = np.lexsort((timestamps, users))

        per_user = [[] for _ in range(user_num)]
        for index in order:
            per_user[int(users[index])].append(int(items[index]))

        history = torch.zeros((user_num, history_len), dtype=torch.long)
        length = torch.zeros(user_num, dtype=torch.long)
        for user, item_list in enumerate(per_user):
            selected = item_list[-history_len:]
            if selected:
                history[user, :len(selected)] = torch.tensor(selected)
                length[user] = len(selected)
        histories.append(history)
        lengths.append(length)
    return torch.stack(histories), torch.stack(lengths)


def _remove_positive(positive, history, history_len):
    keep = history.ne(positive.unsqueeze(1))
    removed = (~keep).sum(dim=1)
    return history * keep.long(), (history_len - removed).clamp_min(0)


class NoiseEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        half = embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer('frequencies', frequencies)

    def forward(self, noise_level):
        values = noise_level.unsqueeze(1) * self.frequencies.unsqueeze(0)
        return torch.cat((values.sin(), values.cos()), dim=1)


class DPMCDR(CrossDomainRecommender):
    r"""A deterministic multi-modal alternative to diffusion-based CDR.

    Multiple preference prototypes define a conditional item distribution.
    Source-domain prototypes are transferred only when a learned reliability
    gate agrees with target-domain evidence. A noise-conditioned reconstruction
    objective regularizes the same item space used by pairwise ranking.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.num_prototypes = config['num_prototypes']
        self.history_length = config['history_len']
        self.score_temperature = config['score_temperature']
        self.source_loss_weight = config['source_loss_weight']
        self.local_loss_weight = config['local_loss_weight']
        self.safe_transfer_weight = config['safe_transfer_weight']
        self.denoise_weight = config['denoise_weight']
        self.alignment_weight = config['alignment_weight']
        self.diversity_weight = config['diversity_weight']
        self.embedding_reg_weight = config['embedding_reg_weight']
        self.max_noise = config['max_noise']
        self.transfer_scale = config['transfer_scale']
        self.identity_score_weight = config['identity_score_weight']
        self.prototype_score_weight = config['prototype_score_weight']
        self.hard_negative_weight = config['hard_negative_weight']
        self.hard_negative_pool_size = config[
            'hard_negative_pool_size'
        ]
        self.transfer_warmup_epochs = config[
            'transfer_warmup_epochs'
        ]
        self.hard_negative_warmup_epochs = config[
            'hard_negative_warmup_epochs'
        ]
        self.generation_score_weight = config[
            'generation_score_weight'
        ]
        self.denoise_mask_probability = config[
            'denoise_mask_probability'
        ]
        self.normalize_scores = config['normalize_scores']
        self.current_epoch = 0

        history, history_len = _build_training_history(
            dataset, self.history_length
        )
        self.register_buffer('history_item_id', history)
        self.register_buffer('history_item_len', history_len)

        self.shared_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.domain_user_embedding = nn.ModuleList([
            nn.Embedding(self.total_num_users, self.embedding_size),
            nn.Embedding(self.total_num_users, self.embedding_size),
        ])
        self.item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.prototype_query = nn.Parameter(
            torch.empty(2, self.num_prototypes, self.embedding_size)
        )
        self.domain_embedding = nn.Parameter(
            torch.empty(2, self.embedding_size)
        )
        self.local_projection = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        self.cross_projection = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        gate_size = self.embedding_size * 3 + 2
        self.reliability_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gate_size, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, 1),
            ),
            nn.Sequential(
                nn.Linear(gate_size, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, 1),
            ),
        ])
        self.noise_embedding = NoiseEmbedding(self.embedding_size)
        self.denoiser = nn.Sequential(
            nn.Linear(self.embedding_size * 3, self.embedding_size * 2),
            nn.GELU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
        )
        self.bpr_loss = BPRLoss()
        self.apply(xavier_normal_initialization)
        nn.init.xavier_normal_(self.prototype_query)
        nn.init.xavier_normal_(self.domain_embedding)
        for projection in (
            list(self.local_projection) + list(self.cross_projection)
        ):
            nn.init.eye_(projection.weight)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def _history_prototypes(
        self, users, domain, positive_for_history=None
    ):
        history = self.history_item_id[domain][users]
        history_len = self.history_item_len[domain][users]
        if positive_for_history is not None:
            history, history_len = _remove_positive(
                positive_for_history, history, history_len
            )

        item_embedding = self.item_embedding(history)
        valid = history.ne(0)
        identity = (
            self.shared_user_embedding(users)
            + self.domain_user_embedding[domain](users)
            + self.domain_embedding[domain]
        )
        query = self.prototype_query[domain].unsqueeze(0) + identity.unsqueeze(1)
        logits = torch.einsum(
            'bld,bkd->blk',
            F.normalize(item_embedding, dim=-1),
            F.normalize(query, dim=-1),
        )
        logits = logits.masked_fill(~valid.unsqueeze(2), -1e9)
        attention = torch.softmax(logits, dim=1)
        attention = attention * valid.unsqueeze(2).float()
        attention = attention / attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-10)
        history_modes = torch.einsum(
            'blk,bld->bkd', attention, item_embedding
        )
        local_modes = self.local_projection[domain](
            history_modes + self.prototype_query[domain].unsqueeze(0)
        )
        return local_modes, history_len, identity

    def _distribution_state(
        self, users, domain, positive_for_history=None
    ):
        local, local_len, identity = self._history_prototypes(
            users, domain, positive_for_history
        )
        auxiliary, auxiliary_len, _ = self._history_prototypes(
            users, 1 - domain
        )
        auxiliary = self.cross_projection[domain](auxiliary)

        local_norm = F.normalize(local, dim=-1)
        auxiliary_norm = F.normalize(auxiliary, dim=-1)
        agreement = local_norm * auxiliary_norm
        length_feature = torch.stack((
            torch.log1p(local_len.float()),
            torch.log1p(auxiliary_len.float()),
        ), dim=1)
        gate_input = torch.cat((
            local_norm,
            auxiliary_norm,
            torch.abs(local_norm - auxiliary_norm),
            length_feature.unsqueeze(1).expand(
                -1, self.num_prototypes, -1
            ),
        ), dim=2)
        gate = torch.sigmoid(self.reliability_gate[domain](gate_input))
        active_transfer_scale = (
            self.transfer_scale
            if self.current_epoch >= self.transfer_warmup_epochs
            else 0.0
        )
        transferred = (
            local
            + active_transfer_scale
            * gate
            * agreement.sum(dim=2, keepdim=True).clamp_min(0)
            * auxiliary
        )
        return transferred, local, gate, identity

    def _score(self, item_embedding, modes, identity):
        score_items = self._score_embedding(item_embedding)
        score_modes = self._score_embedding(modes)
        score_identity = self._score_embedding(identity)
        mode_score = torch.einsum(
            'bd,bkd->bk', score_items, score_modes
        )
        mixture_score = self.score_temperature * torch.logsumexp(
            mode_score / self.score_temperature, dim=1
        )
        identity_score = (
            score_identity * score_items
        ).sum(dim=1)
        generated_score = (
            self._score_embedding(
                self._generated_preference(modes, identity)
            )
            * score_items
        ).sum(dim=1)
        return (
            self.prototype_score_weight * mixture_score
            + self.identity_score_weight * identity_score
            + self.generation_score_weight * generated_score
        )

    def _score_pool(self, item_embedding, modes, identity):
        score_items = self._score_embedding(item_embedding)
        score_modes = self._score_embedding(modes)
        score_identity = self._score_embedding(identity)
        mode_score = torch.einsum(
            'bkd,pd->bkp', score_modes, score_items
        )
        mixture_score = self.score_temperature * torch.logsumexp(
            mode_score / self.score_temperature, dim=1
        )
        identity_score = torch.matmul(
            score_identity, score_items.t()
        )
        generated_score = torch.matmul(
            self._score_embedding(
                self._generated_preference(modes, identity), dim=-1
            ),
            score_items.t(),
        )
        return (
            self.prototype_score_weight * mixture_score
            + self.identity_score_weight * identity_score
            + self.generation_score_weight * generated_score
        )

    def _score_embedding(self, embedding, dim=-1):
        if self.normalize_scores:
            return F.normalize(embedding, dim=dim)
        return embedding

    def _hard_negative_score(
        self, positives, modes, identity
    ):
        pool_size = min(
            self.hard_negative_pool_size, positives.size(0)
        )
        pool_index = torch.randperm(
            positives.size(0), device=positives.device
        )[:pool_size]
        pool_items = positives[pool_index]
        pool_embedding = self.item_embedding(pool_items)
        scores = self._score_pool(pool_embedding, modes, identity)
        scores = scores.masked_fill(
            pool_items.unsqueeze(0).eq(positives.unsqueeze(1)),
            -1e9,
        )
        return scores.max(dim=1).values

    def _denoising_loss(self, positive_embedding, modes, identity):
        noise_level = torch.rand(
            positive_embedding.size(0),
            device=positive_embedding.device,
        ) * self.max_noise
        noisy = (
            positive_embedding
            + noise_level.unsqueeze(1)
            * torch.randn_like(positive_embedding)
        )
        mask = (
            torch.rand(
                positive_embedding.size(0), 1,
                device=positive_embedding.device,
            ) < self.denoise_mask_probability
        )
        noisy = torch.where(mask, torch.zeros_like(noisy), noisy)
        context = modes.mean(dim=1) + identity
        predicted = self.denoiser(torch.cat((
            noisy,
            context,
            self.noise_embedding(noise_level),
        ), dim=1))
        return F.mse_loss(predicted, positive_embedding)

    def _generated_preference(self, modes, identity):
        batch_size = identity.size(0)
        noise_level = identity.new_full(
            (batch_size,), self.max_noise
        )
        return self.denoiser(torch.cat((
            torch.zeros_like(identity),
            modes.mean(dim=1) + identity,
            self.noise_embedding(noise_level),
        ), dim=1))

    def _alignment_loss(self, local, transferred):
        local = F.normalize(local, dim=-1)
        transferred = F.normalize(transferred, dim=-1)
        positive = (local * transferred).sum(dim=2)
        negative = torch.matmul(
            local,
            transferred.transpose(1, 2),
        )
        labels = torch.arange(
            self.num_prototypes, device=local.device
        ).expand(local.size(0), -1)
        return F.cross_entropy(
            negative.reshape(-1, self.num_prototypes),
            labels.reshape(-1),
        ) - positive.mean()

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        transferred, local, _, identity = self._distribution_state(
            users, domain, positive_for_history=positives
        )
        positive_embedding = self.item_embedding(positives)
        negative_embedding = self.item_embedding(negatives)
        positive_score = self._score(
            positive_embedding, transferred, identity
        )
        negative_score = self._score(
            negative_embedding, transferred, identity
        )
        local_positive = self._score(
            positive_embedding, local, identity
        )
        local_negative = self._score(
            negative_embedding, local, identity
        )
        safe_loss = F.relu(
            (local_positive - local_negative).detach()
            - (positive_score - negative_score)
        ).mean()
        if self.current_epoch >= self.hard_negative_warmup_epochs:
            hard_negative_loss = self.bpr_loss(
                positive_score,
                self._hard_negative_score(
                    positives, transferred, identity
                ),
            )
        else:
            hard_negative_loss = positive_score.new_zeros(())
        regularization = (
            identity.pow(2).mean()
            + positive_embedding.pow(2).mean()
            + negative_embedding.pow(2).mean()
        )
        return (
            self.bpr_loss(positive_score, negative_score)
            + self.hard_negative_weight * hard_negative_loss
            + self.local_loss_weight
            * self.bpr_loss(local_positive, local_negative)
            + self.safe_transfer_weight * safe_loss
            + self.denoise_weight
            * self._denoising_loss(
                positive_embedding, transferred, identity
            )
            + (
                self.alignment_weight
                if self.current_epoch >= self.transfer_warmup_epochs
                else 0.0
            )
            * self._alignment_loss(local, transferred)
            + self.embedding_reg_weight * regularization
        )

    def _diversity_loss(self):
        query = F.normalize(self.prototype_query, dim=-1)
        gram = torch.matmul(query, query.transpose(1, 2))
        identity = torch.eye(
            self.num_prototypes, device=query.device
        ).unsqueeze(0)
        return (gram - identity).pow(2).mean()

    def calculate_loss(self, interaction):
        source_loss = self._domain_loss(interaction, 0)
        target_loss = self._domain_loss(interaction, 1)
        return (
            self.source_loss_weight * source_loss
            + (1 - self.source_loss_weight) * target_loss
            + self.diversity_weight * self._diversity_loss()
        )

    def set_train_epoch(self, epoch):
        self.current_epoch = epoch

    def predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        unique_users, inverse = torch.unique(
            users, sorted=False, return_inverse=True
        )
        modes, _, _, identity = self._distribution_state(
            unique_users, 1
        )
        return self._score(
            self.item_embedding(items),
            modes[inverse],
            identity[inverse],
        )

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        modes, _, _, identity = self._distribution_state(users, 1)
        items = self.item_embedding.weight[:self.target_num_items]
        score_items = self._score_embedding(items)
        score_modes = self._score_embedding(modes)
        score_identity = self._score_embedding(identity)
        mode_score = torch.einsum(
            'bkd,id->bki', score_modes, score_items
        )
        mixture = self.score_temperature * torch.logsumexp(
            mode_score / self.score_temperature, dim=1
        )
        identity_score = torch.matmul(
            score_identity, score_items.t()
        )
        generated_score = torch.matmul(
            self._score_embedding(
                self._generated_preference(modes, identity), dim=-1
            ),
            score_items.t(),
        )
        return (
            self.prototype_score_weight * mixture
            + self.identity_score_weight * identity_score
            + self.generation_score_weight * generated_score
        ).reshape(-1)
