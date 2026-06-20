# -*- coding: utf-8 -*-

r"""Iterative Refinement Generator for cross-domain recommendation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class IRGCDR(CrossDomainRecommender):
    r"""Generate a set of item-space preferences without diffusion sampling.

    A shared user condition is adapted per domain, expanded into several
    preference particles, and refined recurrently by a shared residual block.
    Best-of-K reconstruction lets different particles cover different modes
    of a user's item distribution.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.adapter_size = config['adapter_size']
        self.num_particles = config['num_particles']
        self.refinement_steps = config['refinement_steps']
        self.score_temperature = config['score_temperature']
        self.source_loss_weight = config['source_loss_weight']
        self.contrastive_weight = config['contrastive_weight']
        self.reconstruction_weight = config['reconstruction_weight']
        self.safety_weight = config['safety_weight']
        self.balance_weight = config['balance_weight']
        self.diversity_weight = config['diversity_weight']
        self.embedding_reg_weight = config['embedding_reg_weight']
        self.gate_initial_bias = config['gate_initial_bias']

        self.shared_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.domain_user_embedding = nn.ModuleList([
            nn.Embedding(self.total_num_users, self.adapter_size),
            nn.Embedding(self.total_num_users, self.adapter_size),
        ])
        self.item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.domain_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.adapter_size, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, self.embedding_size),
            ),
            nn.Sequential(
                nn.Linear(self.adapter_size, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, self.embedding_size),
            ),
        ])
        self.residual_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.adapter_size),
                nn.GELU(),
                nn.Linear(self.adapter_size, 1),
            ),
            nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.adapter_size),
                nn.GELU(),
                nn.Linear(self.adapter_size, 1),
            ),
        ])
        self.domain_embedding = nn.Parameter(
            torch.empty(2, self.embedding_size)
        )
        self.particle_seed = nn.Parameter(
            torch.empty(self.num_particles, self.embedding_size)
        )
        self.refiner_norm = nn.LayerNorm(self.embedding_size)
        self.refiner = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size * 2),
            nn.GELU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
        )
        self.domain_film_scale = nn.Parameter(
            torch.zeros(2, self.embedding_size)
        )
        self.domain_film_bias = nn.Parameter(
            torch.zeros(2, self.embedding_size)
        )
        self.bpr_loss = BPRLoss()

        self.apply(xavier_normal_initialization)
        nn.init.xavier_normal_(self.domain_embedding)
        nn.init.xavier_normal_(self.particle_seed)
        for gate in self.residual_gate:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, self.gate_initial_bias)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def _condition(self, users, domain):
        shared = self.shared_user_embedding(users)
        specific = self.domain_adapter[domain](
            self.domain_user_embedding[domain](users)
        )
        context = shared + specific + self.domain_embedding[domain]
        gate = torch.sigmoid(
            self.residual_gate[domain](
                torch.cat((shared, specific), dim=1)
            )
        )
        return shared, context, gate

    def _generate_particles(self, condition, domain):
        particles = (
            condition.unsqueeze(1)
            + self.particle_seed.unsqueeze(0)
        )
        expanded_condition = condition.unsqueeze(1).expand(
            -1, self.num_particles, -1
        )
        scale = 1 + 0.1 * torch.tanh(
            self.domain_film_scale[domain]
        )
        bias = self.domain_film_bias[domain]
        for _ in range(self.refinement_steps):
            delta = self.refiner(torch.cat((
                self.refiner_norm(particles),
                expanded_condition,
            ), dim=2))
            particles = particles + (
                delta * scale + bias
            ) / math.sqrt(self.refinement_steps)
        return particles

    def _score(self, item_embedding, particles, shared, gate):
        particle_score = torch.einsum(
            'bd,bkd->bk', item_embedding, particles
        ) / math.sqrt(self.embedding_size)
        distribution_score = self.score_temperature * torch.logsumexp(
            particle_score / self.score_temperature, dim=1
        )
        identity_score = (
            shared * item_embedding
        ).sum(dim=1) / math.sqrt(self.embedding_size)
        return identity_score + gate.squeeze(1) * (
            distribution_score - identity_score
        )

    def _contrastive_loss(self, users, positive_embedding, particles):
        scores = torch.einsum(
            'bkd,nd->bkn', particles, positive_embedding
        ) / math.sqrt(self.embedding_size)
        scores = self.score_temperature * torch.logsumexp(
            scores / self.score_temperature, dim=1
        )
        log_probability = scores - torch.logsumexp(
            scores, dim=1, keepdim=True
        )
        positive_mask = users.unsqueeze(1).eq(users.unsqueeze(0))
        positive_log_probability = log_probability.masked_fill(
            ~positive_mask, -torch.inf
        )
        return -torch.logsumexp(
            positive_log_probability, dim=1
        ).mean()

    def _distribution_losses(self, positive_embedding, particles):
        similarity = torch.einsum(
            'bd,bkd->bk',
            F.normalize(positive_embedding, dim=-1),
            F.normalize(particles, dim=-1),
        )
        reconstruction = (1 - similarity.max(dim=1).values).mean()

        assignment = torch.softmax(
            similarity / self.score_temperature, dim=1
        ).mean(dim=0)
        uniform = assignment.new_full(
            assignment.shape, 1 / self.num_particles
        )
        balance = F.kl_div(
            assignment.clamp_min(1e-10).log(),
            uniform,
            reduction='sum',
        )

        normalized = F.normalize(particles, dim=-1)
        gram = torch.matmul(normalized, normalized.transpose(1, 2))
        identity = torch.eye(
            self.num_particles, device=particles.device
        ).unsqueeze(0)
        diversity = (gram - identity).pow(2).mean()
        return reconstruction, balance, diversity

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        shared, context, gate = self._condition(users, domain)
        particles = self._generate_particles(context, domain)
        positive_embedding = self.item_embedding(positives)
        negative_embedding = self.item_embedding(negatives)
        positive_score = self._score(
            positive_embedding, particles, shared, gate
        )
        negative_score = self._score(
            negative_embedding, particles, shared, gate
        )
        local_positive_score = (
            shared * positive_embedding
        ).sum(dim=1) / math.sqrt(self.embedding_size)
        local_negative_score = (
            shared * negative_embedding
        ).sum(dim=1) / math.sqrt(self.embedding_size)
        safety = F.relu(
            (local_positive_score - local_negative_score).detach()
            - (positive_score - negative_score)
        ).mean()
        contrastive = self._contrastive_loss(
            users, positive_embedding, particles
        )
        reconstruction, balance, diversity = (
            self._distribution_losses(
                positive_embedding, particles
            )
        )
        regularization = (
            shared.pow(2).mean()
            + context.pow(2).mean()
            + positive_embedding.pow(2).mean()
            + negative_embedding.pow(2).mean()
        )
        return (
            self.bpr_loss(positive_score, negative_score)
            + self.contrastive_weight * contrastive
            + self.reconstruction_weight * reconstruction
            + self.safety_weight * safety
            + self.balance_weight * balance
            + self.diversity_weight * diversity
            + self.embedding_reg_weight * regularization
        )

    def calculate_loss(self, interaction):
        source_loss = self._domain_loss(interaction, 0)
        target_loss = self._domain_loss(interaction, 1)
        return (
            self.source_loss_weight * source_loss
            + (1 - self.source_loss_weight) * target_loss
        )

    def predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        unique_users, inverse = torch.unique(
            users, sorted=False, return_inverse=True
        )
        shared, context, gate = self._condition(unique_users, 1)
        particles = self._generate_particles(context, 1)
        return self._score(
            self.item_embedding(items),
            particles[inverse],
            shared[inverse],
            gate[inverse],
        )

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        shared, context, gate = self._condition(users, 1)
        particles = self._generate_particles(context, 1)
        items = self.item_embedding.weight[:self.target_num_items]
        particle_score = torch.einsum(
            'bkd,id->bki', particles, items
        ) / math.sqrt(self.embedding_size)
        distribution_score = self.score_temperature * torch.logsumexp(
            particle_score / self.score_temperature, dim=1
        )
        identity_score = (
            torch.matmul(shared, items.t())
            / math.sqrt(self.embedding_size)
        )
        return (
            identity_score
            + gate * (distribution_score - identity_score)
        ).reshape(-1)
