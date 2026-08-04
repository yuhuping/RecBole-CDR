# -*- coding: utf-8 -*-

r"""CD-CDR: Conditional Diffusion Cross-Domain Recommendation."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


def _extract(values, timesteps, shape):
    result = values.gather(0, timesteps)
    return result.reshape(timesteps.size(0), *((1,) * (len(shape) - 1)))


def _linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


def _cosine_beta_schedule(timesteps, offset=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alpha = torch.cos(((x / timesteps) + offset) / (1 + offset) * math.pi * 0.5) ** 2
    alpha = alpha / alpha[0]
    return torch.clamp(1 - alpha[1:] / alpha[:-1], 1e-4, 0.9999)


def _exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    return 1 - torch.exp(
        -beta_min / timesteps
        - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps)
    )


def _sqrt_beta_schedule(timesteps, max_beta=0.999):
    def alpha_bar(value):
        return 1 - np.sqrt(value + 1e-4)

    betas = []
    for index in range(timesteps):
        start = index / timesteps
        end = (index + 1) / timesteps
        betas.append(min(1 - alpha_bar(end) / alpha_bar(start), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def _build_history(dataset, history_len):
    domain_datasets = (
        dataset.source_domain_dataset,
        dataset.target_domain_dataset,
    )
    user_num = dataset.num_total_user
    histories = []
    lengths = []

    for domain_dataset in domain_datasets:
        users = domain_dataset.inter_feat[domain_dataset.uid_field].numpy()
        items = domain_dataset.inter_feat[domain_dataset.iid_field].numpy()
        timestamps = domain_dataset.inter_feat[domain_dataset.time_field].numpy()
        order = np.lexsort((timestamps, users))

        per_user = [[] for _ in range(user_num)]
        for index in order:
            per_user[int(users[index])].append(int(items[index]))

        history = torch.zeros((user_num, history_len), dtype=torch.long)
        length = torch.zeros(user_num, dtype=torch.long)
        for user, item_list in enumerate(per_user):
            selected = item_list[-history_len:]
            if selected:
                history[user, :len(selected)] = torch.tensor(selected, dtype=torch.long)
                length[user] = len(selected)
        histories.append(history)
        lengths.append(length)

    return torch.stack(histories), torch.stack(lengths)


def _remove_positive(positive, history, history_len):
    mask = history != positive.unsqueeze(1)
    removed = (~mask).sum(dim=1)
    return history * mask.long(), torch.clamp(history_len - removed, min=0)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device) * -scale
        )
        values = timesteps[:, None].float() * frequencies[None, :]
        return torch.cat((values.sin(), values.cos()), dim=-1)


class CDCDR(CrossDomainRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.gamma = config['gamma']
        self.aggregator = config['aggregator']
        self.history_length = config['history_len']
        self.loss_name = config['loss_n']
        self.timesteps = config['timestep']
        self.guidance_weight = config['uncon_w']
        self.unconditional_probability = config['uncon_p']
        self.ddim_stride = config['ddim_stride']
        self.source_loss_weight = config['source_loss_weight']
        self.dropout_probability = config['dropout']

        if self.aggregator not in {
            'mean', 'user_attention', 'self_attention', 'transformer'
        }:
            raise ValueError('Unsupported CDCDR aggregator: %s' % self.aggregator)
        if self.loss_name not in {'bpr', 'bce', 'mse'}:
            raise ValueError('Unsupported CDCDR loss: %s' % self.loss_name)

        history_items, history_lengths = _build_history(dataset, self.history_length)
        self.register_buffer('history_item_id', history_items)
        self.register_buffer('history_item_len', history_lengths)

        self.user_emb = nn.Embedding(self.total_num_users, self.embedding_size)
        self.item_emb = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.none_embedding = nn.Embedding(1, self.embedding_size)

        self.layer_norm_1 = (
            nn.LayerNorm(self.embedding_size) if config['layer_norm'] else nn.Identity()
        )
        self.embedding_dropout = nn.Dropout(self.dropout_probability)

        if self.aggregator == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_size,
                nhead=config['n_heads'],
                dim_feedforward=self.embedding_size,
                dropout=self.dropout_probability,
                activation='gelu',
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config['n_layers']
            )
            self.mean_pooling = config['mean_pooling']
            self.ui_map = nn.Identity()
        else:
            self.ui_map = nn.Linear(
                self.embedding_size, self.embedding_size, bias=False
            )
            if self.aggregator in {'user_attention', 'self_attention'}:
                self.attention_key = nn.Sequential(
                    nn.Linear(self.embedding_size, self.embedding_size),
                    nn.Tanh(),
                )
            if self.aggregator == 'self_attention':
                self.attention_query = nn.Linear(
                    self.embedding_size, 1, bias=False
                )

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, self.embedding_size),
            nn.GELU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        if config['diffuser_type'] == 'mlp1':
            self.diffusion_mlp = nn.Linear(
                self.embedding_size * 3, self.embedding_size
            )
        elif config['diffuser_type'] == 'mlp2':
            self.diffusion_mlp = nn.Sequential(
                nn.Linear(self.embedding_size * 3, self.embedding_size * 2),
                nn.GELU(),
                nn.Linear(self.embedding_size * 2, self.embedding_size),
            )
        else:
            raise ValueError('Unsupported diffuser_type: %s' % config['diffuser_type'])

        self.bpr_loss = BPRLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        beta_schedule = config['beta_sche']
        if beta_schedule == 'linear':
            betas = _linear_beta_schedule(self.timesteps)
        elif beta_schedule == 'cosine':
            betas = _cosine_beta_schedule(self.timesteps)
        elif beta_schedule == 'exp':
            betas = _exp_beta_schedule(self.timesteps)
        elif beta_schedule == 'sqrt':
            betas = _sqrt_beta_schedule(self.timesteps)
        else:
            raise ValueError('Unsupported beta schedule: %s' % beta_schedule)
        self._register_diffusion_schedule(betas)

        self.apply(xavier_normal_initialization)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def _register_diffusion_schedule(self, betas):
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer(
            'sqrt_one_minus_alpha_cumprod',
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
        self.register_buffer('ddim_steps', ddim_steps)
        self.register_buffer('ddim_coefficient_1', coefficient_1)
        self.register_buffer('ddim_coefficient_2', coefficient_2)

    def _aggregate_history(self, user_embedding, history_embedding, history_len):
        valid_mask = history_embedding.abs().sum(dim=-1) > 1e-8
        if self.aggregator == 'mean':
            pooled = history_embedding.sum(dim=1)
            pooled = pooled / history_len.clamp(min=1).unsqueeze(1)
        elif self.aggregator in {'user_attention', 'self_attention'}:
            normalized = self.layer_norm_1(history_embedding)
            key = self.attention_key(normalized)
            if self.aggregator == 'user_attention':
                attention = torch.bmm(
                    key, user_embedding.unsqueeze(2)
                ).squeeze(2)
            else:
                attention = self.attention_query(key).squeeze(2)
            attention = attention.masked_fill(~valid_mask, -1e9)
            weights = torch.softmax(attention, dim=1)
            weights = weights * valid_mask.float()
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
            pooled = torch.bmm(
                weights.unsqueeze(1), normalized
            ).squeeze(1)
        else:
            padding_mask = ~valid_mask
            fully_padded = padding_mask.all(dim=1)
            padding_mask = padding_mask.clone()
            padding_mask[fully_padded, 0] = False
            transformer_input = self.embedding_dropout(
                self.layer_norm_1(history_embedding)
            )
            output = self.transformer_encoder(
                transformer_input, src_key_padding_mask=padding_mask
            )
            if self.mean_pooling:
                output = output * valid_mask.unsqueeze(-1).float()
                pooled = output.sum(dim=1) / history_len.clamp(min=1).unsqueeze(1)
            else:
                last_position = history_len.clamp(min=1) - 1
                pooled = output[
                    torch.arange(output.size(0), device=output.device),
                    last_position,
                ]
                pooled[fully_padded] = 0

        pooled = self.ui_map(pooled)
        return self.gamma * user_embedding + (1 - self.gamma) * pooled

    def _user_representation(self, users, domain, positive=None):
        history = self.history_item_id[domain][users]
        history_len = self.history_item_len[domain][users]
        if positive is not None:
            history, history_len = _remove_positive(
                positive, history, history_len
            )
        return self._aggregate_history(
            self.user_emb(users),
            self.item_emb(history),
            history_len,
        )

    def _denoise_conditioned(self, noisy_embedding, condition, timesteps):
        time_embedding = self.step_mlp(timesteps)
        return self.diffusion_mlp(
            torch.cat((noisy_embedding, condition, time_embedding), dim=1)
        )

    def _denoise_unconditioned(self, noisy_embedding, timesteps):
        condition = self.none_embedding.weight.expand(noisy_embedding.size(0), -1)
        return self._denoise_conditioned(
            noisy_embedding, condition, timesteps
        )

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
            _extract(
                self.sqrt_alpha_cumprod, timesteps, item_embedding.shape
            ) * item_embedding
            + _extract(
                self.sqrt_one_minus_alpha_cumprod,
                timesteps,
                item_embedding.shape,
            ) * noise
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
            unconditioned = self._denoise_unconditioned(
                sample, timesteps
            )
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

        condition = self._user_representation(
            users, domain, positive=positives
        )
        positive_embedding = self.item_emb(positives)
        negative_embedding = self.item_emb(negatives)
        positive_score = (condition * positive_embedding).sum(dim=1)
        negative_score = (condition * negative_embedding).sum(dim=1)

        if self.loss_name == 'bpr':
            recommendation_loss = self.bpr_loss(
                positive_score, negative_score
            )
        elif self.loss_name == 'bce':
            recommendation_loss = (
                self.bce_loss(positive_score, torch.ones_like(positive_score))
                + self.bce_loss(
                    negative_score, torch.zeros_like(negative_score)
                )
            )
        else:
            recommendation_loss = self.mse_loss(
                condition, positive_embedding
            )

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
        return (
            generated[inverse] * self.item_emb(items)
        ).sum(dim=1)

    def full_sort_predict(self, interaction):
        users = interaction[self.TARGET_USER_ID]
        generated = self._target_samples(users)
        target_items = self.item_emb.weight[:self.target_num_items]
        return torch.matmul(generated, target_items.t()).view(-1)
