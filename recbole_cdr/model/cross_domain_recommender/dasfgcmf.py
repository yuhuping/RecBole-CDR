# -*- coding: utf-8 -*-

r"""Domain-discrepancy-aware source-filtered GCMF."""

import torch
import torch.nn.functional as F

from recbole_cdr.model.cross_domain_recommender.gducdrdirect import (
    GDUCDRDirect,
)


def _config_value(config, key, default=None):
    value = config[key]
    return default if value is None else value


def _source_config_value(config, key, default=None):
    value = config["source_domain"].get(key)
    if value is not None:
        return value
    return _config_value(config, "source_" + key, default)


class DASFGCMF(GDUCDRDirect):
    r"""GCMF with source-side transferable signal filtering.

    GCMF already reweights target-domain samples by their gradient-direction
    agreement with the moving target population direction. DASFGCMF keeps that
    target-side path and additionally downweights source-domain BPR samples
    whose pairwise update direction is poorly aligned with the user's target
    domain profile. This directly targets negative transfer when source and
    target interests are discrepant.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.source_filter_tau = _source_config_value(config, "filter_tau", 0.0)
        self.source_filter_temperature = _source_config_value(
            config, "filter_temperature", 0.1
        )
        self.source_filter_min_weight = _source_config_value(
            config, "filter_min_weight", 0.25
        )
        self.source_filter_history_len = _source_config_value(
            config, "filter_history_len", 50
        )
        self._register_target_history(dataset)

    def _register_target_history(self, dataset):
        target_dataset = dataset.target_domain_dataset
        users = target_dataset.inter_feat[self.TARGET_USER_ID].long()
        items = target_dataset.inter_feat[self.TARGET_ITEM_ID].long()
        history_len = int(self.source_filter_history_len)
        history = torch.zeros(self.total_num_users, history_len, dtype=torch.long)
        counts = torch.zeros(self.total_num_users, dtype=torch.long)
        for user, item in zip(users.tolist(), items.tolist()):
            count = int(counts[user].item())
            if count < history_len:
                history[user, count] = item
            else:
                history[user, :-1] = history[user, 1:].clone()
                history[user, -1] = item
            counts[user] += 1
        self.register_buffer("target_history_items", history)
        self.register_buffer(
            "target_history_counts", counts.clamp_max(history_len).float()
        )

    def _target_profile(self, users):
        histories = self.target_history_items[users]
        mask = histories.ne(0).float().unsqueeze(-1)
        history_embedding = self.item_emb(histories)
        profile = (history_embedding * mask).sum(dim=1)
        counts = self.target_history_counts[users].clamp_min(1.0).unsqueeze(1)
        return F.normalize(profile / counts, dim=1)

    def _source_transfer_weight(
        self, users, positive_embedding, negative_embedding
    ):
        source_direction = self._pair_direction(
            positive_embedding.detach(), negative_embedding.detach()
        )
        target_profile = self._target_profile(users).detach()
        alignment = (source_direction * target_profile).sum(dim=1)
        weight = torch.sigmoid(
            (alignment - self.source_filter_tau)
            / self.source_filter_temperature
        )
        weight = self.source_filter_min_weight + (
            1.0 - self.source_filter_min_weight
        ) * weight
        return weight.detach()

    def _domain_loss(self, interaction, domain):
        if domain == 0:
            users = interaction[self.SOURCE_USER_ID]
            positives = interaction[self.SOURCE_ITEM_ID]
            negatives = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positives = interaction[self.TARGET_ITEM_ID]
            negatives = interaction[self.TARGET_NEG_ITEM_ID]

        user_embedding = self.user_emb(users)
        positive_embedding = self.item_emb(positives)
        negative_embedding = self.item_emb(negatives)
        positive_score = self._score(user_embedding, positive_embedding)
        negative_score = self._score(user_embedding, negative_embedding)
        loss = -F.logsigmoid(positive_score - negative_score)
        if domain == 0:
            loss = loss * self._source_transfer_weight(
                users, positive_embedding, negative_embedding
            )
        else:
            loss = loss * self._target_direction_weight(
                positive_embedding, negative_embedding
            )
        return loss.mean()
