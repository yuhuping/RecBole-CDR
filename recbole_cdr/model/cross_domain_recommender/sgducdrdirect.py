# -*- coding: utf-8 -*-

r"""Source-gradient aligned DUCDRDirect."""

import torch
import torch.nn.functional as F

from recbole_cdr.model.cross_domain_recommender.ducdrdirect import (
    DUCDRDirect,
)


class SGDUCDRDirect(DUCDRDirect):
    r"""Filter source-domain BPR updates by target-profile alignment.

    Target-domain supervision is kept intact. Source-domain samples are
    downweighted when their pairwise update direction is not aligned with the
    user's current target-domain profile direction.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.sg_direction_tau = config["sg_direction_tau"]
        self.sg_direction_temperature = config["sg_direction_temperature"]
        self.sg_direction_min_weight = config["sg_direction_min_weight"]
        self._register_target_history(dataset)

    def _register_target_history(self, dataset):
        target_dataset = dataset.target_domain_dataset
        users = target_dataset.inter_feat[self.TARGET_USER_ID].long()
        items = target_dataset.inter_feat[self.TARGET_ITEM_ID].long()
        counts = torch.bincount(users, minlength=self.total_num_users)
        max_len = int(counts.max().item())
        history = torch.zeros(
            self.total_num_users, max_len, dtype=torch.long
        )
        cursor = torch.zeros(self.total_num_users, dtype=torch.long)
        for user, item in zip(users.tolist(), items.tolist()):
            offset = int(cursor[user].item())
            history[user, offset] = item
            cursor[user] += 1
        self.register_buffer("target_history_items", history)
        self.register_buffer("target_history_counts", counts.float())

    @staticmethod
    def _pair_direction(positive_embedding, negative_embedding):
        return F.normalize(positive_embedding - negative_embedding, dim=1)

    def _target_profile(self, users):
        histories = self.target_history_items[users]
        mask = histories.ne(0).float().unsqueeze(-1)
        history_embedding = self.item_emb(histories)
        profile = (history_embedding * mask).sum(dim=1)
        counts = self.target_history_counts[users].clamp_min(1.0).unsqueeze(1)
        return F.normalize(profile / counts, dim=1)

    def _source_direction_weight(
        self, users, positive_embedding, negative_embedding
    ):
        source_direction = self._pair_direction(
            positive_embedding.detach(), negative_embedding.detach()
        )
        target_profile = self._target_profile(users).detach()
        alignment = (source_direction * target_profile).sum(dim=1)
        weight = torch.sigmoid(
            (alignment - self.sg_direction_tau)
            / self.sg_direction_temperature
        )
        weight = self.sg_direction_min_weight + (
            1.0 - self.sg_direction_min_weight
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
            loss = loss * self._source_direction_weight(
                users, positive_embedding, negative_embedding
            )
        return loss.mean()
