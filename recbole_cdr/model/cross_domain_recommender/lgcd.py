# -*- coding: utf-8 -*-

r"""Language-guided pseudo-target DUCDR.

This model keeps DUCDR's direct user-conditioned diffusion backbone and adds
a pseudo target-domain clue path inspired by LGCD. The default pseudo clues
are built from source-to-target co-occurrence in the training interactions:
users with similar source-domain histories vote for likely target-domain
items. Those pseudo target items are then aggregated and injected as a gated
residual into the diffusion condition.

The pseudo item buffer is intentionally built from train_data.dataset inside
model initialization, so validation/test positives are not used.
"""

from collections import defaultdict

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization

from recbole_cdr.model.cross_domain_recommender.ducdr import DUCDR


def _get_config(config, key, default):
    try:
        return config[key]
    except KeyError:
        return default


def _unique_nonzero(values):
    result = []
    seen = set()
    for value in values:
        item = int(value)
        if item == 0 or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


class LGCD(DUCDR):
    r"""DUCDR with source-guided pseudo target interaction clues."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.pseudo_topk = int(_get_config(config, "pseudo_topk", 10))
        self.pseudo_weight = float(_get_config(config, "pseudo_weight", 0.1))
        self.pseudo_min_count = int(_get_config(config, "pseudo_min_count", 1))
        self.pseudo_exclude_known_target = bool(
            _get_config(config, "pseudo_exclude_known_target", True)
        )
        self.pseudo_detach_item_embedding = bool(
            _get_config(config, "pseudo_detach_item_embedding", False)
        )

        pseudo_items, pseudo_lengths = self._build_pseudo_target_items()
        self.register_buffer("pseudo_target_item_id", pseudo_items)
        self.register_buffer("pseudo_target_item_len", pseudo_lengths)

        self.pseudo_layer_norm = nn.LayerNorm(self.embedding_size)
        self.pseudo_map = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.pseudo_gate = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.pseudo_map.apply(xavier_normal_initialization)
        self.pseudo_gate.apply(xavier_normal_initialization)

    def _build_pseudo_target_items(self):
        user_num = self.history_item_id.size(1)
        source_history = self.history_item_id[0].cpu()
        target_history = self.history_item_id[1].cpu()
        source_lengths = self.history_item_len[0].cpu()
        target_lengths = self.history_item_len[1].cpu()

        source_to_target = defaultdict(lambda: defaultdict(int))
        target_popularity = defaultdict(int)

        for user in range(user_num):
            source_len = int(source_lengths[user])
            target_len = int(target_lengths[user])
            if source_len == 0 or target_len == 0:
                continue
            source_items = _unique_nonzero(source_history[user, :source_len].tolist())
            target_items = _unique_nonzero(target_history[user, :target_len].tolist())
            if not source_items or not target_items:
                continue
            for target_item in target_items:
                target_popularity[target_item] += 1
            for source_item in source_items:
                item_counts = source_to_target[source_item]
                for target_item in target_items:
                    item_counts[target_item] += 1

        popular_targets = [
            item
            for item, _ in sorted(
                target_popularity.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ]

        pseudo_items = torch.zeros((user_num, self.pseudo_topk), dtype=torch.long)
        pseudo_lengths = torch.zeros(user_num, dtype=torch.long)

        for user in range(user_num):
            source_len = int(source_lengths[user])
            if source_len == 0:
                candidates = []
            else:
                scores = defaultdict(int)
                source_items = _unique_nonzero(source_history[user, :source_len].tolist())
                for source_item in source_items:
                    for target_item, count in source_to_target[source_item].items():
                        if count >= self.pseudo_min_count:
                            scores[target_item] += count
                candidates = [
                    item
                    for item, _ in sorted(
                        scores.items(), key=lambda pair: (-pair[1], pair[0])
                    )
                ]

            if self.pseudo_exclude_known_target:
                target_len = int(target_lengths[user])
                known_target = set(
                    _unique_nonzero(target_history[user, :target_len].tolist())
                )
                candidates = [item for item in candidates if item not in known_target]
            else:
                known_target = set()

            if len(candidates) < self.pseudo_topk:
                for item in popular_targets:
                    if item in candidates or item in known_target:
                        continue
                    candidates.append(item)
                    if len(candidates) >= self.pseudo_topk:
                        break

            selected = candidates[:self.pseudo_topk]
            if selected:
                pseudo_items[user, :len(selected)] = torch.tensor(
                    selected, dtype=torch.long
                )
                pseudo_lengths[user] = len(selected)

        return pseudo_items, pseudo_lengths

    def _pseudo_target_representation(self, users):
        pseudo_items = self.pseudo_target_item_id[users]
        pseudo_lengths = self.pseudo_target_item_len[users]
        pseudo_embedding = self.item_emb(pseudo_items)
        if self.pseudo_detach_item_embedding:
            pseudo_embedding = pseudo_embedding.detach()

        mask = pseudo_items.ne(0).float().unsqueeze(-1)
        pooled = (pseudo_embedding * mask).sum(dim=1)
        pooled = pooled / pseudo_lengths.clamp(min=1).unsqueeze(1)
        pooled = self.pseudo_layer_norm(pooled)
        return self.pseudo_map(pooled)

    def _user_representation(self, users, domain, positive=None):
        base = self.user_emb(users)
        if domain != 1 or self.pseudo_weight <= 0:
            return base

        pseudo = self._pseudo_target_representation(users)
        gate = torch.sigmoid(self.pseudo_gate(torch.cat((base, pseudo), dim=1)))
        return base + self.pseudo_weight * gate * pseudo
