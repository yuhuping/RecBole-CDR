# -*- coding: utf-8 -*-

r"""Domain-aware direct user-conditioned diffusion."""

import torch.nn as nn

from recbole_cdr.model.cross_domain_recommender.ducdrlite import DUCDRLite


class DADUCDR(DUCDRLite):
    r"""Add a small domain-specific residual to DUCDR's user condition."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.adapter_scale = config["adapter_scale"]
        self.domain_adapter = nn.ModuleList([
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
        ])
        for adapter in self.domain_adapter:
            nn.init.zeros_(adapter.weight)

    def _user_representation(self, users, domain, positive=None):
        user_embedding = self.user_emb(users)
        return (
            user_embedding
            + self.adapter_scale * self.domain_adapter[domain](user_embedding)
        )
