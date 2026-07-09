# -*- coding: utf-8 -*-

r"""Target-only direct user-item matching baseline."""

from recbole_cdr.model.cross_domain_recommender.ducdrdirect import (
    DUCDRDirect,
)


class TargetOnlyDirect(DUCDRDirect):
    r"""Train DUCDRDirect with target-domain supervision only."""

    def calculate_loss(self, interaction):
        return self._domain_loss(interaction, 1)
