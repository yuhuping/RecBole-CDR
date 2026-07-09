# -*- coding: utf-8 -*-

r"""Target-only GCMF.

This model isolates whether GCMF's target-domain gradient-direction weighting
works as a general single-domain recommendation regularizer. It removes all
source-domain supervision and keeps only the target-domain BPR loss used by
GDUCDRDirect/GCMF.
"""

from recbole_cdr.model.cross_domain_recommender.gducdrdirect import (
    GDUCDRDirect,
)


class TargetOnlyGCMF(GDUCDRDirect):
    r"""Train GCMF with target-domain supervision only."""

    def calculate_loss(self, interaction):
        return self._domain_loss(interaction, 1)
