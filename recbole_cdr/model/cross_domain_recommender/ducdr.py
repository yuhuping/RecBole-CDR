# -*- coding: utf-8 -*-

r"""Direct user-conditioned diffusion for cross-domain recommendation."""

from recbole_cdr.model.cross_domain_recommender.cdcdr import CDCDR


class DUCDR(CDCDR):
    r"""A history-free CD-CDR variant.

    CD-CDR mixes user ID and aggregated history as
    ``gamma * user_embedding + (1 - gamma) * history_embedding``. In the
    current setting ``gamma=0.999``, so the history branch contributes only a
    tiny residual while adding sequence modeling noise and computation. DUCDR
    keeps the effective backbone: direct user-conditioned diffusion item-space
    matching.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.source_loss_weight is None:
            self.source_loss_weight = config["source_domain"].get(
                "loss_weight", 0.2
            )

    def _user_representation(self, users, domain, positive=None):
        return self.user_emb(users)
