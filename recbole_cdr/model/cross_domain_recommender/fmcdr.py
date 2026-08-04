# -*- coding: utf-8 -*-

r"""Flow matching item-space generation for cross-domain recommendation."""

import torch
import torch.nn.functional as F

from recbole_cdr.model.cross_domain_recommender.cdcdr import CDCDR


class FMCDR(CDCDR):
    r"""Replace diffusion denoising with flow matching.

    Training learns a velocity field from Gaussian noise to the positive item
    embedding under a direct user condition. Inference uses Euler integration
    to generate an item-like vector, then ranks candidates by item-space dot
    product.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.source_loss_weight is None:
            self.source_loss_weight = config["source_domain"].get(
                "loss_weight", 0.2
            )
        self.flow_loss_weight = config["flow_loss_weight"]
        self.flow_steps = config["flow_steps"]

    def _user_representation(self, users, domain, positive=None):
        return self.user_emb(users)

    def _diffusion_training_loss(self, item_embedding, condition):
        noise = torch.randn_like(item_embedding)
        t = torch.rand(item_embedding.size(0), 1, device=item_embedding.device)
        mixed = (1 - t) * noise + t * item_embedding
        target_velocity = item_embedding - noise
        timesteps = (
            t.squeeze(1) * (self.timesteps - 1)
        ).long().clamp(0, self.timesteps - 1)
        prediction = self._denoise_conditioned(mixed, condition, timesteps)
        return self.flow_loss_weight * F.mse_loss(
            prediction, target_velocity
        )

    @torch.no_grad()
    def _sample_item_embedding(self, condition):
        sample = torch.randn_like(condition)
        for step in range(self.flow_steps):
            t = step / max(self.flow_steps - 1, 1)
            timesteps = torch.full(
                (condition.size(0),),
                int(t * (self.timesteps - 1)),
                device=condition.device,
                dtype=torch.long,
            )
            velocity = self._denoise_conditioned(sample, condition, timesteps)
            sample = sample + velocity / self.flow_steps
        return sample
