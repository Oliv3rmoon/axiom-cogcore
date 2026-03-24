from __future__ import annotations
"""Training loop for β-VAE with capacity annealing."""

import torch
import torch.optim as optim
import numpy as np

import config
from beta_vae.model import BetaVAE


class BetaVAETrainer:
    """Trains the β-VAE with gradual capacity annealing."""

    def __init__(self, model: BetaVAE):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=config.BETA_VAE_LR)
        self.train_step_count = 0
        self.total_encoded = 0
        self._capacity = 0.0
        self._capacity_step = config.BETA_VAE_CAPACITY_MAX / 10000.0  # Anneal over 10k steps
        self._loss_history: list[float] = []

    def train_step(self, batch: torch.Tensor) -> dict:
        """
        One training step on a batch of embeddings.
        batch: (N, embedding_dim) tensor on config.DEVICE
        """
        self.model.train()
        self.optimizer.zero_grad()

        x_recon, x, mu, logvar = self.model(batch)
        losses = BetaVAE.loss_function(
            x_recon, x, mu, logvar,
            beta=config.BETA_VAE_BETA,
            capacity=self._capacity,
        )

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.total_encoded += batch.shape[0]
        self._loss_history.append(losses["loss"].item())

        # Anneal capacity
        self._capacity = min(config.BETA_VAE_CAPACITY_MAX,
                             self._capacity + self._capacity_step)

        return {
            "loss": losses["loss"].item(),
            "recon_loss": losses["recon_loss"].item(),
            "kl": losses["kl"].item(),
            "capacity": self._capacity,
            "step": self.train_step_count,
        }

    def get_stats(self) -> dict:
        recent = self._loss_history[-100:] if self._loss_history else []
        return {
            "total_encoded": self.total_encoded,
            "latent_dim": config.BETA_VAE_LATENT_DIM,
            "beta": config.BETA_VAE_BETA,
            "capacity": round(self._capacity, 2),
            "reconstruction_loss": round(
                np.mean(recent) if recent else 0.0, 4
            ),
            "train_steps": self.train_step_count,
        }
