from __future__ import annotations
"""Decoders / predictors for the world model."""

import torch
import torch.nn as nn

import config


class OutcomePredictor(nn.Module):
    """Predicts outcome embedding from hidden state and latent state."""

    def __init__(self, hidden_dim: int = config.WORLD_MODEL_HIDDEN_DIM,
                 latent_dim: int = config.WORLD_MODEL_LATENT_DIM,
                 obs_dim: int = config.WORLD_MODEL_OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """h: (batch, hidden_dim), z: (batch, latent_dim) -> (batch, obs_dim)"""
        return self.net(torch.cat([h, z], dim=-1))


class SuccessPredictor(nn.Module):
    """Predicts probability of success from hidden state and latent state."""

    def __init__(self, hidden_dim: int = config.WORLD_MODEL_HIDDEN_DIM,
                 latent_dim: int = config.WORLD_MODEL_LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """h: (batch, hidden_dim), z: (batch, latent_dim) -> (batch, 1)"""
        return self.net(torch.cat([h, z], dim=-1))
