from __future__ import annotations
"""Observation and action encoders for the world model."""

import torch
import torch.nn as nn

import config


class ObservationEncoder(nn.Module):
    """Encodes observation embeddings (from sentence-transformers) into the model's hidden dim."""

    def __init__(self, obs_dim: int = config.WORLD_MODEL_OBS_DIM,
                 hidden_dim: int = config.WORLD_MODEL_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

    def forward(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        """obs_embedding: (batch, obs_dim) -> (batch, hidden_dim)"""
        return self.net(obs_embedding)


class ActionEncoder(nn.Module):
    """Encodes action type (integer index) into a dense vector."""

    def __init__(self, num_actions: int = config.WORLD_MODEL_ACTION_DIM,
                 action_embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, action_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(action_embed_dim, action_embed_dim),
            nn.ELU(),
        )

    def forward(self, action_idx: torch.Tensor) -> torch.Tensor:
        """action_idx: (batch,) int -> (batch, action_embed_dim)"""
        return self.net(self.embedding(action_idx))
