from __future__ import annotations
"""Random Network Distillation for novelty detection."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import config


class RNDNetwork(nn.Module):
    """Small MLP used for both target and predictor networks."""

    def __init__(self, input_dim: int = config.WORLD_MODEL_OBS_DIM,
                 hidden_dim: int = config.RND_EMBEDDING_DIM,
                 output_dim: int = config.RND_EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule:
    """
    Random Network Distillation module.

    Fixed random target network f (never trained).
    Predictor network f_hat (trained to match f on seen states).
    High ||f_hat(s) - f(s)||^2 = novel state.
    """

    def __init__(self):
        self.target = RNDNetwork().to(config.DEVICE)
        self.predictor = RNDNetwork().to(config.DEVICE)

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-4)
        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 0

    def compute_novelty(self, state_embedding: np.ndarray) -> float:
        """
        Compute novelty score for a state embedding.
        Returns normalized novelty (0-1 scale).
        """
        with torch.no_grad():
            x = torch.tensor(state_embedding, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
            target_out = self.target(x)
            predictor_out = self.predictor(x)
            error = float(torch.mean((predictor_out - target_out) ** 2).item())

        # Update running statistics for normalization
        self._count += 1
        delta = error - self._running_mean
        self._running_mean += delta / self._count
        self._running_var += delta * (error - self._running_mean)

        # Normalize
        std = max(1e-8, (self._running_var / max(1, self._count)) ** 0.5)
        normalized = (error - self._running_mean) / std
        # Clip and sigmoid to 0-1
        return float(1.0 / (1.0 + np.exp(-np.clip(normalized, -5, 5))))

    def train_on_state(self, state_embedding: np.ndarray):
        """Train the predictor on a seen state (makes it less novel next time)."""
        x = torch.tensor(state_embedding, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            target_out = self.target(x)

        predictor_out = self.predictor(x)
        loss = torch.mean((predictor_out - target_out) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_on_batch(self, state_embeddings: np.ndarray):
        """Train the predictor on a batch of seen states."""
        x = torch.tensor(state_embeddings, dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            target_out = self.target(x)

        predictor_out = self.predictor(x)
        loss = torch.mean((predictor_out - target_out) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
