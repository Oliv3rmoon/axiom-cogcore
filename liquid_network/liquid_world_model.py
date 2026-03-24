from __future__ import annotations
"""LTC/CfC-based world model — augments the existing RSSM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from liquid_network.cfc_layer import CfCCell


class LiquidWorldModel(nn.Module):
    """
    World model using Closed-form Continuous-time dynamics.

    Replaces the GRU in the RSSM with a CfC cell that adapts its
    time constants based on input — fast for rapid changes, slow for
    stable patterns.

    Architecture:
    - CfC cell: h_t = CfC(obs_embedding, h_{t-1}, Δt)
    - Outcome predictor: obs_dim from hidden state
    - Success predictor: P(success) from hidden state
    """

    def __init__(self, input_size: int = config.LTC_INPUT_SIZE,
                 hidden_size: int = config.LTC_HIDDEN_SIZE):
        super().__init__()
        self.cfc = CfCCell(input_size, hidden_size)
        self.outcome_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
        self.success_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        self._hidden_size = hidden_size
        self._input_size = input_size
        self._last_tau_mean = 1.0

    def forward(self, obs: torch.Tensor, h: torch.Tensor,
                time_delta: torch.Tensor) -> dict:
        """
        Single step prediction.
        obs: (batch, input_size) — observation embedding
        h: (batch, hidden_size)
        time_delta: (batch, 1)
        """
        h_new = self.cfc(obs, h, time_delta)
        predicted_outcome = self.outcome_predictor(h_new)
        predicted_success = self.success_predictor(h_new)

        return {
            "h": h_new,
            "predicted_outcome": predicted_outcome,
            "predicted_success": predicted_success,
        }

    def compute_loss(self, obs_sequence: torch.Tensor,
                     outcome_sequence: torch.Tensor,
                     time_deltas: torch.Tensor) -> dict:
        """
        Compute loss over a sequence.
        obs_sequence: (batch, seq_len, input_size)
        outcome_sequence: (batch, seq_len, input_size)
        time_deltas: (batch, seq_len, 1) or (batch, seq_len)
        """
        batch_size, seq_len, _ = obs_sequence.shape
        h = self.initial_state(batch_size)

        total_recon = 0.0
        total_success = 0.0

        for t in range(seq_len):
            dt = time_deltas[:, t]
            if dt.dim() == 1:
                dt = dt.unsqueeze(-1)
            result = self.forward(obs_sequence[:, t], h, dt)
            h = result["h"]
            total_recon += F.mse_loss(result["predicted_outcome"], outcome_sequence[:, t])

        total_recon /= seq_len
        return {"loss": total_recon, "recon_loss": total_recon}

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.cfc.initial_state(batch_size)

    def get_status(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "model_type": "CfC",
            "hidden_size": self._hidden_size,
            "input_size": self._input_size,
            "total_parameters": total_params,
            "device": str(next(self.parameters()).device),
        }
