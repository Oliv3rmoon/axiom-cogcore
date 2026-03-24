from __future__ import annotations
"""Closed-form Continuous-time layer — no ODE solver needed, GPU-efficient."""

import torch
import torch.nn as nn

import config


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time cell (MIT CSAIL).

    h_new = (1 - gate) * h_old + gate * candidate

    where gate and candidate are MLPs taking [x, h, Δt] as input.
    The time delta Δt allows adaptive dynamics:
    - Short Δt → fast dynamics (minor updates)
    - Long Δt → slow dynamics (major state changes)
    """

    def __init__(self, input_size: int = config.LTC_INPUT_SIZE,
                 hidden_size: int = config.LTC_HIDDEN_SIZE):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        combined_size = input_size + hidden_size + 1  # +1 for time delta

        self.gate_net = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.candidate_net = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                time_delta: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_size)
        h: (batch, hidden_size)
        time_delta: (batch, 1)
        Returns: h_new (batch, hidden_size)
        """
        if time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(-1)

        combined = torch.cat([x, h, time_delta], dim=-1)
        gate = self.gate_net(combined)
        candidate = self.candidate_net(combined)
        h_new = (1 - gate) * h + gate * candidate
        return h_new

    def initial_state(self, batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device)
