from __future__ import annotations
"""Liquid Time-Constant cell (PyTorch) — Hasani et al. MIT CSAIL."""

import torch
import torch.nn as nn
import math

import config


class LTCCell(nn.Module):
    """
    Liquid Time-Constant cell.

    dx/dt = -[1/τ + f(x, I, θ)] · x + f(x, I, θ) · A + bias

    The time constant τ varies with both input and hidden state:
    τ_sys = τ / (1 + τ · f(x, I, θ))

    This means the network dynamically adjusts its processing speed.
    """

    def __init__(self, input_size: int = config.LTC_INPUT_SIZE,
                 hidden_size: int = config.LTC_HIDDEN_SIZE):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Learnable base time constant
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

        # Input-dependent modulation: f(x, h, θ)
        self.modulation = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Target activation
        self.target = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                time_delta: torch.Tensor) -> torch.Tensor:
        """
        Forward step.
        x: (batch, input_size)
        h: (batch, hidden_size) — previous hidden state
        time_delta: (batch, 1) — time since last step

        Returns: new hidden state (batch, hidden_size)
        """
        combined = torch.cat([x, h], dim=-1)

        # f(x, h, θ) — modulation signal
        f = self.modulation(combined)

        # A — target activation
        a = self.target(combined)

        # τ — base time constant
        tau = torch.exp(self.log_tau).unsqueeze(0)  # (1, hidden_size)

        # τ_sys = τ / (1 + τ · |f|) — effective time constant
        tau_sys = tau / (1.0 + tau * torch.abs(f))

        # Euler step: h_new = h + dt * [-h/τ_sys + f*A + bias]
        # Simplified: h_new = h * exp(-dt/τ_sys) + (1 - exp(-dt/τ_sys)) * (A + bias)
        dt = time_delta.unsqueeze(-1) if time_delta.dim() == 1 else time_delta
        decay = torch.exp(-dt / (tau_sys + 1e-6))
        h_new = decay * h + (1.0 - decay) * (a + self.bias)

        return h_new

    def initial_state(self, batch_size: int) -> torch.Tensor:
        device = self.log_tau.device
        return torch.zeros(batch_size, self.hidden_size, device=device)

    @property
    def effective_time_constants(self) -> dict:
        """Get statistics about learned time constants."""
        tau = torch.exp(self.log_tau).detach()
        return {
            "min": round(float(tau.min()), 3),
            "max": round(float(tau.max()), 3),
            "mean": round(float(tau.mean()), 3),
        }
