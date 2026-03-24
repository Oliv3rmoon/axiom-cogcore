from __future__ import annotations
"""β-VAE encoder/decoder for disentangled representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class BetaVAEEncoder(nn.Module):
    """Encoder: embedding_dim → hidden → (mu, logvar) of latent_dim."""

    def __init__(self, input_dim: int = config.HOPFIELD_PATTERN_DIM,
                 hidden_dim: int = config.BETA_VAE_HIDDEN_DIM,
                 latent_dim: int = config.BETA_VAE_LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h).clamp(-10, 2)


class BetaVAEDecoder(nn.Module):
    """Decoder: latent_dim → hidden → embedding_dim."""

    def __init__(self, latent_dim: int = config.BETA_VAE_LATENT_DIM,
                 hidden_dim: int = config.BETA_VAE_HIDDEN_DIM,
                 output_dim: int = config.HOPFIELD_PATTERN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BetaVAE(nn.Module):
    """
    β-VAE with controlled capacity for disentangled representations.

    Loss = reconstruction + β * |KL - C|

    When β > 1, the information bottleneck forces disentanglement.
    Capacity C is annealed from 0 to BETA_VAE_CAPACITY_MAX over training.
    """

    def __init__(self, input_dim: int = config.HOPFIELD_PATTERN_DIM):
        super().__init__()
        self.encoder = BetaVAEEncoder(input_dim)
        self.decoder = BetaVAEDecoder(output_dim=input_dim)
        self._input_dim = input_dim
        self._latent_dim = config.BETA_VAE_LATENT_DIM

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to (mu, logvar)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (reconstruction, input, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, x, mu, logvar

    @staticmethod
    def loss_function(x_recon: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor,
                      beta: float = config.BETA_VAE_BETA,
                      capacity: float = 0.0) -> dict:
        """
        Controlled-capacity β-VAE loss.

        L = reconstruction + β * |KL - C|
        """
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        total = recon_loss + beta * torch.abs(kl - capacity)

        return {
            "loss": total,
            "recon_loss": recon_loss,
            "kl": kl,
        }
