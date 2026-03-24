from __future__ import annotations
"""RSSM World Model adapted from DreamerV3 for text-based actions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from world_model.encoder import ObservationEncoder, ActionEncoder
from world_model.decoder import OutcomePredictor, SuccessPredictor


class LatentDistribution(nn.Module):
    """Produces parameters for a diagonal Gaussian latent distribution."""

    def __init__(self, input_dim: int, latent_dim: int = config.WORLD_MODEL_LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
        )
        self.mean = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, logvar) each of shape (batch, latent_dim)."""
        h = self.net(x)
        return self.mean(h), self.logvar(h).clamp(-10, 2)


class RSSMWorldModel(nn.Module):
    """
    Recurrent State-Space Model for text-based action prediction.

    Architecture:
    - GRU recurrent core: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    - Prior (dynamics predictor): p(z|h) - predicts latent from hidden state alone
    - Posterior (encoder): q(z|h, obs) - uses both hidden state and observation
    - Outcome predictor: predicts outcome embedding from (h, z)
    - Success predictor: predicts P(success) from (h, z)
    """

    def __init__(self):
        super().__init__()
        hidden_dim = config.WORLD_MODEL_HIDDEN_DIM
        latent_dim = config.WORLD_MODEL_LATENT_DIM
        obs_dim = config.WORLD_MODEL_OBS_DIM
        action_embed_dim = 64

        # Encoders
        self.obs_encoder = ObservationEncoder(obs_dim, hidden_dim)
        self.action_encoder = ActionEncoder(config.WORLD_MODEL_ACTION_DIM, action_embed_dim)

        # GRU recurrent core
        self.gru = nn.GRUCell(latent_dim + action_embed_dim, hidden_dim)

        # Prior: p(z | h_t) - dynamics predictor
        self.prior = LatentDistribution(hidden_dim, latent_dim)

        # Posterior: q(z | h_t, obs_t) - uses observation
        self.posterior = LatentDistribution(hidden_dim + hidden_dim, latent_dim)

        # Decoders
        self.outcome_predictor = OutcomePredictor(hidden_dim, latent_dim, obs_dim)
        self.success_predictor = SuccessPredictor(hidden_dim, latent_dim)

        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim

    def initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns initial (hidden_state, latent_state)."""
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self._hidden_dim, device=device)
        z = torch.zeros(batch_size, self._latent_dim, device=device)
        return h, z

    def observe_step(self, h_prev: torch.Tensor, z_prev: torch.Tensor,
                     action: torch.Tensor, obs_embedding: torch.Tensor
                     ) -> dict:
        """
        Single step with observation (for training / updating).

        Returns dict with: h, prior_mean, prior_logvar, posterior_mean, posterior_logvar,
                          z_posterior, predicted_outcome, predicted_success
        """
        # Encode action
        a_enc = self.action_encoder(action)

        # GRU step: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        gru_input = torch.cat([z_prev, a_enc], dim=-1)
        h = self.gru(gru_input, h_prev)

        # Prior: p(z | h)
        prior_mean, prior_logvar = self.prior(h)

        # Encode observation
        obs_enc = self.obs_encoder(obs_embedding)

        # Posterior: q(z | h, obs)
        post_input = torch.cat([h, obs_enc], dim=-1)
        post_mean, post_logvar = self.posterior(post_input)

        # Sample z from posterior (reparameterization trick)
        z = self._sample(post_mean, post_logvar)

        # Predict outcome and success
        predicted_outcome = self.outcome_predictor(h, z)
        predicted_success = self.success_predictor(h, z)

        return {
            "h": h,
            "z": z,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "posterior_mean": post_mean,
            "posterior_logvar": post_logvar,
            "predicted_outcome": predicted_outcome,
            "predicted_success": predicted_success,
        }

    def imagine_step(self, h_prev: torch.Tensor, z_prev: torch.Tensor,
                     action: torch.Tensor) -> dict:
        """
        Single step without observation (for prediction / imagination).
        Uses the prior distribution instead of posterior.
        """
        a_enc = self.action_encoder(action)
        gru_input = torch.cat([z_prev, a_enc], dim=-1)
        h = self.gru(gru_input, h_prev)

        prior_mean, prior_logvar = self.prior(h)
        z = self._sample(prior_mean, prior_logvar)

        predicted_outcome = self.outcome_predictor(h, z)
        predicted_success = self.success_predictor(h, z)

        return {
            "h": h,
            "z": z,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "predicted_outcome": predicted_outcome,
            "predicted_success": predicted_success,
        }

    def _sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for Gaussian sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    @staticmethod
    def kl_divergence(post_mean: torch.Tensor, post_logvar: torch.Tensor,
                      prior_mean: torch.Tensor, prior_logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence between two diagonal Gaussians."""
        kl = 0.5 * (
            prior_logvar - post_logvar
            + (post_logvar.exp() + (post_mean - prior_mean).pow(2)) / prior_logvar.exp()
            - 1
        )
        return kl.sum(dim=-1).mean()

    def compute_loss(self, obs_embeddings: torch.Tensor, actions: torch.Tensor,
                     outcome_embeddings: torch.Tensor, successes: torch.Tensor
                     ) -> dict:
        """
        Compute ELBO loss over a sequence.

        obs_embeddings: (batch, seq_len, obs_dim)
        actions: (batch, seq_len) int
        outcome_embeddings: (batch, seq_len, obs_dim)
        successes: (batch, seq_len) float
        """
        batch_size, seq_len = actions.shape
        h, z = self.initial_state(batch_size)

        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_success_loss = 0.0

        for t in range(seq_len):
            result = self.observe_step(h, z, actions[:, t], obs_embeddings[:, t])
            h = result["h"]
            z = result["z"]

            # Reconstruction loss: MSE between predicted and actual outcome
            recon_loss = F.mse_loss(result["predicted_outcome"], outcome_embeddings[:, t])
            total_recon_loss += recon_loss

            # KL divergence between posterior and prior
            kl_loss = self.kl_divergence(
                result["posterior_mean"], result["posterior_logvar"],
                result["prior_mean"], result["prior_logvar"]
            )
            total_kl_loss += kl_loss

            # Success prediction loss (binary cross-entropy)
            success_loss = F.binary_cross_entropy(
                result["predicted_success"].squeeze(-1), successes[:, t]
            )
            total_success_loss += success_loss

        # Average over sequence length
        total_recon_loss /= seq_len
        total_kl_loss /= seq_len
        total_success_loss /= seq_len

        # ELBO loss = reconstruction + beta * KL
        loss = total_recon_loss + config.WORLD_MODEL_KL_BETA * total_kl_loss + total_success_loss

        return {
            "loss": loss,
            "recon_loss": total_recon_loss,
            "kl_loss": total_kl_loss,
            "success_loss": total_success_loss,
        }
