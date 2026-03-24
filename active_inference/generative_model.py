from __future__ import annotations
"""Generative model wrapper — P(observations, states) — wraps the RSSM world model."""

import torch
import torch.nn.functional as F
import numpy as np

import config
from world_model.model import RSSMWorldModel
from shared.embeddings import EmbeddingService


class GenerativeModel:
    """
    Wraps the Phase 1 RSSM world model and adds uncertainty estimation.
    This is the 'generative model' in active inference: P(o, s).
    """

    def __init__(self, world_model: RSSMWorldModel, embedder: EmbeddingService):
        self.world_model = world_model
        self.embedder = embedder

    def predict_with_uncertainty(self, state_text: str, action: str
                                 ) -> tuple[torch.Tensor, float, float]:
        """
        Predict outcome with uncertainty estimation.

        Returns:
            predicted_embedding: (obs_dim,) tensor
            uncertainty: float — entropy of the prior distribution (higher = more uncertain)
            predicted_success: float
        """
        state_emb = self.embedder.embed_to_tensor(state_text)
        action_idx = config.ACTION_TYPES.index(action) if action in config.ACTION_TYPES else 0
        action_tensor = torch.tensor([action_idx], dtype=torch.long, device=config.DEVICE)

        self.world_model.eval()
        with torch.no_grad():
            h, z = self.world_model.initial_state(1)
            result = self.world_model.imagine_step(h, z, action_tensor)

            predicted_emb = result["predicted_outcome"].squeeze(0)
            predicted_success = float(result["predicted_success"].squeeze().item())

            # Uncertainty = entropy of the prior distribution
            # For a diagonal Gaussian: H = 0.5 * sum(log(2πe) + logvar)
            prior_logvar = result["prior_logvar"].squeeze(0)
            entropy = 0.5 * torch.sum(1.0 + prior_logvar + np.log(2 * np.pi)).item()
            # Normalize to 0-1 range
            max_entropy = 0.5 * config.WORLD_MODEL_LATENT_DIM * (1.0 + 2.0 + np.log(2 * np.pi))
            uncertainty = min(1.0, max(0.0, entropy / max(1.0, max_entropy)))

        return predicted_emb, uncertainty, predicted_success

    def compute_goal_similarity(self, predicted_emb: torch.Tensor, goal_text: str) -> float:
        """Cosine similarity between predicted outcome and goal."""
        goal_emb = self.embedder.embed_to_tensor(goal_text)
        sim = F.cosine_similarity(predicted_emb.unsqueeze(0), goal_emb.unsqueeze(0), dim=-1)
        return float(sim.item())
