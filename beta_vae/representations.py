from __future__ import annotations
"""Extract and query disentangled representations from the β-VAE."""

import torch
import numpy as np

import config
from beta_vae.model import BetaVAE
from shared.embeddings import EmbeddingService


class RepresentationEngine:
    """Uses a trained β-VAE to produce and compare disentangled representations."""

    # Labels for latent dimensions (assigned post-hoc after training)
    DIMENSION_LABELS = [
        "action_type", "success", "domain", "complexity", "confidence",
        "novelty", "risk", "effort", "impact", "urgency",
        "dependency", "abstraction_level", "temporal", "social",
        "technical_depth", "creativity", "precision_required", "scope",
        "reversibility", "learning_value", "resource_cost", "priority",
        "context_sensitivity", "formality", "autonomy", "collaboration",
        "iteration_count", "error_prone", "documentation", "testing",
        "deployment", "monitoring",
    ]

    def __init__(self, model: BetaVAE, embedder: EmbeddingService):
        self.model = model
        self.embedder = embedder

    def encode_text(self, text: str) -> dict:
        """
        Encode text into disentangled latent vector.
        Returns latent vector and interpreted dimensions.
        """
        emb = self.embedder.embed(text)
        emb_tensor = torch.tensor(emb, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encode(emb_tensor)
            z = mu.squeeze(0)  # Use mean (no sampling) for deterministic encoding

        latent = z.cpu().numpy().tolist()

        # Map latent dimensions to labels with magnitudes
        dimensions = {}
        for i, val in enumerate(latent):
            if i < len(self.DIMENSION_LABELS):
                label = self.DIMENSION_LABELS[i]
            else:
                label = f"dim_{i}"
            dimensions[label] = round(val, 3)

        return {
            "latent_vector": [round(v, 4) for v in latent],
            "dimensions": dimensions,
        }

    def compute_similarity(self, text_a: str, text_b: str) -> dict:
        """Compare two texts in disentangled latent space."""
        enc_a = self.encode_text(text_a)
        enc_b = self.encode_text(text_b)

        z_a = np.array(enc_a["latent_vector"])
        z_b = np.array(enc_b["latent_vector"])

        # Cosine similarity in latent space
        sim = float(np.dot(z_a, z_b) / (np.linalg.norm(z_a) * np.linalg.norm(z_b) + 1e-8))

        # Identify shared vs different factors (threshold per dimension)
        shared_factors = []
        different_factors = []
        for i, label in enumerate(self.DIMENSION_LABELS[:len(z_a)]):
            diff = abs(z_a[i] - z_b[i])
            if diff < 0.5:
                shared_factors.append(label)
            else:
                different_factors.append(label)

        return {
            "similarity": round(sim, 3),
            "shared_factors": shared_factors[:10],
            "different_factors": different_factors[:10],
        }

    def generate_from_modification(self, base_text: str, modifications: dict) -> dict:
        """
        Modify specific latent dimensions and decode.
        Returns the modified latent vector and a description.
        """
        enc = self.encode_text(base_text)
        z = np.array(enc["latent_vector"])

        # Apply modifications
        for dim_name, target_value in modifications.items():
            if dim_name in self.DIMENSION_LABELS:
                idx = self.DIMENSION_LABELS.index(dim_name)
                if idx < len(z):
                    z[idx] = target_value

        # Decode the modified latent
        z_tensor = torch.tensor(z, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            recon = self.model.decode(z_tensor)

        # Build description of what changed
        changed_dims = list(modifications.keys())
        generated = (
            f"Modified dimensions: {', '.join(changed_dims)}. "
            f"Latent vector shifted to encode new values."
        )

        return {
            "generated": generated,
            "modified_latent": [round(v, 4) for v in z.tolist()],
            "dimensions_changed": changed_dims,
        }
