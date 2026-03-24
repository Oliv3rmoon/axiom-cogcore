from __future__ import annotations
"""Text embedding service using sentence-transformers."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config


class EmbeddingService:
    """Singleton embedding service. Loads model once, reuses everywhere."""

    _instance = None
    _model = None

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if EmbeddingService._model is None:
            EmbeddingService._model = SentenceTransformer(
                config.EMBEDDING_MODEL, device=config.DEVICE
            )
        self.model = EmbeddingService._model
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns numpy array of shape (dim,)."""
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns numpy array of shape (n, dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)

    def embed_to_tensor(self, text: str) -> torch.Tensor:
        """Embed text and return as a PyTorch tensor on config.DEVICE."""
        arr = self.embed(text)
        return torch.tensor(arr, dtype=torch.float32, device=config.DEVICE)

    def embed_batch_to_tensor(self, texts: list[str]) -> torch.Tensor:
        """Embed batch and return as PyTorch tensor on config.DEVICE."""
        arr = self.embed_batch(texts)
        return torch.tensor(arr, dtype=torch.float32, device=config.DEVICE)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        a = self.embed(text_a)
        b = self.embed(text_b)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(cos)
