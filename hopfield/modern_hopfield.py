from __future__ import annotations
"""Modern Hopfield network with exponential capacity."""

import torch
import torch.nn.functional as F
import numpy as np

import config


class ModernHopfieldNetwork:
    """
    Modern Hopfield network with exponential storage capacity.

    Energy: E(ξ) = -lse(β, X^T ξ) + ½||ξ||²
    Retrieval: ξ_new = X · softmax(β · X^T · ξ)

    This IS the transformer attention mechanism with theoretical grounding.
    """

    def __init__(self, pattern_dim: int = config.HOPFIELD_PATTERN_DIM,
                 max_patterns: int = config.HOPFIELD_MAX_PATTERNS,
                 beta: float = config.HOPFIELD_BETA):
        self.pattern_dim = pattern_dim
        self.max_patterns = max_patterns
        self.beta = beta
        # Stored patterns matrix X: (N, D) — grows dynamically
        self._patterns: list[np.ndarray] = []

    @property
    def num_patterns(self) -> int:
        return len(self._patterns)

    def store(self, pattern: np.ndarray) -> int:
        """
        Store a new pattern. Returns index.
        Pattern should be shape (D,) with D = pattern_dim.
        """
        if pattern.shape[0] != self.pattern_dim:
            raise ValueError(f"Pattern dim {pattern.shape[0]} != expected {self.pattern_dim}")

        # Normalize
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm

        self._patterns.append(pattern.astype(np.float32))

        # If over capacity, consolidate
        if len(self._patterns) > self.max_patterns:
            self._consolidate()

        return len(self._patterns) - 1

    def retrieve(self, query: np.ndarray, top_k: int = 5) -> list[tuple[int, float, np.ndarray]]:
        """
        Content-based retrieval using the Hopfield update rule.

        ξ_new = X · softmax(β · X^T · ξ)

        Returns list of (index, similarity, pattern) tuples.
        """
        if not self._patterns:
            return []

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        X = np.array(self._patterns)  # (N, D)
        similarities = self.beta * X @ query  # (N,)

        # Softmax attention weights
        max_sim = np.max(similarities)
        exp_sim = np.exp(similarities - max_sim)
        attention = exp_sim / (np.sum(exp_sim) + 1e-8)  # (N,)

        # Get top-k by attention weight
        top_indices = np.argsort(attention)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(attention[idx]),
                self._patterns[idx].copy(),
            ))
        return results

    def retrieve_single(self, query: np.ndarray) -> np.ndarray:
        """
        Full Hopfield retrieval: ξ_new = X · softmax(β · X^T · ξ)
        Returns the retrieved pattern (weighted combination).
        """
        if not self._patterns:
            return np.zeros(self.pattern_dim, dtype=np.float32)

        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        X = np.array(self._patterns)  # (N, D)
        similarities = self.beta * X @ query  # (N,)

        max_sim = np.max(similarities)
        exp_sim = np.exp(similarities - max_sim)
        attention = exp_sim / (np.sum(exp_sim) + 1e-8)

        # ξ_new = X^T · attention = weighted combination of patterns
        retrieved = X.T @ attention  # (D,)
        return retrieved.astype(np.float32)

    def energy(self, xi: np.ndarray) -> float:
        """
        Compute energy: E(ξ) = -lse(β, X^T ξ) + ½||ξ||²
        """
        if not self._patterns:
            return 0.0

        X = np.array(self._patterns)
        similarities = self.beta * X @ xi
        lse = np.log(np.sum(np.exp(similarities - np.max(similarities)))) + np.max(similarities)
        lse /= self.beta
        return float(-lse + 0.5 * np.dot(xi, xi))

    def find_associations(self, pattern_idx: int, top_k: int = 5) -> list[tuple[int, float]]:
        """Find patterns most associated with the given pattern."""
        if pattern_idx >= len(self._patterns):
            return []

        query = self._patterns[pattern_idx]
        results = self.retrieve(query, top_k=top_k + 1)

        # Exclude self
        return [(idx, sim) for idx, sim, _ in results if idx != pattern_idx][:top_k]

    def _consolidate(self):
        """Merge similar patterns when over capacity."""
        if len(self._patterns) <= self.max_patterns:
            return

        threshold = config.HOPFIELD_CONSOLIDATION_THRESHOLD
        X = np.array(self._patterns)
        # Compute pairwise cosine similarities
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_normed = X / norms

        merged = set()
        new_patterns = []
        for i in range(len(self._patterns)):
            if i in merged:
                continue
            cluster = [self._patterns[i]]
            for j in range(i + 1, len(self._patterns)):
                if j in merged:
                    continue
                sim = float(X_normed[i] @ X_normed[j])
                if sim > threshold:
                    cluster.append(self._patterns[j])
                    merged.add(j)
            # Average the cluster
            avg = np.mean(cluster, axis=0).astype(np.float32)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg /= norm
            new_patterns.append(avg)

        self._patterns = new_patterns[:self.max_patterns]

    def get_all_patterns(self) -> np.ndarray:
        """Return all stored patterns as a matrix."""
        if not self._patterns:
            return np.zeros((0, self.pattern_dim), dtype=np.float32)
        return np.array(self._patterns)
