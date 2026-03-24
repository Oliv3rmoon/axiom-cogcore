from __future__ import annotations
"""Prediction error computation for curiosity signals."""

import numpy as np
import torch

import config


class PredictionErrorTracker:
    """Tracks prediction errors per domain for curiosity scoring."""

    def __init__(self):
        # domain -> list of recent prediction errors
        self._errors: dict[str, list[float]] = {}
        self._window_size = 50

    def record(self, domain: str, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Compute and record prediction error between predicted and actual outcome embeddings.
        Returns the normalized prediction error (0-1 scale).
        """
        error = float(np.mean((predicted - actual) ** 2))
        normalized = self._normalize(error)

        if domain not in self._errors:
            self._errors[domain] = []
        self._errors[domain].append(normalized)
        if len(self._errors[domain]) > self._window_size:
            self._errors[domain] = self._errors[domain][-self._window_size:]

        return normalized

    def get_domain_score(self, domain: str) -> float:
        """Average prediction error for a domain (higher = more curious)."""
        errors = self._errors.get(domain, [])
        if not errors:
            return config.PREDICTION_ERROR_THRESHOLD  # Moderate curiosity for unknown domains
        # Apply decay: more recent errors weighted higher
        weights = [config.CURIOSITY_DECAY ** (len(errors) - i - 1) for i in range(len(errors))]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(e * w for e, w in zip(errors, weights)) / total_weight

    def get_all_scores(self) -> dict[str, float]:
        """Get prediction error scores for all tracked domains."""
        return {domain: self.get_domain_score(domain) for domain in self._errors}

    def _normalize(self, error: float) -> float:
        """Normalize error to 0-1 range using symlog-style normalization."""
        # symlog: sign(x) * log(1 + |x|)
        return float(np.log1p(abs(error)) / (1 + np.log1p(abs(error))))
