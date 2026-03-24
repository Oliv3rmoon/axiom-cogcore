from __future__ import annotations
"""Information gain computation for curiosity."""

import torch
import numpy as np

import config


class InformationGainTracker:
    """
    Approximate information gain as change in world model loss
    before/after incorporating an experience.
    """

    def __init__(self):
        self._gains: dict[str, list[float]] = {}
        self._window_size = 50

    def compute_gain(self, loss_before: float, loss_after: float) -> float:
        """
        Information gain = loss reduction from incorporating this experience.
        Positive = model improved = this area is worth exploring.
        """
        gain = max(0.0, loss_before - loss_after)
        # Normalize to 0-1
        return float(np.log1p(gain) / (1 + np.log1p(gain)))

    def record(self, domain: str, loss_before: float, loss_after: float) -> float:
        """Record an information gain observation for a domain."""
        gain = self.compute_gain(loss_before, loss_after)
        if domain not in self._gains:
            self._gains[domain] = []
        self._gains[domain].append(gain)
        if len(self._gains[domain]) > self._window_size:
            self._gains[domain] = self._gains[domain][-self._window_size:]
        return gain

    def get_domain_score(self, domain: str) -> float:
        """Average information gain for a domain."""
        gains = self._gains.get(domain, [])
        if not gains:
            return 0.5  # Moderate for unknown domains
        weights = [config.CURIOSITY_DECAY ** (len(gains) - i - 1) for i in range(len(gains))]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(g * w for g, w in zip(gains, weights)) / total_weight

    def get_all_scores(self) -> dict[str, float]:
        """Get information gain scores for all tracked domains."""
        return {domain: self.get_domain_score(domain) for domain in self._gains}
