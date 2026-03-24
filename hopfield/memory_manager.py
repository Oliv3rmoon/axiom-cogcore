from __future__ import annotations
"""Manage memory lifecycle and consolidation for Hopfield episodic memory."""

import numpy as np

import config
from hopfield.episodic_store import EpisodicStore


class MemoryManager:
    """
    Manages the lifecycle of episodic memories:
    - Importance-based retention
    - Similarity-based consolidation
    - Access-frequency tracking
    """

    def __init__(self, store: EpisodicStore):
        self.store = store

    def should_consolidate(self) -> bool:
        """Check if memory consolidation is needed."""
        return self.store.hopfield.num_patterns > config.HOPFIELD_MAX_PATTERNS * 0.9

    def consolidate(self) -> dict:
        """
        Run memory consolidation:
        1. Merge similar patterns (cosine > threshold)
        2. Report statistics
        """
        before = self.store.hopfield.num_patterns
        self.store.hopfield._consolidate()
        after = self.store.hopfield.num_patterns

        return {
            "patterns_before": before,
            "patterns_after": after,
            "patterns_merged": before - after,
        }

    def get_memory_health(self) -> dict:
        """Get overall memory health metrics."""
        stats = self.store.get_stats()
        return {
            **stats,
            "needs_consolidation": self.should_consolidate(),
            "health": "good" if stats["utilization"] < 0.8 else "needs_attention",
        }
