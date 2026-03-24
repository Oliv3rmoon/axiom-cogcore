"""Tests for the Modern Hopfield network and episodic store."""
from __future__ import annotations

import pytest
import numpy as np
import asyncio

import config
from hopfield.modern_hopfield import ModernHopfieldNetwork
from hopfield.memory_manager import MemoryManager

import os
os.environ["DB_PATH"] = ":memory:"


class TestModernHopfieldNetwork:
    @pytest.fixture
    def net(self):
        return ModernHopfieldNetwork(pattern_dim=64, max_patterns=100)

    def test_store_pattern(self, net):
        p = np.random.randn(64).astype(np.float32)
        idx = net.store(p)
        assert idx == 0
        assert net.num_patterns == 1

    def test_store_multiple(self, net):
        for i in range(10):
            net.store(np.random.randn(64).astype(np.float32))
        assert net.num_patterns == 10

    def test_wrong_dim_raises(self, net):
        with pytest.raises(ValueError):
            net.store(np.random.randn(32).astype(np.float32))

    def test_retrieve_similar(self, net):
        # Store a known pattern
        base = np.random.randn(64).astype(np.float32)
        net.store(base)
        # Store some different patterns
        for _ in range(5):
            net.store(np.random.randn(64).astype(np.float32))

        # Query with something similar to base
        query = base + np.random.randn(64).astype(np.float32) * 0.1
        results = net.retrieve(query, top_k=3)
        assert len(results) == 3
        # First result should be the most similar (highest attention)
        assert results[0][1] >= results[1][1]

    def test_retrieve_single(self, net):
        base = np.random.randn(64).astype(np.float32)
        net.store(base)
        retrieved = net.retrieve_single(base)
        assert retrieved.shape == (64,)

    def test_retrieve_empty(self, net):
        results = net.retrieve(np.random.randn(64).astype(np.float32))
        assert results == []

    def test_retrieve_single_empty(self, net):
        result = net.retrieve_single(np.random.randn(64).astype(np.float32))
        assert result.shape == (64,)
        assert np.all(result == 0)

    def test_energy(self, net):
        p = np.random.randn(64).astype(np.float32)
        net.store(p)
        e = net.energy(p)
        assert isinstance(e, float)

    def test_energy_empty(self, net):
        assert net.energy(np.random.randn(64).astype(np.float32)) == 0.0

    def test_find_associations(self, net):
        for i in range(5):
            net.store(np.random.randn(64).astype(np.float32))
        assocs = net.find_associations(0, top_k=3)
        assert len(assocs) <= 3
        # Should not include self
        for idx, _ in assocs:
            assert idx != 0

    def test_consolidation(self):
        net = ModernHopfieldNetwork(pattern_dim=64, max_patterns=10)
        # Store 15 very similar patterns (should merge down)
        base = np.random.randn(64).astype(np.float32)
        for i in range(15):
            p = base + np.random.randn(64).astype(np.float32) * 0.01
            net.store(p)
        assert net.num_patterns <= 10

    def test_content_addressed_retrieval(self):
        """Similar queries should return similar results."""
        net = ModernHopfieldNetwork(pattern_dim=64, max_patterns=100)
        # Store distinct patterns
        patterns = [np.random.randn(64).astype(np.float32) for _ in range(20)]
        for p in patterns:
            net.store(p)

        # Two similar queries should return the same top result
        query = patterns[5] + np.random.randn(64).astype(np.float32) * 0.01
        r1 = net.retrieve(query, top_k=1)
        r2 = net.retrieve(patterns[5], top_k=1)
        assert r1[0][0] == r2[0][0]  # Same top pattern index


class TestMemoryManager:
    def test_should_consolidate(self):
        from hopfield.episodic_store import EpisodicStore

        class FakeEmbedder:
            dim = 64
            def embed(self, text):
                return np.random.randn(64).astype(np.float32)

        store = EpisodicStore(FakeEmbedder())
        store.hopfield = ModernHopfieldNetwork(pattern_dim=64, max_patterns=100)
        mgr = MemoryManager(store)
        assert not mgr.should_consolidate()

    def test_memory_health(self):
        from hopfield.episodic_store import EpisodicStore

        class FakeEmbedder:
            dim = 64
            def embed(self, text):
                return np.random.randn(64).astype(np.float32)

        store = EpisodicStore(FakeEmbedder())
        store.hopfield = ModernHopfieldNetwork(pattern_dim=64, max_patterns=100)
        mgr = MemoryManager(store)
        health = mgr.get_memory_health()
        assert health["health"] == "good"
        assert "needs_consolidation" in health
