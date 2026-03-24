"""Tests for the self-model module."""

import pytest
import numpy as np
from unittest.mock import AsyncMock

from self_model.state_tracker import StateTracker
from self_model.capability_model import CapabilityModel
from self_model.transition_model import TransitionModel

import os
os.environ["DB_PATH"] = ":memory:"


class MockBackend:
    """Mock backend for tests."""
    async def get_goals(self):
        return [
            {"id": 1, "goal": "Build dashboard", "status": "active"},
            {"id": 2, "goal": "Fix bug", "status": "completed"},
        ]

    async def get_lessons(self):
        return [
            {"lesson": "Check context first", "action_type": "research", "success": True, "confidence": 0.8, "created_at": "2026-01-01"},
            {"lesson": "Test before deploy", "action_type": "build_and_test", "success": True, "confidence": 0.9, "created_at": "2026-01-02"},
            {"lesson": "Syntax error caught", "action_type": "build_and_test", "success": False, "confidence": 0.4, "created_at": "2026-01-03"},
            {"lesson": "API research", "action_type": "research", "success": True, "confidence": 0.7, "created_at": "2026-01-04"},
            {"lesson": "Deploy failed", "action_type": "deploy", "success": False, "confidence": 0.3, "created_at": "2026-01-05"},
        ]

    async def get_skills(self):
        return [
            {"goal_pattern": "code audit", "approach": "read codebase first", "success_rate": 0.8},
        ]

    async def get_knowledge(self):
        return [{"concept": f"concept_{i}"} for i in range(50)]

    async def get_learning_stats(self):
        return {"lesson_count": 5, "skill_count": 1}


class MockEmbedder:
    def __init__(self, dim=1024):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % 2**31)
        return np.random.randn(self.dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class TestStateTracker:
    @pytest.fixture
    def backend(self):
        return MockBackend()

    @pytest.mark.asyncio
    async def test_get_state(self, backend):
        from shared import db as db_module
        db_module._db_connection = None

        tracker = StateTracker(backend)
        state = await tracker.get_state()

        assert state["active_goals"] == 1
        assert state["lessons_learned"] == 5
        assert state["knowledge_nodes"] == 50
        assert state["skills_acquired"] == 1
        assert 0 <= state["success_rate_overall"] <= 1
        assert state["strongest_capability"] in ["research", "build_and_test", "deploy"]
        assert state["weakest_capability"] in ["research", "build_and_test", "deploy"]

    @pytest.mark.asyncio
    async def test_state_success_rate(self, backend):
        from shared import db as db_module
        db_module._db_connection = None

        tracker = StateTracker(backend)
        state = await tracker.get_state()
        # 3 successes out of 5 = 0.6
        assert state["success_rate_overall"] == 0.6


class TestCapabilityModel:
    @pytest.fixture
    def backend(self):
        return MockBackend()

    @pytest.mark.asyncio
    async def test_update(self, backend):
        cm = CapabilityModel(backend)
        caps = await cm.update()

        assert "research" in caps
        assert "build_and_test" in caps
        assert all(0 <= v <= 1 for v in caps.values())

    @pytest.mark.asyncio
    async def test_get_capabilities(self, backend):
        cm = CapabilityModel(backend)
        caps = await cm.get_capabilities()
        # First call should trigger update
        assert isinstance(caps, dict)


class TestTransitionModel:
    @pytest.fixture
    def backend(self):
        return MockBackend()

    @pytest.fixture
    def embedder(self):
        return MockEmbedder()

    @pytest.mark.asyncio
    async def test_predict_change(self, backend, embedder):
        from shared import db as db_module
        db_module._db_connection = None

        st = StateTracker(backend)
        cm = CapabilityModel(backend)
        tm = TransitionModel(st, cm, embedder, backend)

        result = await tm.predict_change("Add retry logic to build_and_test")
        assert "predicted_effect" in result
        assert "confidence" in result
        assert "side_effects" in result
        assert "recommendation" in result
        assert result["recommendation"] in ["proceed", "defer", "modify"]

    @pytest.mark.asyncio
    async def test_predict_next_state(self, backend, embedder):
        from shared import db as db_module
        db_module._db_connection = None

        st = StateTracker(backend)
        cm = CapabilityModel(backend)
        tm = TransitionModel(st, cm, embedder, backend)

        result = await tm.predict_next_state()
        assert "if_continue_current" in result
        assert "recommendation" in result
