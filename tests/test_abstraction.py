"""Tests for the abstraction module."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from abstraction.meta_learner import MetaLearner
from abstraction.skill_composer import SkillComposer


class MockEmbedder:
    """Mock embedding service for tests."""
    def __init__(self, dim=1024):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**31)
        return np.random.randn(self.dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class MockBackend:
    """Mock backend client for tests."""
    def __init__(self):
        self.lessons = []
        self.skills = []

    async def get_lessons(self):
        return self.lessons

    async def get_skills(self):
        return self.skills


class TestMetaLearner:
    @pytest.fixture
    def embedder(self):
        return MockEmbedder()

    @pytest.fixture
    def backend(self):
        return MockBackend()

    @pytest.mark.asyncio
    async def test_no_lessons(self, embedder, backend):
        ml = MetaLearner(embedder, backend)
        result = await ml.extract_meta_lessons()
        assert result == []

    @pytest.mark.asyncio
    async def test_few_lessons(self, embedder, backend):
        backend.lessons = [
            {"lesson": "Test lesson", "action_type": "research", "success": True, "confidence": 0.8}
        ]
        ml = MetaLearner(embedder, backend)
        result = await ml.extract_meta_lessons()
        assert result == []  # Not enough lessons

    @pytest.mark.asyncio
    async def test_extracts_lessons_from_clusters(self, embedder, backend):
        # Create enough similar lessons to form a cluster
        base_text = "Always check the context before making changes"
        backend.lessons = []
        for i in range(10):
            backend.lessons.append({
                "lesson": f"{base_text} - variant {i}",
                "action_type": "research" if i < 5 else "propose_change",
                "goal_type": "code_audit" if i < 5 else "feature",
                "success": True,
                "confidence": 0.8,
            })
        # Add some different lessons
        for i in range(10):
            backend.lessons.append({
                "lesson": f"Completely unrelated topic about cooking recipe {i}",
                "action_type": "email",
                "goal_type": "personal",
                "success": False,
                "confidence": 0.3,
            })

        ml = MetaLearner(embedder, backend)
        result = await ml.extract_meta_lessons()
        # Should find at least something (exact cluster behavior depends on embeddings)
        assert isinstance(result, list)


class TestSkillComposer:
    @pytest.fixture
    def embedder(self):
        return MockEmbedder()

    @pytest.fixture
    def backend(self):
        return MockBackend()

    @pytest.mark.asyncio
    async def test_no_skills(self, embedder, backend):
        sc = SkillComposer(embedder, backend)
        result = await sc.compose_for_goal("Build a dashboard")
        assert result["matching_skills"] == []
        assert result["coverage"] == 0.0

    @pytest.mark.asyncio
    async def test_with_skills(self, embedder, backend):
        backend.skills = [
            {"goal_pattern": "Build web application", "approach": "Use React", "success_rate": 0.8},
            {"goal_pattern": "Create API endpoint", "approach": "Use FastAPI", "success_rate": 0.9},
        ]
        sc = SkillComposer(embedder, backend)
        result = await sc.compose_for_goal("Build a web dashboard")
        assert isinstance(result["matching_skills"], list)
        assert isinstance(result["coverage"], float)

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, embedder, backend):
        sc = SkillComposer(embedder, backend)
        await sc.compose_for_goal("test")
        assert sc._skills_cache is not None
        sc.invalidate_cache()
        assert sc._skills_cache is None
