"""Tests for the DreamCoder abstraction module."""
from __future__ import annotations

import pytest
import numpy as np

from dreamcoder.library import Library, Primitive
from dreamcoder.wake import wake_solve

import os
os.environ["DB_PATH"] = ":memory:"


class MockEmbedder:
    def __init__(self, dim=384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % 2**31)
        return np.random.randn(self.dim).astype(np.float32)

    def embed_batch(self, texts):
        return np.array([self.embed(t) for t in texts])


class TestPrimitive:
    def test_create(self):
        p = Primitive(name="test", pattern="Test pattern", steps=["a", "b"])
        assert p.name == "test"
        assert len(p.steps) == 2

    def test_to_dict(self):
        p = Primitive(name="test", pattern="Test", steps=["x"], domains=["d1"])
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["domains"] == ["d1"]
        assert "success_rate" in d


class TestLibrary:
    @pytest.fixture
    def embedder(self):
        return MockEmbedder()

    @pytest.fixture
    def library(self, embedder):
        return Library(embedder)

    def test_empty(self, library):
        assert library.size == 0

    def test_add_primitive(self, library):
        p = Primitive(name="read_first", pattern="Read before acting", steps=["read", "act"])
        library.add_primitive(p)
        assert library.size == 1

    def test_merge_duplicate_name(self, library):
        p1 = Primitive(name="read_first", pattern="Read", steps=["read"], frequency=3)
        p2 = Primitive(name="read_first", pattern="Read before", steps=["read", "act"], frequency=2)
        library.add_primitive(p1)
        library.add_primitive(p2)
        assert library.size == 1
        # Frequency should be summed
        prim = list(library.primitives.values())[0]
        assert prim.frequency == 5

    def test_compose_empty(self, library):
        result = library.compose("some task")
        assert result == []

    def test_compose_finds_relevant(self, library):
        library.add_primitive(Primitive(
            name="code_audit", pattern="audit code for issues",
            steps=["read_codebase", "identify_issues", "report"],
            domains=["code"],
        ))
        library.add_primitive(Primitive(
            name="research", pattern="research a topic online",
            steps=["search", "read", "summarize"],
            domains=["research"],
        ))
        result = library.compose("audit the backend code")
        assert len(result) > 0

    def test_max_library_size(self, embedder):
        import config
        old_max = config.DREAMCODER_MAX_LIBRARY_SIZE
        config.DREAMCODER_MAX_LIBRARY_SIZE = 5
        try:
            lib = Library(embedder)
            for i in range(10):
                lib.add_primitive(Primitive(
                    name=f"prim_{i}", pattern=f"Pattern {i}",
                    steps=[f"step_{i}"], frequency=i,
                ))
            assert lib.size <= 5
        finally:
            config.DREAMCODER_MAX_LIBRARY_SIZE = old_max


class TestWakeSolve:
    @pytest.fixture
    def embedder(self):
        return MockEmbedder()

    def test_empty_library(self, embedder):
        lib = Library(embedder)
        result = wake_solve("audit code", lib, embedder)
        assert result["solution_found"] is False
        assert result["solution"] == []

    def test_with_primitives(self, embedder):
        lib = Library(embedder)
        lib.add_primitive(Primitive(
            name="read_first", pattern="read codebase first",
            steps=["read_codebase", "analyze"],
        ))
        lib.add_primitive(Primitive(
            name="propose", pattern="propose code changes",
            steps=["propose_change", "test"],
        ))
        result = wake_solve("audit the code and fix issues", lib, embedder)
        assert result["solution_found"] is True
        assert len(result["solution"]) > 0
        assert len(result["primitives_used"]) > 0
        assert "solution_id" in result
        assert 0 <= result["novelty"] <= 1
