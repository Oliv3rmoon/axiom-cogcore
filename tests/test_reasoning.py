"""Tests for the reasoning module."""

import pytest
import numpy as np
import asyncio

import config
from reasoning.thought_graph import ThoughtNode, ThoughtEdge
from reasoning.causal_graph import CausalGraph
from reasoning.workspace import ReasoningWorkspace

# Reset DB for each test
import os
os.environ["DB_PATH"] = ":memory:"


class MockEmbedder:
    """Mock embedding service for tests."""
    def __init__(self, dim=1024):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % 2**31)
        return np.random.randn(self.dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class TestThoughtNode:
    def test_create_node(self):
        node = ThoughtNode(thought="test", thought_type="observation")
        assert node.thought == "test"
        assert node.thought_type == "observation"
        assert node.confidence == 0.5

    def test_to_dict(self):
        node = ThoughtNode(thought="test", thought_type="hypothesis", confidence=0.8)
        d = node.to_dict()
        assert d["thought"] == "test"
        assert d["thought_type"] == "hypothesis"
        assert d["confidence"] == 0.8

    def test_not_expired_initially(self):
        node = ThoughtNode(thought="test")
        assert not node.is_expired()


class TestThoughtEdge:
    def test_create_edge(self):
        edge = ThoughtEdge(source_id="a", target_id="b", relationship="supports")
        assert edge.source_id == "a"
        assert edge.target_id == "b"

    def test_to_dict(self):
        edge = ThoughtEdge(source_id="a", target_id="b", strength=0.9)
        d = edge.to_dict()
        assert d["strength"] == 0.9


class TestCausalGraph:
    def test_add_and_get_causes(self):
        cg = CausalGraph()
        n1 = ThoughtNode(thought="cause")
        n2 = ThoughtNode(thought="effect")
        cg.add_node(n1)
        cg.add_node(n2)
        cg.add_causal_link(n1.id, n2.id, "enables", 0.9)

        causes = cg.get_causes(n2.id)
        assert len(causes) == 1
        assert causes[0]["relationship"] == "enables"

    def test_get_effects(self):
        cg = CausalGraph()
        n1 = ThoughtNode(thought="cause")
        n2 = ThoughtNode(thought="effect")
        cg.add_node(n1)
        cg.add_node(n2)
        cg.add_causal_link(n1.id, n2.id)

        effects = cg.get_effects(n1.id)
        assert len(effects) == 1

    def test_find_contradictions(self):
        cg = CausalGraph()
        n1 = ThoughtNode(thought="A")
        n2 = ThoughtNode(thought="B")
        cg.add_node(n1)
        cg.add_node(n2)
        cg.add_causal_link(n1.id, n2.id, "contradicts")

        contradictions = cg.find_contradictions()
        assert len(contradictions) == 1

    def test_causal_chain(self):
        cg = CausalGraph()
        nodes = [ThoughtNode(thought=f"node_{i}") for i in range(3)]
        for n in nodes:
            cg.add_node(n)
        cg.add_causal_link(nodes[0].id, nodes[1].id)
        cg.add_causal_link(nodes[1].id, nodes[2].id)

        chains = cg.get_causal_chain(nodes[0].id)
        assert len(chains) > 0


class TestReasoningWorkspace:
    @pytest.fixture
    def embedder(self):
        return MockEmbedder()

    @pytest.mark.asyncio
    async def test_start_workspace(self, embedder):
        # Reset DB connection for fresh in-memory DB
        from shared import db as db_module
        db_module._db_connection = None

        ws = ReasoningWorkspace(embedder)
        result = await ws.start("Build a monitoring dashboard")
        assert "workspace_id" in result
        assert "root_node_id" in result
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_add_thought(self, embedder):
        from shared import db as db_module
        db_module._db_connection = None

        ws = ReasoningWorkspace(embedder)
        start = await ws.start("Test goal")
        wid = start["workspace_id"]
        rid = start["root_node_id"]

        result = await ws.add_thought(wid, "A new observation", rid, "observation")
        assert "node_id" in result
        assert result["total_nodes"] == 2

    @pytest.mark.asyncio
    async def test_add_causal_link(self, embedder):
        from shared import db as db_module
        db_module._db_connection = None

        ws = ReasoningWorkspace(embedder)
        start = await ws.start("Test goal")
        wid = start["workspace_id"]
        rid = start["root_node_id"]

        t1 = await ws.add_thought(wid, "Cause node", rid, "observation")
        t2 = await ws.add_thought(wid, "Effect node", rid, "hypothesis")

        await ws.add_causal_link(wid, t1["node_id"], t2["node_id"], "enables", 0.9)

    @pytest.mark.asyncio
    async def test_get_workspace(self, embedder):
        from shared import db as db_module
        db_module._db_connection = None

        ws = ReasoningWorkspace(embedder)
        start = await ws.start("Test goal")
        wid = start["workspace_id"]
        rid = start["root_node_id"]
        await ws.add_thought(wid, "Observation 1", rid, "observation")
        await ws.add_thought(wid, "Question 1", rid, "question")

        result = await ws.get_workspace(wid)
        assert result["total_nodes"] == 3
        assert len(result["open_questions"]) == 1
        assert result["summary"]

    @pytest.mark.asyncio
    async def test_query(self, embedder):
        from shared import db as db_module
        db_module._db_connection = None

        ws = ReasoningWorkspace(embedder)
        start = await ws.start("Build dashboard")
        wid = start["workspace_id"]
        rid = start["root_node_id"]
        await ws.add_thought(wid, "Need an API endpoint for data", rid, "observation")

        result = await ws.query(wid, "What API endpoints are needed?")
        assert "relevant_nodes" in result
        assert "answer" in result

    @pytest.mark.asyncio
    async def test_workspace_not_found(self, embedder):
        from shared import db as db_module
        db_module._db_connection = None

        ws = ReasoningWorkspace(embedder)
        result = await ws.get_workspace("nonexistent")
        assert "error" in result
