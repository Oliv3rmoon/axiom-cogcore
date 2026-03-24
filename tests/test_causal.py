"""Tests for the Causal Reasoner module."""
from __future__ import annotations

import pytest

from causal.scm import StructuralCausalModel, CausalNode, CausalEdge
from causal.do_calculus import do_intervention
from causal.counterfactual import counterfactual_query


class TestStructuralCausalModel:
    @pytest.fixture
    def scm(self):
        return StructuralCausalModel()

    def test_add_node(self, scm):
        scm.add_node("X")
        assert "X" in scm.nodes
        assert "X" in scm.graph

    def test_add_edge(self, scm):
        edge = scm.add_edge("X", "Y", strength=0.8, evidence_count=10)
        assert edge.cause == "X"
        assert edge.effect == "Y"
        assert edge.strength == 0.8
        assert scm.graph.has_edge("X", "Y")

    def test_add_edge_creates_nodes(self, scm):
        scm.add_edge("A", "B")
        assert "A" in scm.nodes
        assert "B" in scm.nodes

    def test_update_existing_edge(self, scm):
        scm.add_edge("X", "Y", strength=0.6, evidence_count=5)
        scm.add_edge("X", "Y", strength=0.8, evidence_count=5)
        # Strength should be updated (weighted average)
        edges = scm.get_all_edges()
        xy = [e for e in edges if e["cause"] == "X" and e["effect"] == "Y"]
        assert len(xy) == 1
        assert xy[0]["evidence_count"] == 10

    def test_no_cycles(self, scm):
        scm.add_edge("A", "B")
        scm.add_edge("B", "C")
        edge = scm.add_edge("C", "A")  # Would create cycle
        assert edge.strength == 0.0  # Should be rejected

    def test_get_parents(self, scm):
        scm.add_edge("X", "Z")
        scm.add_edge("Y", "Z")
        parents = scm.get_parents("Z")
        assert set(parents) == {"X", "Y"}

    def test_get_children(self, scm):
        scm.add_edge("X", "Y")
        scm.add_edge("X", "Z")
        children = scm.get_children("X")
        assert set(children) == {"Y", "Z"}

    def test_get_all_nodes_edges(self, scm):
        scm.add_edge("A", "B", strength=0.5)
        scm.add_edge("B", "C", strength=0.7)
        assert len(scm.get_all_nodes()) == 3
        assert len(scm.get_all_edges()) == 2


class TestDoCalculus:
    @pytest.fixture
    def scm(self):
        model = StructuralCausalModel()
        # Build a simple causal graph:
        # read_codebase → proposal_success
        # goal_type → proposal_success
        # read_codebase → understanding (confounder)
        model.add_edge("read_codebase", "proposal_success", strength=0.8, evidence_count=20)
        model.add_edge("goal_type", "proposal_success", strength=0.3, evidence_count=15)
        model.add_edge("read_codebase", "understanding", strength=0.9, evidence_count=25)
        model.add_edge("understanding", "proposal_success", strength=0.7, evidence_count=18)
        return model

    def test_do_intervention_positive(self, scm):
        result = do_intervention(scm, "read_codebase", True, "proposal_success")
        assert result["result"] > 0.5  # Positive intervention should help
        assert "causal_path" in result
        assert "explanation" in result

    def test_do_intervention_negative(self, scm):
        result = do_intervention(scm, "read_codebase", False, "proposal_success")
        assert result["result"] < 0.8  # Not reading should hurt

    def test_do_removes_incoming_edges(self, scm):
        # Add a confounding parent to read_codebase
        scm.add_edge("motivation", "read_codebase", strength=0.5)
        result = do_intervention(scm, "read_codebase", True, "proposal_success")
        # Should still work — do() removes motivation → read_codebase
        assert result["result"] > 0.5

    def test_missing_variable(self, scm):
        result = do_intervention(scm, "nonexistent", True, "proposal_success")
        assert result["result"] == 0.5

    def test_no_causal_path(self, scm):
        scm.add_node("isolated")
        result = do_intervention(scm, "isolated", True, "proposal_success")
        assert result["result"] == 0.5

    def test_with_given_conditions(self, scm):
        result = do_intervention(
            scm, "read_codebase", True, "proposal_success",
            given={"goal_type": True}
        )
        assert "result" in result


class TestCounterfactual:
    @pytest.fixture
    def scm(self):
        model = StructuralCausalModel()
        model.add_edge("read_codebase", "success", strength=0.8, evidence_count=20)
        model.add_edge("propose_change", "success", strength=0.4, evidence_count=15)
        return model

    def test_counterfactual_better_action(self, scm):
        result = counterfactual_query(
            scm,
            actual_action="propose_change",
            actual_outcome="failed",
            counterfactual_action="read_codebase",
        )
        assert "counterfactual_outcome" in result
        assert "probability" in result
        assert "explanation" in result
        assert "confidence" in result
        # Reading codebase should have been better
        assert result["probability"] > 0.5

    def test_counterfactual_unknown_action(self, scm):
        result = counterfactual_query(
            scm,
            actual_action="propose_change",
            actual_outcome="failed",
            counterfactual_action="completely_unknown",
        )
        assert result["probability"] == 0.5
        assert result["confidence"] <= 0.3

    def test_counterfactual_already_good(self, scm):
        result = counterfactual_query(
            scm,
            actual_action="read_codebase",
            actual_outcome="success",
            counterfactual_action="propose_change",
        )
        assert "explanation" in result


class TestCausalNode:
    def test_observe(self):
        node = CausalNode(name="X")
        node.observe(True)
        node.observe(True)
        node.observe(False)
        assert node.total_observations == 3
        assert node.observations["True"] == 2


class TestCausalEdge:
    def test_to_dict(self):
        edge = CausalEdge(cause="A", effect="B", strength=0.7, evidence_count=10)
        d = edge.to_dict()
        assert d["cause"] == "A"
        assert d["strength"] == 0.7
