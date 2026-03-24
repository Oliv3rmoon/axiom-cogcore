"""Tests for the Attention Schema module."""
from __future__ import annotations

import pytest
import time

from attention_schema.schema import AttentionSchema, AttentionTarget
from attention_schema.meta_cognition import MetaCognition
from attention_schema.awareness import AwarenessEngine

import config


class TestAttentionTarget:
    def test_create(self):
        t = AttentionTarget(target="test", target_type="goal", attention_strength=0.8)
        assert t.target == "test"
        assert t.attention_strength == 0.8

    def test_to_dict(self):
        t = AttentionTarget(target="x", target_type="step", attention_strength=0.5)
        d = t.to_dict()
        assert d["target"] == "x"
        assert "timestamp" in d


class TestAttentionSchema:
    @pytest.fixture
    def schema(self):
        return AttentionSchema()

    def test_compute_strength(self, schema):
        signals = {
            "curiosity_score": 0.8,
            "prediction_error": 0.6,
            "goal_relevance": 0.9,
            "broadcast_salience": 0.7,
        }
        strength = schema.compute_attention_strength(signals)
        assert 0 < strength < 1
        # 0.3*0.8 + 0.25*0.6 + 0.25*0.9 + 0.2*0.7 = 0.24+0.15+0.225+0.14 = 0.755
        assert abs(strength - 0.755) < 0.01

    def test_compute_strength_empty_signals(self, schema):
        assert schema.compute_attention_strength({}) == 0.0

    def test_update_focus(self, schema):
        result = schema.update_focus("audit code", "goal", {"curiosity_score": 0.9})
        assert "attention_strength" in result
        assert result["attention_strength"] > 0
        assert schema.current_focus is not None
        assert schema.current_focus.target == "audit code"

    def test_update_focus_multiple(self, schema):
        schema.update_focus("task_a", "step", {"curiosity_score": 0.5})
        schema.update_focus("task_b", "step", {"curiosity_score": 0.9})
        assert len(schema.focus_history) == 1  # First focus moved to history

    def test_should_switch_attention(self, schema):
        assert schema.should_switch_attention(0.5, 0.8)  # 0.3 > threshold
        assert not schema.should_switch_attention(0.5, 0.55)  # 0.05 < threshold

    def test_get_focus_empty(self, schema):
        state = schema.get_focus()
        assert state["current_focus"] is None
        assert state["attention_history"] == []

    def test_get_focus_with_data(self, schema):
        schema.update_focus("test", "goal", {"curiosity_score": 0.8})
        state = schema.get_focus()
        assert state["current_focus"] is not None
        assert state["current_focus"]["target"] == "test"
        assert "predicted_next_shift" in state

    def test_attention_weights_tracked(self, schema):
        schema.update_focus("task_a", "code", {"curiosity_score": 0.8})
        schema.update_focus("task_b", "research", {"curiosity_score": 0.6})
        assert "code" in schema.attention_weights
        assert "research" in schema.attention_weights


class TestMetaCognition:
    @pytest.fixture
    def meta(self):
        schema = AttentionSchema()
        return MetaCognition(schema)

    def test_analyze_biases_empty(self, meta):
        biases = meta.analyze_biases()
        assert len(biases) > 0
        assert "No attention data" in biases[0]

    def test_analyze_biases_with_data(self, meta):
        # Heavily attend to one domain
        for _ in range(10):
            meta.schema.update_focus("code task", "code", {"curiosity_score": 0.9})
        meta.schema.update_focus("research task", "research", {"curiosity_score": 0.3})
        biases = meta.analyze_biases()
        assert any("code" in b.lower() for b in biases)

    def test_recommendations_empty(self, meta):
        recs = meta.get_recommendations()
        assert isinstance(recs, list)

    def test_recommendations_with_data(self, meta):
        for _ in range(5):
            meta.schema.update_focus("code", "code", {"curiosity_score": 0.9})
        meta.schema.update_focus("research", "research", {"curiosity_score": 0.2})
        meta.schema.update_focus("email", "email", {"curiosity_score": 0.1})
        recs = meta.get_recommendations()
        assert isinstance(recs, list)


class TestAwarenessEngine:
    @pytest.fixture
    def awareness(self):
        schema = AttentionSchema()
        meta = MetaCognition(schema)
        return AwarenessEngine(schema, meta)

    def test_introspect_no_focus(self, awareness):
        result = awareness.introspect()
        assert "self_report" in result
        assert "meta_observations" in result
        assert "recommendations" in result

    def test_introspect_with_focus(self, awareness):
        awareness.schema.update_focus("audit backend", "goal", {
            "curiosity_score": 0.8, "prediction_error": 0.6
        })
        result = awareness.introspect()
        assert "audit backend" in result["self_report"]

    def test_introspect_deep(self, awareness):
        for i in range(5):
            awareness.schema.update_focus(f"task_{i}", "step", {"curiosity_score": 0.5})
        result = awareness.introspect(depth="deep")
        assert "history" in result["self_report"].lower() or "different" in result["self_report"].lower()
