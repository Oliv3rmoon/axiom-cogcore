"""Tests for the Predictive Processing Hierarchy module."""
from __future__ import annotations

import pytest
import config

from predictive_hierarchy.hierarchy import PredictiveHierarchy
from predictive_hierarchy.precision_weighting import compute_precision_weighted_salience
from predictive_hierarchy.error_propagation import propagate_error_up, propagate_precision_down


class TestPredictiveHierarchy:
    @pytest.fixture
    def hierarchy(self):
        return PredictiveHierarchy()

    def test_initial_state(self, hierarchy):
        state = hierarchy.get_state()
        assert len(state["levels"]) == config.HIERARCHY_LEVELS
        assert state["total_prediction_errors"] == 0
        assert state["global_surprise"] == 0.0

    def test_predict(self, hierarchy):
        result = hierarchy.predict(0, "propose_change step")
        assert "predicted_outcome" in result
        assert "confidence" in result
        assert result["level"] == 0
        assert result["level_name"] == "step_outcomes"

    def test_predict_clamped_level(self, hierarchy):
        result = hierarchy.predict(99, "out of range")
        assert result["level"] == config.HIERARCHY_LEVELS - 1

    def test_update_records_error(self, hierarchy):
        result = hierarchy.update(0, predicted=0.8, actual=0.2, context="test")
        assert "error_at_level_0" in result
        assert result["error_at_level_0"] == pytest.approx(0.6, abs=0.01)
        assert hierarchy.total_predictions[0] == 1

    def test_error_propagates_up(self, hierarchy):
        result = hierarchy.update(0, predicted=0.9, actual=0.1, context="big error")
        # Error should propagate to higher levels
        assert "propagated_to_level_1" in result
        assert result["propagated_to_level_1"] > 0

    def test_precision_updates_down(self, hierarchy):
        # Record some errors
        for _ in range(5):
            hierarchy.update(0, predicted=0.8, actual=0.2)
        # Precision should have decreased due to high error
        assert hierarchy.precisions[0] < 1.0

    def test_precision_stays_above_minimum(self, hierarchy):
        for _ in range(100):
            hierarchy.update(0, predicted=1.0, actual=0.0)  # Worst case
        assert hierarchy.precisions[0] >= 0.1

    def test_precision_weighted_error(self, hierarchy):
        hierarchy.mean_errors[0] = 0.5
        hierarchy.precisions[0] = 0.8
        pwe = hierarchy.get_precision_weighted_error(0)
        assert pwe == pytest.approx(0.4, abs=0.01)  # 0.8 * 0.5

    def test_multiple_levels(self, hierarchy):
        # Update different levels
        hierarchy.update(0, 0.8, 0.2, "step error")
        hierarchy.update(1, 0.7, 0.3, "plan error")
        hierarchy.update(2, 0.6, 0.5, "goal error")

        state = hierarchy.get_state()
        assert state["levels"][0]["total_predictions"] == 1
        assert state["levels"][1]["total_predictions"] == 1
        assert state["levels"][2]["total_predictions"] == 1

    def test_get_state_structure(self, hierarchy):
        hierarchy.update(0, 0.7, 0.3)
        state = hierarchy.get_state()
        level0 = state["levels"][0]
        assert "name" in level0
        assert "description" in level0
        assert "mean_prediction_error" in level0
        assert "precision" in level0
        assert "recent_errors" in level0


class TestPrecisionWeighting:
    def test_compute_salience(self):
        h = PredictiveHierarchy()
        h.mean_errors[0] = 0.5
        h.precisions[0] = 0.9
        result = compute_precision_weighted_salience(h)
        assert "per_level_salience" in result
        assert "global_attention_demand" in result
        assert "most_salient_level" in result
        assert result["per_level_salience"]["step_outcomes"] > 0

    def test_zero_errors(self):
        h = PredictiveHierarchy()
        result = compute_precision_weighted_salience(h)
        assert result["global_attention_demand"] == 0.0


class TestErrorPropagation:
    def test_propagate_up(self):
        h = PredictiveHierarchy()
        updates = propagate_error_up(h, 0, 0.8)
        assert "level_0" in updates
        assert "level_1" in updates
        assert h.mean_errors[1] > 0

    def test_propagate_precision_down(self):
        h = PredictiveHierarchy()
        h.mean_errors[1] = 0.5
        h.mean_errors[0] = 0.3
        updates = propagate_precision_down(h, 1)
        assert "level_1" in updates
        assert "level_0" in updates
        assert h.precisions[1] < 1.0
