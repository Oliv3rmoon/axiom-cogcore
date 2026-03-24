"""Tests for the active inference module."""
from __future__ import annotations

import pytest
import torch
import numpy as np

import config
from active_inference.precision import PrecisionController
from active_inference.policy_selection import compare_policies, recommend_action
from active_inference.expected_free_energy import expected_free_energy
from active_inference.generative_model import GenerativeModel
from world_model.model import RSSMWorldModel


class MockEmbedder:
    """Mock embedder for testing."""
    def __init__(self, dim=config.WORLD_MODEL_OBS_DIM):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % 2**31)
        return np.random.randn(self.dim).astype(np.float32)

    def embed_to_tensor(self, text: str) -> torch.Tensor:
        arr = self.embed(text)
        return torch.tensor(arr, dtype=torch.float32, device=config.DEVICE)

    def embed_batch(self, texts):
        return np.array([self.embed(t) for t in texts])


class TestPrecisionController:
    def test_initial_state(self):
        pc = PrecisionController()
        assert pc.precision == config.AI_PRECISION_INIT
        assert pc.total_inferences == 0

    def test_update_low_error_moves_toward_high(self):
        pc = PrecisionController()
        # Start from low precision so low error should increase it
        pc.precision = 0.5
        pc.update(0.01)  # Very low error → target ~0.99
        assert pc.precision > 0.5  # Should increase

    def test_update_high_error_decreases_precision(self):
        pc = PrecisionController()
        # First increase precision
        for _ in range(20):
            pc.update(0.01)
        high_precision = pc.precision
        # Now give high error
        for _ in range(20):
            pc.update(5.0)
        assert pc.precision < high_precision

    def test_precision_clamped(self):
        pc = PrecisionController()
        for _ in range(100):
            pc.update(0.001)
        assert pc.precision <= 2.0
        for _ in range(100):
            pc.update(100.0)
        assert pc.precision >= 0.1

    def test_exploration_tendency(self):
        pc = PrecisionController()
        pc.precision = 0.1  # Low precision
        assert pc.exploration_tendency > 0.9  # High exploration
        pc.precision = 2.0  # High precision
        assert pc.exploration_tendency < 0.1  # Low exploration

    def test_get_status(self):
        pc = PrecisionController()
        pc.update(0.5)
        status = pc.get_status()
        assert "precision" in status
        assert "exploration_tendency" in status
        assert "total_inferences" in status
        assert status["total_inferences"] == 1


class TestExpectedFreeEnergy:
    @pytest.fixture
    def gen_model(self):
        wm = RSSMWorldModel().to(config.DEVICE)
        return GenerativeModel(wm, MockEmbedder())

    def test_efe_returns_all_fields(self, gen_model):
        result = expected_free_energy(
            gen_model, "current state", "research", "audit code", 1.0
        )
        assert "expected_free_energy" in result
        assert "epistemic_value" in result
        assert "pragmatic_value" in result
        assert "ambiguity" in result
        assert "risk" in result
        assert "confidence" in result

    def test_efe_epistemic_nonnegative(self, gen_model):
        result = expected_free_energy(
            gen_model, "state", "research", "goal", 1.0
        )
        assert result["epistemic_value"] >= 0

    def test_high_precision_favors_pragmatic(self, gen_model):
        high_p = expected_free_energy(
            gen_model, "state", "research", "goal", 2.0
        )
        low_p = expected_free_energy(
            gen_model, "state", "research", "goal", 0.1
        )
        # With high precision, pragmatic should have more influence
        # (EFE should differ between precision levels)
        assert high_p["expected_free_energy"] != low_p["expected_free_energy"]


class TestPolicySelection:
    @pytest.fixture
    def gen_model(self):
        wm = RSSMWorldModel().to(config.DEVICE)
        return GenerativeModel(wm, MockEmbedder())

    def test_compare_empty_policies(self, gen_model):
        result = compare_policies(gen_model, "state", [], "goal", 1.0)
        assert result["ranked_policies"] == []
        assert result["best_action"] == ""

    def test_compare_returns_ranked(self, gen_model):
        result = compare_policies(
            gen_model, "state",
            ["research", "propose_change", "build_and_test"],
            "audit code", 1.0,
        )
        assert len(result["ranked_policies"]) == 3
        assert result["best_action"] in ["research", "propose_change", "build_and_test"]
        assert "exploration_exploitation_ratio" in result

    def test_probabilities_sum_to_one(self, gen_model):
        result = compare_policies(
            gen_model, "state",
            ["research", "propose_change"],
            "goal", 1.0,
        )
        probs = [r["probability"] for r in result["ranked_policies"]]
        assert abs(sum(probs) - 1.0) < 0.01

    def test_recommend_action(self):
        assert recommend_action({"expected_free_energy": -2.0, "epistemic_value": 0.5, "pragmatic_value": 0.3}) == "proceed"
        assert recommend_action({"expected_free_energy": 1.0, "epistemic_value": 0.5, "pragmatic_value": 0.3}) == "reconsider"
