"""Tests for the meta-learning (Reptile) module."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config
from meta_learning.reptile import Reptile


class SimpleModel(nn.Module):
    """Simple model for testing Reptile."""
    def __init__(self, in_dim=16, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TestReptile:
    @pytest.fixture
    def model(self):
        return SimpleModel().to(config.DEVICE)

    @pytest.fixture
    def reptile(self, model):
        return Reptile(model)

    def test_initial_state(self, reptile):
        assert reptile.meta_step_count == 0
        status = reptile.get_status()
        assert status["meta_steps"] == 0
        assert status["domains_seen"] == 0

    def test_meta_step_updates_count(self, reptile, model):
        # Create simple tasks
        tasks = []
        for i in range(2):
            tasks.append({
                "data": torch.randn(8, 16, device=config.DEVICE),
                "domain": f"domain_{i}",
            })

        def loss_fn(m, data):
            out = m(data)
            return F.mse_loss(out, torch.zeros_like(out))

        result = reptile.meta_step(tasks, loss_fn)
        assert result["outer_step"] == 1
        assert result["inner_steps"] == config.REPTILE_INNER_STEPS
        assert "meta_loss" in result
        assert reptile.meta_step_count == 1

    def test_meta_step_moves_params(self, reptile, model):
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        tasks = [{
            "data": torch.randn(8, 16, device=config.DEVICE),
            "domain": "test",
        }]

        def loss_fn(m, data):
            out = m(data)
            return F.mse_loss(out, torch.ones_like(out))

        reptile.meta_step(tasks, loss_fn)

        # Parameters should have changed
        changed = False
        for k in initial_params:
            if not torch.allclose(initial_params[k], model.state_dict()[k]):
                changed = True
                break
        assert changed

    def test_meta_step_toward_task(self, reptile, model):
        """Reptile should move params toward task solution direction."""
        # Fixed target for reproducibility
        target = torch.ones(8, 4, device=config.DEVICE) * 2.0
        data = torch.randn(8, 16, device=config.DEVICE)

        def loss_fn(m, d):
            return F.mse_loss(m(d), target)

        # Initial loss
        with torch.no_grad():
            initial_loss = loss_fn(model, data).item()

        # Multiple meta steps
        for _ in range(10):
            tasks = [{"data": data, "domain": "test"}]
            reptile.meta_step(tasks, loss_fn, outer_lr=0.1)

        # Loss should decrease
        with torch.no_grad():
            final_loss = loss_fn(model, data).item()
        assert final_loss < initial_loss

    def test_adapt_does_not_modify_meta(self, reptile, model):
        """adapt() should return adapted state without modifying meta-model."""
        meta_state = {k: v.clone() for k, v in model.state_dict().items()}

        task_data = torch.randn(8, 16, device=config.DEVICE)

        def loss_fn(m, data):
            return F.mse_loss(m(data), torch.zeros(8, 4, device=config.DEVICE))

        result = reptile.adapt(task_data, loss_fn)

        # Meta-model should be unchanged
        for k in meta_state:
            assert torch.allclose(meta_state[k], model.state_dict()[k])

        # Adapted state should be different
        adapted = result["adapted_state"]
        different = False
        for k in meta_state:
            if not torch.allclose(meta_state[k], adapted[k]):
                different = True
                break
        assert different
        assert result["adaptation_steps"] == config.REPTILE_INNER_STEPS

    def test_empty_tasks(self, reptile):
        def loss_fn(m, data):
            return torch.tensor(0.0)

        result = reptile.meta_step([], loss_fn)
        assert result["meta_loss"] == 0.0

    def test_domains_tracked(self, reptile, model):
        tasks = [
            {"data": torch.randn(4, 16, device=config.DEVICE), "domain": "code_audit"},
            {"data": torch.randn(4, 16, device=config.DEVICE), "domain": "research"},
        ]

        def loss_fn(m, data):
            return F.mse_loss(m(data), torch.zeros(4, 4, device=config.DEVICE))

        reptile.meta_step(tasks, loss_fn)
        status = reptile.get_status()
        assert status["domains_seen"] == 2
        assert "code_audit" in status["domains"]
        assert "research" in status["domains"]

    def test_get_status(self, reptile):
        status = reptile.get_status()
        assert "meta_steps" in status
        assert "domains_seen" in status
        assert "adaptation_speed" in status
        assert "recent_meta_loss" in status
