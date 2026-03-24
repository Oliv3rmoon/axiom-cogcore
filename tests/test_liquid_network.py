"""Tests for the Liquid Time-Constant Network module."""
from __future__ import annotations

import pytest
import torch
import numpy as np

import config
from liquid_network.ltc_cell import LTCCell
from liquid_network.cfc_layer import CfCCell
from liquid_network.liquid_world_model import LiquidWorldModel

INPUT_SIZE = config.LTC_INPUT_SIZE
HIDDEN_SIZE = config.LTC_HIDDEN_SIZE


class TestLTCCell:
    @pytest.fixture
    def cell(self):
        return LTCCell(INPUT_SIZE, HIDDEN_SIZE).to(config.DEVICE)

    def test_forward_shape(self, cell):
        x = torch.randn(4, INPUT_SIZE, device=config.DEVICE)
        h = cell.initial_state(4)
        dt = torch.ones(4, 1, device=config.DEVICE)
        h_new = cell(x, h, dt)
        assert h_new.shape == (4, HIDDEN_SIZE)

    def test_initial_state(self, cell):
        h = cell.initial_state(2)
        assert h.shape == (2, HIDDEN_SIZE)
        assert (h == 0).all()

    def test_time_delta_affects_output(self, cell):
        x = torch.randn(1, INPUT_SIZE, device=config.DEVICE)
        h = cell.initial_state(1)
        h_short = cell(x, h, torch.tensor([[0.1]], device=config.DEVICE))
        h_long = cell(x, h, torch.tensor([[10.0]], device=config.DEVICE))
        # Different time deltas should produce different outputs
        assert not torch.allclose(h_short, h_long, atol=1e-4)

    def test_effective_time_constants(self, cell):
        tc = cell.effective_time_constants
        assert "min" in tc
        assert "max" in tc
        assert "mean" in tc

    def test_1d_time_delta(self, cell):
        x = torch.randn(4, INPUT_SIZE, device=config.DEVICE)
        h = cell.initial_state(4)
        dt = torch.ones(4, device=config.DEVICE)  # 1D
        h_new = cell(x, h, dt)
        assert h_new.shape == (4, HIDDEN_SIZE)


class TestCfCCell:
    @pytest.fixture
    def cell(self):
        return CfCCell(INPUT_SIZE, HIDDEN_SIZE).to(config.DEVICE)

    def test_forward_shape(self, cell):
        x = torch.randn(4, INPUT_SIZE, device=config.DEVICE)
        h = cell.initial_state(4)
        dt = torch.ones(4, 1, device=config.DEVICE)
        h_new = cell(x, h, dt)
        assert h_new.shape == (4, HIDDEN_SIZE)

    def test_initial_state(self, cell):
        h = cell.initial_state(2)
        assert h.shape == (2, HIDDEN_SIZE)
        assert (h == 0).all()

    def test_gate_bounded(self, cell):
        """Gate should be between 0 and 1 (Sigmoid output)."""
        x = torch.randn(4, INPUT_SIZE, device=config.DEVICE)
        h = cell.initial_state(4)
        dt = torch.ones(4, 1, device=config.DEVICE)
        combined = torch.cat([x, h, dt], dim=-1)
        gate = cell.gate_net(combined)
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_1d_time_delta(self, cell):
        x = torch.randn(4, INPUT_SIZE, device=config.DEVICE)
        h = cell.initial_state(4)
        dt = torch.ones(4, device=config.DEVICE)  # 1D
        h_new = cell(x, h, dt)
        assert h_new.shape == (4, HIDDEN_SIZE)

    def test_different_dt_different_output(self, cell):
        x = torch.randn(1, INPUT_SIZE, device=config.DEVICE)
        h = cell.initial_state(1)
        h_short = cell(x, h, torch.tensor([[0.1]], device=config.DEVICE))
        h_long = cell(x, h, torch.tensor([[10.0]], device=config.DEVICE))
        assert not torch.allclose(h_short, h_long, atol=1e-4)


class TestLiquidWorldModel:
    @pytest.fixture
    def model(self):
        return LiquidWorldModel(INPUT_SIZE, HIDDEN_SIZE).to(config.DEVICE)

    def test_forward(self, model):
        obs = torch.randn(4, INPUT_SIZE, device=config.DEVICE)
        h = model.initial_state(4)
        dt = torch.ones(4, 1, device=config.DEVICE)
        result = model(obs, h, dt)
        assert result["h"].shape == (4, HIDDEN_SIZE)
        assert result["predicted_outcome"].shape == (4, INPUT_SIZE)
        assert result["predicted_success"].shape == (4, 1)

    def test_success_bounded(self, model):
        obs = torch.randn(4, INPUT_SIZE, device=config.DEVICE)
        h = model.initial_state(4)
        dt = torch.ones(4, 1, device=config.DEVICE)
        result = model(obs, h, dt)
        s = result["predicted_success"]
        assert (s >= 0).all() and (s <= 1).all()

    def test_compute_loss(self, model):
        model.train()
        batch, seq = 4, 3
        obs = torch.randn(batch, seq, INPUT_SIZE, device=config.DEVICE)
        out = torch.randn(batch, seq, INPUT_SIZE, device=config.DEVICE)
        dt = torch.ones(batch, seq, 1, device=config.DEVICE)
        losses = model.compute_loss(obs, out, dt)
        assert "loss" in losses
        assert losses["loss"].requires_grad

    def test_backward(self, model):
        model.train()
        obs = torch.randn(2, 2, INPUT_SIZE, device=config.DEVICE)
        out = torch.randn(2, 2, INPUT_SIZE, device=config.DEVICE)
        dt = torch.ones(2, 2, 1, device=config.DEVICE)
        losses = model.compute_loss(obs, out, dt)
        losses["loss"].backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_get_status(self, model):
        status = model.get_status()
        assert status["model_type"] == "CfC"
        assert status["hidden_size"] == HIDDEN_SIZE
        assert status["total_parameters"] > 0

    def test_param_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total > 10000  # Should have meaningful params
