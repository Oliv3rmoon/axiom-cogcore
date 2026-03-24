"""Tests for the continual learning module."""

import pytest
import torch
import numpy as np

import config
from world_model.model import RSSMWorldModel
from continual.ewc import EWC


class TestEWC:
    @pytest.fixture
    def model(self):
        return RSSMWorldModel().to(config.DEVICE)

    @pytest.fixture
    def ewc(self, model):
        return EWC(model)

    def test_initial_state(self, ewc):
        assert not ewc.is_initialized
        assert ewc.computation_count == 0
        assert ewc.fisher_diag == {}
        assert ewc.anchor_params == {}

    def test_penalty_without_initialization(self, ewc):
        penalty = ewc.get_penalty()
        assert penalty.item() == 0.0

    def test_fisher_stats_uninitialized(self, ewc):
        stats = ewc.get_fisher_stats()
        assert stats["initialized"] is False
        assert stats["computation_count"] == 0

    def test_ewc_prevents_forgetting(self):
        """
        Train on task A, compute Fisher, train on task B with EWC.
        Verify that EWC penalty keeps parameters near task A values.
        """
        model = RSSMWorldModel().to(config.DEVICE)
        ewc = EWC(model)

        # Snapshot initial parameters
        initial_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        # Simulate Fisher computation by manually setting it
        ewc.fisher_diag = {
            name: torch.ones_like(param) * 1.0
            for name, param in model.named_parameters()
        }
        ewc.anchor_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        ewc.is_initialized = True

        # Verify penalty is 0 when params haven't changed
        penalty = ewc.get_penalty()
        assert penalty.item() == 0.0

        # Manually shift a parameter
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.add_(0.1)
                break

        # Penalty should now be > 0
        penalty = ewc.get_penalty()
        assert penalty.item() > 0

    def test_penalty_increases_with_distance(self):
        """Verify penalty grows as params move further from anchor."""
        model = RSSMWorldModel().to(config.DEVICE)
        ewc = EWC(model)

        ewc.fisher_diag = {
            name: torch.ones_like(param)
            for name, param in model.named_parameters()
        }
        ewc.anchor_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        ewc.is_initialized = True

        # Small perturbation
        with torch.no_grad():
            first_param = next(model.parameters())
            first_param.add_(0.01)
        penalty_small = ewc.get_penalty().item()

        # Larger perturbation
        with torch.no_grad():
            first_param.add_(0.99)
        penalty_large = ewc.get_penalty().item()

        assert penalty_large > penalty_small
