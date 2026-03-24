"""Tests for the Global Workspace module."""
from __future__ import annotations

import pytest

from global_workspace.salience import compute_salience
from global_workspace.module_registry import ModuleRegistry, ModuleInfo
from global_workspace.broadcaster import Broadcaster, Signal
from global_workspace.workspace import GlobalWorkspace

import os
os.environ["DB_PATH"] = ":memory:"


class TestSalience:
    def test_zero_inputs(self):
        assert compute_salience() == 0.0

    def test_max_inputs(self):
        s = compute_salience(surprise=1.0, relevance=1.0, urgency=1.0, novelty=1.0)
        assert abs(s - 1.0) < 1e-6

    def test_weights_sum_to_one(self):
        # Each factor at 1.0 contributes its weight
        s = compute_salience(surprise=1.0, relevance=0.0, urgency=0.0, novelty=0.0)
        assert abs(s - 0.3) < 1e-6

    def test_partial(self):
        s = compute_salience(surprise=0.5, relevance=0.5, urgency=0.5, novelty=0.5)
        assert abs(s - 0.5) < 1e-6


class TestModuleRegistry:
    def test_register(self):
        reg = ModuleRegistry()
        reg.register("test_module", ["signal_a", "signal_b"])
        assert "test_module" in reg.modules

    def test_register_defaults(self):
        reg = ModuleRegistry()
        reg.register_defaults()
        assert "world_model" in reg.modules
        assert "curiosity" in reg.modules
        assert "dreamcoder" in reg.modules

    def test_get_subscribers(self):
        reg = ModuleRegistry()
        reg.register("mod_a", ["prediction_error"])
        reg.register("mod_b", ["prediction_error", "high_curiosity"])
        reg.register("mod_c", ["high_curiosity"])

        subs = reg.get_subscribers("prediction_error")
        assert "mod_a" in subs
        assert "mod_b" in subs
        assert "mod_c" not in subs

    def test_record_stats(self):
        reg = ModuleRegistry()
        reg.register("mod_a")
        reg.record_broadcast_sent("mod_a")
        assert reg.modules["mod_a"].broadcasts_sent == 1
        reg.record_broadcast_received("mod_a")
        assert reg.modules["mod_a"].broadcasts_received == 1

    def test_get_all(self):
        reg = ModuleRegistry()
        reg.register("mod_a")
        reg.register("mod_b")
        all_mods = reg.get_all()
        assert len(all_mods) == 2


class TestBroadcaster:
    def test_broadcast_to_subscribers(self):
        reg = ModuleRegistry()
        reg.register("world_model", ["prediction_error"])
        reg.register("curiosity", ["prediction_error"])
        reg.register("self_model", ["capability_change"])

        bc = Broadcaster(reg)
        signal = Signal("world_model", "prediction_error", {"error": 0.5}, 0.8)
        receivers = bc.broadcast(signal)

        # world_model should not receive its own broadcast
        assert "world_model" not in receivers
        assert "curiosity" in receivers
        # self_model not interested in prediction_error
        assert "self_model" not in receivers

    def test_broadcast_updates_stats(self):
        reg = ModuleRegistry()
        reg.register("source", ["signal_a"])
        reg.register("receiver", ["signal_a"])

        bc = Broadcaster(reg)
        signal = Signal("source", "signal_a", {}, 0.5)
        bc.broadcast(signal)

        assert reg.modules["source"].broadcasts_sent == 1
        assert reg.modules["receiver"].broadcasts_received == 1


class TestGlobalWorkspace:
    @pytest.fixture
    def workspace(self):
        reg = ModuleRegistry()
        reg.register_defaults()
        bc = Broadcaster(reg)
        return GlobalWorkspace(reg, bc)

    def test_submit_above_threshold(self, workspace):
        signal = Signal("world_model", "prediction_error", {"error": 0.8}, 0.9)
        result = workspace.submit(signal)
        assert result["accepted"] is True
        assert result["queue_position"] == 1

    def test_submit_below_threshold(self, workspace):
        signal = Signal("world_model", "prediction_error", {}, 0.1)
        result = workspace.submit(signal)
        assert result["accepted"] is False

    def test_competition_picks_highest_salience(self, workspace):
        s1 = Signal("curiosity", "high_curiosity", {}, 0.5, urgency=0.5)
        s2 = Signal("world_model", "prediction_error", {}, 0.9, urgency=0.8)
        workspace.submit(s1)
        workspace.submit(s2)
        winner = workspace.compete()
        assert winner is not None
        assert winner.source_module == "world_model"

    def test_compete_empty_queue(self, workspace):
        winner = workspace.compete()
        assert winner is None

    def test_broadcast_history(self, workspace):
        for i in range(5):
            signal = Signal(f"mod_{i}", "test", {}, 0.5 + i * 0.05)
            workspace.submit(signal)
            workspace.compete()
        assert workspace.total_broadcasts == 5
        assert len(workspace.history) == 5

    def test_get_current(self, workspace):
        state = workspace.get_current()
        assert "current_broadcast" in state
        assert "queue" in state
        assert "broadcast_history_count" in state

    def test_subscribe(self, workspace):
        workspace.subscribe("new_module", ["test_signal"])
        assert "new_module" in workspace.registry.modules
        subs = workspace.registry.get_subscribers("test_signal")
        assert "new_module" in subs
