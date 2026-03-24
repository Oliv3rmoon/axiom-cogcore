from __future__ import annotations
"""Reptile meta-learning algorithm for rapid domain adaptation."""

import copy
import torch
import torch.nn as nn
import torch.optim as optim

import config


class Reptile:
    """
    Reptile meta-learning: learn a meta-initialization Φ so the model
    can rapidly adapt to new tasks with just a few SGD steps.

    Algorithm:
        Φ ← Φ + ε(W̃ - Φ)
    where W̃ = result of k SGD steps on task T starting from Φ
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.meta_step_count = 0
        self._domains_seen: set[str] = set()
        self._meta_losses: list[float] = []

    def meta_step(self, tasks: list[dict],
                  loss_fn,
                  inner_lr: float = config.REPTILE_INNER_LR,
                  outer_lr: float = config.REPTILE_OUTER_LR,
                  inner_steps: int = config.REPTILE_INNER_STEPS) -> dict:
        """
        One Reptile meta-update step over a batch of tasks.

        Each task is a dict with at least a 'data' key (tensor batch)
        and optionally a 'domain' key.

        loss_fn: callable(model, task_data) -> loss tensor
        """
        if not tasks:
            return {"meta_loss": 0.0, "inner_steps": 0, "outer_step": self.meta_step_count}

        meta_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        accumulated_diff = {k: torch.zeros_like(v) for k, v in meta_state.items()}

        total_loss = 0.0
        for task in tasks:
            # Reset to meta parameters
            self.model.load_state_dict({k: v.clone() for k, v in meta_state.items()})

            # Inner loop: k SGD steps on this task
            inner_opt = optim.SGD(self.model.parameters(), lr=inner_lr)
            task_loss = 0.0
            for step in range(inner_steps):
                loss = loss_fn(self.model, task["data"])
                inner_opt.zero_grad()
                loss.backward()
                inner_opt.step()
                task_loss += loss.item()

            total_loss += task_loss / inner_steps

            # Accumulate (W̃ - Φ)
            for k, v in self.model.state_dict().items():
                accumulated_diff[k] += (v - meta_state[k])

            # Track domains
            if "domain" in task:
                self._domains_seen.add(task["domain"])

        # Average over tasks
        n_tasks = len(tasks)
        for k in accumulated_diff:
            accumulated_diff[k] /= n_tasks

        # Meta-update: Φ ← Φ + ε · avg(W̃ - Φ)
        new_state = {}
        for k in meta_state:
            new_state[k] = meta_state[k] + outer_lr * accumulated_diff[k]

        self.model.load_state_dict(new_state)
        self.meta_step_count += 1

        avg_loss = total_loss / n_tasks
        self._meta_losses.append(avg_loss)

        return {
            "meta_loss": round(avg_loss, 4),
            "inner_steps": inner_steps,
            "outer_step": self.meta_step_count,
        }

    def adapt(self, task_data: torch.Tensor, loss_fn,
              steps: int = config.REPTILE_INNER_STEPS,
              lr: float = config.REPTILE_INNER_LR) -> dict:
        """
        Rapidly adapt current model to a specific task.
        Returns adapted model state (does NOT modify the meta-model).
        """
        # Save meta state
        meta_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        inner_opt = optim.SGD(self.model.parameters(), lr=lr)
        losses = []
        for step in range(steps):
            loss = loss_fn(self.model, task_data)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
            losses.append(loss.item())

        # Capture adapted state
        adapted_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Restore meta state
        self.model.load_state_dict(meta_state)

        return {
            "adapted_state": adapted_state,
            "final_loss": round(losses[-1], 4) if losses else 0.0,
            "loss_trajectory": [round(l, 4) for l in losses],
            "adaptation_steps": steps,
        }

    def get_status(self) -> dict:
        """Get meta-learning status."""
        recent = self._meta_losses[-20:] if self._meta_losses else []
        adaptation_speed = 0.0
        if len(recent) >= 2:
            # Speed = how fast loss is decreasing
            first_half = sum(recent[:len(recent)//2]) / max(1, len(recent)//2)
            second_half = sum(recent[len(recent)//2:]) / max(1, len(recent) - len(recent)//2)
            if first_half > 0:
                adaptation_speed = (first_half - second_half) / first_half

        return {
            "meta_steps": self.meta_step_count,
            "domains_seen": len(self._domains_seen),
            "domains": sorted(self._domains_seen),
            "adaptation_speed": round(max(0, adaptation_speed), 4),
            "recent_meta_loss": round(recent[-1], 4) if recent else 0.0,
        }
