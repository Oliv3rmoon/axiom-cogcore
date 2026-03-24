from __future__ import annotations
"""Elastic Weight Consolidation for preventing catastrophic forgetting."""

import io
import torch
import numpy as np

import config
from world_model.model import RSSMWorldModel
from world_model.buffer import ExperienceBuffer
from shared.db import get_db


class EWC:
    """
    Elastic Weight Consolidation.

    Computes diagonal Fisher Information Matrix to identify important parameters,
    then penalizes changes to those parameters during future training.

    L_EWC = L_new + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
    """

    def __init__(self, model: RSSMWorldModel):
        self.model = model
        self.fisher_diag: dict[str, torch.Tensor] = {}
        self.anchor_params: dict[str, torch.Tensor] = {}
        self.is_initialized = False
        self.computation_count = 0

    async def compute_fisher(self, buffer: ExperienceBuffer, n_samples: int = 500):
        """
        Compute diagonal Fisher Information Matrix from replay buffer.
        F_i = E[(d log p(y|x;theta) / d theta_i)^2]
        Approximated as average squared gradient over mini-batches.
        """
        batch = await buffer.get_all_for_fisher(limit=n_samples)
        if batch is None:
            return

        self.model.eval()

        # Initialize Fisher to zero
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param, device=config.DEVICE)

        # Compute squared gradients over the batch
        n_computed = 0
        batch_size = batch["obs_embeddings"].shape[0]

        # Process in mini-batches of 8
        for start in range(0, batch_size, 8):
            end = min(start + 8, batch_size)
            mini_obs = batch["obs_embeddings"][start:end]
            mini_act = batch["actions"][start:end]
            mini_out = batch["outcome_embeddings"][start:end]
            mini_suc = batch["successes"][start:end]

            self.model.zero_grad()
            losses = self.model.compute_loss(mini_obs, mini_act, mini_out, mini_suc)
            losses["loss"].backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            n_computed += (end - start)

        # Average the Fisher
        if n_computed > 0:
            for name in fisher:
                fisher[name] /= n_computed

        # Store Fisher and anchor parameters
        self.fisher_diag = fisher
        self.anchor_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        self.is_initialized = True
        self.computation_count += 1
        self.model.train()

    def get_penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
        """
        if not self.is_initialized:
            return torch.tensor(0.0, device=config.DEVICE)

        penalty = torch.tensor(0.0, device=config.DEVICE)
        for name, param in self.model.named_parameters():
            if name in self.fisher_diag and name in self.anchor_params:
                penalty += (
                    self.fisher_diag[name] * (param - self.anchor_params[name]).pow(2)
                ).sum()

        return (config.EWC_LAMBDA / 2.0) * penalty

    def get_fisher_stats(self) -> dict:
        """Get statistics about the Fisher Information Matrix."""
        if not self.is_initialized:
            return {"initialized": False, "computation_count": 0}

        total_params = 0
        total_fisher_mass = 0.0
        for name, f in self.fisher_diag.items():
            total_params += f.numel()
            total_fisher_mass += float(f.sum().item())

        return {
            "initialized": True,
            "computation_count": self.computation_count,
            "total_params_tracked": total_params,
            "mean_fisher_value": total_fisher_mass / max(1, total_params),
            "fisher_memory_mb": total_params * 4 / (1024 * 1024),
        }

    async def save_checkpoint(self, version: int):
        """Save Fisher and anchor params to database."""
        db = await get_db()

        fisher_buf = io.BytesIO()
        torch.save(self.fisher_diag, fisher_buf)
        fisher_bytes = fisher_buf.getvalue()

        anchor_buf = io.BytesIO()
        torch.save(self.anchor_params, anchor_buf)
        anchor_bytes = anchor_buf.getvalue()

        import json
        stats_json = json.dumps(self.get_fisher_stats())

        await db.execute(
            """INSERT INTO model_checkpoints (version, fisher_diagonal, anchor_params, stats_json)
               VALUES (?, ?, ?, ?)""",
            (version, fisher_bytes, anchor_bytes, stats_json),
        )
        await db.commit()

    async def load_latest_checkpoint(self) -> bool:
        """Load the latest Fisher/anchor checkpoint from database."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT fisher_diagonal, anchor_params FROM model_checkpoints ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return False

        fisher_bytes, anchor_bytes = row[0], row[1]
        if fisher_bytes and anchor_bytes:
            fisher_buf = io.BytesIO(fisher_bytes)
            self.fisher_diag = torch.load(fisher_buf, map_location=config.DEVICE, weights_only=True)
            anchor_buf = io.BytesIO(anchor_bytes)
            self.anchor_params = torch.load(anchor_buf, map_location=config.DEVICE, weights_only=True)
            self.is_initialized = True
            return True
        return False
