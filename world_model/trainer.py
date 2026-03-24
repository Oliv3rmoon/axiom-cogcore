from __future__ import annotations
"""Training loop for the world model with EWC support."""

import torch
import torch.optim as optim

import config
from world_model.model import RSSMWorldModel
from world_model.buffer import ExperienceBuffer


class WorldModelTrainer:
    """Manages world model training with online learning and EWC."""

    def __init__(self, model: RSSMWorldModel):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=config.WORLD_MODEL_LR)
        self.buffer = ExperienceBuffer()
        self.train_step_count = 0
        self.total_loss_history: list[float] = []

        # EWC state (set externally by consolidator)
        self.fisher_diag: dict[str, torch.Tensor] | None = None
        self.anchor_params: dict[str, torch.Tensor] | None = None

    async def train_step(self, batch_size: int | None = None,
                         seq_len: int | None = None) -> dict | None:
        """
        Run a single training step from the replay buffer.
        Returns loss dict or None if not enough data.
        """
        bs = batch_size or config.WORLD_MODEL_BATCH_SIZE
        sl = seq_len or config.WORLD_MODEL_SEQUENCE_LENGTH

        # Try sequence training first, fall back to single-step
        batch = await self.buffer.sample_batch(bs, sl)
        if batch is None:
            batch = await self.buffer.sample_batch(min(bs, 8), 1)
            if batch is None:
                return None

        self.model.train()
        self.optimizer.zero_grad()

        losses = self.model.compute_loss(
            batch["obs_embeddings"],
            batch["actions"],
            batch["outcome_embeddings"],
            batch["successes"],
        )

        total_loss = losses["loss"]

        # Add EWC penalty if available
        ewc_loss = torch.tensor(0.0, device=config.DEVICE)
        if self.fisher_diag is not None and self.anchor_params is not None:
            for name, param in self.model.named_parameters():
                if name in self.fisher_diag and name in self.anchor_params:
                    fisher = self.fisher_diag[name]
                    anchor = self.anchor_params[name]
                    ewc_loss += (fisher * (param - anchor).pow(2)).sum()
            ewc_loss *= config.EWC_LAMBDA / 2.0
            total_loss = total_loss + ewc_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
        self.optimizer.step()

        self.train_step_count += 1
        loss_val = total_loss.item()
        self.total_loss_history.append(loss_val)

        return {
            "total_loss": loss_val,
            "recon_loss": losses["recon_loss"].item(),
            "kl_loss": losses["kl_loss"].item(),
            "success_loss": losses["success_loss"].item(),
            "ewc_loss": ewc_loss.item(),
            "train_step": self.train_step_count,
        }

    async def maybe_train(self, new_experience_count: int) -> dict | None:
        """Train if we've accumulated enough new experiences."""
        if new_experience_count % config.WORLD_MODEL_TRAIN_EVERY == 0:
            return await self.train_step()
        return None

    def get_stats(self) -> dict:
        """Return training statistics."""
        recent_losses = self.total_loss_history[-100:] if self.total_loss_history else []
        return {
            "train_steps": self.train_step_count,
            "mean_recent_loss": sum(recent_losses) / len(recent_losses) if recent_losses else 0.0,
            "total_experiences_trained": self.train_step_count * config.WORLD_MODEL_BATCH_SIZE,
        }
