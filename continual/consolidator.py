from __future__ import annotations
"""Memory consolidation process that runs periodically."""

import time
from datetime import datetime

import config
from continual.ewc import EWC
from continual.replay import ReplayManager
from world_model.model import RSSMWorldModel
from world_model.trainer import WorldModelTrainer
from world_model.buffer import ExperienceBuffer


class Consolidator:
    """
    Runs memory consolidation cycle:
    1. Compute Fisher Information Matrix
    2. Update EWC anchor parameters
    3. Prune low-value experiences
    4. Update model stats
    """

    def __init__(self, model: RSSMWorldModel, trainer: WorldModelTrainer, ewc: EWC):
        self.model = model
        self.trainer = trainer
        self.ewc = ewc
        self.replay_manager = ReplayManager(trainer.buffer)
        self.consolidation_count = 0
        self.last_consolidation: datetime | None = None
        self.model_version = 0

    async def consolidate(self) -> dict:
        """
        Run a full consolidation cycle.
        """
        start_time = time.time()

        # 1. Compute Fisher Information Matrix
        buffer_count = await self.trainer.buffer.count()
        fisher_updated = False
        if buffer_count >= 10:
            await self.ewc.compute_fisher(self.trainer.buffer)
            fisher_updated = True

            # Push Fisher/anchor to trainer for EWC penalty
            self.trainer.fisher_diag = self.ewc.fisher_diag
            self.trainer.anchor_params = self.ewc.anchor_params

        # 2. Prune low-value experiences
        pruned = await self.replay_manager.prune_low_value(keep_ratio=0.9)

        # 3. Train a few extra steps on replay buffer
        train_losses = []
        for _ in range(5):
            result = await self.trainer.train_step(batch_size=min(16, buffer_count))
            if result:
                train_losses.append(result["total_loss"])

        # 4. Save checkpoint
        self.model_version += 1
        if fisher_updated:
            await self.ewc.save_checkpoint(self.model_version)

        # 5. Update state
        self.consolidation_count += 1
        self.last_consolidation = datetime.utcnow()

        final_buffer_count = await self.trainer.buffer.count()

        # Count params
        ewc_params = sum(p.numel() for p in self.model.parameters())

        return {
            "fisher_updated": fisher_updated,
            "replay_buffer_size": final_buffer_count,
            "pruned_experiences": pruned,
            "ewc_params_anchored": ewc_params if fisher_updated else 0,
            "consolidation_time_seconds": round(time.time() - start_time, 2),
            "model_version": self.model_version,
            "train_losses": train_losses,
        }

    def get_status(self) -> dict:
        """Get consolidation status."""
        fisher_stats = self.ewc.get_fisher_stats()
        hours_since = None
        if self.last_consolidation:
            delta = datetime.utcnow() - self.last_consolidation
            hours_since = round(delta.total_seconds() / 3600, 1)

        return {
            "ewc_active": self.ewc.is_initialized,
            "fisher_matrix_age_hours": hours_since,
            "consolidation_count": self.consolidation_count,
            "last_consolidation": self.last_consolidation.isoformat() if self.last_consolidation else None,
            "model_version": self.model_version,
            "fisher_stats": fisher_stats,
        }
