from __future__ import annotations
"""Experience replay buffer management for continual learning."""

import numpy as np

import config
from shared.db import get_db
from world_model.buffer import ExperienceBuffer, _decode_array


class ReplayManager:
    """
    Manages the experience replay buffer with priority-based sampling.
    Higher prediction error = more likely to be replayed.
    """

    def __init__(self, buffer: ExperienceBuffer):
        self.buffer = buffer

    async def get_stats(self) -> dict:
        """Get replay buffer statistics."""
        db = await get_db()
        count = await self.buffer.count()

        cursor = await db.execute(
            "SELECT AVG(prediction_error), MAX(prediction_error), MIN(prediction_error) FROM experiences"
        )
        row = await cursor.fetchone()
        avg_error = row[0] or 0.0
        max_error = row[1] or 0.0
        min_error = row[2] or 0.0

        cursor = await db.execute(
            "SELECT action, COUNT(*) FROM experiences GROUP BY action ORDER BY COUNT(*) DESC"
        )
        action_dist = {}
        async for row in cursor:
            action_dist[row[0]] = row[1]

        return {
            "total_experiences": count,
            "buffer_fill": round(count / config.REPLAY_BUFFER_SIZE, 3),
            "avg_prediction_error": round(avg_error, 4),
            "max_prediction_error": round(max_error, 4),
            "min_prediction_error": round(min_error, 4),
            "action_distribution": action_dist,
        }

    async def prune_low_value(self, keep_ratio: float = 0.9) -> int:
        """
        Prune low-value experiences (low prediction error + old).
        Returns number of pruned experiences.
        """
        count = await self.buffer.count()
        target = int(count * keep_ratio)
        to_prune = count - target

        if to_prune <= 0:
            return 0

        db = await get_db()
        await db.execute(
            """DELETE FROM experiences WHERE id IN
               (SELECT id FROM experiences
                ORDER BY prediction_error ASC, created_at ASC LIMIT ?)""",
            (to_prune,),
        )
        await db.commit()
        self.buffer._count_cache = None
        return to_prune

    async def get_domain_distribution(self) -> dict[str, int]:
        """Get count of experiences per action type."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT action, COUNT(*) FROM experiences GROUP BY action"
        )
        result = {}
        async for row in cursor:
            result[row[0]] = row[1]
        return result
