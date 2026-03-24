from __future__ import annotations
"""Learned precision (confidence/attention weighting) for active inference."""

import config
from shared.db import get_db


class PrecisionController:
    """
    Manages the precision parameter (γ) for active inference.

    Precision adapts based on prediction accuracy:
    - Accurate predictions → increase precision → exploit more
    - Poor predictions → decrease precision → explore more

    precision ~ inverse prediction error (smoothed)
    """

    def __init__(self):
        self.precision = config.AI_PRECISION_INIT
        self.total_inferences = 0
        self._prediction_errors: list[float] = []

    def update(self, prediction_error: float) -> float:
        """
        Update precision based on observed prediction error.

        precision = precision + lr * (target - precision)
        target = 1 / (1 + error)
        """
        target_precision = 1.0 / (1.0 + prediction_error)
        self.precision += config.AI_PRECISION_LR * (target_precision - self.precision)
        self.precision = max(0.1, min(2.0, self.precision))

        self._prediction_errors.append(prediction_error)
        if len(self._prediction_errors) > 100:
            self._prediction_errors = self._prediction_errors[-100:]

        self.total_inferences += 1
        return self.precision

    @property
    def exploration_tendency(self) -> float:
        """Higher = more exploration (lower precision)."""
        return 1.0 - (self.precision - 0.1) / 1.9  # Map [0.1, 2.0] → [1.0, 0.0]

    def get_status(self) -> dict:
        """Get current precision state."""
        return {
            "precision": round(self.precision, 4),
            "exploration_tendency": round(self.exploration_tendency, 4),
            "total_inferences": self.total_inferences,
            "mean_prediction_error": round(
                sum(self._prediction_errors) / max(1, len(self._prediction_errors)), 4
            ) if self._prediction_errors else 0.0,
        }

    async def persist(self):
        """Save precision state to database."""
        db = await get_db()
        await db.execute(
            """INSERT INTO inference_state (precision, exploration_tendency, total_inferences)
               VALUES (?, ?, ?)""",
            (self.precision, self.exploration_tendency, self.total_inferences),
        )
        await db.commit()

    async def load_latest(self) -> bool:
        """Load latest precision state from database."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT precision, exploration_tendency, total_inferences "
            "FROM inference_state ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row:
            self.precision = row[0]
            self.total_inferences = row[2]
            return True
        return False
