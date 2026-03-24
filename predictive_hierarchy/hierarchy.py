from __future__ import annotations
"""Multi-level prediction error tracking (Clark/Friston predictive processing)."""

import json
from dataclasses import dataclass, field
from collections import deque

import config
from shared.db import get_db


@dataclass
class PredictionRecord:
    """A single prediction and its outcome."""
    level: int = 0
    context: str = ""
    predicted: float = 0.5
    actual: float | None = None
    error: float | None = None

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "context": self.context,
            "predicted": round(self.predicted, 3),
            "actual": round(self.actual, 3) if self.actual is not None else None,
            "error": round(self.error, 3) if self.error is not None else None,
        }


LEVEL_NAMES = ["step_outcomes", "plan_trajectories", "goal_achievability", "self_trajectory"]
LEVEL_DESCRIPTIONS = [
    "Individual step success/failure predictions",
    "Plan-level success predictions",
    "Goal-level outcome predictions",
    "Self-model predictions about own capability evolution",
]
LEVEL_TIMESCALES = ["seconds", "minutes", "hours", "days"]


class PredictiveHierarchy:
    """
    Four levels of prediction at different timescales.

    Level 0: step_outcomes (seconds)
    Level 1: plan_trajectories (minutes)
    Level 2: goal_achievability (hours)
    Level 3: self_trajectory (days)

    Errors propagate UP. Precision propagates DOWN.
    """

    def __init__(self):
        n = config.HIERARCHY_LEVELS
        self.mean_errors = [0.0] * n
        self.precisions = [1.0] * n
        self.total_predictions = [0] * n
        self.recent_errors: list[deque] = [
            deque(maxlen=20) for _ in range(n)
        ]
        self.recent_records: list[deque] = [
            deque(maxlen=10) for _ in range(n)
        ]

    def predict(self, level: int, context: str) -> dict:
        """Generate prediction at specified level."""
        level = max(0, min(level, config.HIERARCHY_LEVELS - 1))
        # Use precision as confidence proxy
        confidence = self.precisions[level]
        # Base prediction on mean error (lower error = predict success)
        predicted = 1.0 - self.mean_errors[level]

        record = PredictionRecord(level=level, context=context, predicted=predicted)
        self.recent_records[level].append(record)

        return {
            "predicted_outcome": "success" if predicted > 0.5 else "failure",
            "confidence": round(confidence, 3),
            "predicted_value": round(predicted, 3),
            "level": level,
            "level_name": LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else f"level_{level}",
        }

    def update(self, level: int, predicted: float, actual: float,
               context: str = "") -> dict:
        """
        Update with actual outcome. Error propagates UP, precision adjusts DOWN.

        error_i+1 = α * error_i + (1-α) * error_i+1
        precision_i = precision_i * (1 - lr * error_i)
        """
        level = max(0, min(level, config.HIERARCHY_LEVELS - 1))
        error = abs(predicted - actual)

        # Update this level
        self.total_predictions[level] += 1
        self.recent_errors[level].append(error)
        self.mean_errors[level] = sum(self.recent_errors[level]) / len(self.recent_errors[level])

        # Store record
        record = PredictionRecord(level=level, context=context,
                                   predicted=predicted, actual=actual, error=error)
        self.recent_records[level].append(record)

        # Propagate error UP the hierarchy
        alpha = config.HIERARCHY_PROPAGATION_RATE
        result = {f"error_at_level_{level}": round(error, 4)}
        propagated_error = error

        for up_level in range(level + 1, config.HIERARCHY_LEVELS):
            propagated_error = alpha * propagated_error + (1 - alpha) * self.mean_errors[up_level]
            self.mean_errors[up_level] = propagated_error
            result[f"propagated_to_level_{up_level}"] = round(propagated_error, 4)

        # Update precision DOWN the hierarchy (from this level down)
        lr = config.HIERARCHY_PRECISION_LR
        precision_updates = {}
        for down_level in range(level, -1, -1):
            err = self.mean_errors[down_level]
            self.precisions[down_level] = max(
                0.1, self.precisions[down_level] * (1 - lr * err)
            )
            precision_updates[f"level_{down_level}"] = round(self.precisions[down_level], 4)

        result["precision_updates"] = precision_updates
        return result

    def get_state(self) -> dict:
        """Get full hierarchy state."""
        levels = []
        for i in range(config.HIERARCHY_LEVELS):
            recent = [r.to_dict() for r in list(self.recent_records[i])[-3:]]
            levels.append({
                "level": i,
                "name": LEVEL_NAMES[i] if i < len(LEVEL_NAMES) else f"level_{i}",
                "description": LEVEL_DESCRIPTIONS[i] if i < len(LEVEL_DESCRIPTIONS) else "",
                "timescale": LEVEL_TIMESCALES[i] if i < len(LEVEL_TIMESCALES) else "unknown",
                "mean_prediction_error": round(self.mean_errors[i], 4),
                "precision": round(self.precisions[i], 4),
                "total_predictions": self.total_predictions[i],
                "recent_errors": recent,
            })

        total = sum(self.total_predictions)
        all_errors = []
        for errs in self.recent_errors:
            all_errors.extend(errs)
        global_surprise = sum(all_errors) / max(1, len(all_errors)) if all_errors else 0.0

        return {
            "levels": levels,
            "total_prediction_errors": total,
            "global_surprise": round(global_surprise, 4),
        }

    def get_precision_weighted_error(self, level: int) -> float:
        """
        Precision-weighted prediction error = attention signal.
        High precision * high error = VERY salient
        Low precision * high error = expected uncertainty (ignore)
        """
        level = max(0, min(level, config.HIERARCHY_LEVELS - 1))
        return self.precisions[level] * self.mean_errors[level]

    async def persist(self):
        """Save hierarchy state to database."""
        db = await get_db()
        for i in range(config.HIERARCHY_LEVELS):
            name = LEVEL_NAMES[i] if i < len(LEVEL_NAMES) else f"level_{i}"
            await db.execute(
                """INSERT INTO prediction_levels
                   (level, level_name, mean_error, precision, total_predictions, updated_at)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (i, name, self.mean_errors[i], self.precisions[i], self.total_predictions[i]),
            )
        await db.commit()
