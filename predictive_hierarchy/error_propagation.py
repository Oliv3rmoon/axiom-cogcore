from __future__ import annotations
"""How prediction errors flow up/down the hierarchy."""

import config
from predictive_hierarchy.hierarchy import PredictiveHierarchy


def propagate_error_up(hierarchy: PredictiveHierarchy, source_level: int,
                       error: float) -> dict:
    """
    Propagate error from source_level upward.
    error_i+1 = α * error_from_below + (1-α) * current_error_i+1
    """
    alpha = config.HIERARCHY_PROPAGATION_RATE
    propagated = error
    updates = {f"level_{source_level}": round(error, 4)}

    for level in range(source_level + 1, config.HIERARCHY_LEVELS):
        propagated = alpha * propagated + (1 - alpha) * hierarchy.mean_errors[level]
        hierarchy.mean_errors[level] = propagated
        updates[f"level_{level}"] = round(propagated, 4)

    return updates


def propagate_precision_down(hierarchy: PredictiveHierarchy,
                             source_level: int) -> dict:
    """
    Propagate precision updates from source_level downward.
    Higher levels constrain lower levels.
    """
    lr = config.HIERARCHY_PRECISION_LR
    updates = {}

    for level in range(source_level, -1, -1):
        err = hierarchy.mean_errors[level]
        hierarchy.precisions[level] = max(
            0.1, hierarchy.precisions[level] * (1 - lr * err)
        )
        updates[f"level_{level}"] = round(hierarchy.precisions[level], 4)

    return updates
