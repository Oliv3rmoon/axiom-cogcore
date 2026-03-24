from __future__ import annotations
"""Precision = attention = confidence in predictions."""

from predictive_hierarchy.hierarchy import PredictiveHierarchy


def compute_precision_weighted_salience(hierarchy: PredictiveHierarchy) -> dict:
    """
    Compute precision-weighted error across all levels.
    This is computationally equivalent to attention allocation.

    High precision * high error = attend urgently
    Low precision * high error = expected, ignore
    High precision * low error = all good, low attention needed
    """
    import config
    salience = {}
    for level in range(config.HIERARCHY_LEVELS):
        pwe = hierarchy.get_precision_weighted_error(level)
        from predictive_hierarchy.hierarchy import LEVEL_NAMES
        name = LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else f"level_{level}"
        salience[name] = round(pwe, 4)

    # Global attention = max salience across levels
    global_attention = max(salience.values()) if salience else 0.0

    return {
        "per_level_salience": salience,
        "global_attention_demand": round(global_attention, 4),
        "most_salient_level": max(salience, key=salience.get) if salience else "none",
    }
