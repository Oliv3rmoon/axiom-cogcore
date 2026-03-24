from __future__ import annotations
"""Score which signals win broadcast access in the Global Workspace."""


def compute_salience(surprise: float = 0.0, relevance: float = 0.0,
                     urgency: float = 0.0, novelty: float = 0.0) -> float:
    """
    Salience = how important a signal is for current processing.

    salience = 0.3*surprise + 0.3*relevance + 0.2*urgency + 0.2*novelty
    """
    return (0.3 * surprise + 0.3 * relevance + 0.2 * urgency + 0.2 * novelty)
