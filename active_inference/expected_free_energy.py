from __future__ import annotations
"""Expected Free Energy computation — G(π) = ambiguity + risk."""

import torch
import torch.nn.functional as F
import numpy as np

import config
from active_inference.generative_model import GenerativeModel


def expected_free_energy(gen_model: GenerativeModel,
                         current_state: str,
                         action: str,
                         goal: str,
                         precision: float) -> dict:
    """
    Compute Expected Free Energy for a single policy (action).

    G(π) = -(precision * pragmatic + (1-precision) * epistemic)

    Lower G = better policy (more informative AND more goal-aligned).

    Returns dict with: efe, epistemic_value, pragmatic_value, ambiguity, risk,
                       predicted_success, confidence
    """
    # 1. Predict outcome with uncertainty
    predicted_emb, uncertainty, predicted_success = gen_model.predict_with_uncertainty(
        current_state, action
    )

    # 2. Epistemic value (curiosity) — how much would we learn?
    # High uncertainty = high learning value
    epistemic = uncertainty

    # 3. Pragmatic value (goal-directed) — does it serve our goals?
    goal_similarity = gen_model.compute_goal_similarity(predicted_emb, goal)
    pragmatic = max(0.0, goal_similarity)  # Clamp negative similarities

    # 4. Ambiguity and risk (decomposition of EFE)
    ambiguity = uncertainty  # Same as epistemic — how uncertain are outcomes
    risk = 1.0 - pragmatic  # How far from preferred outcomes

    # 5. Precision-weighted combination
    # High precision → exploit (favor pragmatic)
    # Low precision → explore (favor epistemic)
    clamped_precision = max(0.1, min(2.0, precision))
    normalized_precision = clamped_precision / 2.0  # Map [0.1, 2.0] → [0.05, 1.0]

    efe = -(normalized_precision * pragmatic + (1.0 - normalized_precision) * epistemic)

    # Confidence based on prediction certainty
    confidence = predicted_success * (1.0 - uncertainty * 0.5)

    return {
        "expected_free_energy": round(efe, 4),
        "epistemic_value": round(epistemic, 4),
        "pragmatic_value": round(pragmatic, 4),
        "ambiguity": round(ambiguity, 4),
        "risk": round(risk, 4),
        "predicted_success": round(predicted_success, 4),
        "confidence": round(confidence, 4),
    }
