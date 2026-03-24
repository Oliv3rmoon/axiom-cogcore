from __future__ import annotations
"""Select actions via softmax over -G(π)."""

import torch
import torch.nn.functional as F
import numpy as np

import config
from active_inference.generative_model import GenerativeModel
from active_inference.expected_free_energy import expected_free_energy


def compare_policies(gen_model: GenerativeModel,
                     current_state: str,
                     policies: list[str],
                     goal: str,
                     precision: float,
                     gamma: float = config.AI_GAMMA) -> dict:
    """
    Compare multiple action options using Expected Free Energy.

    P(π) = softmax(-γ · G(π))

    Returns ranked policies with EFE decomposition.
    """
    if not policies:
        return {
            "ranked_policies": [],
            "best_action": "",
            "exploration_exploitation_ratio": 0.5,
        }

    results = []
    efes = []
    for action in policies:
        efe_result = expected_free_energy(gen_model, current_state, action, goal, precision)
        results.append({
            "action": action,
            "efe": efe_result["expected_free_energy"],
            "epistemic": efe_result["epistemic_value"],
            "pragmatic": efe_result["pragmatic_value"],
            "confidence": efe_result["confidence"],
        })
        efes.append(efe_result["expected_free_energy"])

    # Softmax over negative EFE (lower EFE = higher probability)
    efe_tensor = torch.tensor(efes, dtype=torch.float32)
    probs = F.softmax(-gamma * efe_tensor, dim=0).tolist()

    # Attach probabilities
    for i, r in enumerate(results):
        r["probability"] = round(float(probs[i]), 4)

    # Sort by EFE (lower is better)
    results.sort(key=lambda x: x["efe"])

    # Exploration-exploitation ratio
    # High ratio = more exploration (epistemic-driven)
    total_epistemic = sum(r["epistemic"] for r in results)
    total_pragmatic = sum(r["pragmatic"] for r in results)
    if total_epistemic + total_pragmatic > 0:
        ratio = total_epistemic / (total_epistemic + total_pragmatic)
    else:
        ratio = 0.5

    return {
        "ranked_policies": results,
        "best_action": results[0]["action"] if results else "",
        "exploration_exploitation_ratio": round(ratio, 3),
    }


def recommend_action(efe_result: dict) -> str:
    """Generate a recommendation string based on EFE analysis."""
    efe = efe_result["expected_free_energy"]
    epistemic = efe_result["epistemic_value"]
    pragmatic = efe_result["pragmatic_value"]

    if efe < -1.0:
        return "proceed"
    elif efe < 0.0:
        if epistemic > pragmatic:
            return "explore"
        else:
            return "proceed"
    else:
        return "reconsider"
