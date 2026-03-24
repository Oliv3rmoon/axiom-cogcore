from __future__ import annotations
"""Attention Schema — tracks what AXIOM is attending to (Graziano AST)."""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import config
from shared.db import get_db


@dataclass
class AttentionTarget:
    """What AXIOM is currently focusing on."""
    target: str = ""
    target_type: str = "unknown"  # goal, step, domain, signal
    attention_strength: float = 0.0
    signals: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "target_type": self.target_type,
            "attention_strength": round(self.attention_strength, 3),
            "duration_seconds": round(self.duration_seconds, 1),
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
        }


class AttentionSchema:
    """
    Graziano's Attention Schema Theory implementation.

    AXIOM models her own attention process — knows what she's focusing on,
    predicts attention shifts, and uses that self-knowledge for meta-cognitive control.
    """

    def __init__(self):
        self.current_focus: AttentionTarget | None = None
        self.focus_history: deque[AttentionTarget] = deque(
            maxlen=config.ATTENTION_HISTORY_SIZE
        )
        self.attention_weights: dict[str, float] = {}  # domain -> cumulative attention
        self._domain_counts: dict[str, int] = {}

    def compute_attention_strength(self, signals: dict) -> float:
        """
        Attention = weighted combination of:
        - curiosity_score (0.3)
        - prediction_error (0.25)
        - goal_relevance (0.25)
        - broadcast_salience (0.2)
        """
        curiosity = signals.get("curiosity_score", 0.0)
        pred_error = signals.get("prediction_error", 0.0)
        goal_rel = signals.get("goal_relevance", 0.0)
        salience = signals.get("broadcast_salience", 0.0)

        return (0.3 * curiosity + 0.25 * pred_error +
                0.25 * goal_rel + 0.2 * salience)

    def update_focus(self, target: str, target_type: str,
                     signals: dict) -> dict:
        """Update what AXIOM is attending to."""
        strength = self.compute_attention_strength(signals)

        # Close out previous focus duration
        if self.current_focus:
            self.current_focus.duration_seconds = time.time() - self.current_focus.timestamp
            self.focus_history.append(self.current_focus)

        new_target = AttentionTarget(
            target=target,
            target_type=target_type,
            attention_strength=strength,
            signals=signals,
        )

        # Track per-domain attention
        domain = target_type if target_type != "unknown" else target.split()[0] if target else "unknown"
        self.attention_weights[domain] = self.attention_weights.get(domain, 0.0) + strength
        self._domain_counts[domain] = self._domain_counts.get(domain, 0) + 1

        # Check if we should switch from current to a competing target
        competing = self._get_competing_targets()
        should_switch = True
        if self.current_focus and self.current_focus.target == target:
            should_switch = False
        elif self.current_focus:
            should_switch = self.should_switch_attention(
                self.current_focus.attention_strength, strength
            )

        if should_switch:
            self.current_focus = new_target

        return {
            "attention_strength": round(strength, 3),
            "should_switch": should_switch,
            "competing_targets": competing[:3],
        }

    def should_switch_attention(self, current_strength: float,
                                new_strength: float) -> bool:
        """Switch when new target exceeds current by > threshold (hysteresis)."""
        return (new_strength - current_strength) > config.ATTENTION_SWITCH_THRESHOLD

    def get_focus(self) -> dict:
        """Get current attention state."""
        history = [t.to_dict() for t in list(self.focus_history)[-10:]]
        predicted = self._predict_next_shift()

        if self.current_focus:
            self.current_focus.duration_seconds = time.time() - self.current_focus.timestamp

        return {
            "current_focus": self.current_focus.to_dict() if self.current_focus else None,
            "attention_history": history,
            "predicted_next_shift": predicted,
        }

    def _predict_next_shift(self) -> dict:
        """Predict where attention will shift next."""
        if len(self.focus_history) < 2:
            return {"likely_target": "unknown", "probability": 0.5, "reason": "Insufficient history"}

        # Find domains with high attention weight but not currently focused
        current_domain = ""
        if self.current_focus:
            current_domain = self.current_focus.target_type

        candidates = []
        for domain, weight in self.attention_weights.items():
            if domain != current_domain:
                avg_weight = weight / max(1, self._domain_counts.get(domain, 1))
                candidates.append((domain, avg_weight))

        if not candidates:
            return {"likely_target": "unknown", "probability": 0.3, "reason": "No competing domains"}

        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        return {
            "likely_target": best[0],
            "probability": round(min(0.9, best[1]), 2),
            "reason": f"Domain '{best[0]}' has accumulated attention weight",
        }

    def _get_competing_targets(self) -> list[dict]:
        """Get targets competing for attention."""
        if len(self.focus_history) < 1:
            return []
        recent = list(self.focus_history)[-5:]
        seen = set()
        competing = []
        for t in reversed(recent):
            if t.target not in seen:
                # Decay strength over time
                age = time.time() - t.timestamp
                decayed = t.attention_strength * (config.ATTENTION_DECAY_RATE ** (age / 60.0))
                competing.append({"target": t.target, "strength": round(decayed, 3)})
                seen.add(t.target)
        return competing

    async def persist_focus(self):
        """Save current focus to database."""
        if not self.current_focus:
            return
        db = await get_db()
        await db.execute(
            """INSERT INTO attention_history
               (target, target_type, attention_strength, signals, duration_seconds)
               VALUES (?, ?, ?, ?, ?)""",
            (self.current_focus.target, self.current_focus.target_type,
             self.current_focus.attention_strength,
             json.dumps(self.current_focus.signals),
             self.current_focus.duration_seconds),
        )
        await db.commit()
