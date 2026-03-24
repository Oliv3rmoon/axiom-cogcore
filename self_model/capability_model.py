from __future__ import annotations
"""Model of AXIOM's capabilities and confidence levels."""

import time
from collections import Counter

import config
from shared.backend_client import BackendClient


class CapabilityModel:
    """
    For each action_type, computes capability confidence:

    confidence = 0.5 * success_rate + 0.3 * recency_weight + 0.2 * lesson_quality
    """

    def __init__(self, backend: BackendClient):
        self.backend = backend
        self._capabilities: dict[str, float] = {}
        self._last_update: float = 0.0

    async def update(self) -> dict[str, float]:
        """Recompute capability confidences from backend data."""
        lessons = await self.backend.get_lessons()

        # Group by action type
        action_data: dict[str, list[dict]] = {}
        for lesson in lessons:
            action = lesson.get("action_type", "unknown")
            if action not in action_data:
                action_data[action] = []
            action_data[action].append(lesson)

        capabilities = {}
        for action, action_lessons in action_data.items():
            # Success rate
            total = len(action_lessons)
            successes = len([l for l in action_lessons if l.get("success")])
            success_rate = successes / max(1, total)

            # Recency weight: more recent usage = higher weight
            timestamps = []
            for l in action_lessons:
                ts = l.get("created_at", "")
                if ts:
                    timestamps.append(ts)
            if timestamps:
                # Simple proxy: how many lessons in last week
                recent_count = min(10, total)  # Cap at 10
                recency = recent_count / 10.0
            else:
                recency = 0.0

            # Lesson quality: average confidence
            confidences = [l.get("confidence", 0.5) for l in action_lessons]
            quality = sum(confidences) / max(1, len(confidences))

            # Combined score
            confidence = 0.5 * success_rate + 0.3 * recency + 0.2 * quality
            capabilities[action] = round(confidence, 2)

        self._capabilities = capabilities
        self._last_update = time.time()
        return capabilities

    async def get_capabilities(self) -> dict[str, float]:
        """Get current capabilities, updating if stale."""
        if time.time() - self._last_update > config.CAPABILITY_UPDATE_INTERVAL:
            return await self.update()
        return self._capabilities

    async def should_update(self) -> bool:
        """Check if capabilities need updating."""
        return time.time() - self._last_update > config.CAPABILITY_UPDATE_INTERVAL
