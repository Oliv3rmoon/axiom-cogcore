from __future__ import annotations
"""Track AXIOM's internal state by querying backend APIs."""

import json
from datetime import datetime
from collections import Counter

from shared.backend_client import BackendClient
from shared.db import get_db


class StateTracker:
    """
    Tracks AXIOM's internal state by fetching data from the backend.

    Maintains a snapshot of:
    - Active goals count and types
    - Lessons learned (total, by action_type)
    - Success rate (overall, per action)
    - Strongest and weakest capabilities
    - Dominant failure modes
    - Knowledge graph size
    """

    def __init__(self, backend: BackendClient):
        self.backend = backend
        self._last_state: dict | None = None
        self._last_update: datetime | None = None

    async def get_state(self) -> dict:
        """Compute and return current internal state."""
        goals = await self.backend.get_goals()
        lessons = await self.backend.get_lessons()
        knowledge = await self.backend.get_knowledge()

        # Active goals
        active_goals = [g for g in goals if g.get("status") == "active"]

        # Lesson stats
        total_lessons = len(lessons)
        success_lessons = [l for l in lessons if l.get("success")]
        success_rate = len(success_lessons) / max(1, total_lessons)

        # Per-action stats
        action_counts = Counter()
        action_successes = Counter()
        for lesson in lessons:
            action = lesson.get("action_type", "unknown")
            action_counts[action] += 1
            if lesson.get("success"):
                action_successes[action] += 1

        action_success_rates = {}
        for action, count in action_counts.items():
            action_success_rates[action] = round(action_successes[action] / count, 2)

        # Find strongest and weakest capabilities
        strongest = max(action_success_rates, key=action_success_rates.get) if action_success_rates else "none"
        weakest = min(action_success_rates, key=action_success_rates.get) if action_success_rates else "none"

        # Dominant failure mode
        failure_actions = Counter()
        for lesson in lessons:
            if not lesson.get("success"):
                failure_actions[lesson.get("action_type", "unknown")] += 1
        dominant_failure = failure_actions.most_common(1)[0][0] if failure_actions else "none"

        # Recent success rate (last 10)
        recent = sorted(lessons, key=lambda x: x.get("created_at", ""), reverse=True)[:10]
        recent_success = len([l for l in recent if l.get("success")]) / max(1, len(recent))

        state = {
            "active_goals": len(active_goals),
            "lessons_learned": total_lessons,
            "skills_acquired": 0,  # Will be updated from skills API
            "knowledge_nodes": len(knowledge),
            "success_rate_overall": round(success_rate, 2),
            "success_rate_last_10": round(recent_success, 2),
            "dominant_failure_mode": f"{dominant_failure} failures",
            "strongest_capability": strongest,
            "weakest_capability": weakest,
            "action_success_rates": action_success_rates,
        }

        # Update skills count
        skills = await self.backend.get_skills()
        state["skills_acquired"] = len(skills)

        self._last_state = state
        self._last_update = datetime.utcnow()

        # Persist snapshot
        await self._save_snapshot(state, action_success_rates)

        return state

    async def _save_snapshot(self, state: dict, capabilities: dict):
        """Save state snapshot to database."""
        db = await get_db()
        await db.execute(
            "INSERT INTO self_snapshots (state_json, capability_json) VALUES (?, ?)",
            (json.dumps(state), json.dumps(capabilities)),
        )
        await db.commit()

    async def get_history(self, limit: int = 10) -> list[dict]:
        """Get recent state snapshots."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT state_json, capability_json, created_at FROM self_snapshots "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        history = []
        async for row in cursor:
            history.append({
                "state": json.loads(row[0]),
                "capabilities": json.loads(row[1]),
                "timestamp": row[2],
            })
        return history
