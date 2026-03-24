from __future__ import annotations
"""Register cognitive modules as workspace participants."""

import json
from datetime import datetime

from shared.db import get_db


class ModuleInfo:
    """Tracks a registered module's state."""

    def __init__(self, name: str, interests: list[str] = None):
        self.name = name
        self.status = "active"
        self.interests = interests or []
        self.broadcasts_sent = 0
        self.broadcasts_received = 0
        self.last_broadcast: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "interests": self.interests,
            "broadcasts_sent": self.broadcasts_sent,
            "broadcasts_received": self.broadcasts_received,
            "last_broadcast": self.last_broadcast,
        }


class ModuleRegistry:
    """Registry of all cognitive modules participating in the Global Workspace."""

    # Default modules and their interests
    DEFAULT_MODULES = {
        "world_model": ["prediction_error", "state_transition", "goal_achieved"],
        "curiosity": ["high_curiosity", "novelty_detected", "prediction_error"],
        "active_inference": ["policy_evaluation", "precision_update", "prediction_error"],
        "self_model": ["capability_change", "failure_mode", "state_transition"],
        "abstraction": ["new_principle", "pattern_detected", "goal_achieved"],
        "hopfield": ["memory_retrieved", "association_found", "prediction_error"],
        "beta_vae": ["representation_change", "new_factor", "prediction_error"],
        "dreamcoder": ["new_primitive", "solution_found", "pattern_detected"],
        "reasoning": ["conclusion_reached", "contradiction_found", "hypothesis"],
        "causal": ["causal_link_found", "intervention_result", "counterfactual"],
    }

    def __init__(self):
        self.modules: dict[str, ModuleInfo] = {}

    def register_defaults(self):
        """Register all default cognitive modules."""
        for name, interests in self.DEFAULT_MODULES.items():
            self.register(name, interests)

    def register(self, name: str, interests: list[str] = None):
        """Register a module."""
        self.modules[name] = ModuleInfo(name, interests or [])

    def unregister(self, name: str):
        """Remove a module."""
        self.modules.pop(name, None)

    def get_subscribers(self, signal_type: str) -> list[str]:
        """Get modules interested in a signal type."""
        return [
            name for name, mod in self.modules.items()
            if signal_type in mod.interests
        ]

    def record_broadcast_sent(self, module_name: str):
        """Record that a module sent a broadcast."""
        mod = self.modules.get(module_name)
        if mod:
            mod.broadcasts_sent += 1
            mod.last_broadcast = datetime.utcnow().isoformat()

    def record_broadcast_received(self, module_name: str):
        """Record that a module received a broadcast."""
        mod = self.modules.get(module_name)
        if mod:
            mod.broadcasts_received += 1

    def get_all(self) -> list[dict]:
        """Get all modules as dicts."""
        return [mod.to_dict() for mod in self.modules.values()]

    async def save_all(self):
        """Persist registry to database."""
        db = await get_db()
        for mod in self.modules.values():
            await db.execute(
                """INSERT OR REPLACE INTO workspace_modules
                   (name, status, interests, broadcasts_sent, broadcasts_received, last_broadcast)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (mod.name, mod.status, json.dumps(mod.interests),
                 mod.broadcasts_sent, mod.broadcasts_received, mod.last_broadcast),
            )
        await db.commit()

    async def load_from_db(self):
        """Load registry from database."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT name, status, interests, broadcasts_sent, broadcasts_received, last_broadcast "
            "FROM workspace_modules"
        )
        async for row in cursor:
            mod = ModuleInfo(row[0], json.loads(row[2]) if row[2] else [])
            mod.status = row[1]
            mod.broadcasts_sent = row[3]
            mod.broadcasts_received = row[4]
            mod.last_broadcast = row[5]
            self.modules[mod.name] = mod
