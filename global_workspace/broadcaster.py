from __future__ import annotations
"""Publish-subscribe broadcast mechanism for the Global Workspace."""

import json
import uuid
from datetime import datetime

from shared.db import get_db
from global_workspace.module_registry import ModuleRegistry


class Signal:
    """A signal submitted to the workspace for broadcast consideration."""

    def __init__(self, source_module: str, signal_type: str,
                 content: dict, salience: float, urgency: float = 0.5):
        self.id = str(uuid.uuid4())
        self.source_module = source_module
        self.signal_type = signal_type
        self.content = content
        self.salience = salience
        self.urgency = urgency
        self.created_at = datetime.utcnow().isoformat()
        self.received_by: list[str] = []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source_module,
            "signal_type": self.signal_type,
            "content": self.content,
            "salience": round(self.salience, 3),
            "urgency": round(self.urgency, 3),
            "broadcast_at": self.created_at,
            "received_by": self.received_by,
        }


class Broadcaster:
    """Handles broadcast distribution to subscribers."""

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry

    def broadcast(self, signal: Signal) -> list[str]:
        """
        Broadcast a signal to all interested subscribers.
        Returns list of module names that received the signal.
        """
        subscribers = self.registry.get_subscribers(signal.signal_type)
        # Don't send back to source
        receivers = [s for s in subscribers if s != signal.source_module]

        signal.received_by = receivers

        # Update module stats
        self.registry.record_broadcast_sent(signal.source_module)
        for receiver in receivers:
            self.registry.record_broadcast_received(receiver)

        return receivers

    async def persist_broadcast(self, signal: Signal):
        """Save a broadcast to the database."""
        db = await get_db()
        await db.execute(
            """INSERT INTO workspace_broadcasts
               (id, source_module, signal_type, content, salience, urgency, received_by)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (signal.id, signal.source_module, signal.signal_type,
             json.dumps(signal.content), signal.salience, signal.urgency,
             json.dumps(signal.received_by)),
        )
        await db.commit()
