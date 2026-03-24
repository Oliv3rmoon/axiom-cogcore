from __future__ import annotations
"""Central broadcast buffer with competition — the Global Workspace."""

from collections import deque
from datetime import datetime

import config
from global_workspace.broadcaster import Signal, Broadcaster
from global_workspace.module_registry import ModuleRegistry
from global_workspace.salience import compute_salience


class GlobalWorkspace:
    """
    Global Workspace Theory implementation.

    Multiple specialized modules compete for access to a central broadcast hub.
    The winning signal (highest salience) gets broadcast to all other modules.
    This is the "ignition" — the moment a signal enters the workspace.
    """

    def __init__(self, registry: ModuleRegistry, broadcaster: Broadcaster):
        self.registry = registry
        self.broadcaster = broadcaster
        self.current_broadcast: Signal | None = None
        self.queue: list[Signal] = []
        self.history: deque[Signal] = deque(maxlen=config.GW_BROADCAST_HISTORY_SIZE)
        self.total_broadcasts = 0

    def submit(self, signal: Signal) -> dict:
        """
        Module submits a signal for broadcast consideration.
        Gets queued if salience > threshold.
        """
        if signal.salience < config.GW_SALIENCE_THRESHOLD:
            return {"accepted": False, "reason": "Below salience threshold"}

        self.queue.append(signal)
        # Sort by combined salience * urgency
        self.queue.sort(key=lambda s: s.salience * s.urgency, reverse=True)

        position = self.queue.index(signal) + 1

        return {
            "accepted": True,
            "broadcast_id": signal.id,
            "queue_position": position,
        }

    def compete(self) -> Signal | None:
        """
        Competition round: highest salience * urgency wins.
        Winner gets broadcast to all subscribers.
        """
        if not self.queue:
            return None

        # Winner = highest salience * urgency (already sorted)
        winner = self.queue.pop(0)

        # Broadcast to subscribers
        receivers = self.broadcaster.broadcast(winner)
        winner.received_by = receivers

        # Update state
        self.current_broadcast = winner
        self.history.append(winner)
        self.total_broadcasts += 1

        return winner

    async def submit_and_compete(self, signal: Signal) -> dict:
        """Submit a signal and immediately run competition."""
        submit_result = self.submit(signal)
        if not submit_result["accepted"]:
            return submit_result

        winner = self.compete()
        if winner and winner.id == signal.id:
            await self.broadcaster.persist_broadcast(winner)
            return {
                "accepted": True,
                "broadcast_id": signal.id,
                "queue_position": 0,
                "was_broadcast": True,
                "received_by": winner.received_by,
            }

        return submit_result

    def get_current(self) -> dict:
        """Get the current workspace state."""
        return {
            "current_broadcast": self.current_broadcast.to_dict() if self.current_broadcast else None,
            "queue": [s.to_dict() for s in self.queue[:10]],
            "broadcast_history_count": len(self.history),
            "total_broadcasts": self.total_broadcasts,
        }

    def subscribe(self, module_name: str, interests: list[str]) -> bool:
        """Register a module to receive broadcasts."""
        self.registry.register(module_name, interests)
        return True
