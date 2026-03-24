from __future__ import annotations
"""Thought graph data structures for reasoning workspace."""

import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Literal

import config

ThoughtType = Literal["observation", "hypothesis", "conclusion", "question", "action", "causal"]
RelationType = Literal["requires", "enables", "contradicts", "supports"]


@dataclass
class ThoughtNode:
    """A single thought in the reasoning graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: str = ""
    thought: str = ""
    thought_type: ThoughtType = "observation"
    confidence: float = 0.5
    parent_id: str | None = None
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=config.THOUGHT_TTL_HOURS)
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "thought": self.thought,
            "thought_type": self.thought_type,
            "confidence": self.confidence,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


@dataclass
class ThoughtEdge:
    """A relationship between two thoughts."""
    source_id: str = ""
    target_id: str = ""
    relationship: RelationType = "supports"
    strength: float = 0.5
    workspace_id: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "strength": self.strength,
        }
