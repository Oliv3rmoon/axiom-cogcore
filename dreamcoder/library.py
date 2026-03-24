from __future__ import annotations
"""Primitive library — stores extracted reusable action patterns."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

import config
from shared.db import get_db
from shared.embeddings import EmbeddingService


@dataclass
class Primitive:
    """A reusable action pattern discovered from solved problems."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    pattern: str = ""
    steps: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    frequency: int = 1
    success_rate: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "pattern": self.pattern,
            "steps": self.steps,
            "domains": self.domains,
            "frequency": self.frequency,
            "success_rate": round(self.success_rate, 3),
        }


class Library:
    """Collection of reusable primitives with embedding-based lookup."""

    def __init__(self, embedder: EmbeddingService):
        self.embedder = embedder
        self.primitives: dict[str, Primitive] = {}
        self._embeddings: dict[str, np.ndarray] = {}

    def add_primitive(self, prim: Primitive):
        """Add or merge a primitive. If a similar one exists, merge by updating frequency."""
        # Check for existing similar primitive
        if prim.name in {p.name for p in self.primitives.values()}:
            for pid, existing in self.primitives.items():
                if existing.name == prim.name:
                    existing.frequency += prim.frequency
                    existing.success_rate = (
                        existing.success_rate * 0.7 + prim.success_rate * 0.3
                    )
                    existing.domains = sorted(set(existing.domains + prim.domains))
                    return
        # Enforce max library size
        if len(self.primitives) >= config.DREAMCODER_MAX_LIBRARY_SIZE:
            # Remove least frequent
            least = min(self.primitives.values(), key=lambda p: p.frequency)
            self.primitives.pop(least.id, None)
            self._embeddings.pop(least.id, None)

        self.primitives[prim.id] = prim
        emb = self.embedder.embed(prim.pattern + " " + " ".join(prim.steps))
        self._embeddings[prim.id] = emb

    def compose(self, task_text: str, top_k: int = 5) -> list[Primitive]:
        """Find most relevant primitives for a task using embedding similarity."""
        if not self.primitives:
            return []

        task_emb = self.embedder.embed(task_text)
        scored = []
        for pid, prim in self.primitives.items():
            emb = self._embeddings.get(pid)
            if emb is None:
                emb = self.embedder.embed(prim.pattern + " " + " ".join(prim.steps))
                self._embeddings[pid] = emb
            sim = float(np.dot(task_emb, emb) / (
                np.linalg.norm(task_emb) * np.linalg.norm(emb) + 1e-8
            ))
            scored.append((prim, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]

    async def save_all(self):
        """Persist all primitives to database."""
        db = await get_db()
        for prim in self.primitives.values():
            await db.execute(
                """INSERT OR REPLACE INTO library_primitives
                   (id, name, pattern, steps, domains, frequency, success_rate, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (prim.id, prim.name, prim.pattern,
                 json.dumps(prim.steps), json.dumps(prim.domains),
                 prim.frequency, prim.success_rate),
            )
        await db.commit()

    async def load_from_db(self):
        """Load primitives from database."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT id, name, pattern, steps, domains, frequency, success_rate, created_at "
            "FROM library_primitives"
        )
        async for row in cursor:
            prim = Primitive(
                id=row[0], name=row[1], pattern=row[2],
                steps=json.loads(row[3]) if row[3] else [],
                domains=json.loads(row[4]) if row[4] else [],
                frequency=row[5], success_rate=row[6],
                created_at=row[7] or "",
            )
            self.primitives[prim.id] = prim
            emb = self.embedder.embed(prim.pattern + " " + " ".join(prim.steps))
            self._embeddings[prim.id] = emb

    @property
    def size(self) -> int:
        return len(self.primitives)
