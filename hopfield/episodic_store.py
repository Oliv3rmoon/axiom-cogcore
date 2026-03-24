from __future__ import annotations
"""Store and retrieve experience episodes using Modern Hopfield network."""

import uuid
import numpy as np
from datetime import datetime

import config
from shared.db import get_db
from shared.embeddings import EmbeddingService
from hopfield.modern_hopfield import ModernHopfieldNetwork


class Episode:
    """A stored experience episode."""

    def __init__(self, content: str, context: str = "",
                 importance: float = 0.5, pattern_id: str = ""):
        self.pattern_id = pattern_id or str(uuid.uuid4())
        self.content = content
        self.context = context
        self.importance = importance
        self.created_at = datetime.utcnow()
        self.access_count = 0


class EpisodicStore:
    """
    Episodic memory backed by Modern Hopfield network + SQLite persistence.
    """

    def __init__(self, embedder: EmbeddingService):
        self.embedder = embedder
        self.hopfield = ModernHopfieldNetwork()
        # Map: hopfield index → episode metadata
        self._index_to_episode: dict[int, Episode] = {}
        # Map: pattern_id → hopfield index
        self._id_to_index: dict[str, int] = {}

    async def store(self, content: str, context: str = "",
                    importance: float = 0.5) -> dict:
        """Store an experience episode."""
        embedding = self.embedder.embed(content)
        episode = Episode(content, context, importance)

        idx = self.hopfield.store(embedding)
        self._index_to_episode[idx] = episode
        self._id_to_index[episode.pattern_id] = idx

        # Persist to database
        await self._save_episode(episode, embedding)

        return {
            "pattern_id": episode.pattern_id,
            "total_patterns": self.hopfield.num_patterns,
        }

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Content-based retrieval of similar episodes."""
        query_emb = self.embedder.embed(query)
        results = self.hopfield.retrieve(query_emb, top_k=top_k)

        retrieved = []
        for idx, similarity, pattern in results:
            episode = self._index_to_episode.get(idx)
            if episode:
                episode.access_count += 1
                retrieved.append({
                    "pattern_id": episode.pattern_id,
                    "content": episode.content,
                    "context": episode.context,
                    "similarity": round(similarity, 4),
                    "importance": episode.importance,
                    "stored_at": episode.created_at.isoformat(),
                    "access_count": episode.access_count,
                })

        # If we got results from hopfield but no episode metadata,
        # return what we can
        if not retrieved and results:
            for idx, similarity, _ in results:
                retrieved.append({
                    "pattern_id": f"pattern_{idx}",
                    "content": "(metadata not loaded)",
                    "similarity": round(similarity, 4),
                })

        return retrieved

    async def associate(self, pattern_id: str, top_k: int = 5) -> list[dict]:
        """Find episodes associated with a given pattern."""
        idx = self._id_to_index.get(pattern_id)
        if idx is None:
            return []

        associations = self.hopfield.find_associations(idx, top_k)
        results = []
        for assoc_idx, strength in associations:
            episode = self._index_to_episode.get(assoc_idx)
            if episode:
                results.append({
                    "pattern_id": episode.pattern_id,
                    "content": episode.content,
                    "association_strength": round(strength, 4),
                })

        return results

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "total_patterns": self.hopfield.num_patterns,
            "capacity": config.HOPFIELD_MAX_PATTERNS,
            "utilization": round(self.hopfield.num_patterns / config.HOPFIELD_MAX_PATTERNS, 4),
            "temperature": config.HOPFIELD_BETA,
            "pattern_dim": config.HOPFIELD_PATTERN_DIM,
        }

    async def _save_episode(self, episode: Episode, embedding: np.ndarray):
        """Persist episode to SQLite."""
        db = await get_db()
        await db.execute(
            """INSERT OR REPLACE INTO hopfield_patterns
               (id, pattern, content, context, importance, access_count, last_accessed, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.pattern_id,
                embedding.astype(np.float32).tobytes(),
                episode.content,
                episode.context,
                episode.importance,
                episode.access_count,
                datetime.utcnow().isoformat(),
                episode.created_at.isoformat(),
            ),
        )
        await db.commit()

    async def load_from_db(self):
        """Load all patterns from database into the Hopfield network."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT id, pattern, content, context, importance, access_count, created_at "
            "FROM hopfield_patterns ORDER BY created_at"
        )
        async for row in cursor:
            pattern_id = row[0]
            pattern = np.frombuffer(row[1], dtype=np.float32).copy()
            if len(pattern) != config.HOPFIELD_PATTERN_DIM:
                continue

            episode = Episode(
                content=row[2],
                context=row[3] or "",
                importance=row[4],
                pattern_id=pattern_id,
            )
            episode.access_count = row[5]

            idx = self.hopfield.store(pattern)
            self._index_to_episode[idx] = episode
            self._id_to_index[pattern_id] = idx
