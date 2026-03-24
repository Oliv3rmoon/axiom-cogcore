from __future__ import annotations
"""Extract domain-general principles from lessons."""

import json
import numpy as np
from sklearn.cluster import DBSCAN

import config
from shared.embeddings import EmbeddingService
from shared.backend_client import BackendClient
from shared.db import get_db


class PrincipleExtractor:
    """
    Extracts domain-general principles from patterns in lesson data.

    Algorithm:
    1. Embed all lessons
    2. Cluster with DBSCAN
    3. For each cluster spanning 2+ domains, extract as a principle
    4. Store principles in local SQLite
    """

    def __init__(self, embedder: EmbeddingService, backend: BackendClient):
        self.embedder = embedder
        self.backend = backend

    async def extract_principles(self) -> list[dict]:
        """
        Analyze all lessons and extract domain-general principles.
        Returns list of new principles found.
        """
        lessons = await self.backend.get_lessons()
        if len(lessons) < config.META_LESSON_MIN_EXAMPLES:
            return []

        # Build lesson list
        lesson_data = []
        for lesson in lessons:
            text = lesson.get("lesson", lesson.get("description", ""))
            if not text:
                continue
            lesson_data.append({
                "text": text,
                "action_type": lesson.get("action_type", "unknown"),
                "goal_type": lesson.get("goal_type", "unknown"),
                "success": lesson.get("success", False),
                "confidence": lesson.get("confidence", 0.5),
            })

        if len(lesson_data) < config.META_LESSON_MIN_EXAMPLES:
            return []

        texts = [ld["text"] for ld in lesson_data]
        embeddings = self.embedder.embed_batch(texts)

        # Cluster
        clustering = DBSCAN(eps=0.5, min_samples=3, metric="cosine").fit(embeddings)
        labels = clustering.labels_

        new_principles = []
        unique_labels = set(labels)
        unique_labels.discard(-1)

        for label in unique_labels:
            mask = labels == label
            cluster = [ld for ld, m in zip(lesson_data, mask) if m]
            cluster_emb = embeddings[mask]

            if len(cluster) < 3:
                continue

            # Check domain span
            domains = set()
            for item in cluster:
                if item["action_type"] != "unknown":
                    domains.add(item["action_type"])
                if item["goal_type"] != "unknown":
                    domains.add(item["goal_type"])

            if len(domains) < 2:
                continue

            # Find representative text
            centroid = cluster_emb.mean(axis=0)
            dists = np.linalg.norm(cluster_emb - centroid, axis=1)
            rep_idx = int(np.argmin(dists))

            # Compute success correlation
            success_with = [c for c in cluster if c["success"]]
            success_rate = len(success_with) / len(cluster)

            # Build actionable rule
            if success_rate > 0.7:
                actionable = f"Following this pattern leads to {int(success_rate*100)}% success"
            elif success_rate < 0.3:
                actionable = f"Avoid this pattern - only {int(success_rate*100)}% success rate"
            else:
                actionable = f"Mixed results ({int(success_rate*100)}% success) - apply with caution"

            confidence = min(1.0, success_rate * 0.7 + len(cluster) / 20.0 * 0.3)

            if confidence < config.PRINCIPLE_CONFIDENCE_THRESHOLD:
                continue

            principle = {
                "principle": cluster[rep_idx]["text"],
                "evidence_count": len(cluster),
                "domains": sorted(domains),
                "confidence": round(confidence, 2),
                "actionable_rule": actionable,
            }

            # Save to database
            await self._save_principle(principle)
            new_principles.append(principle)

        return new_principles

    async def get_all_principles(self) -> list[dict]:
        """Load all stored principles from database."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT id, principle, evidence_count, domains, confidence, actionable_rule, times_applied "
            "FROM principles ORDER BY confidence DESC"
        )
        principles = []
        async for row in cursor:
            principles.append({
                "id": row[0],
                "principle": row[1],
                "evidence_count": row[2],
                "domains": json.loads(row[3]) if row[3] else [],
                "confidence": row[4],
                "actionable_rule": row[5],
                "times_applied": row[6],
            })
        return principles

    async def apply_principles(self, goal: str, action: str = "") -> dict:
        """
        Find principles relevant to a goal/action and return suggestions.
        """
        principles = await self.get_all_principles()
        if not principles:
            return {
                "relevant_principles": [],
                "suggested_approach": "No principles extracted yet. More experience data needed.",
            }

        # Embed goal and compute similarity to each principle
        goal_emb = self.embedder.embed(goal)
        principle_texts = [p["principle"] for p in principles]
        principle_embs = self.embedder.embed_batch(principle_texts)

        # Cosine similarities
        sims = np.dot(principle_embs, goal_emb) / (
            np.linalg.norm(principle_embs, axis=1) * np.linalg.norm(goal_emb) + 1e-8
        )

        # Get top relevant principles (similarity > 0.3)
        relevant = []
        for i, sim in enumerate(sims):
            if sim > 0.3:
                p = principles[i].copy()
                p["relevance"] = round(float(sim), 3)
                relevant.append(p)

        relevant.sort(key=lambda x: x["relevance"], reverse=True)
        relevant = relevant[:5]

        # Generate suggested approach
        if relevant:
            top = relevant[0]
            suggested = (
                f"Based on {top['evidence_count']} past experiences: "
                f"{top['actionable_rule']}. "
                f"Confidence: {top['confidence']:.0%}."
            )
            # Mark as applied
            db = await get_db()
            for r in relevant:
                await db.execute(
                    "UPDATE principles SET times_applied = times_applied + 1 WHERE id = ?",
                    (r["id"],),
                )
            await db.commit()
        else:
            suggested = "No strongly relevant principles found for this goal."

        return {
            "relevant_principles": relevant,
            "suggested_approach": suggested,
        }

    async def _save_principle(self, principle: dict):
        """Save a principle to the database."""
        db = await get_db()
        await db.execute(
            """INSERT INTO principles (principle, evidence_count, domains, confidence, actionable_rule)
               VALUES (?, ?, ?, ?, ?)""",
            (
                principle["principle"],
                principle["evidence_count"],
                json.dumps(principle["domains"]),
                principle["confidence"],
                principle["actionable_rule"],
            ),
        )
        await db.commit()
