from __future__ import annotations
"""Extract meta-lessons across domains from backend lesson data."""

import numpy as np
from sklearn.cluster import DBSCAN

from shared.embeddings import EmbeddingService
from shared.backend_client import BackendClient
import config


class MetaLearner:
    """
    Analyzes lessons across domains to find cross-domain patterns.
    Groups lessons by action type, clusters by embedding similarity,
    identifies patterns that span multiple domains.
    """

    def __init__(self, embedder: EmbeddingService, backend: BackendClient):
        self.embedder = embedder
        self.backend = backend
        self.meta_lessons: list[dict] = []

    async def extract_meta_lessons(self) -> list[dict]:
        """
        Load lessons from backend, group by domain, find cross-domain patterns.
        """
        lessons = await self.backend.get_lessons()
        if len(lessons) < config.META_LESSON_MIN_EXAMPLES:
            return []

        # Extract text and metadata from lessons
        lesson_texts = []
        lesson_meta = []
        for lesson in lessons:
            text = lesson.get("lesson", lesson.get("description", ""))
            if not text:
                continue
            lesson_texts.append(text)
            lesson_meta.append({
                "action_type": lesson.get("action_type", "unknown"),
                "goal_type": lesson.get("goal_type", "unknown"),
                "success": lesson.get("success", False),
                "confidence": lesson.get("confidence", 0.5),
                "text": text,
            })

        if len(lesson_texts) < config.META_LESSON_MIN_EXAMPLES:
            return []

        # Embed all lessons
        embeddings = self.embedder.embed_batch(lesson_texts)

        # Cluster with DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=3, metric="cosine").fit(embeddings)
        labels = clustering.labels_

        meta_lessons = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in unique_labels:
            mask = labels == label
            cluster_meta = [m for m, is_in in zip(lesson_meta, mask) if is_in]
            cluster_embeddings = embeddings[mask]

            if len(cluster_meta) < 3:
                continue

            # Check how many domains this cluster spans
            action_types = set(m["action_type"] for m in cluster_meta)
            goal_types = set(m["goal_type"] for m in cluster_meta)
            all_domains = action_types | goal_types

            # Find the most representative lesson (closest to centroid)
            centroid = cluster_embeddings.mean(axis=0)
            dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            representative_idx = np.argmin(dists)
            representative = cluster_meta[representative_idx]

            # Compute success rate within cluster
            successes = [m for m in cluster_meta if m["success"]]
            success_rate = len(successes) / len(cluster_meta) if cluster_meta else 0

            meta_lesson = {
                "pattern": representative["text"],
                "example_count": len(cluster_meta),
                "domains": sorted(all_domains - {"unknown"}),
                "is_cross_domain": len(all_domains - {"unknown"}) >= 2,
                "success_rate": round(success_rate, 2),
                "avg_confidence": round(
                    np.mean([m["confidence"] for m in cluster_meta]), 2
                ),
            }
            meta_lessons.append(meta_lesson)

        # Sort by cross-domain first, then by example count
        meta_lessons.sort(
            key=lambda x: (x["is_cross_domain"], x["example_count"]), reverse=True
        )

        self.meta_lessons = meta_lessons
        return meta_lessons

    def get_cross_domain_lessons(self) -> list[dict]:
        """Return only lessons that span multiple domains."""
        return [m for m in self.meta_lessons if m["is_cross_domain"]]
