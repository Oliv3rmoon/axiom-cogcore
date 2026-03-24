from __future__ import annotations
"""Find solutions by composing library primitives for new tasks."""

import json
import numpy as np

from shared.embeddings import EmbeddingService
from shared.db import get_db
from dreamcoder.library import Library


async def compose_solution(task: str, domain: str, library: Library,
                           embedder: EmbeddingService) -> dict:
    """
    Given a new task, suggest a solution by composing library primitives.
    Also finds similar previously solved tasks.
    """
    # Find relevant primitives
    relevant = library.compose(task, top_k=5)

    if not relevant:
        return {
            "suggested_steps": [],
            "confidence": 0.0,
            "based_on_primitives": [],
            "similar_solved_tasks": [],
        }

    # Build step sequence from primitives
    steps = []
    prim_names = []
    seen = set()
    for prim in relevant:
        for step in prim.steps:
            if step not in seen:
                steps.append(step)
                seen.add(step)
        prim_names.append(prim.name)

    # Confidence based on primitive quality
    avg_success = np.mean([p.success_rate for p in relevant]) if relevant else 0.0
    avg_freq = np.mean([p.frequency for p in relevant]) if relevant else 0
    confidence = float(avg_success * 0.6 + min(1.0, avg_freq / 10.0) * 0.4)

    # Find similar solved tasks
    similar_tasks = await _find_similar_tasks(task, embedder)

    return {
        "suggested_steps": steps,
        "confidence": round(confidence, 3),
        "based_on_primitives": prim_names,
        "similar_solved_tasks": similar_tasks,
    }


async def _find_similar_tasks(task: str, embedder: EmbeddingService,
                               top_k: int = 3) -> list[str]:
    """Find previously solved tasks similar to the given task."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT task FROM solved_tasks WHERE was_successful = 1 "
        "ORDER BY created_at DESC LIMIT 50"
    )
    tasks = []
    async for row in cursor:
        tasks.append(row[0])

    if not tasks:
        return []

    task_emb = embedder.embed(task)
    task_embs = embedder.embed_batch(tasks)

    sims = np.dot(task_embs, task_emb) / (
        np.linalg.norm(task_embs, axis=1) * np.linalg.norm(task_emb) + 1e-8
    )

    top_indices = np.argsort(sims)[::-1][:top_k]
    return [tasks[i] for i in top_indices if sims[i] > 0.3]
