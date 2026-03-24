from __future__ import annotations
"""Wake phase — solve tasks by composing library primitives."""

import uuid

from shared.embeddings import EmbeddingService
from dreamcoder.library import Library, Primitive


def wake_solve(task: str, library: Library, embedder: EmbeddingService) -> dict:
    """
    Given a task, compose a solution from library primitives.

    1. Embed the task
    2. Find top-k relevant primitives
    3. Compose them into a step sequence
    4. Score the solution
    """
    relevant = library.compose(task, top_k=5)

    if not relevant:
        # No primitives yet — return a generic plan
        return {
            "solution_found": False,
            "solution": [],
            "primitives_used": [],
            "novelty": 1.0,
            "solution_id": str(uuid.uuid4()),
        }

    # Build solution by chaining primitive steps
    solution_steps = []
    primitives_used = []
    seen_steps = set()
    for prim in relevant:
        for step in prim.steps:
            if step not in seen_steps:
                solution_steps.append(step)
                seen_steps.add(step)
        primitives_used.append(prim.name)

    # Novelty = 1 - coverage (what fraction of steps come from known primitives)
    total_possible = sum(len(p.steps) for p in relevant)
    novelty = 1.0 - (len(solution_steps) / max(1, total_possible + len(solution_steps)))

    return {
        "solution_found": True,
        "solution": solution_steps,
        "primitives_used": primitives_used,
        "novelty": round(novelty, 3),
        "solution_id": str(uuid.uuid4()),
    }
