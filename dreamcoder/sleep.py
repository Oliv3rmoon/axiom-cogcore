from __future__ import annotations
"""Sleep phase — extract common patterns across solutions into library primitives."""

import json
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN

import config
from shared.db import get_db
from shared.embeddings import EmbeddingService
from shared.backend_client import BackendClient
from dreamcoder.library import Library, Primitive


async def abstraction_sleep(library: Library, embedder: EmbeddingService,
                            backend: BackendClient,
                            min_solutions: int = config.DREAMCODER_MIN_SOLUTIONS
                            ) -> list[Primitive]:
    """
    Sleep phase: extract common sub-patterns from solved tasks.

    1. Collect all step sequences from recent completed goals
    2. Embed each step
    3. Find common sub-patterns using clustering
    4. For patterns appearing in 3+ solutions across 2+ domains → new primitive
    """
    # Load solved tasks from DB
    db = await get_db()
    cursor = await db.execute(
        "SELECT task, domain, solution_steps, was_successful FROM solved_tasks "
        "WHERE was_successful = 1 ORDER BY created_at DESC LIMIT 200"
    )
    solutions = []
    async for row in cursor:
        steps = json.loads(row[2]) if row[2] else []
        if steps:
            solutions.append({
                "task": row[0],
                "domain": row[1] or "unknown",
                "steps": steps,
            })

    # Also try to get data from backend lessons/skills
    try:
        lessons = await backend.get_lessons()
        skills = await backend.get_skills()
        for skill in skills:
            steps_template = skill.get("steps_template", "")
            if steps_template:
                try:
                    steps = json.loads(steps_template) if isinstance(steps_template, str) else steps_template
                except (json.JSONDecodeError, TypeError):
                    steps = [steps_template]
                if isinstance(steps, list) and steps:
                    solutions.append({
                        "task": skill.get("goal_pattern", ""),
                        "domain": skill.get("goal_pattern", "unknown"),
                        "steps": [str(s) for s in steps],
                    })
    except Exception:
        pass

    if len(solutions) < min_solutions:
        return []

    # Embed all steps across solutions
    all_steps = []
    step_metadata = []  # Track which solution/domain each step came from
    for sol in solutions:
        for step in sol["steps"]:
            all_steps.append(step)
            step_metadata.append({"domain": sol["domain"], "task": sol["task"]})

    if not all_steps:
        return []

    embeddings = embedder.embed_batch(all_steps)

    # Cluster steps by embedding similarity
    clustering = DBSCAN(eps=0.5, min_samples=config.DREAMCODER_MIN_PATTERN_FREQ,
                        metric="cosine").fit(embeddings)
    labels = clustering.labels_

    new_primitives = []
    unique_labels = set(labels)
    unique_labels.discard(-1)

    for label in unique_labels:
        mask = labels == label
        cluster_steps = [s for s, m in zip(all_steps, mask) if m]
        cluster_meta = [m for m, is_in in zip(step_metadata, mask) if is_in]

        if len(cluster_steps) < config.DREAMCODER_MIN_PATTERN_FREQ:
            continue

        # Check domain span
        domains = set(m["domain"] for m in cluster_meta)
        domains.discard("unknown")
        if len(domains) < config.DREAMCODER_MIN_DOMAINS:
            continue

        # Find representative step (most common)
        step_counts = Counter(cluster_steps)
        representative = step_counts.most_common(1)[0][0]

        # Build primitive name from pattern
        name = representative.lower().replace(" ", "_")[:30]

        prim = Primitive(
            name=name,
            pattern=representative,
            steps=list(set(cluster_steps))[:5],
            domains=sorted(domains),
            frequency=len(cluster_steps),
            success_rate=0.7,  # Default for newly extracted patterns
        )
        library.add_primitive(prim)
        new_primitives.append(prim)

    if new_primitives:
        await library.save_all()

    return new_primitives
