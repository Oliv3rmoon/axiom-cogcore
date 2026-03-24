from __future__ import annotations
"""Learn causal structure from AXIOM's experience data."""

from collections import defaultdict

import config
from shared.backend_client import BackendClient
from causal.scm import StructuralCausalModel, CausalEdge


async def learn_causal_structure(scm: StructuralCausalModel,
                                  backend: BackendClient,
                                  min_evidence: int = config.CAUSAL_MIN_EVIDENCE
                                  ) -> dict:
    """
    Discover causal relationships from AXIOM's experience data.

    For each pair of variables (A, B):
    1. Compute P(B=success | A=true) and P(B=success | A=false)
    2. If difference > threshold, A may cause B
    3. Add edges for significant causal relationships
    """
    lessons = await backend.get_lessons()
    if len(lessons) < min_evidence:
        return {"new_relationships": 0, "updated_relationships": 0, "total_relationships": 0}

    # Extract variables from lessons
    # Variable: action_type -> success
    action_success: dict[str, list[bool]] = defaultdict(list)
    # Variable: goal_type -> success
    goal_success: dict[str, list[bool]] = defaultdict(list)

    for lesson in lessons:
        action = lesson.get("action_type", "unknown")
        success = lesson.get("success", False)
        goal_type = lesson.get("goal_type", "unknown")

        action_success[action].append(bool(success))
        if goal_type != "unknown":
            goal_success[goal_type].append(bool(success))

    new_count = 0
    updated_count = 0

    # Learn action -> success relationships
    for action, outcomes in action_success.items():
        if len(outcomes) < min_evidence:
            continue
        success_rate = sum(outcomes) / len(outcomes)
        overall_rate = sum(
            sum(v) for v in action_success.values()
        ) / sum(len(v) for v in action_success.values())

        causal_strength = success_rate - overall_rate

        if abs(causal_strength) >= config.CAUSAL_STRENGTH_THRESHOLD:
            existing = scm.get_edge_data(action, "success")
            if existing:
                updated_count += 1
            else:
                new_count += 1

            mechanism = (
                f"{action} has {success_rate:.0%} success rate vs "
                f"{overall_rate:.0%} baseline ({causal_strength:+.0%} effect)"
            )
            scm.add_edge(
                action, "success",
                strength=abs(causal_strength),
                evidence_count=len(outcomes),
                mechanism=mechanism,
            )

    # Learn goal_type -> success relationships
    for goal_type, outcomes in goal_success.items():
        if len(outcomes) < min_evidence:
            continue
        success_rate = sum(outcomes) / len(outcomes)
        overall_rate = sum(
            sum(v) for v in goal_success.values()
        ) / sum(len(v) for v in goal_success.values())

        causal_strength = success_rate - overall_rate

        if abs(causal_strength) >= config.CAUSAL_STRENGTH_THRESHOLD:
            existing = scm.get_edge_data(f"goal_{goal_type}", "success")
            if existing:
                updated_count += 1
            else:
                new_count += 1

            scm.add_edge(
                f"goal_{goal_type}", "success",
                strength=abs(causal_strength),
                evidence_count=len(outcomes),
                mechanism=f"Goal type '{goal_type}' has {causal_strength:+.0%} effect on success",
            )

    # Learn action-pair relationships (does doing A before B help?)
    # Group lessons by sequence/time if available
    action_pairs: dict[tuple[str, str], list[bool]] = defaultdict(list)
    sorted_lessons = sorted(lessons, key=lambda l: l.get("created_at", ""))
    for i in range(1, len(sorted_lessons)):
        prev_action = sorted_lessons[i - 1].get("action_type", "unknown")
        curr_action = sorted_lessons[i].get("action_type", "unknown")
        curr_success = bool(sorted_lessons[i].get("success", False))
        if prev_action != curr_action:
            action_pairs[(prev_action, curr_action)].append(curr_success)

    for (prev, curr), outcomes in action_pairs.items():
        if len(outcomes) < min_evidence:
            continue
        pair_rate = sum(outcomes) / len(outcomes)
        base_rate = sum(action_success.get(curr, [False])) / max(1, len(action_success.get(curr, [False])))
        strength = pair_rate - base_rate

        if abs(strength) >= config.CAUSAL_STRENGTH_THRESHOLD:
            existing = scm.get_edge_data(f"after_{prev}", f"{curr}_success")
            if existing:
                updated_count += 1
            else:
                new_count += 1

            scm.add_edge(
                f"after_{prev}", f"{curr}_success",
                strength=abs(strength),
                evidence_count=len(outcomes),
                mechanism=f"Doing {prev} before {curr} has {strength:+.0%} effect",
            )

    # Save to DB
    await scm.save_all()

    # Find strongest causes
    all_edges = scm.get_all_edges()
    strongest = sorted(all_edges, key=lambda e: e.get("strength", 0), reverse=True)[:5]

    return {
        "new_relationships": new_count,
        "updated_relationships": updated_count,
        "total_relationships": len(all_edges),
        "strongest_causes": strongest,
    }
