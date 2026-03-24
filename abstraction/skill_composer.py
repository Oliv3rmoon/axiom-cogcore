from __future__ import annotations
"""Compose new skills from existing skill primitives."""

import numpy as np

from shared.embeddings import EmbeddingService
from shared.backend_client import BackendClient


class SkillComposer:
    """
    Composes new skills from existing ones by finding complementary
    skill combinations that cover a goal's requirements.
    """

    def __init__(self, embedder: EmbeddingService, backend: BackendClient):
        self.embedder = embedder
        self.backend = backend
        self._skills_cache: list[dict] | None = None

    async def _load_skills(self) -> list[dict]:
        """Load and cache skills from backend."""
        if self._skills_cache is None:
            self._skills_cache = await self.backend.get_skills()
        return self._skills_cache

    def invalidate_cache(self):
        """Force reload of skills on next access."""
        self._skills_cache = None

    async def compose_for_goal(self, goal: str) -> dict:
        """
        Given a goal, find the best composition of existing skills.

        Returns:
        - matching_skills: skills that directly apply
        - composed_approach: a suggested multi-skill approach
        - coverage: how much of the goal is covered by existing skills
        """
        skills = await self._load_skills()
        if not skills:
            return {
                "matching_skills": [],
                "composed_approach": "No skills available yet.",
                "coverage": 0.0,
            }

        # Embed the goal
        goal_emb = self.embedder.embed(goal)

        # Embed all skills
        skill_texts = []
        for skill in skills:
            text = skill.get("goal_pattern", "") + " " + skill.get("approach", "")
            skill_texts.append(text.strip())

        if not skill_texts:
            return {
                "matching_skills": [],
                "composed_approach": "No skills with descriptions available.",
                "coverage": 0.0,
            }

        skill_embs = self.embedder.embed_batch(skill_texts)

        # Compute similarities
        sims = np.dot(skill_embs, goal_emb) / (
            np.linalg.norm(skill_embs, axis=1) * np.linalg.norm(goal_emb) + 1e-8
        )

        # Rank skills by relevance
        ranked_indices = np.argsort(sims)[::-1]
        matching = []
        for idx in ranked_indices[:5]:
            sim = float(sims[idx])
            if sim > 0.2:
                skill = skills[idx].copy()
                skill["relevance"] = round(sim, 3)
                skill["success_rate"] = skill.get("success_rate", 0.0)
                matching.append(skill)

        # Compute coverage: max similarity as a proxy
        coverage = float(sims.max()) if len(sims) > 0 else 0.0

        # Build composed approach
        if matching:
            steps = []
            for i, skill in enumerate(matching[:3], 1):
                approach = skill.get("approach", "unknown")
                rate = skill.get("success_rate", 0)
                steps.append(
                    f"{i}. {approach} (success rate: {rate:.0%})"
                )
            composed = "Suggested approach using existing skills:\n" + "\n".join(steps)
        else:
            composed = "No existing skills match this goal. This is novel territory."

        return {
            "matching_skills": matching,
            "composed_approach": composed,
            "coverage": round(coverage, 3),
        }
