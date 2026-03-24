from __future__ import annotations
"""Predict effects of self-modifications on AXIOM's capabilities."""

import json
from shared.backend_client import BackendClient
from shared.embeddings import EmbeddingService
from self_model.capability_model import CapabilityModel
from self_model.state_tracker import StateTracker


class TransitionModel:
    """
    Predicts how proposed changes will affect AXIOM's state.
    Uses historical data to estimate effects of modifications.
    """

    def __init__(self, state_tracker: StateTracker,
                 capability_model: CapabilityModel,
                 embedder: EmbeddingService,
                 backend: BackendClient):
        self.state_tracker = state_tracker
        self.capability_model = capability_model
        self.embedder = embedder
        self.backend = backend

    async def predict_change(self, proposed_change: str,
                             change_type: str = "code_modification") -> dict:
        """
        Predict the effect of a proposed change on AXIOM's capabilities.
        """
        # Get current state
        state = await self.state_tracker.get_state()
        capabilities = await self.capability_model.get_capabilities()

        # Analyze the proposed change
        change_embedding = self.embedder.embed(proposed_change)

        # Get lessons related to this type of change
        lessons = await self.backend.get_lessons()

        # Find similar past experiences
        similar_lessons = []
        if lessons:
            lesson_texts = [l.get("lesson", l.get("description", "")) for l in lessons]
            lesson_texts = [t for t in lesson_texts if t]
            if lesson_texts:
                lesson_embeddings = self.embedder.embed_batch(lesson_texts)
                import numpy as np
                sims = np.dot(lesson_embeddings, change_embedding) / (
                    np.linalg.norm(lesson_embeddings, axis=1) * np.linalg.norm(change_embedding) + 1e-8
                )
                top_indices = np.argsort(sims)[::-1][:5]
                for idx in top_indices:
                    if sims[idx] > 0.3 and idx < len(lessons):
                        similar_lessons.append(lessons[idx])

        # Estimate effects based on similar past experiences
        if similar_lessons:
            past_successes = [l for l in similar_lessons if l.get("success")]
            past_rate = len(past_successes) / len(similar_lessons)

            # Identify affected capability
            affected_actions = set()
            for l in similar_lessons:
                a = l.get("action_type")
                if a:
                    affected_actions.add(a)

            if affected_actions:
                main_action = list(affected_actions)[0]
                current_rate = capabilities.get(main_action, 0.5)
                predicted_rate = current_rate + (past_rate - current_rate) * 0.3
                effect = (
                    f"{main_action} capability would change from "
                    f"{current_rate:.0%} to ~{predicted_rate:.0%}"
                )
            else:
                effect = "Effect unclear - no matching capability area"
                predicted_rate = 0.5

            side_effects = self._estimate_side_effects(change_type, similar_lessons)
            confidence = min(0.9, 0.3 + len(similar_lessons) * 0.1)
        else:
            effect = "No similar past changes found. Effect is unpredictable."
            side_effects = ["Unknown impact due to lack of similar precedents"]
            confidence = 0.3
            predicted_rate = 0.5

        # Recommendation
        if confidence > 0.6 and predicted_rate > 0.5:
            recommendation = "proceed"
        elif confidence < 0.4:
            recommendation = "defer"
        else:
            recommendation = "modify"

        return {
            "predicted_effect": effect,
            "confidence": round(confidence, 2),
            "side_effects": side_effects,
            "recommendation": recommendation,
        }

    async def predict_next_state(self) -> dict:
        """Predict AXIOM's next state based on current trajectory."""
        state = await self.state_tracker.get_state()
        capabilities = await self.capability_model.get_capabilities()

        # Get learning stats
        stats = await self.backend.get_learning_stats()

        # Simple trajectory prediction based on current trends
        current_success = state.get("success_rate_overall", 0.5)
        weakest = state.get("weakest_capability", "unknown")
        weakest_rate = capabilities.get(weakest, 0.3)

        predictions = {
            "if_continue_current": (
                f"Success rate will {'improve' if current_success < 0.7 else 'plateau'} "
                f"as {weakest} lessons accumulate. "
                f"Estimated rate: {min(0.95, current_success + 0.05):.0%}"
            ),
            "if_focus_weakness": (
                f"Focusing on {weakest} (currently {weakest_rate:.0%}) "
                f"could improve it to ~{min(0.8, weakest_rate + 0.2):.0%} "
                f"with targeted practice"
            ),
            "recommendation": self._trajectory_recommendation(state, capabilities),
        }

        return predictions

    def _estimate_side_effects(self, change_type: str, similar: list[dict]) -> list[str]:
        """Estimate side effects based on change type and past experience."""
        effects = []
        if change_type == "code_modification":
            effects.append("May require testing before deployment")
        elif change_type == "config_change":
            effects.append("Immediate effect on behavior")
        elif change_type == "model_retrain":
            effects.append("Temporary performance dip during retraining")
            effects.append("May affect other capabilities (catastrophic forgetting risk)")

        # Check for failures in similar past changes
        failures = [l for l in similar if not l.get("success")]
        if failures:
            effects.append(f"{len(failures)}/{len(similar)} similar past changes had issues")

        return effects if effects else ["No significant side effects expected"]

    def _trajectory_recommendation(self, state: dict, capabilities: dict) -> str:
        """Generate recommendation based on trajectory."""
        success = state.get("success_rate_overall", 0.5)
        lessons = state.get("lessons_learned", 0)

        if lessons < 20:
            return "Focus on accumulating more diverse experiences before optimization"
        elif success < 0.5:
            return "Success rate is low - analyze failure modes and adjust approach"
        elif success > 0.8:
            return "Strong performance - explore new domains to expand capabilities"
        else:
            return "Moderate performance - continue current trajectory with targeted improvements"
