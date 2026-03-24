from __future__ import annotations
"""Unified curiosity signal manager combining prediction error, RND, and info gain."""

import numpy as np

import config
from shared.db import get_db
from shared.embeddings import EmbeddingService
from curiosity.prediction_error import PredictionErrorTracker
from curiosity.rnd import RNDModule
from curiosity.information_gain import InformationGainTracker


class CuriosityManager:
    """
    Combines three curiosity signals into one unified score:
    - Prediction error (0.4 weight): how wrong was the world model?
    - RND novelty (0.3 weight): is this state novel?
    - Information gain (0.3 weight): would exploring here improve the model?
    """

    WEIGHT_PRED = 0.4
    WEIGHT_RND = 0.3
    WEIGHT_INFO = 0.3

    def __init__(self):
        self.pred_tracker = PredictionErrorTracker()
        self.rnd = RNDModule()
        self.info_tracker = InformationGainTracker()

    def record_experience(self, domain: str, state_embedding: np.ndarray,
                          predicted_outcome: np.ndarray, actual_outcome: np.ndarray,
                          loss_before: float = 0.0, loss_after: float = 0.0) -> dict:
        """
        Record a full curiosity observation from one experience.
        Returns the individual and combined curiosity scores.
        """
        pred_score = self.pred_tracker.record(domain, predicted_outcome, actual_outcome)
        rnd_score = self.rnd.compute_novelty(state_embedding)
        info_score = self.info_tracker.record(domain, loss_before, loss_after)

        # Train RND predictor on this state (reduce its novelty for next time)
        self.rnd.train_on_state(state_embedding)

        combined = (
            self.WEIGHT_PRED * pred_score
            + self.WEIGHT_RND * rnd_score
            + self.WEIGHT_INFO * info_score
        )

        return {
            "prediction_error": pred_score,
            "rnd_novelty": rnd_score,
            "information_gain": info_score,
            "combined_score": combined,
            "domain": domain,
        }

    def get_domain_curiosity(self, domain: str) -> dict:
        """Get current curiosity scores for a specific domain."""
        pred = self.pred_tracker.get_domain_score(domain)
        rnd = self._estimate_rnd_for_domain(domain)
        info = self.info_tracker.get_domain_score(domain)
        combined = self.WEIGHT_PRED * pred + self.WEIGHT_RND * rnd + self.WEIGHT_INFO * info

        return {
            "domain": domain,
            "prediction_error": pred,
            "rnd_novelty": rnd,
            "information_gain": info,
            "combined_score": combined,
        }

    def get_all_signals(self) -> list[dict]:
        """Get curiosity signals for all tracked domains, sorted by score."""
        all_domains = set()
        all_domains.update(self.pred_tracker.get_all_scores().keys())
        all_domains.update(self.info_tracker.get_all_scores().keys())

        signals = [self.get_domain_curiosity(d) for d in all_domains]
        signals.sort(key=lambda x: x["combined_score"], reverse=True)
        return signals

    def evaluate_goal(self, goal_text: str, embedder: EmbeddingService) -> dict:
        """
        Evaluate curiosity for a proposed goal.
        Uses embedding similarity to known domains.
        """
        goal_embedding = embedder.embed(goal_text)

        # Check RND novelty of the goal itself
        rnd_score = self.rnd.compute_novelty(goal_embedding)

        # Find closest domain
        all_signals = self.get_all_signals()
        if not all_signals:
            return {
                "curiosity_score": 0.8,
                "novelty": rnd_score,
                "expected_information_gain": 0.7,
                "recommendation": "high_priority",
                "reason": "No prior experience data - completely novel territory",
            }

        # Use the average curiosity across domains as a baseline
        avg_curiosity = np.mean([s["combined_score"] for s in all_signals])

        # Combine RND novelty with domain curiosity
        curiosity_score = 0.5 * rnd_score + 0.5 * avg_curiosity

        if curiosity_score > 0.7:
            recommendation = "high_priority"
            reason = "High novelty and expected learning value"
        elif curiosity_score > 0.4:
            recommendation = "moderate_priority"
            reason = "Moderate learning opportunity"
        else:
            recommendation = "low_priority"
            reason = "Familiar territory with limited learning potential"

        return {
            "curiosity_score": round(curiosity_score, 3),
            "novelty": round(rnd_score, 3),
            "expected_information_gain": round(
                self.WEIGHT_INFO * avg_curiosity + self.WEIGHT_RND * rnd_score, 3
            ),
            "recommendation": recommendation,
            "reason": reason,
        }

    async def persist_signals(self):
        """Save current curiosity signals to database."""
        db = await get_db()
        for signal in self.get_all_signals():
            await db.execute(
                """INSERT OR REPLACE INTO curiosity_signals
                   (domain, prediction_error_avg, rnd_novelty_avg,
                    info_gain_avg, combined_score, sample_count, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (
                    signal["domain"],
                    signal["prediction_error"],
                    signal["rnd_novelty"],
                    signal["information_gain"],
                    signal["combined_score"],
                    len(self.pred_tracker._errors.get(signal["domain"], [])),
                ),
            )
        await db.commit()

    def _estimate_rnd_for_domain(self, domain: str) -> float:
        """Estimate RND novelty for a domain (using domain name as proxy)."""
        # For domains we haven't explicitly computed RND for, use a moderate default
        return 0.5

    def get_novelty_by_action(self) -> dict[str, float]:
        """Get novelty scores grouped by action type."""
        pred_scores = self.pred_tracker.get_all_scores()
        result = {}
        for action in config.ACTION_TYPES:
            if action in pred_scores:
                result[action] = round(pred_scores[action], 3)
        return result
