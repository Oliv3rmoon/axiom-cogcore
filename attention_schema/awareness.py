from __future__ import annotations
"""Functional awareness — 'what am I focusing on and why?'"""

from attention_schema.schema import AttentionSchema
from attention_schema.meta_cognition import MetaCognition


class AwarenessEngine:
    """
    Generates introspective self-reports about AXIOM's attention state.
    This is the 'consciousness' layer — AXIOM's model of her own attention.
    """

    def __init__(self, schema: AttentionSchema, meta: MetaCognition):
        self.schema = schema
        self.meta = meta

    def introspect(self, depth: str = "normal") -> dict:
        """
        Generate a self-report about current attention state.
        depth: "shallow", "normal", or "deep"
        """
        focus = self.schema.current_focus
        biases = self.meta.analyze_biases()
        recommendations = self.meta.get_recommendations()

        # Build self-report
        if focus:
            focus_desc = (
                f"I'm focused on '{focus.target}' (type: {focus.target_type}) "
                f"with attention strength {focus.attention_strength:.2f}."
            )
            if focus.signals:
                drivers = []
                for signal, value in sorted(focus.signals.items(),
                                            key=lambda x: x[1], reverse=True):
                    if value > 0.5:
                        drivers.append(f"{signal} ({value:.2f})")
                if drivers:
                    focus_desc += f" Driven by: {', '.join(drivers[:3])}."
        else:
            focus_desc = "I don't have a current focus target."

        # Add history analysis for deep introspection
        if depth == "deep":
            history = list(self.schema.focus_history)
            if history:
                unique_targets = len(set(t.target for t in history))
                avg_duration = sum(t.duration_seconds for t in history) / max(1, len(history))
                focus_desc += (
                    f" Over my recent history, I've attended to {unique_targets} "
                    f"different targets with an average focus duration of {avg_duration:.0f}s."
                )

        return {
            "self_report": focus_desc,
            "meta_observations": biases,
            "recommendations": recommendations,
        }
