from __future__ import annotations
"""Meta-cognitive control — adjust attention based on self-model."""

from attention_schema.schema import AttentionSchema


class MetaCognition:
    """
    Analyzes AXIOM's own attention patterns and provides meta-cognitive insights.
    """

    def __init__(self, schema: AttentionSchema):
        self.schema = schema

    def analyze_biases(self) -> list[str]:
        """Detect attention biases — over/under-attended domains."""
        observations = []
        weights = self.schema.attention_weights
        counts = self.schema._domain_counts

        if not weights:
            return ["No attention data yet"]

        total = sum(weights.values())
        if total == 0:
            return ["No attention weight accumulated"]

        for domain, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            share = weight / total
            count = counts.get(domain, 0)
            if share > 0.4 and count > 3:
                observations.append(
                    f"Attention bias toward '{domain}' ({share:.0%} of total attention)"
                )
            elif share < 0.05 and count > 0:
                observations.append(
                    f"Under-attending to '{domain}' ({share:.0%} of total attention)"
                )

        if not observations:
            observations.append("Attention distribution appears balanced")

        return observations

    def get_recommendations(self) -> list[str]:
        """Generate meta-cognitive recommendations."""
        recommendations = []
        weights = self.schema.attention_weights
        counts = self.schema._domain_counts

        if not weights:
            return ["Start building attention history by processing more tasks"]

        total = sum(weights.values())
        if total == 0:
            return []

        # Find under-attended domains
        for domain, weight in weights.items():
            share = weight / total
            if share < 0.1 and counts.get(domain, 0) >= 2:
                recommendations.append(
                    f"Consider switching to '{domain}' — low attention but has prior experience"
                )

        # Check attention diversity
        n_domains = len(weights)
        if n_domains > 0:
            ideal_share = 1.0 / n_domains
            max_share = max(weights.values()) / total
            if max_share > 3 * ideal_share:
                recommendations.append(
                    "Attention is concentrated — consider exploring other domains"
                )

        return recommendations
