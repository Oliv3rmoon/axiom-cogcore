from __future__ import annotations
"""'What would have happened if...?' counterfactual reasoning."""

from causal.scm import StructuralCausalModel
from causal.do_calculus import do_intervention


def counterfactual_query(scm: StructuralCausalModel,
                         actual_action: str, actual_outcome: str,
                         counterfactual_action: str) -> dict:
    """
    Pearl's three-step counterfactual procedure:
    1. ABDUCTION: Given actual evidence, infer context
    2. ACTION: Modify the model with the counterfactual intervention
    3. PREDICTION: Compute the outcome in the modified model

    Simplified for AXIOM's text-based domain.
    """
    # Step 1: ABDUCTION — understand what happened
    # Look up the actual action in the causal graph
    actual_outcome_bool = actual_outcome.lower() in ("success", "true", "succeeded", "1")

    # Step 2: ACTION — what if we had done counterfactual_action instead?
    # Use do-calculus to compute P(success | do(action = counterfactual))
    success_vars = [n for n in scm.get_all_nodes()
                    if "success" in n.lower() or "outcome" in n.lower()]
    query_var = success_vars[0] if success_vars else "success"

    # Ensure the query variable exists
    if query_var not in scm.graph:
        scm.add_node(query_var)

    # Check if counterfactual action exists in graph
    if counterfactual_action not in scm.graph:
        # Try to find a related node
        related = [n for n in scm.get_all_nodes()
                   if counterfactual_action.replace("_", " ") in n.replace("_", " ") or
                   n.replace("_", " ") in counterfactual_action.replace("_", " ")]
        if related:
            counterfactual_action = related[0]
        else:
            # No data — make a reasonable estimate
            return {
                "counterfactual_outcome": "unknown",
                "probability": 0.5,
                "explanation": f"No causal data for '{counterfactual_action}'. Cannot compute counterfactual.",
                "confidence": 0.2,
            }

    # Step 3: PREDICTION — compute counterfactual outcome
    cf_result = do_intervention(
        scm, counterfactual_action, True, query_var
    )

    cf_prob = cf_result["result"]

    # Determine outcome
    if cf_prob > 0.6:
        cf_outcome = "likely_success"
    elif cf_prob < 0.4:
        cf_outcome = "likely_failure"
    else:
        cf_outcome = "uncertain"

    # Compare with what actually happened
    actual_str = "success" if actual_outcome_bool else "failure"
    if actual_outcome_bool and cf_prob > 0.6:
        explanation = f"The counterfactual action would likely have also succeeded (P={cf_prob:.2f})"
    elif not actual_outcome_bool and cf_prob > 0.6:
        explanation = (
            f"Had you used '{counterfactual_action}' instead of '{actual_action}', "
            f"the outcome would likely have been success (P={cf_prob:.2f})"
        )
    elif actual_outcome_bool and cf_prob < 0.4:
        explanation = (
            f"The counterfactual '{counterfactual_action}' would likely have failed (P={cf_prob:.2f}), "
            f"so the actual choice of '{actual_action}' was better"
        )
    else:
        explanation = f"Counterfactual outcome is uncertain (P={cf_prob:.2f})"

    # Confidence based on evidence
    path_edges = cf_result.get("causal_path", [])
    evidence_counts = []
    for i in range(len(path_edges) - 1):
        ed = scm.get_edge_data(path_edges[i], path_edges[i + 1])
        if ed:
            evidence_counts.append(ed.get("evidence_count", 0))
    avg_evidence = sum(evidence_counts) / max(1, len(evidence_counts)) if evidence_counts else 0
    confidence = min(0.95, avg_evidence / 20.0 + 0.3)

    return {
        "counterfactual_outcome": cf_outcome,
        "probability": round(cf_prob, 3),
        "explanation": explanation,
        "confidence": round(confidence, 3),
    }
