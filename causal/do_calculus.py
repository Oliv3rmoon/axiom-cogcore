from __future__ import annotations
"""do(X=x) operator — intervention vs observation."""

import networkx as nx
from causal.scm import StructuralCausalModel


def do_intervention(scm: StructuralCausalModel, intervention_var: str,
                    value, query_var: str,
                    given: dict = None) -> dict:
    """
    P(Y | do(X=x)) ≠ P(Y | X=x)

    do(X=x) means:
    1. Remove all incoming edges to X (cut it from its causes)
    2. Set X = x
    3. Propagate effects forward through the graph

    Returns the estimated probability and explanation.
    """
    if intervention_var not in scm.graph:
        return {
            "result": 0.5,
            "explanation": f"Variable '{intervention_var}' not in causal graph",
            "causal_path": [],
            "recommendation": "Add more experience data",
        }

    if query_var not in scm.graph:
        return {
            "result": 0.5,
            "explanation": f"Query variable '{query_var}' not in causal graph",
            "causal_path": [],
            "recommendation": "Add more experience data",
        }

    # 1. Copy the graph and remove incoming edges to intervention variable
    modified = scm.graph.copy()
    incoming = list(modified.predecessors(intervention_var))
    modified.remove_edges_from([(pred, intervention_var) for pred in incoming])

    # 2. Compute causal effect by traversing paths from intervention to query
    try:
        paths = list(nx.all_simple_paths(modified, intervention_var, query_var, cutoff=5))
    except nx.NetworkXError:
        paths = []

    if not paths:
        # No causal path exists
        # Check if there was an observational correlation removed by do-calculus
        try:
            obs_paths = list(nx.all_simple_paths(scm.graph, intervention_var, query_var, cutoff=5))
        except nx.NetworkXError:
            obs_paths = []

        if obs_paths:
            explanation = (
                f"Observational correlation exists between {intervention_var} and {query_var}, "
                f"but the do-operator reveals no direct causal effect. "
                f"The correlation was due to confounding through: {', '.join(incoming)}"
            )
        else:
            explanation = f"No causal path from {intervention_var} to {query_var}"

        return {
            "result": 0.5,  # No effect = baseline
            "explanation": explanation,
            "causal_path": [],
            "recommendation": f"Intervening on {intervention_var} has no causal effect on {query_var}",
        }

    # 3. Estimate effect strength by multiplying edge strengths along paths
    best_path = None
    best_effect = 0.0
    for path in paths:
        effect = 1.0
        for i in range(len(path) - 1):
            edge_data = modified.edges.get((path[i], path[i + 1]), {})
            effect *= edge_data.get("strength", 0.5)
        if effect > best_effect:
            best_effect = effect
            best_path = path

    # Adjust result based on intervention value
    if value in (True, 1, "true", "yes"):
        result = 0.5 + best_effect * 0.5  # Positive intervention
    elif value in (False, 0, "false", "no"):
        result = 0.5 - best_effect * 0.5  # Negative intervention
    else:
        result = 0.5 + best_effect * 0.3  # Partial intervention

    # Apply conditioning on given variables
    if given:
        for gvar, gval in given.items():
            edge_data = scm.get_edge_data(gvar, query_var)
            if edge_data:
                modifier = edge_data.get("strength", 0.0) * 0.2
                if gval in (False, 0, "false", "no"):
                    result -= modifier
                else:
                    result += modifier

    result = max(0.0, min(1.0, result))

    # Build explanation
    path_str = " → ".join(best_path) if best_path else "none"
    recommendation = (
        f"Intervening on {intervention_var}={'yes' if value else 'no'} "
        f"{'increases' if result > 0.5 else 'decreases'} {query_var} probability to {result:.0%}"
    )

    return {
        "result": round(result, 3),
        "explanation": f"Causal path: {path_str} (strength={best_effect:.2f}). "
                       f"Confounders removed: {', '.join(incoming) if incoming else 'none'}",
        "causal_path": best_path or [],
        "recommendation": recommendation,
    }
