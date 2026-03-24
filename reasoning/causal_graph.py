from __future__ import annotations
"""Causal relationship tracking using NetworkX."""

import networkx as nx
from reasoning.thought_graph import ThoughtNode, ThoughtEdge


class CausalGraph:
    """
    Manages causal relationships between thoughts using a directed graph.
    Supports do-calculus-inspired reasoning about interventions.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node: ThoughtNode):
        """Add a thought node to the causal graph."""
        self.graph.add_node(node.id, **node.to_dict())

    def add_causal_link(self, cause_id: str, effect_id: str,
                        relationship: str = "enables", strength: float = 0.5):
        """Add a causal link between two nodes."""
        self.graph.add_edge(
            cause_id, effect_id,
            relationship=relationship,
            strength=strength,
        )

    def get_causes(self, node_id: str) -> list[dict]:
        """Get all direct causes of a node."""
        if node_id not in self.graph:
            return []
        causes = []
        for pred in self.graph.predecessors(node_id):
            edge_data = self.graph.edges[pred, node_id]
            node_data = self.graph.nodes[pred]
            causes.append({
                "node": node_data,
                "relationship": edge_data.get("relationship", "enables"),
                "strength": edge_data.get("strength", 0.5),
            })
        return causes

    def get_effects(self, node_id: str) -> list[dict]:
        """Get all direct effects of a node."""
        if node_id not in self.graph:
            return []
        effects = []
        for succ in self.graph.successors(node_id):
            edge_data = self.graph.edges[node_id, succ]
            node_data = self.graph.nodes[succ]
            effects.append({
                "node": node_data,
                "relationship": edge_data.get("relationship", "enables"),
                "strength": edge_data.get("strength", 0.5),
            })
        return effects

    def get_causal_chain(self, start_id: str, max_depth: int = 5) -> list[list[str]]:
        """
        Get all causal chains starting from a node.
        Returns list of paths (each path is a list of node IDs).
        """
        if start_id not in self.graph:
            return []
        paths = []
        for target in self.graph.nodes:
            if target == start_id:
                continue
            try:
                for path in nx.all_simple_paths(self.graph, start_id, target, cutoff=max_depth):
                    paths.append(path)
            except nx.NetworkXError:
                continue
        return paths

    def find_contradictions(self) -> list[tuple[str, str]]:
        """Find pairs of nodes connected by 'contradicts' relationships."""
        contradictions = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("relationship") == "contradicts":
                contradictions.append((u, v))
        return contradictions

    def get_all_edges(self) -> list[dict]:
        """Get all edges as dicts."""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source_id": u,
                "target_id": v,
                "relationship": data.get("relationship", "supports"),
                "strength": data.get("strength", 0.5),
            })
        return edges
