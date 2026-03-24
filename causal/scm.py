from __future__ import annotations
"""Structural Causal Model — DAG with functional relationships."""

import uuid
import json
from dataclasses import dataclass, field

import networkx as nx

import config
from shared.db import get_db


@dataclass
class CausalNode:
    """A variable in the causal graph."""
    name: str
    values: list = field(default_factory=lambda: [True, False])
    observations: dict = field(default_factory=dict)  # value -> count

    def observe(self, value):
        key = str(value)
        self.observations[key] = self.observations.get(key, 0) + 1

    @property
    def total_observations(self) -> int:
        return sum(self.observations.values())


@dataclass
class CausalEdge:
    """A causal relationship: cause → effect."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cause: str = ""
    effect: str = ""
    strength: float = 0.0
    evidence_count: int = 0
    mechanism: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cause": self.cause,
            "effect": self.effect,
            "strength": round(self.strength, 3),
            "evidence_count": self.evidence_count,
            "mechanism": self.mechanism,
        }


class StructuralCausalModel:
    """
    Structural Causal Model: M = (U, V, F, P(U))

    Maintains a directed acyclic graph of causal relationships.
    Each edge represents: cause → effect with a measured strength.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: dict[str, CausalNode] = {}
        self.edges: dict[str, CausalEdge] = {}  # id -> edge

    def add_node(self, name: str, values: list = None):
        """Add a variable to the causal model."""
        if name not in self.nodes:
            node = CausalNode(name=name, values=values or [True, False])
            self.nodes[name] = node
            self.graph.add_node(name)

    def add_edge(self, cause: str, effect: str, strength: float = 0.5,
                 evidence_count: int = 1, mechanism: str = "") -> CausalEdge:
        """Add a causal relationship."""
        # Ensure nodes exist
        self.add_node(cause)
        self.add_node(effect)

        # Check for existing edge
        for eid, edge in self.edges.items():
            if edge.cause == cause and edge.effect == effect:
                # Update existing
                edge.strength = (edge.strength * edge.evidence_count + strength * evidence_count) / (
                    edge.evidence_count + evidence_count
                )
                edge.evidence_count += evidence_count
                if mechanism:
                    edge.mechanism = mechanism
                self.graph[cause][effect]["strength"] = edge.strength
                self.graph[cause][effect]["evidence_count"] = edge.evidence_count
                return edge

        # Check for cycle
        if effect in self.graph and cause in nx.descendants(self.graph, effect):
            # Would create a cycle — skip
            return CausalEdge(cause=cause, effect=effect, strength=0.0)

        edge = CausalEdge(
            cause=cause, effect=effect, strength=strength,
            evidence_count=evidence_count, mechanism=mechanism,
        )
        self.edges[edge.id] = edge
        self.graph.add_edge(cause, effect, strength=strength,
                            evidence_count=evidence_count,
                            edge_id=edge.id)
        return edge

    def get_parents(self, node: str) -> list[str]:
        """Get direct causes of a node."""
        if node not in self.graph:
            return []
        return list(self.graph.predecessors(node))

    def get_children(self, node: str) -> list[str]:
        """Get direct effects of a node."""
        if node not in self.graph:
            return []
        return list(self.graph.successors(node))

    def get_edge_data(self, cause: str, effect: str) -> dict | None:
        """Get edge data between two nodes."""
        if self.graph.has_edge(cause, effect):
            return dict(self.graph.edges[cause, effect])
        return None

    def get_all_nodes(self) -> list[str]:
        return list(self.graph.nodes)

    def get_all_edges(self) -> list[dict]:
        result = []
        for u, v, data in self.graph.edges(data=True):
            eid = data.get("edge_id", "")
            edge = self.edges.get(eid)
            if edge:
                result.append(edge.to_dict())
            else:
                result.append({
                    "cause": u, "effect": v,
                    "strength": data.get("strength", 0.5),
                    "evidence_count": data.get("evidence_count", 0),
                })
        return result

    async def save_all(self):
        """Persist all edges to database."""
        db = await get_db()
        for edge in self.edges.values():
            await db.execute(
                """INSERT OR REPLACE INTO causal_edges
                   (id, cause, effect, strength, evidence_count, mechanism, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (edge.id, edge.cause, edge.effect, edge.strength,
                 edge.evidence_count, edge.mechanism),
            )
        await db.commit()

    async def load_from_db(self):
        """Load causal graph from database."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT id, cause, effect, strength, evidence_count, mechanism FROM causal_edges"
        )
        async for row in cursor:
            edge = CausalEdge(
                id=row[0], cause=row[1], effect=row[2],
                strength=row[3], evidence_count=row[4], mechanism=row[5] or "",
            )
            self.add_node(edge.cause)
            self.add_node(edge.effect)
            self.edges[edge.id] = edge
            if not self.graph.has_edge(edge.cause, edge.effect):
                self.graph.add_edge(edge.cause, edge.effect,
                                    strength=edge.strength,
                                    evidence_count=edge.evidence_count,
                                    edge_id=edge.id)
