from __future__ import annotations
"""Graph-based reasoning workspace for multi-step persistent reasoning."""

import uuid
import json
from datetime import datetime, timedelta

import numpy as np
import networkx as nx

import config
from shared.embeddings import EmbeddingService
from shared.db import get_db
from reasoning.thought_graph import ThoughtNode, ThoughtEdge
from reasoning.causal_graph import CausalGraph


class ReasoningWorkspace:
    """
    Persistent reasoning workspace backed by a NetworkX directed graph and SQLite.

    Supports:
    - Adding thoughts connected to parent thoughts
    - Causal relationships between thoughts
    - Querying by embedding similarity
    - Summarization
    - Auto-expiry after THOUGHT_TTL_HOURS
    """

    def __init__(self, embedder: EmbeddingService):
        self.embedder = embedder
        # workspace_id -> (graph, causal_graph, goal)
        self._workspaces: dict[str, dict] = {}

    async def start(self, goal: str, initial_context: str = "") -> dict:
        """Create a new reasoning workspace with a root node."""
        workspace_id = str(uuid.uuid4())

        graph = nx.DiGraph()
        causal = CausalGraph()

        # Create root node
        root = ThoughtNode(
            workspace_id=workspace_id,
            thought=goal,
            thought_type="observation",
            confidence=1.0,
        )
        root_emb = self.embedder.embed(goal)
        root.embedding = root_emb.tolist()

        graph.add_node(root.id, **root.to_dict())
        causal.add_node(root)

        self._workspaces[workspace_id] = {
            "graph": graph,
            "causal": causal,
            "goal": goal,
            "root_id": root.id,
            "created_at": datetime.utcnow(),
        }

        # Persist root node
        await self._save_node(root)

        # If there's initial context, add it as a child
        if initial_context:
            await self.add_thought(
                workspace_id, initial_context, root.id, "observation", 0.8
            )

        return {
            "workspace_id": workspace_id,
            "root_node_id": root.id,
            "status": "active",
        }

    async def add_thought(self, workspace_id: str, thought: str,
                          parent_id: str | None = None,
                          thought_type: str = "observation",
                          confidence: float = 0.5) -> dict:
        """Add a thought node to the workspace."""
        ws = self._workspaces.get(workspace_id)
        if ws is None:
            ws = await self._load_workspace(workspace_id)
            if ws is None:
                raise ValueError(f"Workspace {workspace_id} not found")

        graph = ws["graph"]

        # Check node limit
        if graph.number_of_nodes() >= config.MAX_THOUGHT_NODES:
            # Remove oldest expired nodes
            self._prune_expired(graph)
            if graph.number_of_nodes() >= config.MAX_THOUGHT_NODES:
                raise ValueError(f"Workspace has reached maximum of {config.MAX_THOUGHT_NODES} nodes")

        node = ThoughtNode(
            workspace_id=workspace_id,
            thought=thought,
            thought_type=thought_type,
            confidence=confidence,
            parent_id=parent_id,
        )
        node_emb = self.embedder.embed(thought)
        node.embedding = node_emb.tolist()

        graph.add_node(node.id, **node.to_dict())

        if parent_id and parent_id in graph:
            graph.add_edge(parent_id, node.id, relationship="supports", strength=0.8)

        ws["causal"].add_node(node)
        await self._save_node(node)

        if parent_id:
            edge = ThoughtEdge(
                source_id=parent_id, target_id=node.id,
                relationship="supports", strength=0.8, workspace_id=workspace_id,
            )
            await self._save_edge(edge)

        return {
            "node_id": node.id,
            "total_nodes": graph.number_of_nodes(),
        }

    async def add_causal_link(self, workspace_id: str, cause_id: str,
                              effect_id: str, relationship: str = "enables",
                              strength: float = 0.5):
        """Add a causal link between two existing nodes."""
        ws = self._workspaces.get(workspace_id)
        if ws is None:
            ws = await self._load_workspace(workspace_id)
            if ws is None:
                raise ValueError(f"Workspace {workspace_id} not found")

        graph = ws["graph"]
        if cause_id not in graph or effect_id not in graph:
            raise ValueError("Both cause and effect nodes must exist in workspace")

        graph.add_edge(cause_id, effect_id, relationship=relationship, strength=strength)
        ws["causal"].add_causal_link(cause_id, effect_id, relationship, strength)

        edge = ThoughtEdge(
            source_id=cause_id, target_id=effect_id,
            relationship=relationship, strength=strength,
            workspace_id=workspace_id,
        )
        await self._save_edge(edge)

    async def get_workspace(self, workspace_id: str) -> dict:
        """Get full workspace state."""
        ws = self._workspaces.get(workspace_id)
        if ws is None:
            ws = await self._load_workspace(workspace_id)
            if ws is None:
                return {"error": "Workspace not found"}

        graph = ws["graph"]
        nodes = [graph.nodes[n] for n in graph.nodes]
        edges = []
        for u, v, data in graph.edges(data=True):
            edges.append({
                "source_id": u,
                "target_id": v,
                "relationship": data.get("relationship", "supports"),
                "strength": data.get("strength", 0.5),
            })

        conclusions = [n for n in nodes if n.get("thought_type") == "conclusion"]
        open_questions = [n for n in nodes if n.get("thought_type") == "question"]

        summary = self._summarize(nodes, edges)

        return {
            "workspace_id": workspace_id,
            "goal": ws.get("goal", ""),
            "nodes": nodes,
            "edges": edges,
            "conclusions": conclusions,
            "open_questions": open_questions,
            "summary": summary,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    async def query(self, workspace_id: str, question: str) -> dict:
        """Find nodes relevant to a question using embedding similarity."""
        ws = self._workspaces.get(workspace_id)
        if ws is None:
            ws = await self._load_workspace(workspace_id)
            if ws is None:
                return {"error": "Workspace not found", "relevant_nodes": [], "answer": ""}

        graph = ws["graph"]
        question_emb = self.embedder.embed(question)

        scored_nodes = []
        for node_id in graph.nodes:
            node_data = graph.nodes[node_id]
            embedding = node_data.get("embedding")
            if embedding is None:
                # Re-embed from thought text
                thought = node_data.get("thought", "")
                if thought:
                    emb = self.embedder.embed(thought)
                    sim = float(np.dot(question_emb, emb) / (
                        np.linalg.norm(question_emb) * np.linalg.norm(emb) + 1e-8
                    ))
                else:
                    sim = 0.0
            else:
                # Stored embedding may be a list
                emb = np.array(embedding, dtype=np.float32) if isinstance(embedding, list) else embedding
                sim = float(np.dot(question_emb, emb) / (
                    np.linalg.norm(question_emb) * np.linalg.norm(emb) + 1e-8
                ))

            scored_nodes.append((node_data, sim))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, s in scored_nodes[:5] if s > 0.2]

        if top_nodes:
            thoughts = [n.get("thought", "") for n in top_nodes]
            answer = (
                f"Based on the reasoning graph, the most relevant thoughts are: "
                + "; ".join(thoughts[:3])
            )
        else:
            answer = "No strongly relevant thoughts found in the workspace."

        return {
            "relevant_nodes": top_nodes,
            "answer": answer,
        }

    def _summarize(self, nodes: list[dict], edges: list[dict]) -> str:
        """Generate a text summary of the reasoning chain."""
        if not nodes:
            return "Empty workspace."

        observations = [n["thought"] for n in nodes if n.get("thought_type") == "observation"]
        hypotheses = [n["thought"] for n in nodes if n.get("thought_type") == "hypothesis"]
        conclusions = [n["thought"] for n in nodes if n.get("thought_type") == "conclusion"]
        questions = [n["thought"] for n in nodes if n.get("thought_type") == "question"]

        parts = []
        if observations:
            parts.append(f"Observations ({len(observations)}): {'; '.join(observations[:3])}")
        if hypotheses:
            parts.append(f"Hypotheses ({len(hypotheses)}): {'; '.join(hypotheses[:3])}")
        if conclusions:
            parts.append(f"Conclusions ({len(conclusions)}): {'; '.join(conclusions[:3])}")
        if questions:
            parts.append(f"Open questions ({len(questions)}): {'; '.join(questions[:3])}")
        parts.append(f"Total: {len(nodes)} thoughts, {len(edges)} connections.")

        return " | ".join(parts)

    def _prune_expired(self, graph: nx.DiGraph):
        """Remove expired nodes from graph."""
        now = datetime.utcnow()
        expired = []
        for node_id in graph.nodes:
            expires = graph.nodes[node_id].get("expires_at")
            if expires and isinstance(expires, str):
                try:
                    if datetime.fromisoformat(expires) < now:
                        expired.append(node_id)
                except ValueError:
                    pass
        for node_id in expired:
            graph.remove_node(node_id)

    async def _save_node(self, node: ThoughtNode):
        """Persist a node to SQLite."""
        db = await get_db()
        emb_blob = None
        if node.embedding is not None:
            emb_blob = np.array(node.embedding, dtype=np.float32).tobytes()
        await db.execute(
            """INSERT OR REPLACE INTO thought_nodes
               (id, workspace_id, thought, thought_type, confidence,
                parent_id, embedding, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.id, node.workspace_id, node.thought, node.thought_type,
                node.confidence, node.parent_id, emb_blob,
                node.created_at.isoformat(), node.expires_at.isoformat(),
            ),
        )
        await db.commit()

    async def _save_edge(self, edge: ThoughtEdge):
        """Persist an edge to SQLite."""
        db = await get_db()
        await db.execute(
            """INSERT INTO thought_edges
               (workspace_id, source_id, target_id, relationship, strength)
               VALUES (?, ?, ?, ?, ?)""",
            (edge.workspace_id, edge.source_id, edge.target_id,
             edge.relationship, edge.strength),
        )
        await db.commit()

    async def _load_workspace(self, workspace_id: str) -> dict | None:
        """Load a workspace from SQLite."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT id, thought, thought_type, confidence, parent_id, embedding, created_at, expires_at "
            "FROM thought_nodes WHERE workspace_id = ?",
            (workspace_id,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return None

        graph = nx.DiGraph()
        causal = CausalGraph()
        goal = ""

        for row in rows:
            node_id = row[0]
            node_dict = {
                "id": node_id,
                "workspace_id": workspace_id,
                "thought": row[1],
                "thought_type": row[2],
                "confidence": row[3],
                "parent_id": row[4],
                "created_at": row[6],
                "expires_at": row[7],
            }
            graph.add_node(node_id, **node_dict)
            if not goal:
                goal = row[1]

        # Load edges
        cursor = await db.execute(
            "SELECT source_id, target_id, relationship, strength "
            "FROM thought_edges WHERE workspace_id = ?",
            (workspace_id,),
        )
        async for row in cursor:
            if row[0] in graph and row[1] in graph:
                graph.add_edge(row[0], row[1], relationship=row[2], strength=row[3])
                causal.add_causal_link(row[0], row[1], row[2], row[3])

        root_id = ""
        for n in graph.nodes:
            if graph.nodes[n].get("parent_id") is None:
                root_id = n
                break

        ws = {
            "graph": graph,
            "causal": causal,
            "goal": goal,
            "root_id": root_id,
            "created_at": datetime.utcnow(),
        }
        self._workspaces[workspace_id] = ws
        return ws
