from __future__ import annotations
"""Client for the AXIOM backend API."""

import httpx
import config


class BackendClient:
    """Async HTTP client for the AXIOM backend."""

    def __init__(self):
        self.base_url = config.BACKEND_URL
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=30.0
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_lessons(self) -> list[dict]:
        """Fetch all lessons from backend."""
        client = await self._get_client()
        try:
            resp = await client.get("/api/lessons")
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("lessons", [])
        except Exception:
            return []

    async def get_skills(self) -> list[dict]:
        """Fetch all skills from backend."""
        client = await self._get_client()
        try:
            resp = await client.get("/api/skills")
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("skills", [])
        except Exception:
            return []

    async def get_goals(self) -> list[dict]:
        """Fetch all goals from backend."""
        client = await self._get_client()
        try:
            resp = await client.get("/api/goals")
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("goals", [])
        except Exception:
            return []

    async def get_journal(self, limit: int = 50) -> list[dict]:
        """Fetch journal entries."""
        client = await self._get_client()
        try:
            resp = await client.get(f"/api/journal?limit={limit}")
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            return []

    async def get_knowledge(self) -> list[dict]:
        """Fetch knowledge nodes."""
        client = await self._get_client()
        try:
            resp = await client.get("/api/knowledge")
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("knowledge", [])
        except Exception:
            return []

    async def get_learning_stats(self) -> dict:
        """Fetch learning statistics."""
        client = await self._get_client()
        try:
            resp = await client.get("/api/learning/stats")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    async def get_training_data(self) -> list[dict]:
        """Fetch JSONL training data export."""
        client = await self._get_client()
        try:
            resp = await client.get("/api/training-data")
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception:
            return []
