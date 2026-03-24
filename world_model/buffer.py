from __future__ import annotations
"""Experience replay buffer for world model training."""

import json
import struct
import numpy as np
import torch

import config
from shared.db import get_db


class Experience:
    """Single experience tuple."""

    def __init__(self, state_embedding: np.ndarray, action: str,
                 action_details: str, outcome_embedding: np.ndarray,
                 was_successful: bool, prediction_error: float = 0.0,
                 prediction_id: str = ""):
        self.state_embedding = state_embedding
        self.action = action
        self.action_details = action_details
        self.outcome_embedding = outcome_embedding
        self.was_successful = was_successful
        self.prediction_error = prediction_error
        self.prediction_id = prediction_id


def _encode_array(arr: np.ndarray) -> bytes:
    """Encode numpy array to bytes."""
    return arr.astype(np.float32).tobytes()


def _decode_array(data: bytes, dim: int) -> np.ndarray:
    """Decode bytes to numpy array."""
    return np.frombuffer(data, dtype=np.float32).copy().reshape(-1)[:dim]


class ExperienceBuffer:
    """Priority replay buffer backed by SQLite."""

    def __init__(self):
        self._count_cache: int | None = None

    async def add(self, exp: Experience):
        """Add an experience to the buffer."""
        db = await get_db()
        await db.execute(
            """INSERT OR REPLACE INTO experiences
               (state_embedding, action, action_details, outcome_embedding,
                was_successful, prediction_error, prediction_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                _encode_array(exp.state_embedding),
                exp.action,
                exp.action_details,
                _encode_array(exp.outcome_embedding),
                int(exp.was_successful),
                exp.prediction_error,
                exp.prediction_id,
            ),
        )
        await db.commit()
        self._count_cache = None

        # Prune if over capacity
        count = await self.count()
        if count > config.REPLAY_BUFFER_SIZE:
            await self._prune(count - config.REPLAY_BUFFER_SIZE)

    async def count(self) -> int:
        """Total experiences in buffer."""
        if self._count_cache is not None:
            return self._count_cache
        db = await get_db()
        cursor = await db.execute("SELECT COUNT(*) FROM experiences")
        row = await cursor.fetchone()
        self._count_cache = row[0]
        return self._count_cache

    async def sample_batch(self, batch_size: int, seq_len: int = 1,
                           obs_dim: int = config.WORLD_MODEL_OBS_DIM
                           ) -> dict | None:
        """
        Sample a batch of experiences. For seq_len > 1, samples contiguous sequences.
        Returns dict of tensors or None if not enough data.
        """
        count = await self.count()
        if count < batch_size * seq_len:
            return None

        db = await get_db()

        if seq_len == 1:
            # Priority sampling: higher prediction_error = more likely sampled
            cursor = await db.execute(
                """SELECT state_embedding, action, outcome_embedding,
                          was_successful, prediction_error
                   FROM experiences ORDER BY
                   (prediction_error + 0.01) * RANDOM() DESC LIMIT ?""",
                (batch_size,),
            )
            rows = await cursor.fetchall()
            return self._rows_to_batch(rows, obs_dim)
        else:
            # Sample contiguous sequences
            max_id_cursor = await db.execute("SELECT MAX(id), MIN(id) FROM experiences")
            max_row = await max_id_cursor.fetchone()
            max_id, min_id = max_row[0], max_row[1]

            all_obs = []
            all_actions = []
            all_outcomes = []
            all_successes = []

            attempts = 0
            while len(all_obs) < batch_size and attempts < batch_size * 3:
                attempts += 1
                start_id = np.random.randint(min_id, max(min_id + 1, max_id - seq_len + 1))
                cursor = await db.execute(
                    """SELECT state_embedding, action, outcome_embedding,
                              was_successful, prediction_error
                       FROM experiences WHERE id >= ? ORDER BY id LIMIT ?""",
                    (start_id, seq_len),
                )
                rows = await cursor.fetchall()
                if len(rows) < seq_len:
                    continue

                obs_seq = []
                act_seq = []
                out_seq = []
                suc_seq = []
                for row in rows:
                    obs_seq.append(_decode_array(row[0], obs_dim))
                    act_idx = config.ACTION_TYPES.index(row[1]) if row[1] in config.ACTION_TYPES else 0
                    act_seq.append(act_idx)
                    out_seq.append(_decode_array(row[2], obs_dim))
                    suc_seq.append(float(row[3]))

                all_obs.append(obs_seq)
                all_actions.append(act_seq)
                all_outcomes.append(out_seq)
                all_successes.append(suc_seq)

            if len(all_obs) < batch_size:
                return None

            return {
                "obs_embeddings": torch.tensor(
                    np.array(all_obs[:batch_size]), dtype=torch.float32, device=config.DEVICE
                ),
                "actions": torch.tensor(
                    np.array(all_actions[:batch_size]), dtype=torch.long, device=config.DEVICE
                ),
                "outcome_embeddings": torch.tensor(
                    np.array(all_outcomes[:batch_size]), dtype=torch.float32, device=config.DEVICE
                ),
                "successes": torch.tensor(
                    np.array(all_successes[:batch_size]), dtype=torch.float32, device=config.DEVICE
                ),
            }

    def _rows_to_batch(self, rows, obs_dim: int) -> dict:
        """Convert DB rows to tensor batch (seq_len=1)."""
        obs_list, act_list, out_list, suc_list = [], [], [], []
        for row in rows:
            obs_list.append(_decode_array(row[0], obs_dim))
            act_idx = config.ACTION_TYPES.index(row[1]) if row[1] in config.ACTION_TYPES else 0
            act_list.append(act_idx)
            out_list.append(_decode_array(row[2], obs_dim))
            suc_list.append(float(row[3]))

        return {
            "obs_embeddings": torch.tensor(
                np.array(obs_list), dtype=torch.float32, device=config.DEVICE
            ).unsqueeze(1),
            "actions": torch.tensor(
                act_list, dtype=torch.long, device=config.DEVICE
            ).unsqueeze(1),
            "outcome_embeddings": torch.tensor(
                np.array(out_list), dtype=torch.float32, device=config.DEVICE
            ).unsqueeze(1),
            "successes": torch.tensor(
                suc_list, dtype=torch.float32, device=config.DEVICE
            ).unsqueeze(1),
        }

    async def _prune(self, n: int):
        """Remove n oldest, lowest-error experiences."""
        db = await get_db()
        await db.execute(
            """DELETE FROM experiences WHERE id IN
               (SELECT id FROM experiences
                ORDER BY prediction_error ASC, created_at ASC LIMIT ?)""",
            (n,),
        )
        await db.commit()
        self._count_cache = None

    async def get_all_for_fisher(self, limit: int = 500,
                                  obs_dim: int = config.WORLD_MODEL_OBS_DIM) -> dict | None:
        """Get a batch of recent experiences for Fisher computation."""
        db = await get_db()
        cursor = await db.execute(
            """SELECT state_embedding, action, outcome_embedding, was_successful
               FROM experiences ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return None
        return self._rows_to_batch(rows, obs_dim)
