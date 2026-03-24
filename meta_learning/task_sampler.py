from __future__ import annotations
"""Sample learning tasks from experience for Reptile meta-learning."""

import numpy as np
import torch

import config
from shared.db import get_db
from world_model.buffer import _decode_array


class TaskSampler:
    """
    Samples learning tasks from the experience buffer.

    Each task = a subset of experiences filtered by domain (action_type).
    Tasks for meta-learning:
    - Outcome prediction per action type
    - Success prediction per action type
    """

    def __init__(self):
        pass

    async def sample_tasks(self, n_tasks: int = config.REPTILE_TASKS_PER_UPDATE,
                           samples_per_task: int = 16,
                           obs_dim: int = config.WORLD_MODEL_OBS_DIM
                           ) -> list[dict]:
        """
        Sample n_tasks from the experience buffer, each containing
        samples_per_task examples from a single domain.
        """
        db = await get_db()

        # Get available domains with enough data
        cursor = await db.execute(
            "SELECT action, COUNT(*) as cnt FROM experiences GROUP BY action HAVING cnt >= ?",
            (samples_per_task,),
        )
        domains = []
        async for row in cursor:
            domains.append(row[0])

        if not domains:
            return []

        tasks = []
        rng = np.random.default_rng()
        chosen_domains = rng.choice(domains, size=min(n_tasks, len(domains)), replace=len(domains) < n_tasks)

        for domain in chosen_domains:
            cursor = await db.execute(
                """SELECT state_embedding, action, outcome_embedding, was_successful
                   FROM experiences WHERE action = ?
                   ORDER BY RANDOM() LIMIT ?""",
                (domain, samples_per_task),
            )
            rows = await cursor.fetchall()
            if len(rows) < 2:
                continue

            obs_list = []
            out_list = []
            suc_list = []
            act_list = []
            for row in rows:
                obs_list.append(_decode_array(row[0], obs_dim))
                act_idx = config.ACTION_TYPES.index(row[1]) if row[1] in config.ACTION_TYPES else 0
                act_list.append(act_idx)
                out_list.append(_decode_array(row[2], obs_dim))
                suc_list.append(float(row[3]))

            task_data = {
                "obs": torch.tensor(np.array(obs_list), dtype=torch.float32, device=config.DEVICE).unsqueeze(1),
                "actions": torch.tensor(act_list, dtype=torch.long, device=config.DEVICE).unsqueeze(1),
                "outcomes": torch.tensor(np.array(out_list), dtype=torch.float32, device=config.DEVICE).unsqueeze(1),
                "successes": torch.tensor(suc_list, dtype=torch.float32, device=config.DEVICE).unsqueeze(1),
            }

            tasks.append({
                "data": task_data,
                "domain": domain,
            })

        return tasks

    async def get_available_domains(self) -> list[dict]:
        """Get domains with their sample counts."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT action, COUNT(*) as cnt FROM experiences GROUP BY action ORDER BY cnt DESC"
        )
        domains = []
        async for row in cursor:
            domains.append({"domain": row[0], "count": row[1]})
        return domains
