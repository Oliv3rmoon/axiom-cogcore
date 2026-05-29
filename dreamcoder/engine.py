from __future__ import annotations
"""DreamCoderEngine -- wake/sleep loop tying together the typed DSL, MDL
compositional library learning, and the neural recognition model, with
persistence to the cogcore DB.

A *task* is a solved goal: its text (-> embedding) plus the observed action trace.
WAKE   : synthesize the MAP program whose eval reproduces each trace (parse_trace).
SLEEP  : (abstraction) compress the solution corpus into reusable macros, then
         (dream) rebuild + train the recognition network on (goal, program) replays.
COMPOSE: for a NEW goal, condition the grammar on the goal embedding and enumerate
         the most-likely program -> a synthesized plan over the learned library.
"""
import os, io, json, base64
from collections import Counter
import numpy as np
import torch

import config
from shared.db import get_db
from .dsl import (Grammar, parse_trace, enumerate_solve, enumerate_plan,
                  prog_to_str, macros_used, decisions, SEQ, NAT)
from .compression import abstract
from .recognition import Recognition, train_recognition, condition, production_accuracy


class DreamCoderEngine:
    def __init__(self, embedder):
        self.embedder = embedder
        self.dim = getattr(embedder, "dim", config.LTC_INPUT_SIZE)
        self.grammar = Grammar(actions=[])
        self.recog = None
        self.meta = {}

    # ---------- persistence ----------
    async def _ensure_table(self):
        db = await get_db()
        await db.execute("CREATE TABLE IF NOT EXISTS dreamcoder_state "
                         "(id INTEGER PRIMARY KEY CHECK (id=1), grammar TEXT, recog TEXT, meta TEXT)")
        await db.commit()

    def _build_recog(self):
        self.recog = Recognition(self.dim, self.grammar.production_keys())

    def _recog_blob(self):
        if self.recog is None:
            return None
        buf = io.BytesIO(); torch.save(self.recog.state_dict(), buf)
        return base64.b64encode(buf.getvalue()).decode()

    def _load_recog(self, blob):
        try:
            self._build_recog()
            buf = io.BytesIO(base64.b64decode(blob))
            self.recog.load_state_dict(torch.load(buf, map_location="cpu"))
        except Exception:
            self.recog = None

    async def load(self):
        await self._ensure_table()
        db = await get_db()
        cur = await db.execute("SELECT grammar, recog, meta FROM dreamcoder_state WHERE id=1")
        row = await cur.fetchone()
        if row and row[0]:
            self.grammar = Grammar.from_json(row[0])
            self.meta = json.loads(row[2]) if row[2] else {}
            if row[1]:
                self._load_recog(row[1])

    async def save(self):
        await self._ensure_table()
        db = await get_db()
        await db.execute("INSERT OR REPLACE INTO dreamcoder_state (id, grammar, recog, meta) VALUES (1,?,?,?)",
                         (self.grammar.to_json(), self._recog_blob(), json.dumps(self.meta)))
        await db.commit()

    async def reset(self):
        await self._ensure_table()
        db = await get_db()
        await db.execute("DELETE FROM dreamcoder_state")
        await db.commit()
        self.grammar = Grammar(actions=[]); self.recog = None; self.meta = {}

    # ---------- data ----------
    async def _load_solutions(self, limit=1000):
        db = await get_db()
        cur = await db.execute(
            "SELECT task, domain, solution_steps FROM solved_tasks "
            "WHERE was_successful = 1 ORDER BY rowid DESC LIMIT ?", (limit,))
        out = []
        async for r in cur:
            try:
                steps = json.loads(r[2]) if r[2] else []
            except Exception:
                steps = []
            steps = [str(s) for s in steps if str(s).strip()]
            if steps:
                out.append({"task": r[0] or "", "domain": r[1] or "", "steps": steps})
        return out

    def _ingest_alphabet(self, sols):
        toks = [t for s in sols for t in s["steps"]]
        actions = list(dict.fromkeys(toks))
        self.grammar = Grammar(actions=actions, nats=(2, 3),
                               macros=dict(self.grammar.macros), weights=dict(self.grammar.weights))

    def _mle_update(self, traces, smoothing=1.0):
        cnt = Counter()
        for t in traces:
            _, p = parse_trace(self.grammar, t)
            if p is None:
                continue
            for (_typ, key) in decisions(p):
                cnt[key] += 1
        for k in list(self.grammar.weights):
            self.grammar.weights[k] = float(np.log(cnt.get(k, 0) + smoothing))

    def _embed(self, text):
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        return np.asarray(self.embedder.embed(text[:300]), dtype=np.float32)

    # ---------- sleep (pure core, no DB) ----------
    def _sleep_core(self, sols, max_new=6, min_count=2, epochs=60):
        if not sols:
            return {"error": "no solutions"}
        self._ingest_alphabet(sols)
        traces = [s["steps"] for s in sols]

        self._mle_update(traces)
        before_dl = sum(-parse_trace(self.grammar, t)[0] for t in traces)

        g2, new_macros, info = abstract(self.grammar, traces, max_new=max_new, min_count=min_count)
        self.grammar = g2
        self._mle_update(traces)
        after_dl = sum(-parse_trace(self.grammar, t)[0] for t in traces)

        self._build_recog()
        pairs = []
        for s in sols:
            _, p = parse_trace(self.grammar, s["steps"])
            if p is not None:
                pairs.append((self._embed(s["task"]), p))
        rec_hist = train_recognition(self.recog, self.grammar, pairs, epochs=epochs) if pairs else []
        acc = production_accuracy(self.grammar, self.recog, pairs) if pairs else {}

        self.meta = {"solutions": len(sols), "library_size": len(self.grammar.macros)}
        return {
            "solutions": len(sols),
            "alphabet_size": len(self.grammar.actions),
            "new_abstractions": new_macros,
            "library_size": len(self.grammar.macros),
            "compression": info,
            "dl_before_macros": round(before_dl, 3),
            "dl_after_macros": round(after_dl, 3),
            "recognition_loss_start": round(rec_hist[0], 4) if rec_hist else None,
            "recognition_loss_end": round(rec_hist[-1], 4) if rec_hist else None,
            "recognition_production_top1": acc.get("recognition_top1"),
            "base_production_top1": acc.get("base_top1"),
        }

    async def sleep(self, max_new=6, min_count=2, epochs=60):
        sols = await self._load_solutions()
        m = self._sleep_core(sols, max_new=max_new, min_count=min_count, epochs=epochs)
        if "error" not in m:
            m["recognition_benefit"] = self.recognition_benefit(sols)
            await self.save()
        return m

    # ---------- recognition value (honest, exact) ----------
    def recognition_benefit(self, sols, sample=12, budget=4000):
        nb, nr, db_, dr = [], [], [], []
        for s in sols[:sample]:
            steps = s["steps"]
            _, en_b, _ = enumerate_solve(self.grammar, steps, budget=budget)
            ll_b, _ = parse_trace(self.grammar, steps)
            if self.recog is not None and s["task"]:
                g = condition(self.grammar, self.recog, torch.tensor(self._embed(s["task"]), dtype=torch.float32))
            else:
                g = self.grammar
            _, en_r, _ = enumerate_solve(g, steps, budget=budget)
            ll_r, _ = parse_trace(g, steps)
            nb.append(en_b); nr.append(en_r); db_.append(-ll_b); dr.append(-ll_r)
        if not nb:
            return {}
        return {
            "tasks": len(nb),
            "mean_search_nodes_base": round(float(np.mean(nb)), 1),
            "mean_search_nodes_recognition": round(float(np.mean(nr)), 1),
            "mean_dl_base": round(float(np.mean(db_)), 3),
            "mean_dl_recognition": round(float(np.mean(dr)), 3),
        }

    # ---------- synthesis / compose ----------
    def synthesize_for_trace(self, steps, task_text="", use_recognition=True):
        g = self.grammar
        if use_recognition and self.recog is not None and task_text:
            g = condition(self.grammar, self.recog, torch.tensor(self._embed(task_text), dtype=torch.float32))
        ll, p = parse_trace(g, list(steps))
        return {"program": prog_to_str(p), "log_prior": round(ll, 3),
                "abstractions_used": macros_used(p) if p else [], "matches": (g.eval(p) == list(steps)) if p else False}

    def compose(self, task_text, min_len=2, max_len=12, budget=6000):
        if self.recog is not None and task_text:
            g = condition(self.grammar, self.recog, torch.tensor(self._embed(task_text), dtype=torch.float32))
        else:
            g = self.grammar
        if not g.actions:
            return {"error": "empty library -- run sleep first"}
        prog, nodes, ll = enumerate_plan(g, min_len=min_len, max_len=max_len, budget=budget)
        steps = g.eval(prog) if prog else []
        conf = float(np.exp(ll / max(1, len(steps)))) if prog else 0.0
        return {
            "program": prog_to_str(prog) if prog else None,
            "suggested_steps": steps,
            "abstractions_used": macros_used(prog) if prog else [],
            "confidence": round(min(1.0, conf), 3),
            "log_prior": round(ll, 3) if prog else None,
            "search_nodes": nodes,
        }

    # ---------- library view ----------
    def library_view(self):
        z = self.grammar._logZ()
        base = [{"type": "action", "name": a,
                 "prob": round(float(np.exp(self.grammar.logprob(f"act::{a}", z))), 4)}
                for a in self.grammar.actions]
        macros = [{"type": "abstraction", "name": m, "definition": prog_to_str(defn),
                   "expansion": list(self.grammar.expansion(m)),
                   "uses_abstractions": macros_used(defn),
                   "prob": round(float(np.exp(self.grammar.logprob(f"macro::{m}", z))), 4)}
                  for m, defn in self.grammar.macros.items()]
        base.sort(key=lambda x: x["prob"], reverse=True)
        macros.sort(key=lambda x: x["prob"], reverse=True)
        return {"actions": base, "abstractions": macros,
                "alphabet_size": len(self.grammar.actions), "library_size": len(self.grammar.macros)}
