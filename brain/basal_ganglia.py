"""
BASAL GANGLIA — a real action-selection mechanism (Go/NoGo gating by learned value).

Not a prompt and not a fixed mapping. The cortex (the conversational LLM) proposes; this
region GATES which response *program* (stance) is released, using a value function it LEARNS
from reward. Reward = the emotional consequence of the last action (the Amygdala's read of the
user's next turn): "when I did X, the reaction was Y." Positive valence reinforces (Go),
negative suppresses (NoGo). Procedural memory persists across sessions on the volume.

Mechanism: a contextual linear policy. For each discrete action a, a weight vector w_a over the
context embedding. value(s,a) = w_a · embed(s). Select argmax (epsilon-greedy). On reward r,
delta-rule update toward r: w_a += lr * (r - value(s,a)) * embed(s).
"""
from __future__ import annotations
import os, hashlib
import numpy as np


class BasalGanglia:
    # Discrete response programs ("motor programs") the region selects among.
    ACTIONS = ["DIRECT", "PROBE", "MIRROR", "PUSH", "LIGHTEN"]

    def __init__(self, embed_fn, dim, path=None, lr=0.15, epsilon=0.1, seed=0):
        self.embed_fn = embed_fn
        self.dim = int(dim)
        self.path = path
        self.lr = float(lr)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((len(self.ACTIONS), self.dim), dtype=np.float32)
        self.n_updates = 0
        if path and os.path.exists(path):
            try: self.load()
            except Exception: pass
        self.ready = True

    def _featurize(self, text):
        v = np.asarray(self.embed_fn(text or ""), dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            out = np.zeros(self.dim, dtype=np.float32)
            n = min(self.dim, v.shape[0]); out[:n] = v[:n]; v = out
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 1e-8 else v

    def values(self, text):
        x = self._featurize(text)
        return {a: float(self.W[i] @ x) for i, a in enumerate(self.ACTIONS)}

    def select(self, text, explore=True):
        x = self._featurize(text)
        vals = {a: float(self.W[i] @ x) for i, a in enumerate(self.ACTIONS)}
        if explore and self.rng.random() < self.epsilon:
            action = self.ACTIONS[int(self.rng.integers(len(self.ACTIONS)))]
            explored = True
        else:
            action = max(vals, key=vals.get)
            explored = False
        margin = sorted(vals.values(), reverse=True)
        conf = round(float(margin[0] - margin[1]), 4) if len(margin) > 1 else 0.0
        return {"action": action, "values": {k: round(v, 4) for k, v in vals.items()},
                "explore": explored, "confidence": conf}

    def reward(self, text, action, r):
        if action not in self.ACTIONS:
            return {"error": "unknown action", "action": action}
        i = self.ACTIONS.index(action)
        x = self._featurize(text)
        pred = float(self.W[i] @ x)
        r = float(max(-1.0, min(1.0, r)))
        self.W[i] += self.lr * (r - pred) * x
        new = float(self.W[i] @ x)
        self.n_updates += 1
        if self.path:
            try: self.save()
            except Exception: pass
        return {"action": action, "reward": r, "value_before": round(pred, 4),
                "value_after": round(new, 4), "n_updates": self.n_updates}

    def save(self):
        np.savez(self.path, W=self.W, n=np.array([self.n_updates]))

    def load(self):
        d = np.load(self.path)
        W = d["W"]
        if W.shape == self.W.shape:
            self.W = W.astype(np.float32)
            self.n_updates = int(d["n"][0]) if "n" in d.files else 0


def _hash_embed(text, dim=64):
    """Deterministic standalone featurizer for tests (no model load)."""
    h = hashlib.sha256((text or "").encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.standard_normal(dim).astype(np.float32)


_singleton = None


def get_basal_ganglia(embed_fn=None, dim=384, path=None):
    global _singleton
    if _singleton is None:
        _singleton = BasalGanglia(embed_fn or (lambda t: _hash_embed(t, dim)), dim, path=path)
    return _singleton


class CandidateSelector:
    """Value head over (context, candidate). The cortex proposes candidate replies; this
    head scores the ACTUAL generated candidates and releases the best (Go), suppressing the
    rest (NoGo). Linear over phi = [ctx, cand, ctx*cand] — the multiplicative term lets it
    learn context-dependent candidate preferences. Learns online from reward; persists."""

    def __init__(self, embed_fn, dim, path=None, lr=0.1, epsilon=0.0, seed=0):
        self.embed_fn = embed_fn
        self.dim = int(dim)
        self.fdim = 3 * self.dim
        self.path = path
        self.lr = float(lr)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(self.fdim, dtype=np.float32)
        self.n_updates = 0
        if path and os.path.exists(path):
            try: self.load()
            except Exception: pass
        self.ready = True

    def _emb(self, t):
        v = np.asarray(self.embed_fn(t or ""), dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            out = np.zeros(self.dim, dtype=np.float32)
            n = min(self.dim, v.shape[0]); out[:n] = v[:n]; v = out
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 1e-8 else v

    def _phi(self, c, x):
        return np.concatenate([c, x, c * x]).astype(np.float32)

    def value(self, context, candidate):
        return float(self.w @ self._phi(self._emb(context), self._emb(candidate)))

    def score(self, context, candidates, explore=True):
        c = self._emb(context)
        vals = [float(self.w @ self._phi(c, self._emb(cand))) for cand in candidates]
        if explore and len(candidates) and self.rng.random() < self.epsilon:
            best = int(self.rng.integers(len(candidates))); explored = True
        else:
            best = int(np.argmax(vals)) if vals else 0; explored = False
        return {"values": [round(v, 4) for v in vals], "best_index": best,
                "explore": explored, "n_updates": self.n_updates}

    def learn(self, context, candidate, reward):
        phi = self._phi(self._emb(context), self._emb(candidate))
        pred = float(self.w @ phi)
        r = float(max(-1.0, min(1.0, reward)))
        self.w += self.lr * (r - pred) * phi
        new = float(self.w @ phi)
        self.n_updates += 1
        if self.path:
            try: self.save()
            except Exception: pass
        return {"reward": r, "value_before": round(pred, 4),
                "value_after": round(new, 4), "n_updates": self.n_updates}

    def save(self):
        np.savez(self.path, w=self.w, n=np.array([self.n_updates]))

    def load(self):
        d = np.load(self.path)
        if d["w"].shape == self.w.shape:
            self.w = d["w"].astype(np.float32)
            self.n_updates = int(d["n"][0]) if "n" in d.files else 0


_selector_singleton = None


def get_candidate_selector(embed_fn=None, dim=384, path=None):
    global _selector_singleton
    if _selector_singleton is None:
        _selector_singleton = CandidateSelector(embed_fn or (lambda t: _hash_embed(t, dim)), dim, path=path)
    return _selector_singleton


if __name__ == "__main__":
    dim = 64
    bg = BasalGanglia(lambda t: _hash_embed(t, dim), dim, path=None, epsilon=0.0, seed=1)
    ctx = "I keep second-guessing whether any of this is real."
    s0 = bg.select(ctx, explore=False)
    print("initial select:", s0["action"], "| values:", s0["values"])
    for _ in range(8): bg.reward(ctx, "PUSH", +1.0)
    s1 = bg.select(ctx, explore=False)
    print("after +PUSH x8:", s1["action"], "conf=", s1["confidence"], "| values:", s1["values"])
    for _ in range(8):
        bg.reward(ctx, "PUSH", -1.0); bg.reward(ctx, "MIRROR", +1.0)
    s2 = bg.select(ctx, explore=False)
    print("after -PUSH/+MIRROR x8:", s2["action"], "conf=", s2["confidence"], "| values:", s2["values"])
    other = "what did we decide about the build pipeline?"
    print("unrelated ctx values:", {k: round(v, 3) for k, v in bg.values(other).items()})
    print("DONE")
