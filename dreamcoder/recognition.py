from __future__ import annotations
"""Neural recognition model -- DreamCoder's recognition network.

Maps a task encoding (goal embedding) to per-production log-weights, producing a
task-conditioned grammar. Trained supervised on (goal embedding, solution program)
replays from solved tasks so that, for a new goal, synthesis is biased toward the
productions that goal is likely to need.
"""
import numpy as np
import torch
import torch.nn as nn

from .dsl import SEQ, NAT, decisions


class Recognition(nn.Module):
    def __init__(self, in_dim, prod_keys, hidden=128):
        super().__init__()
        self.prod_keys = list(prod_keys)
        self.idx = {k: i for i, k in enumerate(self.prod_keys)}
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, len(self.prod_keys)),
        )

    def forward(self, e):
        return self.net(e)


def _type_index_lists(grammar, idx):
    out = {}
    for t, ks in grammar.prods_by_type().items():
        out[t] = [idx[k] for k in ks if k in idx]
    return out


def _base_logits(grammar, prod_keys):
    return torch.tensor([grammar.weights.get(k, 0.0) for k in prod_keys], dtype=torch.float32)


def train_recognition(model, base_grammar, pairs, epochs=60, lr=1e-2):
    if not pairs:
        return []
    type_idx = _type_index_lists(base_grammar, model.idx)
    base = _base_logits(base_grammar, model.prod_keys)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    embs = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    progs = [p[1] for p in pairs]
    hist = []
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(embs) + base.unsqueeze(0)
        loss = torch.zeros((), dtype=torch.float32)
        ndec = 0
        for bi, prog in enumerate(progs):
            for (typ, key) in decisions(prog):
                idxs = type_idx.get(typ, [])
                if not idxs or key not in model.idx:
                    continue
                sub = logits[bi, idxs]
                logp = torch.log_softmax(sub, dim=0)
                loss = loss - logp[idxs.index(model.idx[key])]
                ndec += 1
        if ndec > 0:
            loss = loss / ndec
            loss.backward(); opt.step()
            hist.append(float(loss.item()))
        else:
            hist.append(0.0)
    return hist


def condition(base_grammar, model, emb_tensor):
    """Return a task-conditioned grammar (base weights + recognition logits)."""
    g = base_grammar.copy()
    with torch.no_grad():
        logits = model(emb_tensor.unsqueeze(0))[0]
    for k in g.weights:
        if k in model.idx:
            g.weights[k] = base_grammar.weights.get(k, 0.0) + float(logits[model.idx[k]])
    return g


def production_accuracy(base_grammar, model, pairs):
    """Top-1 production prediction accuracy per program node: does the recognition-
    conditioned grammar's most likely production match the one actually used, vs the
    base grammar? An honest held-out measure of what the recognition model learned."""
    type_idx = _type_index_lists(base_grammar, model.idx)
    base = _base_logits(base_grammar, model.prod_keys)
    cr = cb = tot = 0
    for emb, prog in pairs:
        et = torch.tensor(emb, dtype=torch.float32)
        with torch.no_grad():
            rl = model(et.unsqueeze(0))[0] + base
        for (typ, key) in decisions(prog):
            idxs = type_idx.get(typ, [])
            if not idxs or key not in model.idx:
                continue
            pick_r = model.prod_keys[idxs[int(torch.argmax(rl[idxs]))]]
            pick_b = model.prod_keys[idxs[int(torch.argmax(base[idxs]))]]
            cr += (pick_r == key); cb += (pick_b == key); tot += 1
    return {"nodes": tot,
            "base_top1": round(cb / tot, 3) if tot else None,
            "recognition_top1": round(cr / tot, 3) if tot else None}
