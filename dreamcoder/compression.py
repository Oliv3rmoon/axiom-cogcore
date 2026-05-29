from __future__ import annotations
"""MDL-driven compositional library learning -- DreamCoder's 'abstraction' sleep.

Greedily extract the fragment whose naming most reduces the corpus description
length (DL = negative log-prior under a uniform PCFG, which parse_trace minimizes,
so DL tracks minimum program size while charging for a growing library). A new
macro's definition is parsed under the current grammar, so it may reuse earlier
macros -> abstractions compose into a hierarchy. Iterate until no fragment helps.
"""
from collections import Counter
from .dsl import parse_trace, prog_to_str


def _uniform(g):
    for k in g.weights:
        g.weights[k] = 0.0
    return g


def _corpus_dl(grammar, traces):
    total = 0.0; progs = []
    for t in traces:
        ll, p = parse_trace(grammar, t)
        progs.append(p)
        total += (-ll) if p is not None else 1e6
    return total, progs


def _candidates(traces, min_len=2, max_len=8):
    c = Counter()
    for t in traces:
        n = len(t)
        for L in range(min_len, min(max_len, n) + 1):
            for i in range(0, n - L + 1):
                c[tuple(t[i:i + L])] += 1
    return c


def abstract(grammar, traces, max_new=8, min_count=2, top_k=25):
    g = _uniform(grammar.copy())
    base_dl, _ = _corpus_dl(g, traces)
    new_names = []; steps = []
    while len(new_names) < max_new:
        cur_dl, _ = _corpus_dl(g, traces)
        existing = {tuple(g.expansion(m)) for m in g.macros}
        cnt = _candidates(traces)
        cands = [(frag, cnt[frag]) for frag in cnt
                 if cnt[frag] >= min_count and tuple(frag) not in existing and len(frag) >= 2]
        cands.sort(key=lambda fc: fc[1] * (len(fc[0]) - 1), reverse=True)
        best = None
        for frag, _cc in cands[:top_k]:
            trial = _uniform(g.copy())
            ll, defn = parse_trace(trial, list(frag))
            if defn is None:
                continue
            name = f"m{len(g.macros) + 1}"
            trial.add_macro(name, defn)
            _uniform(trial)
            dl, _ = _corpus_dl(trial, traces)
            sav = cur_dl - dl
            if best is None or sav > best[0]:
                best = (sav, name, defn, list(frag))
        if best is None or best[0] <= 1e-9:
            break
        sav, name, defn, frag = best
        g.add_macro(name, defn)
        _uniform(g)
        new_names.append(name)
        steps.append({
            "macro": name, "expansion": frag,
            "definition": prog_to_str(defn), "dl_savings": round(sav, 3),
        })
    final_dl, _ = _corpus_dl(g, traces)
    ratio = base_dl / final_dl if final_dl > 0 else 1.0
    return g, new_names, {
        "dl_before": round(base_dl, 3), "dl_after": round(final_dl, 3),
        "compression_ratio": round(ratio, 3), "steps": steps,
    }
