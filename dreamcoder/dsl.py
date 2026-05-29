from __future__ import annotations
"""Typed DSL, PCFG grammar, and synthesizers for DreamCoder-style program
synthesis over AXIOM action traces.

A program denotes a sequence of action tokens (type Seq):
    act a                       : Seq   -> [a]
    chain s1 s2                 : Seq   -> eval(s1) ++ eval(s2)
    repeat n s   (n in NATS)    : Seq   -> eval(s) * n
    macro m  (learned)          : Seq   -> eval(defs[m])   (compositional)

The grammar is a PCFG: each production has a weight; P(prod) = softmax of the
weights within the production's result type. Synthesis finds the maximum-prior
program whose evaluation equals a target trace, either exactly (DP parser) or by
type-directed best-first enumeration (the search the recognition model guides).
"""
import math, heapq, json
from itertools import count

SEQ = "Seq"
NAT = "Nat"

# ---- program constructors (programs are nested tuples) ----
def p_act(tok):      return ("act", tok)
def p_macro(name):   return ("macro", name)
def p_chain(a, b):   return ("chain", a, b)
def p_repeat(n, s):  return ("repeat", n, s)

def prog_to_str(p):
    if p is None: return "None"
    k = p[0]
    if k == "act":    return str(p[1])
    if k == "macro":  return str(p[1]).upper()
    if k == "chain":  return f"({prog_to_str(p[1])} ; {prog_to_str(p[2])})"
    if k == "repeat": return f"repeat{p[1]}({prog_to_str(p[2])})"
    return "?"

def decisions(p):
    """Yield (result_type, production_key) for every node of a program."""
    k = p[0]
    if k == "act":
        yield (SEQ, f"act::{p[1]}")
    elif k == "macro":
        yield (SEQ, f"macro::{p[1]}")
    elif k == "chain":
        yield (SEQ, "chain"); yield from decisions(p[1]); yield from decisions(p[2])
    elif k == "repeat":
        yield (SEQ, "repeat"); yield (NAT, f"nat::{p[1]}"); yield from decisions(p[2])

def macros_used(p):
    out = []
    def rec(x):
        if not isinstance(x, tuple): return
        if x[0] == "macro": out.append(x[1]); return
        for c in x[1:]: rec(c)
    rec(p)
    return list(dict.fromkeys(out))

def _prog_pack(p):
    return [p[0]] + [(_prog_pack(x) if isinstance(x, tuple) else x) for x in p[1:]]
def _prog_unpack(l):
    return (l[0],) + tuple(_prog_unpack(x) if isinstance(x, list) else x for x in l[1:])

# ---- hole helpers (for best-first enumeration) ----
def _first_hole_type(p):
    if not isinstance(p, tuple): return None
    if p[0] == "hole": return p[1]
    for x in p[1:]:
        t = _first_hole_type(x)
        if t: return t
    return None

def _holes_by_type(p):
    if not isinstance(p, tuple): return {}
    if p[0] == "hole": return {p[1]: 1}
    out = {}
    for x in p[1:]:
        for t, c in _holes_by_type(x).items():
            out[t] = out.get(t, 0) + c
    return out

def _fill_leftmost(p, repl):
    if not isinstance(p, tuple): return p, False
    if p[0] == "hole": return repl, True
    if p[0] in ("act", "macro"): return p, False
    new = []; done = False
    for x in p[1:]:
        if not done and isinstance(x, tuple):
            nx, d = _fill_leftmost(x, repl); new.append(nx); done = done or d
        else:
            new.append(x)
    return (p[0], *new), done

def _size_h(p):
    if not isinstance(p, tuple): return 1
    if p[0] in ("hole", "act", "macro"): return 1
    return 1 + sum(_size_h(x) for x in p[1:])


class Grammar:
    def __init__(self, actions, nats=(2, 3), macros=None, weights=None):
        self.actions = list(dict.fromkeys(actions))
        self.nats = list(nats)
        self.macros = dict(macros or {})        # name -> definition program
        self.weights = dict(weights or {})
        self._exp_cache = {}
        for k in self.production_keys():
            self.weights.setdefault(k, 0.0)

    def production_keys(self):
        ks = [f"act::{a}" for a in self.actions] + ["chain", "repeat"]
        ks += [f"nat::{n}" for n in self.nats]
        ks += [f"macro::{m}" for m in self.macros]
        return ks

    def prods_by_type(self):
        seq = [f"act::{a}" for a in self.actions] + ["chain", "repeat"] + [f"macro::{m}" for m in self.macros]
        nat = [f"nat::{n}" for n in self.nats]
        return {SEQ: seq, NAT: nat}

    def result_type(self, key):
        return NAT if key.startswith("nat::") else SEQ

    def _logZ(self):
        z = {}
        for t, ks in self.prods_by_type().items():
            if not ks:
                z[t] = 0.0; continue
            m = max(self.weights[k] for k in ks)
            z[t] = m + math.log(sum(math.exp(self.weights[k] - m) for k in ks) + 1e-12)
        return z

    def logprob(self, key, _z=None):
        z = _z or self._logZ()
        return self.weights[key] - z[self.result_type(key)]

    def expansion(self, name, _depth=0):
        if name in self._exp_cache: return self._exp_cache[name]
        if _depth > 60: return tuple()
        toks = tuple(self.eval(self.macros[name], _depth + 1))
        self._exp_cache[name] = toks
        return toks

    def eval(self, p, _depth=0):
        if _depth > 300: return []
        k = p[0]
        if k == "act":    return [p[1]]
        if k == "macro":  return list(self.expansion(p[1], _depth))
        if k == "chain":  return self.eval(p[1], _depth + 1) + self.eval(p[2], _depth + 1)
        if k == "repeat": return self.eval(p[2], _depth + 1) * int(p[1])
        return []

    def size(self, p):
        k = p[0]
        if k in ("act", "macro"): return 1
        if k == "chain":  return 1 + self.size(p[1]) + self.size(p[2])
        if k == "repeat": return 2 + self.size(p[2])
        return 1

    def loglik(self, p, _z=None):
        z = _z or self._logZ()
        k = p[0]
        if k == "act":    return self.logprob(f"act::{p[1]}", z)
        if k == "macro":  return self.logprob(f"macro::{p[1]}", z)
        if k == "chain":  return self.logprob("chain", z) + self.loglik(p[1], z) + self.loglik(p[2], z)
        if k == "repeat": return self.logprob("repeat", z) + self.logprob(f"nat::{p[1]}", z) + self.loglik(p[2], z)
        return -1e9

    def add_macro(self, name, definition):
        self.macros[name] = definition
        self._exp_cache.pop(name, None)
        self.weights.setdefault(f"macro::{name}", 0.0)

    def copy(self):
        return Grammar(self.actions, self.nats, dict(self.macros), dict(self.weights))

    def to_json(self):
        return json.dumps({
            "actions": self.actions, "nats": self.nats,
            "macro_defs": {k: _prog_pack(v) for k, v in self.macros.items()},
            "weights": self.weights,
        })

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        macros = {k: _prog_unpack(v) for k, v in d.get("macro_defs", {}).items()}
        return Grammar(d["actions"], tuple(d.get("nats", [2, 3])), macros, d.get("weights", {}))


def parse_trace(grammar, target):
    """Exact maximum-loglik program whose eval == target (CYK-style DP over spans).
    Returns (loglik, program) or (-1e9, None)."""
    z = grammar._logZ()
    target = list(target); n = len(target)
    if n == 0: return (-1e9, None)
    macs = [(m, list(grammar.expansion(m))) for m in grammar.macros]
    best = [[(-1e18, None)] * (n + 1) for _ in range(n + 1)]
    for L in range(1, n + 1):
        for i in range(0, n - L + 1):
            j = i + L; span = target[i:j]; cur = (-1e18, None)
            if L == 1:
                key = f"act::{span[0]}"
                if key in grammar.weights:
                    ll = grammar.logprob(key, z)
                    if ll > cur[0]: cur = (ll, ("act", span[0]))
            for m, exp in macs:
                if exp == span:
                    ll = grammar.logprob(f"macro::{m}", z)
                    if ll > cur[0]: cur = (ll, ("macro", m))
            for nn in grammar.nats:
                if L % nn == 0:
                    Ls = L // nn
                    if target[i:i + Ls] * nn == span:
                        sub = best[i][i + Ls]
                        if sub[1] is not None:
                            ll = grammar.logprob("repeat", z) + grammar.logprob(f"nat::{nn}", z) + sub[0]
                            if ll > cur[0]: cur = (ll, ("repeat", nn, sub[1]))
            for k in range(i + 1, j):
                l = best[i][k]; r = best[k][j]
                if l[1] is not None and r[1] is not None:
                    ll = grammar.logprob("chain", z) + l[0] + r[0]
                    if ll > cur[0]: cur = (ll, ("chain", l[1], r[1]))
            best[i][j] = cur
    return best[0][n]


def _expansions_for(grammar, T, z):
    out = []
    if T == SEQ:
        for a in grammar.actions: out.append((("act", a), grammar.logprob(f"act::{a}", z)))
        for m in grammar.macros:  out.append((("macro", m), grammar.logprob(f"macro::{m}", z)))
        out.append((("chain", ("hole", SEQ), ("hole", SEQ)), grammar.logprob("chain", z)))
        out.append((("repeat", ("hole", NAT), ("hole", SEQ)), grammar.logprob("repeat", z)))
    else:
        for n in grammar.nats: out.append((n, grammar.logprob(f"nat::{n}", z)))
    return out


def _best_first(grammar, accept, budget=5000, max_size=60, target=None):
    """Type-directed best-first enumeration of complete Seq programs in ~decreasing
    prior order. Returns (program|None, expansions, loglik). `accept(prog, tokens)`
    decides if a complete program is a solution. When `target` is given, partial
    programs whose committed prefix already diverges from the target are pruned, so
    enumeration is efficient and a sharper (recognition-conditioned) grammar reaches
    the solution in fewer pops -- the neural-guided search of DreamCoder."""
    z = grammar._logZ()
    hmax = {t: max((grammar.logprob(k, z) for k in ks), default=0.0)
            for t, ks in grammar.prods_by_type().items()}
    def heur(p):
        return sum(hmax.get(t, 0.0) * c for t, c in _holes_by_type(p).items())
    def cprefix(p):
        if not isinstance(p, tuple): return [], False
        k = p[0]
        if k == "hole":  return [], True
        if k == "act":   return [p[1]], False
        if k == "macro": return list(grammar.expansion(p[1])), False
        if k == "chain":
            pa, ha = cprefix(p[1])
            if ha: return pa, True
            pb, hb = cprefix(p[2]); return pa + pb, hb
        if k == "repeat":
            n = p[1]
            if isinstance(n, tuple): return [], True
            pa, ha = cprefix(p[2])
            if ha: return pa, True
            return pa * int(n), False
        return [], False
    def consistent(p):
        if target is None: return True
        pref, _ = cprefix(p)
        return len(pref) <= len(target) and pref == target[:len(pref)]
    tie = count()
    start = ("hole", SEQ)
    h = [(-(0.0 + heur(start)), next(tie), start, 0.0)]
    expansions = 0
    while h and expansions < budget:
        _, _, prog, ll = heapq.heappop(h)
        T = _first_hole_type(prog)
        expansions += 1
        if T is None:
            toks = grammar.eval(prog)
            if accept(prog, toks):
                return prog, expansions, ll
            continue
        for repl, lp in _expansions_for(grammar, T, z):
            np_, _ = _fill_leftmost(prog, repl)
            if _size_h(np_) > max_size: continue
            if not consistent(np_): continue
            nll = ll + lp
            heapq.heappush(h, (-(nll + heur(np_)), next(tie), np_, nll))
    return None, expansions, -1e9


def enumerate_solve(grammar, target, budget=5000, max_size=60):
    target = list(target)
    return _best_first(grammar, lambda p, toks: toks == target, budget, max_size, target=target)


def enumerate_plan(grammar, min_len=2, max_len=12, budget=6000, max_size=60):
    return _best_first(grammar, lambda p, toks: min_len <= len(toks) <= max_len, budget, max_size)
