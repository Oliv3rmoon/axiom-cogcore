"""
Cingulate — contradiction / conflict monitor.

A dedicated Natural Language Inference net (entailment / neutral / CONTRADICTION)
scores whether a new statement asserts the OPPOSITE of an earlier claim. NLI alone
mistakes "topically unrelated" for "contradiction", so we gate it with embedding
similarity: a genuine contradiction must be the opposite claim ABOUT THE SAME SUBJECT.
contradiction := nli_contradiction >= threshold  AND  cos_sim(statement, prior) >= sim_threshold

When contradiction is detected the controller raises an INTERRUPT — it stops the
reflexive answer and forces a re-evaluation that surfaces the discrepancy. This is a
real mechanism (a classifier + a similarity gate), not a prompt.
"""
import os, time

if os.path.isdir("/app/data"):
    os.environ.setdefault("HF_HOME", "/app/data/hf")

_CANDIDATES = [
    "cross-encoder/nli-roberta-base",   # roberta-base, BPE (no sentencepiece), small
    "roberta-large-mnli",               # reliable fallback, larger
]


class Cingulate:
    def __init__(self, threshold: float = 0.55, sim_threshold: float = 0.25, embed_fn=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.threshold = float(threshold)
        self.sim_threshold = float(sim_threshold)
        self.embed_fn = embed_fn
        self._torch = torch
        last = None
        for mid in _CANDIDATES:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                mdl = AutoModelForSequenceClassification.from_pretrained(mid)
                mdl.eval()
                id2 = {int(k): str(v) for k, v in mdl.config.id2label.items()}
                contra = next((i for i, l in id2.items() if "contra" in l.lower()), 0)
                self.tok, self.model = tok, mdl
                self.id2label, self.contra_idx = id2, contra
                self.model_id = mid
                break
            except Exception as e:
                last = e
                continue
        else:
            raise RuntimeError(f"Cingulate: no NLI model could be loaded ({last})")

    def _contra(self, premise: str, hypothesis: str) -> float:
        enc = self.tok(premise, hypothesis, return_tensors="pt",
                       truncation=True, max_length=256)
        with self._torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = self._torch.softmax(logits, dim=-1)
        return float(probs[self.contra_idx])

    def _embed(self, text: str):
        import numpy as np
        v = np.asarray(self.embed_fn(text), dtype=float)
        return v[0] if v.ndim > 1 else v

    @staticmethod
    def _cos(a, b) -> float:
        import numpy as np
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def check(self, statement: str, against, threshold: float = None):
        """Opposite claim (NLI) AND same subject (embedding sim) => contradiction."""
        t0 = time.time()
        thr = self.threshold if threshold is None else float(threshold)
        statement = (statement or "").strip()
        priors = [(p or "").strip() for p in (against or [])]
        s_vec = self._embed(statement) if (self.embed_fn and statement) else None
        rows = []
        for p in priors:
            if not p or not statement:
                rows.append({"nli": 0.0, "sim": 0.0}); continue
            nli = max(self._contra(p, statement), self._contra(statement, p))
            sim = self._cos(s_vec, self._embed(p)) if s_vec is not None else 1.0
            rows.append({"nli": round(nli, 4), "sim": round(sim, 4)})
        flagged = [(i, r) for i, r in enumerate(rows)
                   if r["nli"] >= thr and r["sim"] >= self.sim_threshold]
        if flagged:
            best_i, best = max(flagged, key=lambda ir: ir[1]["nli"]); contradiction = True
        elif rows:
            best_i = max(range(len(rows)), key=lambda i: rows[i]["nli"])
            best = rows[best_i]; contradiction = False
        else:
            best_i, best, contradiction = -1, {"nli": 0.0, "sim": 0.0}, False
        return {
            "contradiction": bool(contradiction),
            "score": best["nli"], "sim": best["sim"],
            "threshold": thr, "sim_threshold": self.sim_threshold,
            "with_index": best_i,
            "with_statement": (priors[best_i] if best_i >= 0 else None),
            "nli_scores": [r["nli"] for r in rows],
            "sims": [r["sim"] for r in rows],
            "model": self.model_id, "ms": int((time.time() - t0) * 1000),
        }


_cingulate = None
def get_cingulate(threshold: float = 0.55, sim_threshold: float = 0.25, embed_fn=None):
    global _cingulate
    if _cingulate is None:
        _cingulate = Cingulate(threshold=threshold, sim_threshold=sim_threshold, embed_fn=embed_fn)
    return _cingulate
