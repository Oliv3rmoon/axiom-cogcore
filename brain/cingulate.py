"""
Cingulate — contradiction / conflict monitor.

The anterior cingulate's job here is NOT to phrase anything. It is a dedicated
Natural Language Inference net (entailment / neutral / CONTRADICTION) that scores
whether a new statement conflicts with things said earlier. When contradiction is
detected the controller raises an INTERRUPT — it stops the reflexive answer and
forces a re-evaluation pass that must surface the discrepancy.

This is a real mechanism (a separate classifier), not a prompt. It loads the first
NLI model that will actually load in this environment (safetensors-friendly,
no sentencepiece dependency), preferring the smaller one.
"""
import os, time

# Persist weights on the Railway volume if present (same pattern as the amygdala).
if os.path.isdir("/app/data"):
    os.environ.setdefault("HF_HOME", "/app/data/hf")

_CANDIDATES = [
    "cross-encoder/nli-roberta-base",   # roberta-base, BPE (no sentencepiece), small
    "roberta-large-mnli",               # reliable fallback, larger
]


class Cingulate:
    def __init__(self, threshold: float = 0.55):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.threshold = float(threshold)
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
            except Exception as e:  # format block / missing dep / network → try next
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

    def check(self, statement: str, against, threshold: float = None):
        """Score `statement` against each prior claim; return the strongest conflict.
        NLI is directional, so we take the max over both orderings."""
        t0 = time.time()
        thr = self.threshold if threshold is None else float(threshold)
        statement = (statement or "").strip()
        priors = [p for p in (against or [])]
        scores = []
        for p in priors:
            ps = (p or "").strip()
            if not ps or not statement:
                scores.append(0.0); continue
            s = max(self._contra(ps, statement), self._contra(statement, ps))
            scores.append(round(s, 4))
        best_i = max(range(len(scores)), key=lambda i: scores[i]) if scores else -1
        best = scores[best_i] if best_i >= 0 else 0.0
        return {
            "contradiction": bool(best >= thr),
            "score": round(float(best), 4),
            "threshold": thr,
            "with_index": best_i,
            "with_statement": (priors[best_i] if best_i >= 0 else None),
            "scores": scores,
            "model": self.model_id,
            "ms": int((time.time() - t0) * 1000),
        }


_cingulate = None
def get_cingulate(threshold: float = 0.55):
    global _cingulate
    if _cingulate is None:
        _cingulate = Cingulate(threshold=threshold)
    return _cingulate
