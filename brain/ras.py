"""
RAS — Reticular Activating System: arousal-gated attention over perception channels.

The RAS is the brain's salience filter: of the ~11M bits/sec of afferent sense data, only
a tiny attended subset reaches awareness, and arousal narrows that aperture. Here it scores
each perception channel's salience by relevance to the current context and admits the top-K,
where K shrinks as arousal rises (tunnel vision under arousal). A real gate (embedding
relevance + an arousal-modulated capacity), not a prompt. The admitted channels pass into the
context; the rest are suppressed. Context + all channels are embedded in ONE batched call so a
full ~33-channel sweep stays well under a second.
"""


class Ras:
    def __init__(self, embed_fn, embed_batch_fn=None, base_k=10, min_k=3):
        self.embed_fn = embed_fn
        self.embed_batch_fn = embed_batch_fn
        self.base_k = int(base_k)
        self.min_k = int(min_k)

    def _embed_many(self, texts):
        import numpy as np
        if self.embed_batch_fn is not None:
            return np.asarray(self.embed_batch_fn(texts), dtype=float)
        return np.asarray([self.embed_fn(t) for t in texts], dtype=float)

    @staticmethod
    def _cos(a, b):
        import numpy as np
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

    def gate(self, context, channels, arousal=0.5, base_k=None):
        import time
        import numpy as np
        t0 = time.time()
        bk = int(base_k) if base_k else self.base_k
        chans = [c for c in (channels or []) if isinstance(c, dict) and c.get("name")]
        ctx = (context or "").strip()
        texts = [ctx] + [(c.get("text") or "").strip() for c in chans]
        vecs = self._embed_many(texts)  # (1 + n, dim), one batched forward pass
        cv = vecs[0] if len(vecs) else None
        scored = []
        for i, c in enumerate(chans):
            txt = (c.get("text") or "").strip()
            s = self._cos(cv, vecs[i + 1]) if (txt and cv is not None) else 0.0
            scored.append({"name": c["name"], "text": c.get("text", ""), "score": round(s, 4)})
        a = min(1.0, max(0.0, float(arousal)))
        # arousal narrows the aperture: K runs from base_k (calm) down to base_k/2 (peak arousal)
        eff_k = max(self.min_k, min(bk, round(bk * (1.0 - 0.5 * a))))
        order = sorted(range(len(scored)), key=lambda i: scored[i]["score"], reverse=True)
        selected = [scored[i] for i in order[:eff_k]]
        suppressed = [scored[i]["name"] for i in order[eff_k:]]
        return {
            "selected": selected,
            "suppressed": suppressed,
            "effective_k": eff_k,
            "n_channels": len(scored),
            "arousal": round(a, 4),
            "ms": int((time.time() - t0) * 1000),
        }


_ras = None
def get_ras(embed_fn, embed_batch_fn=None, base_k=10, min_k=3):
    global _ras
    if _ras is None:
        _ras = Ras(embed_fn, embed_batch_fn=embed_batch_fn, base_k=base_k, min_k=min_k)
    return _ras
