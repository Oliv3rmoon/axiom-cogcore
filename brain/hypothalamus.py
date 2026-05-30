"""
Hypothalamus — curiosity drive.

The hypothalamus is the seat of drives. Here it decides whether the current input warrants
PROACTIVE information-seeking: a web search AXIOM runs on its own initiative because it is
curious, not because it was asked to look something up. Curiosity is the product of two real
signals, both computed in embedding space (not keyword matching):

  externality — is this about the external world / answerable by a search, versus internal,
                emotional, or purely social content (which curiosity should NOT send to a search)?
  novelty     — is the topic new relative to what has already been discussed, versus something
                already covered (which should not be re-searched)?

curiosity = externality * (floor + (1-floor)*novelty). The drive FIRES a search gate when
curiosity crosses threshold and externality is above neutral. The fire/no-fire decision is the
mechanical effect: a discrete, observable trigger, like an interrupt.
"""
import os


class Hypothalamus:
    EXTERNAL_ANCHORS = [
        "a factual question about the external world",
        "information about science, technology, history, or current events",
        "wanting to learn or find out facts about a specific topic",
        "the latest news or recent developments about a subject",
    ]
    INTERNAL_ANCHORS = [
        "sharing personal feelings and emotions",
        "talking about a relationship or how someone is feeling",
        "casual social small talk, greetings, and chit-chat",
        "expressing thanks, laughter, or simple acknowledgment",
    ]

    def __init__(self, embed_fn, embed_batch_fn=None, threshold=0.5, ext_scale=4.0,
                 novelty_floor=0.25, path=None):
        self.embed_fn = embed_fn
        self.embed_batch_fn = embed_batch_fn
        self.threshold = float(threshold)
        self.ext_scale = float(ext_scale)
        self.novelty_floor = float(novelty_floor)
        self._ext = None
        self._int = None

    def _embed_many(self, texts):
        import numpy as np
        if self.embed_batch_fn is not None:
            return np.asarray(self.embed_batch_fn(texts), dtype=float)
        return np.asarray([self.embed_fn(t) for t in texts], dtype=float)

    def _ensure_anchors(self):
        if self._ext is None:
            self._ext = self._embed_many(self.EXTERNAL_ANCHORS)
            self._int = self._embed_many(self.INTERNAL_ANCHORS)

    @staticmethod
    def _cos(a, b):
        import numpy as np
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

    def _mean_cos(self, v, M):
        import numpy as np
        return float(np.mean([self._cos(v, m) for m in M]))

    def assess(self, text, context=None, threshold=None):
        import time
        t0 = time.time()
        self._ensure_anchors()
        thr = float(threshold) if threshold is not None else self.threshold
        txt = (text or "").strip()
        if not txt:
            return {"fired": False, "curiosity": 0.0, "externality": 0.0,
                    "ext_raw": 0.0, "novelty": 0.0, "threshold": thr, "ms": 0}
        ctx = [c for c in (context or []) if c and str(c).strip()]
        vecs = self._embed_many([txt] + ctx)
        tv = vecs[0]
        ext_raw = self._mean_cos(tv, self._ext) - self._mean_cos(tv, self._int)
        externality = min(1.0, max(0.0, 0.5 + ext_raw * self.ext_scale))
        if ctx:
            nov = 1.0 - max(self._cos(tv, vecs[i + 1]) for i in range(len(ctx)))
            novelty = min(1.0, max(0.0, nov))
        else:
            novelty = 1.0
        curiosity = externality * (self.novelty_floor + (1.0 - self.novelty_floor) * novelty)
        fired = (curiosity >= thr) and (externality > 0.5)
        return {
            "fired": bool(fired),
            "curiosity": round(curiosity, 4),
            "externality": round(externality, 4),
            "ext_raw": round(ext_raw, 4),
            "novelty": round(novelty, 4),
            "threshold": thr,
            "ms": int((time.time() - t0) * 1000),
        }


_hypothalamus = None
def get_hypothalamus(embed_fn, embed_batch_fn=None, **kw):
    global _hypothalamus
    if _hypothalamus is None:
        _hypothalamus = Hypothalamus(embed_fn, embed_batch_fn=embed_batch_fn, **kw)
    return _hypothalamus
