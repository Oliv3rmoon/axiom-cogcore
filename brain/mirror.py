"""
Mirror Neurons — perception->expression coupling (emotional attunement).

Mirror neurons fire both when an action/affect is observed and when it is expressed; they are
the substrate of emotional contagion and empathy. Here the region is a coupled dynamical state:
AXIOM's EXPRESSED affect (valence, arousal) is pulled toward the user's PERCEIVED affect each
turn (convergence with a mirroring gain), then mapped to a Phoenix emotion label that drives the
avatar's face/voice.

It is attunement, not naive contagion. Valence direction is mirrored, and positive energy is
matched, but arousal under NEGATIVE valence is damped (a steadying, co-regulating presence rather
than catching the user's panic/anger). The convergence lag is the mirror signature: a felt
expression with inertia, distinct from the amygdala's stateless per-turn read.
"""
import os


class MirrorNeurons:
    # Phoenix emotion palette in (valence, arousal) space
    PALETTE = {
        "elated":    (0.90, 0.80),
        "excited":   (0.70, 0.85),
        "content":   (0.50, 0.35),
        "neutral":   (0.00, 0.25),
        "surprised": (0.30, 0.90),
        "sad":       (-0.60, 0.30),
        "dejected":  (-0.75, 0.15),
        "scared":    (-0.50, 0.85),
        "angry":     (-0.65, 0.85),
    }

    def __init__(self, gain=0.5, distress_damp=0.5, path=None):
        self.gain = float(gain)
        self.distress_damp = float(distress_damp)
        self.v = 0.50   # expressed valence — resting state is warm/content ("never flat")
        self.a = 0.35   # expressed arousal
        self.n = 0
        self.path = path
        self._load()

    def _nearest(self, v, a):
        best, bd = None, 1e9
        for label, (pv, pa) in self.PALETTE.items():
            d = (v - pv) ** 2 + (a - pa) ** 2
            if d < bd:
                bd, best = d, label
        return best, bd ** 0.5

    def mirror(self, user_valence, user_arousal, update=True, reset=False):
        if reset:
            self.v, self.a, self.n = 0.50, 0.35, 0
        uv = min(1.0, max(-1.0, float(user_valence)))
        ua = min(1.0, max(0.0, float(user_arousal)))
        g = self.gain
        target_v = uv
        # co-regulation: mirror arousal when valence is non-negative; damp it under distress
        target_a = ua if uv >= 0 else ua * self.distress_damp
        v_new = (1 - g) * self.v + g * target_v
        a_new = (1 - g) * self.a + g * target_a
        v_new = min(1.0, max(-1.0, v_new))
        a_new = min(1.0, max(0.0, a_new))
        label, dist = self._nearest(v_new, a_new)
        if update:
            self.v, self.a, self.n = v_new, a_new, self.n + 1
            self._save()
        return {
            "emotion": label,
            "expressed_valence": round(v_new, 4),
            "expressed_arousal": round(a_new, 4),
            "target_valence": round(target_v, 4),
            "target_arousal": round(target_a, 4),
            "user_valence": round(uv, 4),
            "user_arousal": round(ua, 4),
            "gain": g,
            "distress_regulated": uv < 0,
            "dist": round(dist, 4),
            "n": self.n,
        }

    def peek(self):
        label, _ = self._nearest(self.v, self.a)
        return {"emotion": label, "expressed_valence": round(self.v, 4),
                "expressed_arousal": round(self.a, 4), "n": self.n}

    def _save(self):
        if not self.path:
            return
        try:
            import numpy as np
            np.savez(self.path, v=self.v, a=self.a, n=self.n)
        except Exception:
            pass

    def _load(self):
        if not self.path:
            return
        try:
            import numpy as np
            if os.path.exists(self.path):
                d = np.load(self.path)
                self.v, self.a, self.n = float(d["v"]), float(d["a"]), int(d["n"])
        except Exception:
            pass


_mirror = None
def get_mirror(path=None):
    global _mirror
    if _mirror is None:
        _mirror = MirrorNeurons(path=path)
    return _mirror
