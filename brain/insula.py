"""
Insula — interoception.

The insula integrates the body's internal afferents into a felt state and bridges
arousal to behavior. Here it is a homeostatic integrator (NOT a prompt): each turn it
folds the amygdala's instantaneous arousal/valence plus internal drive pressure into a
slow-moving felt arousal A (momentum + decay toward a neutral baseline), then maps A
onto the REAL sampling temperature via a monotone law. High felt arousal -> hotter,
more variable, expressive sampling; low arousal -> cooler, more measured output.

The momentum is the point: a single spike does not snap the body to maximum arousal;
arousal builds and then decays. That temporal inertia is what makes this interoception
rather than a restatement of the amygdala's per-turn read.
"""
import os


class Insula:
    def __init__(self, base_temp=0.7, k=2.5, a_ref=0.40, tmin=0.45, tmax=1.05,
                 momentum=0.5, decay=0.10, path=None):
        self.base_temp = float(base_temp)
        self.k = float(k)
        self.a_ref = float(a_ref)  # felt-arousal operating midpoint (calibrated to amygdala VA range)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.momentum = float(momentum)   # how much prior felt arousal carries over
        self.decay = float(decay)          # pull back toward neutral baseline each step
        self.A = 0.5                       # felt arousal state (0..1), neutral start
        self.V = 0.0                       # felt valence state (-1..1)
        self.n = 0
        self.path = path
        self._load()

    def _temp(self, A):
        # centered on the felt-arousal operating point; steep enough to span a usable temp range
        t = self.base_temp + self.k * (A - self.a_ref)
        return min(self.tmax, max(self.tmin, t))

    def sense(self, arousal=0.5, valence=0.0, drive=0.0, update=True, reset=False):
        """Integrate one set of interoceptive afferents and return the felt state + temperature.
        arousal: instantaneous arousal (0..1) from amygdala family VA.
        valence: instantaneous valence (-1..1).
        drive:   extra internal pressure, e.g. psyche fear/desire intensity (0..1)."""
        if reset:
            self.A, self.V, self.n = 0.5, 0.0, 0
        a = min(1.0, max(0.0, float(arousal)))
        d = min(1.0, max(0.0, float(drive)))
        v = min(1.0, max(-1.0, float(valence)))
        afferent = min(1.0, max(0.0, 0.7 * a + 0.3 * d))
        # homeostatic integration: momentum * prior + (1-momentum) * afferent, then decay -> neutral
        A_new = self.momentum * self.A + (1.0 - self.momentum) * afferent
        A_new = A_new + self.decay * (0.5 - A_new)
        A_new = min(1.0, max(0.0, A_new))
        V_new = self.momentum * self.V + (1.0 - self.momentum) * v
        V_new = min(1.0, max(-1.0, V_new))
        if update:
            self.A, self.V, self.n = A_new, V_new, self.n + 1
            self._save()
        return {
            "arousal": round(A_new, 4),          # felt (integrated) arousal -> drives temperature
            "valence": round(V_new, 4),
            "afferent": round(afferent, 4),      # this turn's instantaneous interoceptive input
            "temperature": round(self._temp(A_new), 4),
            "temperature_instant": round(self._temp(afferent), 4),  # momentum-free reference
            "base_temp": self.base_temp, "n": self.n,
        }

    def peek(self):
        return {"arousal": round(self.A, 4), "valence": round(self.V, 4),
                "temperature": round(self._temp(self.A), 4), "n": self.n,
                "base_temp": self.base_temp, "tmin": self.tmin, "tmax": self.tmax}

    def _save(self):
        if not self.path:
            return
        try:
            import numpy as np
            np.savez(self.path, A=self.A, V=self.V, n=self.n)
        except Exception:
            pass

    def _load(self):
        if not self.path:
            return
        try:
            import numpy as np
            if os.path.exists(self.path):
                d = np.load(self.path)
                self.A, self.V, self.n = float(d["A"]), float(d["V"]), int(d["n"])
        except Exception:
            pass


_insula = None
def get_insula(path=None):
    global _insula
    if _insula is None:
        _insula = Insula(path=path)
    return _insula
