"""
AMYGDALA — a real emotion-detection mechanism.

This is deliberately NOT a prompt to the conversational LLM. It is a dedicated
neural network (a fine-tuned DistilRoBERTa sequence classifier) that reads text
and returns an emotion distribution + affect dimensions. Its output DRIVES
control flow in the per-turn controller (route / param / instruct / interrupt),
it is not pasted into a system prompt as decoration.
"""
from __future__ import annotations
import os
# Persist the HF model cache on the Railway volume so we don't re-download each deploy.
if os.path.isdir("/app/data"):
    os.environ.setdefault("HF_HOME", "/app/data/hf")
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Russell circumplex priors per label: valence in [-1,1], arousal in [0,1].
_VA = {
    "anger":    (-0.6, 0.85),
    "disgust":  (-0.55, 0.55),
    "fear":     (-0.6, 0.85),
    "joy":      (0.8, 0.7),
    "neutral":  (0.0, 0.3),
    "sadness":  (-0.7, 0.25),
    "surprise": (0.3, 0.7),
}


class Amygdala:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(_MODEL)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(_MODEL)
            .to(device)
            .eval()
        )
        self.labels = [
            self.model.config.id2label[i]
            for i in range(self.model.config.num_labels)
        ]
        self.ready = True

    @torch.no_grad()
    def read(self, text: str) -> dict:
        t0 = time.time()
        if not text or not text.strip():
            return {"emotion": "neutral", "intensity": 0.0, "valence": 0.0,
                    "arousal": 0.3, "scores": {}, "acute": False, "ms": 0.0}
        enc = self.tok(text, truncation=True, max_length=256,
                       return_tensors="pt").to(self.device)
        probs = torch.softmax(self.model(**enc).logits, dim=-1)[0]
        scores = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        top = max(scores, key=scores.get)
        p = scores[top]
        v, a = _VA.get(top, (0.0, 0.3))
        arousal = round(a * (0.5 + 0.5 * p), 3)
        acute = bool((top in ("fear", "anger") and p > 0.6)
                     or (top == "sadness" and p > 0.7))
        return {"emotion": top, "intensity": round(float(p), 3), "valence": v,
                "arousal": arousal,
                "scores": {k: round(val, 3) for k, val in scores.items()},
                "acute": acute, "ms": round((time.time() - t0) * 1000, 1)}


_singleton = None


def get_amygdala(device: str = "cpu") -> Amygdala:
    global _singleton
    if _singleton is None:
        _singleton = Amygdala(device=device)
    return _singleton
