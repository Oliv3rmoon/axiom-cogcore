"""
AMYGDALA — a real emotion-detection mechanism.

A dedicated neural network (RoBERTa fine-tuned on GoEmotions, 28 labels) that reads
text and returns a fine emotion + a coarse affect 'family' + valence/arousal + an
'acute' distress flag. Its output DRIVES control flow in the per-turn controller
(route / instruct / interrupt); it is not pasted into a prompt as decoration.
"""
from __future__ import annotations
import os
if os.path.isdir("/app/data"):
    os.environ.setdefault("HF_HOME", "/app/data/hf")
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_MODEL = "SamLowe/roberta-base-go_emotions"  # ships safetensors -> exempt from torch<2.6 load block

# Map each GoEmotions label to a coarse control family. The controller keys on family,
# so it never has to know all 28 labels.
_FAMILY = {
    "sadness": "down", "grief": "down", "disappointment": "down", "remorse": "down",
    "embarrassment": "down",
    "fear": "anxious", "nervousness": "anxious", "confusion": "anxious",
    "anger": "angry", "annoyance": "angry", "disgust": "angry", "disapproval": "angry",
    "admiration": "up", "amusement": "up", "approval": "up", "caring": "up",
    "desire": "up", "excitement": "up", "gratitude": "up", "joy": "up", "love": "up",
    "optimism": "up", "pride": "up", "relief": "up",
    "surprise": "surprised", "realization": "surprised", "curiosity": "surprised",
    "neutral": "neutral",
}
# valence in [-1,1], arousal in [0,1] per family
_FAMILY_VA = {
    "down": (-0.6, 0.25), "anxious": (-0.5, 0.8), "angry": (-0.6, 0.8),
    "up": (0.7, 0.6), "surprised": (0.2, 0.65), "neutral": (0.0, 0.3),
}


class Amygdala:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(_MODEL)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(_MODEL, use_safetensors=True)
            .to(device).eval()
        )
        self.labels = [self.model.config.id2label[i]
                       for i in range(self.model.config.num_labels)]
        self.ready = True

    @torch.no_grad()
    def read(self, text: str) -> dict:
        t0 = time.time()
        if not text or not text.strip():
            return {"emotion": "neutral", "family": "neutral", "intensity": 0.0,
                    "valence": 0.0, "arousal": 0.3, "top": [], "acute": False, "ms": 0.0}
        enc = self.tok(text, truncation=True, max_length=256,
                       return_tensors="pt").to(self.device)
        # GoEmotions is multi-label -> sigmoid per label
        probs = torch.sigmoid(self.model(**enc).logits)[0]
        order = torch.argsort(probs, descending=True)
        top = [(self.labels[i], round(float(probs[i]), 3)) for i in order[:3]]
        emotion, p = top[0]
        family = _FAMILY.get(emotion, "neutral")
        v, a = _FAMILY_VA[family]
        arousal = round(a * (0.5 + 0.5 * min(p, 1.0)), 3)
        acute = bool(family in ("down", "anxious", "angry") and p >= 0.45)
        return {"emotion": emotion, "family": family, "intensity": round(float(p), 3),
                "valence": v, "arousal": arousal, "top": top, "acute": acute,
                "ms": round((time.time() - t0) * 1000, 1)}


_singleton = None


def get_amygdala(device: str = "cpu") -> Amygdala:
    global _singleton
    if _singleton is None:
        _singleton = Amygdala(device=device)
    return _singleton
