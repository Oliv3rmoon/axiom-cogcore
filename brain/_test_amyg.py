import time
from brain.amygdala import get_amygdala
print("loading amygdala net...", flush=True)
t0 = time.time()
a = get_amygdala()
print(f"loaded in {time.time()-t0:.1f}s | labels={a.labels}\n", flush=True)
tests = [
    "hey, how's it going?",
    "I'm exhausted, today was rough.",
    "do you ever feel like you're not real?",
    "I'm so excited about this, it's finally working!",
    "I can't do this anymore, everything is falling apart.",
    "remind me what we decided about the dream engine",
    "stop lying to me, I know what you did",
]
a.read("warmup")
for t in tests:
    r = a.read(t)
    print(f'{r["emotion"]:>9} p={r["intensity"]:.2f} v={r["valence"]:+.2f} '
          f'arousal={r["arousal"]:.2f} acute={str(r["acute"]):>5} {r["ms"]:.0f}ms  <- {t!r}', flush=True)
print("\nDONE", flush=True)
