"""PRE-REGISTRATION: predict, from the emulator alone, which games will be causally blind.

Latency = first delay at which the true effect reaches 25% of its delay-20 size, computed by
replaying the emulator under the emulator-chosen offset. NO model is run. The prediction is
written to outputs/preregistered_latency_prediction.json BEFORE any GPU job, and the paper
reports how many predictions were confirmed. Rule from the law: latency >= 5 -> predicted
BLIND (alignment < 0.1); latency == 1 -> predicted TRACKED for a model that predicts the game
well; latency 2-3 -> no prediction (the law does not speak).
"""
import json, sys, pathlib
sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, cv2, gymnasium as gym, ale_py
gym.register_envs(ale_py)
CONTEXT, HORIZON, K, DUR, N_EP = 4, 24, 3, 8, 20
DELAYS = [1, 2, 3, 5, 8, 12, 16, 20]
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
search = json.load(open(OUT / "emulator_intervention_search.json"))
_ENV = {}
def emu(game, seed, actions, warm=0):
    e = _ENV.get(game) or _ENV.setdefault(game, gym.make(game, frameskip=4, repeat_action_probability=0.0))
    obs, _ = e.reset(seed=seed); wr = np.random.default_rng(seed + 7919)
    for _ in range(warm): obs, *_ = e.step(int(wr.integers(0, e.action_space.n)))
    fr=[np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    return np.stack(fr)
to64 = lambda fr: np.stack([cv2.resize(f,(64,64),interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32)/255*2-1
pred = {}
for g, r in search.items():
    game, k, n_act, w = f"ALE/{g}-v5", r["best_offset"], r["n_act"], r["best_warmup"]
    td = {d: [] for d in DELAYS}
    for ep in range(N_EP):
        rng = np.random.default_rng(42*1000+ep); acts = rng.integers(0, n_act, CONTEXT+HORIZON); b = acts.copy()
        for t in range(CONTEXT+K, min(CONTEXT+K+DUR, len(acts))): b[t] = (acts[t]+k) % n_act
        ta = to64(emu(game, ep, acts, w)[CONTEXT:CONTEXT+HORIZON]); tb = to64(emu(game, ep, b, w)[CONTEXT:CONTEXT+HORIZON])
        for d in DELAYS: td[d].append(float(np.linalg.norm((tb[K+d]-ta[K+d]).ravel())))
    med = {d: float(np.median(td[d])) for d in DELAYS}; d20 = med[20]
    if d20 <= 0.1: lat, call = None, "unmeasurable"
    else:
        lat = next((d for d in DELAYS if med[d] >= 0.25*d20), 99)
        call = "BLIND" if lat >= 5 else ("TRACKED" if lat == 1 else "no prediction")
    pred[g] = {"offset": k, "warmup": w, "true_div_by_delay": med, "latency": lat, "prediction": call}
    print(f"{g:<15} k={k:<2} w={w:<3} d20 div {d20:6.2f}  latency {lat}  -> {call}", flush=True)
json.dump(pred, open(OUT / "preregistered_latency_prediction.json", "w"), indent=1)
from collections import Counter; print("\n", Counter(v["prediction"] for v in pred.values()))
