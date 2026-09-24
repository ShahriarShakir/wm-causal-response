"""Emulator-only search for a MEASURABLE intervention per game, on PBS, written incrementally.

Two hypotheses about why 14/26 games show zero true divergence, both model-blind:
  H1 OFFSET  -- the fixed action offset k=3 happens to be a no-op in that game.
               First (login-node) run: for 11 of 22 games EVERY offset gave 0. Mostly refuted.
  H2 WARM-UP -- we intervene 7 steps after reset, inside an intro/idle window where NO action
               changes the screen. Test: run W random warm-up steps before the context window.
For each game: best offset at W=0, then for W in (30, 60, 120, 240) the median true divergence
at delay 20 under the best offset. JSON is written after EVERY game so a kill loses nothing.
"""
import json, sys, pathlib
sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, cv2, gymnasium as gym, ale_py
gym.register_envs(ale_py)
import os
DRY = os.environ.get("DRY") == "1"
CONTEXT, HORIZON, K, DUR, N_EP, D20 = 4, 24, 3, 8, (2 if DRY else 20), 20
WARMUPS = ((30,) if DRY else (30, 60, 120, 240))
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"; DEST = OUT / "emulator_intervention_search.json"
GAMES = [p.stem for p in sorted((OUT.parent / "models").glob("*.pt"))]
if DRY: GAMES = GAMES[:1]; DEST = OUT / "emulator_intervention_search_DRY.json"
_ENV = {}
def rollout(game, seed, warm, actions):
    e = _ENV.get(game) or _ENV.setdefault(game, gym.make(game, frameskip=4, repeat_action_probability=0.0))
    obs, _ = e.reset(seed=seed); rng = np.random.default_rng(seed + 7919)
    for _ in range(warm): obs, *_ = e.step(int(rng.integers(0, e.action_space.n)))   # same warm-up in both arms
    fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    return np.stack(fr)
to64 = lambda fr: np.stack([cv2.resize(f,(64,64),interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32)/255*2-1
def med_div(game, n_act, k, warm):
    divs = []
    for ep in range(N_EP):
        rng = np.random.default_rng(42*1000+ep); acts = rng.integers(0, n_act, CONTEXT+HORIZON); b = acts.copy()
        for t in range(CONTEXT+K, min(CONTEXT+K+DUR, len(acts))): b[t] = (acts[t]+k) % n_act
        ta = to64(rollout(game, ep, warm, acts)[CONTEXT:CONTEXT+HORIZON]); tb = to64(rollout(game, ep, warm, b)[CONTEXT:CONTEXT+HORIZON])
        divs.append(float(np.linalg.norm((tb[K+D20]-ta[K+D20]).ravel())))
    return float(np.median(divs))
res = json.load(open(DEST)) if DEST.exists() else {}
for g in GAMES:
    if g in res: continue
    game = f"ALE/{g}-v5"; e = gym.make(game); n_act = int(e.action_space.n); e.close()   # int(): np.int64 is not JSON-serialisable
    by_k = {k: med_div(game, n_act, k, 0) for k in (range(1, min(3, n_act)) if DRY else range(1, n_act))}
    kbest = max(by_k, key=by_k.get)
    by_w = {0: by_k[kbest]}
    for w in WARMUPS: by_w[w] = med_div(game, n_act, kbest, w)
    wbest = max(by_w, key=by_w.get)
    res[g] = {"n_act": n_act, "div_by_offset_w0": by_k, "best_offset": kbest, "div_by_warmup_bestk": by_w,
              "best_warmup": wbest, "best_div": by_w[wbest], "fixed_k3_w0_div": by_k.get(3, by_k[1])}
    json.dump(res, open(DEST, "w"), indent=1)
    print(f"{g:<15} k3/w0 {res[g]['fixed_k3_w0_div']:6.2f} | best k={kbest:<2} w0 {by_k[kbest]:6.2f} | "
          f"warm-ups {' '.join(f'{w}:{by_w[w]:.1f}' for w in WARMUPS)} | BEST w={wbest} -> {by_w[wbest]:.2f}", flush=True)
n0 = sum(1 for r in res.values() if r["fixed_k3_w0_div"] > 0.1); n1 = sum(1 for r in res.values() if r["best_div"] > 0.1)
print(f"\nmeasurable: fixed intervention {n0}/{len(res)} -> emulator-chosen offset+warm-up {n1}/{len(res)}")
