"""Let the EMULATOR choose the intervention, model-blind.

Arm B has always been b[t] = (a[t] + k) mod n_act over the intervention window, with k = 3
fixed for every game. In 14 of 26 games that particular k happens to change nothing visible, so
those games were unmeasurable -- a property of OUR choice, not of the game. This searches k over
all n_act-1 offsets per game and keeps the one with the largest median true divergence at
delay 20. No model is involved anywhere, so the choice cannot favour any architecture.
"""
import json, sys, pathlib
sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, cv2, gymnasium as gym, ale_py
gym.register_envs(ale_py)
CONTEXT, HORIZON, K, DUR, N_EP, D20 = 4, 24, 3, 8, 20, 20
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
GAMES = [p.stem for p in sorted((OUT.parent / "models").glob("*.pt"))]

_ENV = {}
def emu(game, seed, actions):
    # one env per game, reset per rollout -- identical trajectories to a fresh env (ALE reset(seed) is
    # deterministic; verified by the v1 determinism gate), ~40x faster than gym.make per rollout
    e = _ENV.get(game) or _ENV.setdefault(game, gym.make(game, frameskip=4, repeat_action_probability=0.0))
    obs, _ = e.reset(seed=seed); fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    return np.stack(fr)
def to64(fr):
    return np.stack([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1

res = {}
for g in GAMES:
    game = f"ALE/{g}-v5"; e = gym.make(game); n_act = e.action_space.n; e.close()
    best = {}
    for k in range(1, n_act):
        divs = []
        for ep in range(N_EP):
            rng = np.random.default_rng(42 * 1000 + ep)        # same base sequences as seed 42 of the sweep
            acts = rng.integers(0, n_act, CONTEXT + HORIZON); b = acts.copy()
            for t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(acts))): b[t] = (acts[t] + k) % n_act
            ta = to64(emu(game, ep, acts)[CONTEXT:CONTEXT + HORIZON]); tb = to64(emu(game, ep, b)[CONTEXT:CONTEXT + HORIZON])
            i = K + D20; divs.append(float(np.linalg.norm((tb[i] - ta[i]).ravel())))
        best[k] = float(np.median(divs))
    kbest = max(best, key=best.get)
    res[g] = {"n_act": n_act, "median_true_div_by_offset": best, "best_offset": kbest,
              "best_median_div": best[kbest], "current_offset_3_div": best.get(3, best.get(1))}
    print(f"{g:<15} n_act={n_act:<3} k=3 -> {res[g]['current_offset_3_div']:.2f} | best k={kbest} -> {best[kbest]:.2f}", flush=True)
json.dump(res, open(OUT / "emulator_offset_search.json", "w"), indent=1)
n_now = sum(1 for r in res.values() if r["current_offset_3_div"] > 0.1); n_best = sum(1 for r in res.values() if r["best_median_div"] > 0.1)
print(f"\nmeasurable games (median true div > 0.1 at d20): current offset {n_now}/26 -> emulator-chosen offset {n_best}/26")
