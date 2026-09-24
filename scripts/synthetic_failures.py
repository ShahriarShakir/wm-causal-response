"""Which score detects which failure? Synthetic predictors built from the emulator's own futures
(no model), scored with every metric in the paper. A score 'flags' a failure when its value moves away from the
value it gives the perfect predictor. Emulator-only; chosen interventions; CPU.
Usage: synthetic_failures.py [--dry]"""
import json, pathlib, sys, os
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, cv2, gymnasium as gym, ale_py
from ccaus.joint import measure_pair, prediction_psnr, bootstrap_median
DRY = "--dry" in sys.argv
SEARCH = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json"))
GAMES = ["Pong"] if DRY else ["Alien", "Pong", "BankHeist", "Boxing", "Breakout", "Freeway", "Gopher", "Seaquest"]
CONTEXT, HORIZON, K, DUR, IMG = 4, 24, 3, 8, 64; DELAYS = [1, 2, 3, 5, 8, 12, 16, 20]; N_EP = 2 if DRY else 30
def emu(game, warm, seed, actions):
    e = gym.make(f"ALE/{game}-v5", frameskip=4, repeat_action_probability=0.0); obs, _ = e.reset(seed=seed)
    _wr = np.random.default_rng(seed + 7919)
    for _ in range(warm): obs, *_ = e.step(int(_wr.integers(0, e.action_space.n)))
    fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    n = e.action_space.n; e.close(); return np.stack(fr), n
def to64(fr): return (np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2.0 - 1.0)
flat = lambda x: x.reshape(x.shape[0], -1)
def noise_for_psnr(shape, psnr, rng):     # additive noise giving exactly this PSNR (data range 2)
    rmse = 2.0 / 10 ** (psnr / 20); n = rng.standard_normal(shape).astype(np.float32); return n / np.sqrt((n ** 2).mean()) * rmse
def predictors(ta, tb, tc, ctx, rng):
    """Each returns (p, p'): both arms' predicted futures [H, D]."""
    dt = tb - ta; n30 = noise_for_psnr(ta.shape, 30.0, rng); N = lambda x: np.clip(x + n30, -1, 1)   # every realistic predictor sits at 30 dB, like the models
    delayed = ta.copy(); delayed[K + 2:] = tb[K + 2:] * 0 + (ta[K + 2:] + dt[K:-2])          # the true change, arriving two steps late
    other = np.roll(dt, 1, axis=0); other = other * (np.linalg.norm(dt, axis=1, keepdims=True) / (np.linalg.norm(other, axis=1, keepdims=True) + 1e-8))
    orth = other - (np.sum(other * dt, 1, keepdims=True) / (np.sum(dt * dt, 1, keepdims=True) + 1e-8)) * dt   # a change of the same size, orthogonal to the true one
    return {
        "perfect (noiseless)": (ta, tb),
        "correct":           (N(ta), N(tb)),                           # exact response, common 30 dB noise in both arms
        "action-ignoring":   (N(ta), N(ta)),                           # the same future whatever the action
        "wrong action":      (N(ta), N(tc)),                           # responds, but as if to a different action
        "delayed by 2":      (N(ta), N(delayed)),
        "reversed":          (N(ta), N(ta - dt)),
        "orthogonal (world-sized)": (N(ta), N(ta + orth)),             # a change as large as the truth, unrelated to it
        "correct + irrelevant motion": (N(ta), N(tb + orth)),
        "half-size correct": (N(ta), N(ta + 0.5 * dt)),
    }
res = {}
for g in GAMES:
    k, w = int(SEARCH[g]["best_offset"]), int(SEARCH[g]["best_warmup"]); per = {}
    for ep in range(N_EP):
        rng = np.random.default_rng(ep); _, n_act = emu(g, w, ep, [])
        acts = rng.integers(0, n_act, CONTEXT + HORIZON); b = acts.copy(); c = acts.copy()
        for t in range(CONTEXT + K, CONTEXT + K + DUR): b[t] = (acts[t] + k) % n_act
        ctx = to64(emu(g, w, ep, acts[:CONTEXT - 1])[0][:CONTEXT]); ta = to64(emu(g, w, ep, acts)[0][CONTEXT:CONTEXT + HORIZON]); tb = to64(emu(g, w, ep, b)[0][CONTEXT:CONTEXT + HORIZON])
        fa, fb, fx = flat(ta), flat(tb), flat(ctx); i20 = K + 20
        if np.linalg.norm(fb[i20] - fa[i20]) < 1e-6: continue                                # no true response at d20: uninformative episode
        fc = None                                                                           # a third arm whose future differs from BOTH a and b (the 'wrong action')
        for k2 in range(1, n_act):
            if k2 == k: continue
            c = acts.copy()
            for t in range(CONTEXT + K, CONTEXT + K + DUR): c[t] = (acts[t] + k2) % n_act
            tc = flat(to64(emu(g, w, ep, c)[0][CONTEXT:CONTEXT + HORIZON]))
            if np.linalg.norm(tc[i20] - fb[i20]) > 1e-6 and np.linalg.norm(tc[i20] - fa[i20]) > 1e-6: fc = tc; break
        if fc is None: continue
        for name, (pa, pb) in predictors(fa, fb, fc, fx, rng).items():
            pm = measure_pair(pa, pb, fa, fb, fx, K, DELAYS, check_causality_direction=False)
            for d in (8, 20):
                A, R = pm.causal_alignment.get(d), pm.causal_magnitude.get(d)
                per.setdefault(name, {}).setdefault(str(d), []).append(dict(A=A, R=R, err=(R * R + 1 - 2 * A) if (A is not None and R is not None) else None,
                    psnr=pm.prediction_psnr.get(d), consistency=pm.consistency.get(d), dpsnr=(pm.prediction_psnr.get(d) - pm.cross_psnr.get(d)) if pm.cross_psnr else None))
    res[g] = {name: {d: {q: bootstrap_median([r[q] for r in rows if r.get(q) is not None and r[q] == r[q]]).as_dict() for q in ("A", "R", "err", "psnr", "consistency", "dpsnr")} for d, rows in dd.items()} for name, dd in per.items()}
    print(g, "episodes used:", len(next(iter(per.values()))["20"]) if per else 0, flush=True)
    for name in res[g]:
        r = res[g][name]["20"]; print(f"  {name:<28} d20  A {r['A']['median']:+.2f}  R {r['R']['median']:.2f}  err {r['err']['median']:.2f} | PSNR {r['psnr']['median']:.1f}  cons {r['consistency']['median']:.3f}  dPSNR {r['dpsnr']['median']:+.2f}", flush=True)
name = "synthetic_failures" + ("_DRY" if DRY else "")
json.dump({"experiment": "synthetic_failures", "games": GAMES, "n_ep": N_EP, "delays": [8, 20], "results": res}, open(ROOT / "outputs" / f"{name}.json", "w"), indent=1)
print(f"wrote outputs/{name}.json")
