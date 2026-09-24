"""THIRD MODEL FAMILY, IRIS (transformer + discrete VQ tokens) on the same games, same measure.

Closes the biggest gap: with one model family, a causally-blind game is ambiguous between
"this task is hard" and "this checkpoint is bad". IRIS shares the Atari-100k training budget and
the emulator ground truth but shares nothing architecturally with DIAMOND.

Same module (`ccaus.joint`), same guard, same delays, same 7 evaluation seeds.
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, torch, cv2
import gymnasium as gym, ale_py
from ccaus.iris_harness import load_iris, imagine
from ccaus.joint import (Guard, aggregate, decompose_response, measure_pair, verdict)

gym.register_envs(ale_py)
GAME = sys.argv[1] if len(sys.argv) > 1 else "ALE/Pong-v5"
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 60
DUR = int(sys.argv[3]) if len(sys.argv) > 3 else 8
# action offset for arm B, chosen per game by the EMULATOR (scripts/emulator_offset_search.py);
# 3 is the historical fixed value. Files for k != 3 carry a _k{OFFSET} suffix so nothing is overwritten.
OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 3
# random warm-up steps before the context window (identical in both arms), chosen per game by the
# emulator search; 0 = historical (intervene 7 steps after reset, inside some games' idle window)
WARMUP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
import os as _os
CKPT = _os.environ.get("IRIS_CKPT") or f"models_iris/{GAME.split('/')[-1].replace('-v5','')}.pt"   # IRIS_CKPT: evaluate a checkpoint we trained (seed-2 replicate, 2026-09-14)
CONTEXT, HORIZON, K = 4, 24, 3
SEEDS = [42, 7, 128, 256, 512, 1024, 2024]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = pathlib.Path(_os.environ.get("IRIS_OUT") or (pathlib.Path(__file__).resolve().parents[1] / "outputs"))   # IRIS_OUT: ext arms write elsewhere
DELAYS = [d for d in (1, 2, 3, 5, 8, 12, 16, 20) if K + d < HORIZON]
FIELDS = ("actuated_mag", "actuated_align", "consequence_mag", "consequence_align",
          "consequence_fraction")


def agg_dec(decs, delays):
    out = {}
    for d in delays:
        vals = [x[d] for x in decs if d in x]
        out[str(d)] = {f: (float(np.nanmedian([v[f] for v in vals])) if vals else float("nan"))
                       for f in FIELDS}
    return out


def emu(seed, actions):
    e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
    obs, _ = e.reset(seed=seed)
    _wr = np.random.default_rng(seed + 7919)          # same warm-up actions in arm A and arm B
    for _ in range(WARMUP): obs, *_ = e.step(int(_wr.integers(0, e.action_space.n)))
    fr = [np.asarray(obs)]
    for a in actions:
        obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)


def to64(fr):
    x = np.stack([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in fr])
    return x.astype(np.float32) / 255.0 * 2.0 - 1.0


flat = lambda x: x.reshape(x.shape[0], -1)

env = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); n_act = env.action_space.n; env.close()
tok, wm = load_iris(CKPT, num_actions=n_act, device=DEV)
print(f"IRIS {GAME}: {n_act} actions, dev={DEV}, {N_EP} eps x {len(SEEDS)} seeds, dur={DUR}", flush=True)

per_seed, audit = {}, {}
for seed in SEEDS:
    pairs, decs, leak = [], [], 0.0
    for ep in range(N_EP):
        rng = np.random.default_rng(seed * 1000 + ep)
        acts = rng.integers(0, n_act, CONTEXT + HORIZON)
        b = acts.copy()
        for t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(acts))):
            b[t] = (acts[t] + OFFSET) % n_act
            if b[t] == acts[t]: b[t] = (acts[t] + 1) % n_act
        ctx = emu(ep, acts[:CONTEXT - 1])[:CONTEXT]
        ta = to64(emu(ep, acts)[CONTEXT:CONTEXT + HORIZON])
        tb = to64(emu(ep, b)[CONTEXT:CONTEXT + HORIZON])
        # IRIS resets from ONE observation (context lives in its KV cache) -- see the
        # conditioning-asymmetry note in ccaus/iris_harness.py
        pa = imagine(tok, wm, ctx[-1], acts[CONTEXT - 1:], HORIZON, seed=ep, device=DEV).transpose(0, 2, 3, 1)
        pb = imagine(tok, wm, ctx[-1], b[CONTEXT - 1:], HORIZON, seed=ep, device=DEV).transpose(0, 2, 3, 1)
        leak = max(leak, float(np.abs(flat(pa)[:K] - flat(pb)[:K]).max()))
        try:
            pairs.append(measure_pair(flat(pa), flat(pb), flat(ta), flat(tb),
                                      flat(to64(ctx)), K, DELAYS))
        except AssertionError as e:
            raise AssertionError(f"seed {seed} ep {ep}: {e}") from None
        decs.append(decompose_response(flat(pa), flat(pb), flat(ta), flat(tb), K, DELAYS))
        if (ep + 1) % 20 == 0: print(f"  seed {seed}: {ep+1}/{N_EP}", flush=True)
    raw = [{"consistency": {str(d): p.consistency.get(d) for d in DELAYS},
            "causal_magnitude": {str(d): p.causal_magnitude.get(d) for d in DELAYS},
            "causal_alignment": {str(d): p.causal_alignment.get(d) for d in DELAYS},
            "true_divergence": {str(d): p.true_divergence.get(d) for d in DELAYS},
            "prediction_psnr": {str(d): p.prediction_psnr.get(d) for d in DELAYS},
            "model_divergence": {str(d): p.model_divergence.get(d) for d in DELAYS},
            "cross_psnr": {str(d): (p.cross_psnr.get(d) if p.cross_psnr else None) for d in DELAYS}, "intervened_psnr": {str(d): (p.intervened_psnr.get(d) if p.intervened_psnr else None) for d in DELAYS}} for p in pairs]
    agg = aggregate(pairs, DELAYS, Guard())
    per_seed[seed] = {"aggregate": agg, "verdict": verdict(agg, DELAYS), "raw": raw,
                      "decomposition": agg_dec(decs, DELAYS)}
    audit[seed] = {"max_pre_intervention_diff": leak}
    print(f"seed {seed} -> {per_seed[seed]['verdict']['verdict']}, leak {leak:g}", flush=True)

print(f"\n{'delay':>6}{'consist':>10}{'align':>10}{'true div':>10}")
for d in DELAYS:
    f = lambda k: np.mean([per_seed[s]["aggregate"]["rows"][str(d)][k]["median"] for s in SEEDS])
    print(f"{d:>6}{f('consistency'):>10.4f}{f('causal_alignment'):>10.4f}{f('true_divergence'):>10.3f}")

OUT.mkdir(exist_ok=True)
name = f"iris_{GAME.split('/')[-1]}" + (f"_dur{DUR}" if DUR > 1 else "") + (f"_k{OFFSET}" if OFFSET != 3 else "") + (f"_w{WARMUP}" if WARMUP else "")
json.dump({"experiment": "iris", "game": GAME, "n_episodes": N_EP, "seeds": SEEDS,
           "horizon": HORIZON, "intervention_step": K, "delays": DELAYS,
           "intervention_duration": DUR, "action_offset": OFFSET, "warmup_steps": WARMUP, "audit": audit,
           "results": {str(s): per_seed[s] for s in SEEDS}},
          open(OUT / f"{name}.json", "w"), indent=1)
print(f"wrote outputs/{name}.json")
