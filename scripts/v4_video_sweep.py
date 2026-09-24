"""VIDEO ARM, the measurement, on a diffusion video world model.

Same design as DMControl, new substrate, and deliberately the SAME code: every number here
comes from `ccaus.joint`, the module the DMControl arm uses and the one `tests/test_joint.py`
gates. A second, hand-rolled copy of the estimator living in this script is exactly how two
arms of one paper end up not comparable, so there isn't one.

For each episode, two futures share an identical 4-frame context and identical actions except
ONE flipped action inside the horizon.

  TRUE  divergence: from the ALE emulator, replayed with each action sequence (EXACT ground
                    truth -- verified pixel-identical under replay, no VLM judge anywhere).
  MODEL divergence: DIAMOND's imagined futures under the same two action sequences, with
                    COMMON RANDOM NUMBERS so diffusion noise cannot masquerade as a response
                    (verified: null control returns exactly 0.000000).

The claim is the CONTRAST: the emulator's divergence PERSISTS (Pong screens at 85.3% of later
steps) while the model's response DECAYS TO ZERO -- measured on the same rollouts, in one pass,
so nothing but the two metrics differs between the curves.
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, torch, cv2
import gymnasium as gym, ale_py
from ccaus.diamond_harness import load_diamond, imagine, CONTEXT, IMG
from ccaus.joint import (Guard, aggregate, decompose_response, measure_pair,
                         verdict)

gym.register_envs(ale_py)
GAME = sys.argv[1] if len(sys.argv) > 1 else "ALE/Pong-v5"
import os as _os
CKPT = _os.environ.get("DIAMOND_CKPT") or f"models/{GAME.split('/')[-1].replace('-v5','')}.pt"   # DIAMOND_CKPT: evaluate a checkpoint we trained (seed-2 replicate, 2026-09-14)
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
# DURATION of the intervention. 1 = a single flipped action. A single flip leaves the ALE
# screen UNCHANGED in 15 of 26 games (median true divergence exactly 0.0000), so most of the
# suite is unmeasurable and the correlation test has no power. A SUSTAINED flip is still a
# clean intervention -- identical context, identical actions except on a known window -- and
# it makes the cause durable enough to leave a trace in the emulator.
DUR = int(sys.argv[3]) if len(sys.argv) > 3 else 1
# action offset for arm B, chosen per game by the EMULATOR (scripts/emulator_offset_search.py);
# 3 is the historical fixed value. Files for k != 3 carry a _k{OFFSET} suffix so nothing is overwritten.
OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 3
# random warm-up steps before the context window (identical in both arms), chosen per game by the
# emulator search; 0 = historical (intervene 7 steps after reset, inside some games' idle window)
WARMUP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
import os
HORIZON, K = int(os.environ.get("HORIZON", 24)), 3   # HORIZON env override for the horizon ablation
# 7 EVALUATION seeds. NOTE: DIAMOND releases ONE checkpoint per game, so TRAINING-seed
# variation is impossible without retraining 26 models; what varies here is the initial
# state and action sequence, which is what determines measurement stability.
SEEDS = [42, 7, 128, 256, 512, 1024, 2024]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = pathlib.Path(_os.environ.get("DIAMOND_OUT") or (pathlib.Path(__file__).resolve().parents[1] / "outputs"))   # DIAMOND_OUT: ext arms write elsewhere
DELAYS = [d for d in (1, 2, 3, 5, 8, 12, 16, 20, 28, 36, 44) if K + d < HORIZON]


FIELDS = ("actuated_mag", "actuated_align", "consequence_mag", "consequence_align",
          "consequence_fraction")


def agg_dec(decs, delays):
    """Median over episodes of each decomposition field, per delay."""
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
    x = np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr])
    return x.astype(np.float32) / 255.0 * 2.0 - 1.0


def flat(x):
    """[T, H, W, C] -> [T, D]. joint.measure_pair takes [T, D] and nothing else."""
    return x.reshape(x.shape[0], -1)


env = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); n_act = env.action_space.n; env.close()
sampler = load_diamond(CKPT, num_actions=n_act, device=DEV)
print(f"{GAME}: {n_act} actions, device={DEV}, {N_EP} episodes x {len(SEEDS)} seeds", flush=True)

per_seed, audit = {}, {}
for seed in SEEDS:
    pairs, decs, leak = [], [], 0.0
    for ep in range(N_EP):
        rng = np.random.default_rng(seed * 1000 + ep)
        acts = rng.integers(0, n_act, CONTEXT + HORIZON)
        b = acts.copy()
        for _t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(acts))):
            b[_t] = (acts[_t] + OFFSET) % n_act
            if b[_t] == acts[_t]: b[_t] = (acts[_t] + 1) % n_act
        ctx = emu(ep, acts[:CONTEXT - 1])[:CONTEXT]
        ta = to64(emu(ep, acts)[CONTEXT:CONTEXT + HORIZON])
        tb = to64(emu(ep, b)[CONTEXT:CONTEXT + HORIZON])
        pa = imagine(sampler, ctx, acts, HORIZON, seed=ep, device=DEV).transpose(0, 2, 3, 1)
        pb = imagine(sampler, ctx, b, HORIZON, seed=ep, device=DEV).transpose(0, 2, 3, 1)
        leak = max(leak, float(np.abs(flat(pa)[:K] - flat(pb)[:K]).max()))
        try:
            # raises if the two futures differ BEFORE the intervention -- a leaking harness,
            # not a result. Reported with the episode so it is diagnosable, not a bare trace.
            pairs.append(measure_pair(flat(pa), flat(pb), flat(ta), flat(tb),
                                      flat(to64(ctx)), K, DELAYS))
        except AssertionError as e:
            raise AssertionError(f"seed {seed} ep {ep}: {e}") from None
        decs.append(decompose_response(flat(pa), flat(pb), flat(ta), flat(tb), K, DELAYS))
        if (ep + 1) % 5 == 0: print(f"  seed {seed}: {ep+1}/{N_EP}", flush=True)

    # RAW per-episode dump: makes guard / rel_tol / delay-grid sensitivity a RE-ANALYSIS
    # instead of a GPU re-run. Small (a few numbers per episode per delay) and it is the
    # difference between "we fixed 0.25 and hoped" and a real sensitivity sweep.
    raw = [{"consistency": {str(d): p.consistency.get(d) for d in DELAYS},
            "causal_magnitude": {str(d): p.causal_magnitude.get(d) for d in DELAYS},
            "causal_alignment": {str(d): p.causal_alignment.get(d) for d in DELAYS},
            "true_divergence": {str(d): p.true_divergence.get(d) for d in DELAYS},
            "prediction_psnr": {str(d): p.prediction_psnr.get(d) for d in DELAYS},
            "model_divergence": {str(d): p.model_divergence.get(d) for d in DELAYS},
            "cross_psnr": {str(d): (p.cross_psnr.get(d) if p.cross_psnr else None) for d in DELAYS}, "intervened_psnr": {str(d): (p.intervened_psnr.get(d) if p.intervened_psnr else None) for d in DELAYS}}
           for p in pairs]
    agg = aggregate(pairs, DELAYS, Guard())
    v = verdict(agg, DELAYS)
    per_seed[seed] = {"aggregate": agg, "verdict": v, "raw": raw,
                      "decomposition": agg_dec(decs, DELAYS)}
    audit[seed] = {"max_pre_intervention_diff": leak}
    g = agg["guard"]
    print(f"  guard floor {g['floor']:.3f} (0.25 x median true divergence); "
          f"kept {g['n_kept']}, dropped {g['n_dropped']}", flush=True)
    print(f"  max pre-intervention difference: {leak:.3g} (must be 0 -- causality has a direction)")
    print(f"seed {seed} -> {v['verdict']}", flush=True)

print(f"\n{'delay':>6} | " + " ".join(f"{'seed '+str(s):>34}" for s in SEEDS) + "   true div")
print(f"{'':>6} | " + " ".join(f"{'consistency      response':>34}" for _ in SEEDS))
for d in DELAYS:
    cells = []
    for s in SEEDS:
        r = per_seed[s]["aggregate"]["rows"][str(d)]
        c, m = r["consistency"], r["causal_magnitude"]
        cells.append(f"{c['median']:.4f}   {m['median']:.4f}[{m['lo']:.3f},{m['hi']:.3f}]n{m['n']}")
    td = np.mean([per_seed[s]["aggregate"]["rows"][str(d)]["true_divergence"]["median"]
                  for s in SEEDS])
    print(f"{d:>6} | " + " ".join(f"{c:>34}" for c in cells) + f"   {td:8.3f}")

print(f"\n{'delay':>6} | {'ACTUATED align':>15} {'CONSEQUENCE align':>18} {'conseq share':>13}")
for d in DELAYS:
    dd = [per_seed[s]["decomposition"][str(d)] for s in SEEDS]
    print(f"{d:>6} | {np.nanmean([x['actuated_align'] for x in dd]):>15.4f}"
          f" {np.nanmean([x['consequence_align'] for x in dd]):>18.4f}"
          f" {np.nanmean([x['consequence_fraction'] for x in dd]):>13.4f}")

print("\nVERDICT: " + ", ".join(f"seed {s}: {per_seed[s]['verdict']['verdict']}" for s in SEEDS))

OUT.mkdir(exist_ok=True)
json.dump({"experiment": "video_diamond", "game": GAME, "n_episodes": N_EP,
           "seeds": SEEDS, "horizon": HORIZON, "intervention_step": K, "delays": DELAYS,
           "intervention_duration": DUR, "action_offset": OFFSET, "warmup_steps": WARMUP,
           "audit": audit, "results": {str(s): per_seed[s] for s in SEEDS}},
          open(OUT / (f"video_{GAME.split(chr(47))[-1]}"
                       + (f"_dur{DUR}" if DUR > 1 else "") + (f"_k{OFFSET}" if OFFSET != 3 else "") + (f"_w{WARMUP}" if WARMUP else "") + (f"_h{HORIZON}" if HORIZON != 24 else "") + ".json"), "w"), indent=1)
print(f"wrote outputs/video_{GAME.split('/')[-1]}.json")
