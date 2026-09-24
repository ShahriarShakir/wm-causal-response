"""Representation probe on IRIS: pixel-level AND representation-level causal alignment, same rollouts.

Per episode and delay it records pixel alignment (as before) and representation alignment
(ccaus.repr_probe), so each (game, delay) can be classified as tracked / decoding failure /
representation failure. Emulator-chosen (offset, warm-up) per game; 7 seeds; raw dump.
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, torch, cv2, gymnasium as gym, ale_py
from ccaus.iris_harness import load_iris, imagine, encode_repr
from ccaus.joint import measure_pair, bootstrap_median
from ccaus.repr_probe import repr_alignment, repr_magnitude
gym.register_envs(ale_py)
GAME = sys.argv[1]; N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 60; DUR = 8
OFFSET = int(sys.argv[3]) if len(sys.argv) > 3 else 3; WARMUP = int(sys.argv[4]) if len(sys.argv) > 4 else 0
DRY = "--dry" in sys.argv
CKPT = f"models_iris/{GAME.split('/')[-1].replace('-v5','')}.pt"; CONTEXT, HORIZON, K = 4, 24, 3
SEEDS = [42] if DRY else [42, 7, 128, 256, 512, 1024, 2024]; N_EP = 2 if DRY else N_EP
DEV = "cuda" if torch.cuda.is_available() else "cpu"; OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
DELAYS = [1, 2, 3, 5, 8, 12, 16, 20]
def emu(seed, actions):
    e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); obs, _ = e.reset(seed=seed)
    wr = np.random.default_rng(seed + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, e.action_space.n)))
    fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)
to64 = lambda fr: np.stack([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1
flat = lambda x: x.reshape(x.shape[0], -1)
env = gym.make(GAME); n_act = env.action_space.n; env.close()
tok, wm = load_iris(CKPT, num_actions=n_act, device=DEV)
print(f"IRIS repr probe {GAME}: k={OFFSET} w={WARMUP} dev={DEV} {N_EP} eps x {len(SEEDS)} seeds", flush=True)
res = {}
for seed in SEEDS:
    raw = []
    for ep in range(N_EP):
        rng = np.random.default_rng(seed * 1000 + ep); acts = rng.integers(0, n_act, CONTEXT + HORIZON); b = acts.copy()
        for t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(acts))):
            b[t] = (acts[t] + OFFSET) % n_act
            if b[t] == acts[t]: b[t] = (acts[t] + 1) % n_act
        ctx = emu(ep, acts[:CONTEXT - 1])[:CONTEXT]
        TA = emu(ep, acts)[CONTEXT:CONTEXT + HORIZON]; TB = emu(ep, b)[CONTEXT:CONTEXT + HORIZON]     # uint8 true frames
        ta, tb = to64(TA), to64(TB)
        pa, za = imagine(tok, wm, ctx[-1], acts[CONTEXT - 1:], HORIZON, seed=ep, device=DEV, return_repr=True)
        pb, zb = imagine(tok, wm, ctx[-1], b[CONTEXT - 1:], HORIZON, seed=ep, device=DEV, return_repr=True)
        pa, pb = pa.transpose(0, 2, 3, 1), pb.transpose(0, 2, 3, 1)
        zta, ztb = encode_repr(tok, TA, DEV), encode_repr(tok, TB, DEV)                                 # true frames -> same space
        pm = measure_pair(flat(pa), flat(pb), flat(ta), flat(tb), flat(to64(ctx)), K, DELAYS)
        rec = {"pixel_align": {}, "repr_align": {}, "repr_mag": {}, "true_div": {}, "repr_true_div": {}}
        for d in DELAYS:
            i = K + d
            rec["pixel_align"][str(d)] = pm.causal_alignment.get(d); rec["true_div"][str(d)] = pm.true_divergence.get(d)
            rec["repr_align"][str(d)] = repr_alignment(za[i], zb[i], zta[i], ztb[i])
            rec["repr_mag"][str(d)] = repr_magnitude(za[i], zb[i], zta[i], ztb[i])
            rec["repr_true_div"][str(d)] = float(np.linalg.norm(ztb[i] - zta[i]))
        raw.append(rec)
        if (ep + 1) % 20 == 0: print(f"  seed {seed}: {ep+1}/{N_EP}", flush=True)
    agg = {str(d): {"pixel_align": bootstrap_median([r["pixel_align"][str(d)] for r in raw]).as_dict(),
                    "repr_align": bootstrap_median([r["repr_align"][str(d)] for r in raw]).as_dict()} for d in DELAYS}
    res[str(seed)] = {"aggregate": agg, "raw": raw}
    print(f"seed {seed}: d20 pixel {agg['20']['pixel_align']['median']:.3f} | repr {agg['20']['repr_align']['median']:.3f}", flush=True)
name = f"irisrepr_{GAME.split('/')[-1]}" + (f"_k{OFFSET}" if OFFSET != 3 else "") + (f"_w{WARMUP}" if WARMUP else "") + ("_DRY" if DRY else "")
json.dump({"experiment": "iris_repr_probe", "game": GAME, "seeds": SEEDS, "n_episodes": N_EP, "delays": DELAYS,
           "action_offset": OFFSET, "warmup_steps": WARMUP, "results": res}, open(OUT / f"{name}.json", "w"), indent=1)
print(f"wrote outputs/{name}.json")
