"""IRIS hidden-state (pre-sampling) representation probe on the same paired rollouts."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, torch, cv2, gymnasium as gym, ale_py
from ccaus.iris_harness import load_iris
from ccaus.iris_hidden import imagine_hidden, true_hidden
from ccaus.joint import measure_pair, bootstrap_median
from ccaus.repr_probe import repr_alignment
gym.register_envs(ale_py)
GAME, OFFSET, WARMUP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); DRY = "--dry" in sys.argv
N_EP = 2 if DRY else 60; SEEDS = [42] if DRY else [42, 7, 128, 256, 512, 1024, 2024]
CONTEXT, HORIZON, K, DUR = 4, 24, 3, 8; DELAYS = [1, 2, 3, 5, 8, 12, 16, 20]; DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"; G = GAME.split("/")[-1].replace("-v5", "")
def emu(seed, actions):
    e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); obs, _ = e.reset(seed=seed); wr = np.random.default_rng(seed + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, e.action_space.n)))
    fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)
to64 = lambda fr: np.stack([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1
flat = lambda x: x.reshape(x.shape[0], -1)
env = gym.make(GAME); n_act = int(env.action_space.n); env.close(); tok, wm = load_iris(f"models_iris/{G}.pt", num_actions=n_act, device=DEV)
res = {}
for seed in SEEDS:
    raw = []
    for ep in range(N_EP):
        r = np.random.default_rng(seed * 1000 + ep); a = r.integers(0, n_act, CONTEXT + HORIZON); b = a.copy()
        for t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(a))):
            b[t] = (a[t] + OFFSET) % n_act
            if b[t] == a[t]: b[t] = (a[t] + 1) % n_act
        ctx = emu(ep, a[:CONTEXT - 1])[:CONTEXT]; TA = emu(ep, a)[CONTEXT - 1:CONTEXT + HORIZON]; TB = emu(ep, b)[CONTEXT - 1:CONTEXT + HORIZON]
        ta, tb = to64(TA[1:]), to64(TB[1:])
        pa, za = imagine_hidden(tok, wm, ctx[-1], a[CONTEXT - 1:], HORIZON, ep, DEV); pb, zb = imagine_hidden(tok, wm, ctx[-1], b[CONTEXT - 1:], HORIZON, ep, DEV)
        pa, pb = pa.transpose(0, 2, 3, 1), pb.transpose(0, 2, 3, 1)
        zta, ztb = true_hidden(tok, wm, TA, a[CONTEXT - 1:CONTEXT - 1 + HORIZON], DEV), true_hidden(tok, wm, TB, b[CONTEXT - 1:CONTEXT - 1 + HORIZON], DEV)
        pm = measure_pair(flat(pa), flat(pb), flat(ta), flat(tb), flat(to64(ctx)), K, DELAYS)
        rec = {"pixel_align": {}, "repr_align": {}, "repr_true_div": {}, "true_div": {}}
        for d in DELAYS:
            i = K + d; rec["pixel_align"][str(d)] = pm.causal_alignment.get(d); rec["true_div"][str(d)] = pm.true_divergence.get(d)
            rec["repr_align"][str(d)] = repr_alignment(za[i], zb[i], zta[i], ztb[i]); rec["repr_true_div"][str(d)] = float(np.linalg.norm(ztb[i] - zta[i]))
        raw.append(rec)
    agg = {str(d): {"pixel_align": bootstrap_median([x["pixel_align"][str(d)] for x in raw]).as_dict(), "repr_align": bootstrap_median([x["repr_align"][str(d)] for x in raw]).as_dict()} for d in DELAYS}
    res[str(seed)] = {"aggregate": agg, "raw": raw}
    print(f"seed {seed}: d1 pixel {agg['1']['pixel_align']['median']:.3f} hidden {agg['1']['repr_align']['median']:.3f} | d20 pixel {agg['20']['pixel_align']['median']:.3f} hidden {agg['20']['repr_align']['median']:.3f}", flush=True)
name = f"irishidden_{G}_k{OFFSET}_w{WARMUP}" + ("_DRY" if DRY else "")
json.dump({"experiment": "iris_hidden_probe", "game": GAME, "seeds": SEEDS, "n_episodes": N_EP, "delays": DELAYS, "action_offset": OFFSET, "warmup_steps": WARMUP, "results": res}, open(OUT / f"{name}.json", "w"), indent=1)
print(f"wrote outputs/{name}.json")
