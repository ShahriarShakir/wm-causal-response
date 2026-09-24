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
CONTEXT, HORIZON, K, DUR = 4, 24, 3, 8; DELAYS = [1, 2, 3, 5, 8]; DEV = "cuda" if torch.cuda.is_available() else "cpu"
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

# ---------------------------------------------------------------------------------------------------------
# v12: WRONG-ACTION CONTROL for the pre-sampling hidden-state probe (required by the pre-declared plan in hidden_probe_eval.py before a
# positive is quoted). Third arm C = a different offset whose TRUE future differs from both A and B at d20.
#   B_vs_trueB : alignment of the model's hidden response to B with the true hidden response to B (the probe itself)
#   C_vs_trueB : alignment of the model's hidden response to C with the true hidden response to B (wrong action)
#   B_vs_trueC : the converse. Action-specific tracking requires B_vs_trueB >> C_vs_trueB (mirrors repr_control_wrong_action.py).
# ---------------------------------------------------------------------------------------------------------
N_EP = 2 if DRY else 30
def third_arm(a, k, n_act, ep):
    for k2 in list(range(1, n_act)):
        if k2 == k: continue
        c = a.copy()
        for t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(a))):
            c[t] = (a[t] + k2) % n_act
            if c[t] == a[t]: c[t] = (a[t] + 1) % n_act
        TC = emu(ep, c)[CONTEXT - 1:CONTEXT + HORIZON]
        yield c, TC
res = {}
for seed in SEEDS:
    raw = []
    for ep in range(N_EP):
        r = np.random.default_rng(seed * 1000 + ep); a = r.integers(0, n_act, CONTEXT + HORIZON); b = a.copy()
        for t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(a))):
            b[t] = (a[t] + OFFSET) % n_act
            if b[t] == a[t]: b[t] = (a[t] + 1) % n_act
        ctx = emu(ep, a[:CONTEXT - 1])[:CONTEXT]; TA = emu(ep, a)[CONTEXT - 1:CONTEXT + HORIZON]; TB = emu(ep, b)[CONTEXT - 1:CONTEXT + HORIZON]
        i20 = K + min(20, HORIZON - K - 1)
        if np.abs(to64(TB[1:])[i20] - to64(TA[1:])[i20]).max() < 1e-6: continue
        pick = None
        for c, TC in third_arm(a, OFFSET, n_act, ep):
            tc, tb, ta = to64(TC[1:]), to64(TB[1:]), to64(TA[1:])
            if np.abs(tc[i20] - tb[i20]).max() > 1e-6 and np.abs(tc[i20] - ta[i20]).max() > 1e-6: pick = (c, TC); break
        if pick is None: continue
        c, TC = pick
        _, za = imagine_hidden(tok, wm, ctx[-1], a[CONTEXT - 1:], HORIZON, ep, DEV); _, zb = imagine_hidden(tok, wm, ctx[-1], b[CONTEXT - 1:], HORIZON, ep, DEV); _, zc = imagine_hidden(tok, wm, ctx[-1], c[CONTEXT - 1:], HORIZON, ep, DEV)
        zta, ztb, ztc = true_hidden(tok, wm, TA, a[CONTEXT - 1:CONTEXT - 1 + HORIZON], DEV), true_hidden(tok, wm, TB, b[CONTEXT - 1:CONTEXT - 1 + HORIZON], DEV), true_hidden(tok, wm, TC, c[CONTEXT - 1:CONTEXT - 1 + HORIZON], DEV)
        rec = {"B_vs_trueB": {}, "C_vs_trueB": {}, "B_vs_trueC": {}, "cos_trueB_trueC": {}, "repr_true_div_B": {}}
        for d in DELAYS:
            i = K + d
            if i >= len(za): continue
            rec["B_vs_trueB"][str(d)] = repr_alignment(za[i], zb[i], zta[i], ztb[i]); rec["C_vs_trueB"][str(d)] = repr_alignment(za[i], zc[i], zta[i], ztb[i])
            rec["B_vs_trueC"][str(d)] = repr_alignment(za[i], zb[i], zta[i], ztc[i])
            u, v = ztb[i] - zta[i], ztc[i] - zta[i]; nn = float(np.linalg.norm(u) * np.linalg.norm(v)); rec["cos_trueB_trueC"][str(d)] = float(np.dot(u, v) / nn) if nn > 0 else None
            rec["repr_true_div_B"][str(d)] = float(np.linalg.norm(u))
        raw.append(rec)
    agg = {k: {str(d): bootstrap_median([x[k][str(d)] for x in raw if x[k].get(str(d)) is not None and x[k][str(d)] == x[k][str(d)]]).as_dict() for d in DELAYS if any(str(d) in x[k] for x in raw)} for k in ("B_vs_trueB", "C_vs_trueB", "B_vs_trueC", "cos_trueB_trueC")}
    res[str(seed)] = {"aggregate": agg, "raw": raw, "n_used": len(raw)}
    print(f"seed {seed}: n {len(raw)} | " + " ".join(f"d{d}: B|B {agg['B_vs_trueB'][str(d)]['median']:.2f} C|B {agg['C_vs_trueB'][str(d)]['median']:.2f}" for d in DELAYS if str(d) in agg["B_vs_trueB"]), flush=True)
name = f"irishiddenctl_{G}_k{OFFSET}_w{WARMUP}" + ("_DRY" if DRY else "")
json.dump({"experiment": "iris_hidden_wrong_action_control", "game": GAME, "seeds": SEEDS, "n_episodes": N_EP, "delays": DELAYS, "action_offset": OFFSET, "warmup_steps": WARMUP, "results": res}, open(OUT / f"{name}.json", "w"))
print(f"wrote outputs/{name}.json")
