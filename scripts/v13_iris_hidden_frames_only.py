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

# ---------------------------------------------------------------------------------------------------------
# v13: FRAMES-ONLY control for the IRIS pre-sampling probe (the same control as for the RSSM).
# The transformer's hidden state at a position receives the action token too, so the 'true hidden change' between arms
# can be the echo of the action token rather than visual evidence. Control: teacher-force the TRUE tokens of arm b with
# the ACTIONS of arm a (frames-only change) and of arm a with the actions of b (actions-only change).
#   frames_only_align  = alignment of the imagined hidden change with the frames-only true change
#   actions_only_align = alignment of the imagined hidden change with the actions-only true change
#   share              = ||frames-only change|| / ||full change||  (quote frames_only_align only where share >= 0.25)
# ---------------------------------------------------------------------------------------------------------
N_EP = 2 if DRY else 30
res = {}
for seed in SEEDS:
    raw = []
    for ep in range(N_EP):
        r = np.random.default_rng(seed * 1000 + ep); a = r.integers(0, n_act, CONTEXT + HORIZON); b = a.copy()
        for t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(a))):
            b[t] = (a[t] + OFFSET) % n_act
            if b[t] == a[t]: b[t] = (a[t] + 1) % n_act
        ctx = emu(ep, a[:CONTEXT - 1])[:CONTEXT]; TA = emu(ep, a)[CONTEXT - 1:CONTEXT + HORIZON]; TB = emu(ep, b)[CONTEXT - 1:CONTEXT + HORIZON]
        _, za = imagine_hidden(tok, wm, ctx[-1], a[CONTEXT - 1:], HORIZON, ep, DEV); _, zb = imagine_hidden(tok, wm, ctx[-1], b[CONTEXT - 1:], HORIZON, ep, DEV)
        zta, ztb = true_hidden(tok, wm, TA, a[CONTEXT - 1:CONTEXT - 1 + HORIZON], DEV), true_hidden(tok, wm, TB, b[CONTEXT - 1:CONTEXT - 1 + HORIZON], DEV)
        ztb_A = true_hidden(tok, wm, TB, a[CONTEXT - 1:CONTEXT - 1 + HORIZON], DEV)      # frames of b, actions of a
        zta_B = true_hidden(tok, wm, TA, b[CONTEXT - 1:CONTEXT - 1 + HORIZON], DEV)      # frames of a, actions of b
        rec = {"full_align": {}, "frames_only_align": {}, "actions_only_align": {}, "share": {}, "cos_full_vs_frames_only": {}}
        for d in DELAYS:
            i = K + d
            if i >= len(za): continue
            u, uf, ua = ztb[i] - zta[i], ztb_A[i] - zta[i], zta_B[i] - zta[i]
            rec["full_align"][str(d)] = repr_alignment(za[i], zb[i], zta[i], ztb[i])
            rec["frames_only_align"][str(d)] = repr_alignment(za[i], zb[i], zta[i], ztb_A[i])
            rec["actions_only_align"][str(d)] = repr_alignment(za[i], zb[i], zta[i], zta_B[i])
            nu = float(np.linalg.norm(u)); rec["share"][str(d)] = float(np.linalg.norm(uf) / nu) if nu > 0 else None
            nn = float(np.linalg.norm(u) * np.linalg.norm(uf)); rec["cos_full_vs_frames_only"][str(d)] = float(np.dot(u, uf) / nn) if nn > 0 else None
        raw.append(rec)
    agg = {k: {str(d): bootstrap_median([x[k][str(d)] for x in raw if x[k].get(str(d)) is not None and x[k][str(d)] == x[k][str(d)]]).as_dict() for d in DELAYS if any(str(d) in x[k] for x in raw)} for k in raw[0]}
    res[str(seed)] = {"aggregate": agg, "raw": raw}
    print(f"seed {seed}: " + " ".join(f"d{d}: full {agg['full_align'][str(d)]['median']:.2f} frames-only {agg['frames_only_align'][str(d)]['median']:.2f} actions-only {agg['actions_only_align'][str(d)]['median']:.2f} share {agg['share'][str(d)]['median']:.2f}" for d in (1, 3, 8, 20) if str(d) in agg["full_align"]), flush=True)
name = f"irishiddenfo_{G}_k{OFFSET}_w{WARMUP}" + ("_DRY" if DRY else "")
json.dump({"experiment": "iris_hidden_frames_only_control", "game": GAME, "seeds": SEEDS, "n_episodes": N_EP, "delays": DELAYS, "action_offset": OFFSET, "warmup_steps": WARMUP, "results": res}, open(OUT / f"{name}.json", "w"))
print(f"wrote outputs/{name}.json")
