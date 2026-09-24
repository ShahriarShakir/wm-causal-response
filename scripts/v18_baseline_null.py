"""Shared-baseline null for the three-arm cosine: with the SAME actions and three
different model seeds s1, s2, s3, the two 'responses' d2 = p(s2) - p(s1) and d3 = p(s3) - p(s1) share the baseline p(s1) but carry no action
change. If cos(d2, d3) under this null is close to the observed cos(dp_b, dp_c) ~ 0.46, the reading 'action-specific to the world's degree'
is not supported and only 'different outputs under different actions' is. Also records ||d2|| relative to ||dp_b|| (sampling
noise scale vs response scale). Usage: v18_baseline_null.py {diamond|iris|storm} ALE/Game-v5 [--dry]"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
import numpy as np, torch, cv2, gymnasium as gym, ale_py
ARCH, GAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv; G = GAME.split("/")[-1].replace("-v5", ""); OUT = ROOT / "outputs"
S = json.load(open(OUT / "emulator_intervention_search.json")); OFFSET, WARMUP = int(S[G]["best_offset"]), int(S[G]["best_warmup"])
CONTEXT, HORIZON, K, DUR, IMG = 4, 24, 3, 8, 64; DELAYS = [1, 3, 8, 20]; N_EP = 2 if DRY else 40; DEV = "cuda" if torch.cuda.is_available() else "cpu"
if ARCH == "storm":
    import env_wrapper as _sw
    def _make_env(): return _sw.MaxLast2FrameSkipWrapper(gym.make(GAME, frameskip=1, repeat_action_probability=0.0, full_action_space=False), skip=4)
else:
    def _make_env(): return gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
def emu(seed, actions):
    e = _make_env(); obs, _ = e.reset(seed=seed); wr = np.random.default_rng(seed + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, e.action_space.n)))
    fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)
to64 = lambda fr: np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1
flat = lambda x: x.reshape(x.shape[0], -1)
n_act = gym.make(GAME, frameskip=4).action_space.n
if ARCH == "diamond":
    from ccaus.diamond_harness import load_diamond, imagine; sampler = load_diamond(str(ROOT / f"models/{G}.pt"), num_actions=n_act, device=DEV)
    predict = lambda ctx, acts, seed: imagine(sampler, ctx, acts, HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
elif ARCH == "storm":
    from ccaus.storm_harness import load_storm, imagine as si, CONTEXT as SC; CONTEXT = SC; WARMUP = max(0, WARMUP - (SC - 4))
    swm = load_storm(f"<STORM_SRC>/ckpt/{G}-life_done-wm_2L512D8H-100k-seed1/world_model_100000.pth", num_actions=n_act, device=DEV)
    predict = lambda ctx, acts, seed: si(swm, ctx, acts, HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
else:
    from ccaus.iris_harness import load_iris, imagine; tok, wm = load_iris(str(ROOT / f"models_iris/{G}.pt"), num_actions=n_act, device=DEV)
    predict = lambda ctx, acts, seed: imagine(tok, wm, ctx[-1], acts[CONTEXT - 1:], HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
def cos(u, v):
    n = float(np.linalg.norm(u) * np.linalg.norm(v)); return float(u @ v / n) if n > 0 else None
raw = []
for ep in range(N_EP):
    rng = np.random.default_rng(42 * 1000 + ep); a = rng.integers(0, n_act, CONTEXT + HORIZON); b = a.copy()
    for t in range(CONTEXT + K, CONTEXT + K + DUR): b[t] = (a[t] + OFFSET) % n_act
    ctx = emu(ep, a[:CONTEXT - 1])[:CONTEXT]
    p1, p2, p3 = flat(predict(ctx, a, ep)), flat(predict(ctx, a, 10_000 + ep)), flat(predict(ctx, a, 20_000 + ep))   # same actions, three seeds
    pb = flat(predict(ctx, b, ep))                                                                                    # the real intervention under seed 1
    rec = {"null_cos": {}, "noise_over_response": {}}
    for d in DELAYS:
        i = K + d; d2, d3, dpb = p2[i] - p1[i], p3[i] - p1[i], pb[i] - p1[i]
        rec["null_cos"][str(d)] = cos(d2, d3); rec["noise_over_response"][str(d)] = float(np.linalg.norm(d2) / np.linalg.norm(dpb)) if np.linalg.norm(dpb) > 0 else None
    raw.append(rec)
agg = {k: {str(d): float(np.nanmedian([r[k][str(d)] for r in raw if r[k][str(d)] is not None])) for d in DELAYS} for k in raw[0]}
print(f"{ARCH} {G}: null cos(d2,d3) " + " ".join(f"d{d} {agg['null_cos'][str(d)]:.2f}" for d in DELAYS) + " | noise/response " + " ".join(f"d{d} {agg['noise_over_response'][str(d)]:.2f}" for d in DELAYS), flush=True)
json.dump({"experiment": "shared_baseline_null_v18", "arch": ARCH, "game": GAME, "n_episodes": N_EP, "delays": DELAYS, "aggregate": agg, "raw": raw}, open(OUT / f"null_{ARCH}_{G}{'_DRY' if DRY else ''}.json", "w")); print("wrote")
