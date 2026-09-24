"""CLOSED-LOOP planning inside the emulator (ext arm C; rules fixed 2026-09-11 before running).
Random-shooting MPC: at each decision step the world model imagines K candidate action sequences of length H from the current
emulator context (common random numbers across candidates), the FROZEN per-game scorer (outputs/scorers/<G>_dense.pt, trained on
emulator frames, shared by all architectures) reads each imagined future's return, the first action of the best candidate is EXECUTED
in the emulator, and the loop advances. Outcome = the emulator's return over T decision steps. Controls: (i) a random policy, (ii) an
'oracle' MPC that imagines with the emulator itself (upper reference), same K, H, scorer. Per game: median return over N episodes with
bootstrap CI, for each architecture and both controls. A usefulness claim needs the model's return CI to exclude the random policy's.
Usage: v20_closed_loop.py {diamond|iris|storm|random|oracle} ALE/Game-v5 [--dry]"""
import json, pathlib, sys, os
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
import numpy as np, torch, torch.nn as nn, cv2, gymnasium as gym, ale_py
from ccaus.joint import bootstrap_median
ARCH, GAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv; G = GAME.split("/")[-1].replace("-v5", ""); OUT = ROOT / "ext" / "outputs"; OUT.mkdir(exist_ok=True)
S = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json")); WARMUP = int(S[G]["best_warmup"]) if G in S else 0
IMG = 64; DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, H, T, N_EP = (3, 4, 3, 2) if DRY else (8, 12, 30, 20); SEEDS = [42] if DRY else [42, 7, 128]
CONTEXT = 8 if ARCH == "storm" else 4
if ARCH == "storm":
    import env_wrapper as _sw
    def _make_env(): return _sw.MaxLast2FrameSkipWrapper(gym.make(GAME, frameskip=1, repeat_action_probability=0.0, full_action_space=False), skip=4)
else:
    def _make_env(): return gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
n_act = gym.make(GAME, frameskip=4).action_space.n
to64 = lambda fr: np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1
# ---- frozen scorer (same architecture as v14) ----
class Scorer(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(6, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(), nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.SiLU(), nn.Linear(256, 3))
    def forward(self, prev, nxt): return self.net(torch.cat([prev, nxt], 1))
SCK = ROOT / "outputs" / "scorers" / f"{G}_dense.pt"
if not SCK.exists(): SCK = ROOT / "outputs" / "scorers" / f"{G}.pt"
assert SCK.exists() or DRY, f"frozen scorer missing for {G}: run the cFS2 chain first"
scorer = Scorer().to(DEV).eval()
if SCK.exists(): scorer.load_state_dict(torch.load(SCK, map_location=DEV))
def scored_return(frames):                                   # frames [1+H, 64, 64, 3] in [-1, 1]: context last frame + H imagined
    with torch.no_grad():
        f = torch.as_tensor(np.asarray(frames, np.float32)).permute(0, 3, 1, 2).to(DEV)
        e = torch.softmax(scorer(f[:-1], f[1:]), -1) @ torch.tensor([-1.0, 0.0, 1.0], device=DEV)
    return float(e.sum().cpu())
# ---- world models ----
if ARCH == "diamond":
    from ccaus.diamond_harness import load_diamond, imagine; sampler = load_diamond(str(ROOT / f"models/{G}.pt"), num_actions=n_act, device=DEV)
    predict = lambda ctx, acts, seed: imagine(sampler, ctx, acts, H, seed=seed, device=DEV).transpose(0, 2, 3, 1)
elif ARCH == "iris":
    from ccaus.iris_harness import load_iris, imagine; tok, wm = load_iris(str(ROOT / f"models_iris/{G}.pt"), num_actions=n_act, device=DEV)
    predict = lambda ctx, acts, seed: imagine(tok, wm, ctx[-1], acts[CONTEXT - 1:], H, seed=seed, device=DEV).transpose(0, 2, 3, 1)
elif ARCH == "storm":
    from ccaus.storm_harness import load_storm, imagine as si; swm = load_storm(f"<STORM_SRC>/ckpt/{G}-life_done-wm_2L512D8H-100k-seed1/world_model_100000.pth", num_actions=n_act, device=DEV)
    predict = lambda ctx, acts, seed: si(swm, ctx, acts, H, seed=seed, device=DEV).transpose(0, 2, 3, 1)
def run_episode(seed, ep):
    e = _make_env(); obs, _ = e.reset(seed=ep); ale = e.unwrapped.ale; wr = np.random.default_rng(ep + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, n_act)))
    hist = [np.asarray(obs)]; acts_hist = []; rng = np.random.default_rng(seed * 1000 + ep); total = 0.0
    for _ in range(CONTEXT - 1):                                                     # fill the context with random actions
        a = int(rng.integers(0, n_act)); obs, r, term, trunc, _ = e.step(a); hist.append(np.asarray(obs)); acts_hist.append(a); total += float(r)
        if term or trunc: e.close(); return total, 0
    decisions = 0
    for t in range(T):
        cands = rng.integers(0, n_act, (K, H))
        if ARCH == "random": best = cands[0]
        else:
            ctx = np.stack(hist[-CONTEXT:]); acts_ctx = acts_hist[-(CONTEXT - 1):]
            scores = []
            for k in range(K):
                acts_full = np.concatenate([acts_ctx, cands[k]])
                if ARCH == "oracle":
                    st = ale.cloneState(); fr = []
                    for a in cands[k]: o, *_ = e.step(int(a)); fr.append(np.asarray(o))
                    ale.restoreState(st); f = np.concatenate([to64(ctx[-1:]), to64(np.stack(fr))])
                else:
                    p = predict(ctx, acts_full, seed=ep * 100 + t); f = np.concatenate([to64(ctx[-1:]), p])
                scores.append(scored_return(f))
            best = cands[int(np.argmax(scores))]
        a = int(best[0]); obs, r, term, trunc, _ = e.step(a); hist.append(np.asarray(obs)); acts_hist.append(a); total += float(r); decisions += 1
        if term or trunc: break
    e.close(); return total, decisions
res = {}
for seed in SEEDS:
    rets = []
    for ep in range(N_EP):
        ret, dec = run_episode(seed, ep); rets.append(dict(ret=ret, decisions=dec))
        if DRY: print(f"  ep {ep}: return {ret} over {dec} decisions", flush=True)
    res[str(seed)] = dict(returns=[x["ret"] for x in rets], decisions=[x["decisions"] for x in rets], agg=bootstrap_median([x["ret"] for x in rets]).as_dict())
    print(f"{ARCH} {G} seed {seed}: median return {res[str(seed)]['agg']['median']:.2f} [{res[str(seed)]['agg']['lo']:.2f}, {res[str(seed)]['agg']['hi']:.2f}] over {N_EP} episodes", flush=True)
json.dump({"experiment": "closed_loop_mpc_v20", "arch": ARCH, "game": GAME, "K": K, "H": H, "T": T, "n_episodes": N_EP, "seeds": SEEDS, "scorer": SCK.name, "results": res}, open(OUT / f"cl_{ARCH}_{G}{'_DRY' if DRY else ''}.json", "w")); print("wrote")
