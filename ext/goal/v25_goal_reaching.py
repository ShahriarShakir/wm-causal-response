"""Experiment R7 (rule R7 in rebuttal/README.md, fixed before any job ran): goal reaching at the paper's diagnostic states.
Each trial: reset with the trial's seed, the game's emulator-chosen warm-up, a random-action context of the model's length; 8 random candidate
action sequences of length 12; one candidate's true frame after 12 steps is the goal; the model predicts the frame after 12 steps for every
candidate from the same context (one model seed per trial) and picks the one nearest the goal in squared error at 64x64. Success: the chosen
candidate's true frame equals the goal. Model loading and harness calls are those of ext/visited/v24_visited_states.py, unchanged.
Usage: v25_goal_reaching.py {diamond|iris|storm} ALE/Game-v5 [--dry]   Writes ext/outputs_goal/goal_<arch>_<Game>[_DRY].json"""
import json, os, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
import numpy as np, torch, cv2, gymnasium as gym, ale_py
ARCH, GAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv; G = GAME.split("/")[-1].replace("-v5", "")
assert ARCH in ("diamond", "iris", "storm")
OUT = ROOT / "ext" / "outputs_goal"; OUT.mkdir(exist_ok=True)
S = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json")); WARMUP = int(S[G]["best_warmup"]) if G in S else 0
IMG = 64; DEV = "cuda" if torch.cuda.is_available() else "cpu"; DEVICE_NAME = torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"
FLUSH = os.environ.get("FLUSH_DENORMAL") == "1"                                                # off by default; see rebuttal/README.md, R7
if FLUSH: torch.set_flush_denormal(True)                                                       # subnormal floats -> 0 (CPU speed only)
K, H, N = 8, 12, (2 if DRY else 50)
CONTEXT = 8 if ARCH == "storm" else 4
if ARCH == "storm":
    import env_wrapper as _sw
    def _make_env(): return _sw.MaxLast2FrameSkipWrapper(gym.make(GAME, frameskip=1, repeat_action_probability=0.0, full_action_space=False), skip=4)
else:
    def _make_env(): return gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
n_act = gym.make(GAME, frameskip=4).action_space.n
to64 = lambda fr: np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1
# ---- world models (the calls of v24) ----
if ARCH == "diamond":
    from ccaus.diamond_harness import load_diamond, imagine; sampler = load_diamond(str(ROOT / f"models/{G}.pt"), num_actions=n_act, device=DEV)
    roll = lambda ctx, acts, h, seed: imagine(sampler, ctx, acts, h, seed=seed, device=DEV).transpose(0, 2, 3, 1)
elif ARCH == "iris":
    from ccaus.iris_harness import load_iris, imagine; tok, wm = load_iris(str(ROOT / f"models_iris/{G}.pt"), num_actions=n_act, device=DEV)
    roll = lambda ctx, acts, h, seed: imagine(tok, wm, ctx[-1], acts[CONTEXT - 1:], h, seed=seed, device=DEV).transpose(0, 2, 3, 1)
else:
    from ccaus.storm_harness import load_storm, imagine as si; swm = load_storm(f"<STORM_SRC>/ckpt/{G}-life_done-wm_2L512D8H-100k-seed1/world_model_100000.pth", num_actions=n_act, device=DEV)
    roll = lambda ctx, acts, h, seed: si(swm, ctx, acts, h, seed=seed, device=DEV).transpose(0, 2, 3, 1)

def true_rollout(e, ale, actions):
    """Step the emulator from its current state through `actions`, then restore it. Returns (frames, native return, terminated)."""
    st = ale.cloneState(); fr, R, done = [], 0.0, False
    for a in actions:
        o, r, term, trunc, _ = e.step(int(a)); fr.append(np.asarray(o)); R += float(r)
        if term or trunc: done = True; break
    ale.restoreState(st); return fr, R, done

def trial(i):
    e = _make_env(); obs, _ = e.reset(seed=i); ale = e.unwrapped.ale; wr = np.random.default_rng(i + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, n_act)))
    hist = [np.asarray(obs)]; acts_hist = []; rng = np.random.default_rng(10 ** 8 + i)          # the trial's own stream
    for _ in range(CONTEXT - 1):                                                                  # random-action context, as in v20/v24
        a = int(rng.integers(0, n_act)); obs, r, term, trunc, _ = e.step(a); hist.append(np.asarray(obs)); acts_hist.append(a)
        if term or trunc: e.close(); return dict(trial=i, skipped="terminated in the context")
    cands = rng.integers(0, n_act, (K, H)); kstar = int(rng.integers(0, K))
    tf = []
    for k in range(K):
        fr, _, done = true_rollout(e, ale, cands[k])
        if done or len(fr) < H: e.close(); return dict(trial=i, skipped="terminated within the horizon")
        tf.append(to64(np.stack([fr[-1]]))[0])
    goal = tf[kstar]; eq = [bool(np.max(np.abs(t - goal)) < 1e-6) for t in tf]; chance = float(np.mean(eq))
    ctx = np.stack(hist[-CONTEXT:]); acts_ctx = np.asarray(acts_hist[-(CONTEXT - 1):], int)
    dist = []
    for k in range(K):
        p = roll(ctx, np.concatenate([acts_ctx, cands[k]]), H, 10 ** 6 + i)                     # one model seed per trial: common random numbers
        dist.append(float(np.sum((p[-1] - goal) ** 2)))
    choice = int(np.argmin(dist)); e.close()
    return dict(trial=i, goal=kstar, choice=choice, success=bool(eq[choice]), chance=chance, informative=chance < 1.0, dist=dist, equal_to_goal=eq)

NAME = f"goal_{ARCH}_{G}" + ("_DRY" if DRY else "")
t0 = time.time(); trials = []
def cell_stat():
    inf = [t for t in trials if t.get("informative")]
    return (float(np.mean([(t["success"] - t["chance"]) / (1 - t["chance"]) for t in inf])) if len(inf) >= 10 else None), len(inf)
def save(done):
    g, n_inf = cell_stat()
    json.dump({"experiment": "goal_reaching_v25", "rule": "rebuttal/README.md R7", "arch": ARCH, "game": GAME, "K": K, "H": H, "N": N,
               "warmup": WARMUP, "context": CONTEXT, "complete": done, "n_trials_done": len(trials), "n_informative": n_inf,
               "n_skipped": sum(1 for t in trials if "skipped" in t), "G": g, "trials": trials, "seconds": time.time() - t0, "device": DEVICE_NAME, "flush_denormal": FLUSH},
              open(OUT / f"{NAME}.json", "w"))
for i in range(N):
    trials.append(trial(i)); save(False)
save(True)
g, n_inf = cell_stat()
print(f"wrote {NAME}.json: G {g}, informative {n_inf}/{N}, skipped {sum(1 for t in trials if 'skipped' in t)}, success {sum(1 for t in trials if t.get('success'))}, {time.time() - t0:.0f}s", flush=True)
