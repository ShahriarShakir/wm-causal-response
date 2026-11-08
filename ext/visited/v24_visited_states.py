"""Experiment A (rule R6 in rebuttal/README.md, fixed before any job ran): the paired-intervention response and the reward
ranking at the states each model's OWN closed-loop planner visits.
The planner is ext/closed_loop/v20_closed_loop.py unchanged (K=8 candidates, horizon 12, 30 decisions, the frozen per-game scorer, the
same warm-up and random streams), rerun for seed 42, episodes 0-9. The measurement uses separate random streams and every harness
reseeds torch per step, so the planner's choices are unaffected; the executed actions are logged so every visited state replays exactly.
  Measurement 1, at decisions t = 0, 3, ..., 27: the paper's paired intervention from the visited state (context = the planner's last
    frames; window start K=3, duration 8, the game's emulator-chosen action offset; true arms by clone/restore; model arms under common
    random numbers), scored with ccaus.joint.measure_pair / aggregate at delays 1, 2, 3, 5, 8, 12.
  Measurement 2, at every decision: for the planner's own 8 candidates, the model's scored return (what the planner used), the
    emulator's native 12-step game return, and the frozen scorer on the true frames.
Usage: v24_visited_states.py {diamond|iris|storm} ALE/Game-v5 [--dry]   Writes ext/outputs/visited_<arch>_<Game>[_DRY].json,
or the same name under ext/$VISITED_OUTDIR/ when VISITED_OUTDIR is set."""
import json, os, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
import numpy as np, torch, torch.nn as nn, cv2, gymnasium as gym, ale_py
from ccaus.joint import Guard, aggregate, measure_pair
ARCH, GAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv; G = GAME.split("/")[-1].replace("-v5", ""); OUT = ROOT / "ext" / "outputs"; OUT.mkdir(exist_ok=True)
WOUT = ROOT / "ext" / os.environ.get("VISITED_OUTDIR", "outputs"); WOUT.mkdir(exist_ok=True)   # where this run's file is written; OUT is still read
assert ARCH in ("diamond", "iris", "storm")
S = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json")); WARMUP = int(S[G]["best_warmup"]) if G in S else 0; OFFSET = int(S[G]["best_offset"])
IMG = 64; DEV = "cuda" if torch.cuda.is_available() else "cpu"; DEVICE_NAME = torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"
K, H, T, N_EP = (8, 12, 2, 1) if DRY else (8, 12, 30, 10); SEED = 42                 # planner settings of v20 (seed 42, first 10 episodes)
CONTEXT = 8 if ARCH == "storm" else 4
KW, DUR, HM, DELAYS_M, EVERY = 3, 8, 16, [1, 2, 3, 5, 8, 12], 3                       # measurement: window start, duration, horizon, delays
if ARCH == "storm":
    import env_wrapper as _sw
    def _make_env(): return _sw.MaxLast2FrameSkipWrapper(gym.make(GAME, frameskip=1, repeat_action_probability=0.0, full_action_space=False), skip=4)
else:
    def _make_env(): return gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
n_act = gym.make(GAME, frameskip=4).action_space.n
to64 = lambda fr: np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1
flat = lambda x: x.reshape(x.shape[0], -1)
# ---- frozen scorer (v20, unchanged) ----
class Scorer(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(6, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(), nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.SiLU(), nn.Linear(256, 3))
    def forward(self, prev, nxt): return self.net(torch.cat([prev, nxt], 1))
SCK = ROOT / "outputs" / "scorers" / f"{G}_dense.pt"
if not SCK.exists(): SCK = ROOT / "outputs" / "scorers" / f"{G}.pt"
assert SCK.exists(), f"frozen scorer missing for {G}"
scorer = Scorer().to(DEV).eval(); scorer.load_state_dict(torch.load(SCK, map_location=DEV))
def scored_return(frames):
    with torch.no_grad():
        f = torch.as_tensor(np.asarray(frames, np.float32)).permute(0, 3, 1, 2).to(DEV)
        e = torch.softmax(scorer(f[:-1], f[1:]), -1) @ torch.tensor([-1.0, 0.0, 1.0], device=DEV)
    return float(e.sum().cpu())
# ---- world models (v20 calls; the measurement uses the same harness with horizon HM) ----
if ARCH == "diamond":
    from ccaus.diamond_harness import load_diamond, imagine; sampler = load_diamond(str(ROOT / f"models/{G}.pt"), num_actions=n_act, device=DEV)
    roll = lambda ctx, acts, h, seed: imagine(sampler, ctx, acts, h, seed=seed, device=DEV).transpose(0, 2, 3, 1)
elif ARCH == "iris":
    from ccaus.iris_harness import load_iris, imagine; tok, wm = load_iris(str(ROOT / f"models_iris/{G}.pt"), num_actions=n_act, device=DEV)
    roll = lambda ctx, acts, h, seed: imagine(tok, wm, ctx[-1], acts[CONTEXT - 1:], h, seed=seed, device=DEV).transpose(0, 2, 3, 1)
else:
    from ccaus.storm_harness import load_storm, imagine as si; swm = load_storm(f"<STORM_SRC>/ckpt/{G}-life_done-wm_2L512D8H-100k-seed1/world_model_100000.pth", num_actions=n_act, device=DEV)
    roll = lambda ctx, acts, h, seed: si(swm, ctx, acts, h, seed=seed, device=DEV).transpose(0, 2, 3, 1)
predict = lambda ctx, acts, seed: roll(ctx, acts, H, seed)

def true_rollout(e, ale, actions):
    """Step the emulator from its current state through `actions`, then restore it. Returns (frames, native return, terminated)."""
    st = ale.cloneState(); fr, R, done = [], 0.0, False
    for a in actions:
        o, r, term, trunc, _ = e.step(int(a)); fr.append(np.asarray(o)); R += float(r)
        if term or trunc: done = True; break
    ale.restoreState(st); return fr, R, done

def measure_here(e, ale, hist, acts_hist, ep, t):
    """The paper's paired intervention from the current (visited) state; returns a PairMeasurement or a reason string."""
    mr = np.random.default_rng(10 ** 7 + SEED * 10 ** 4 + ep * 100 + t)               # separate stream: the planner's rng is untouched
    ctx = np.stack(hist[-CONTEXT:]); acts = np.concatenate([np.asarray(acts_hist[-(CONTEXT - 1):], int), mr.integers(0, n_act, HM + 1)])
    b = acts.copy()
    for i in range(CONTEXT + KW, CONTEXT + KW + DUR):
        b[i] = (acts[i] + OFFSET) % n_act
        if b[i] == acts[i]: b[i] = (acts[i] + 1) % n_act
    fa, _, da = true_rollout(e, ale, acts[CONTEXT - 1:CONTEXT - 1 + HM]); fb, _, db = true_rollout(e, ale, b[CONTEXT - 1:CONTEXT - 1 + HM])
    if da or db or len(fa) < HM or len(fb) < HM: return "terminated"
    ms = 10 ** 6 + ep * 100 + t
    pa, pb = roll(ctx, acts, HM, ms), roll(ctx, b, HM, ms)                               # common random numbers across the arms
    try:
        return measure_pair(flat(pa), flat(pb), flat(to64(np.stack(fa))), flat(to64(np.stack(fb))), flat(to64(ctx)), KW, DELAYS_M, atol=1e-5)
    except AssertionError as err:
        return f"pre-window difference: {err}"

def run_episode(ep):
    e = _make_env(); obs, _ = e.reset(seed=ep); ale = e.unwrapped.ale; wr = np.random.default_rng(ep + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, n_act)))
    hist = [np.asarray(obs)]; acts_hist = []; rng = np.random.default_rng(SEED * 1000 + ep); total = 0.0
    rec = dict(episode=ep, warmup=WARMUP, context_actions=[], executed=[], ranking=[], pairs=[], skipped=[])
    for _ in range(CONTEXT - 1):                                                         # fill the context with random actions (v20)
        a = int(rng.integers(0, n_act)); obs, r, term, trunc, _ = e.step(a); hist.append(np.asarray(obs)); acts_hist.append(a); total += float(r)
        rec["context_actions"].append(a)
        if term or trunc: e.close(); rec.update(ret=total, decisions=0); return rec, []
    decisions, pairs = 0, []
    for t in range(T):
        cands = rng.integers(0, n_act, (K, H))
        ctx = np.stack(hist[-CONTEXT:]); acts_ctx = acts_hist[-(CONTEXT - 1):]
        scores = []
        for k in range(K):
            acts_full = np.concatenate([acts_ctx, cands[k]])
            p = predict(ctx, acts_full, seed=ep * 100 + t); f = np.concatenate([to64(ctx[-1:]), p])
            scores.append(scored_return(f))
        # measurement 2: the same candidates on the emulator (native return) and the scorer on the true frames
        nat, tfs = [], []
        for k in range(K):
            fr, R, _ = true_rollout(e, ale, cands[k]); nat.append(R); tfs.append(scored_return(np.concatenate([to64(ctx[-1:]), to64(np.stack(fr))])))
        rec["ranking"].append(dict(t=t, model=[float(x) for x in scores], native=nat, true_frames=tfs, chosen=int(np.argmax(scores))))
        # measurement 1: paired intervention from this visited state
        if t % EVERY == 0:
            m = measure_here(e, ale, hist, acts_hist, ep, t)
            if isinstance(m, str): rec["skipped"].append(dict(t=t, reason=m))
            else: pairs.append(m); rec["pairs"].append(dict(t=t, **{f: {str(d): (None if v is None else float(v)) for d, v in (getattr(m, f) or {}).items()} for f in ("causal_alignment", "causal_magnitude", "true_divergence", "prediction_psnr", "consistency")}))
        best = cands[int(np.argmax(scores))]
        a = int(best[0]); obs, r, term, trunc, _ = e.step(a); hist.append(np.asarray(obs)); acts_hist.append(a); total += float(r); decisions += 1
        rec["executed"].append(a)
        if term or trunc: break
    e.close(); rec.update(ret=total, decisions=decisions); return rec, pairs

import os
EP0, EP1 = int(os.environ.get("EP_START", 0)), int(os.environ.get("EP_END", N_EP))       # optional episode range, to split a long cell
NAME = f"visited_{ARCH}_{G}" + ("" if (EP0, EP1) == (0, N_EP) else f"_e{EP0}-{EP1}") + ("_DRY" if DRY else "")
fcl = OUT / f"cl_{ARCH}_{G}.json"
STORED = json.load(open(fcl))["results"].get(str(SEED), {}).get("returns", []) if fcl.exists() else None
t0 = time.time(); episodes, all_pairs = [], []
def save(done):
    """Written after every episode, so a walltime kill keeps the finished episodes."""
    stored = None if STORED is None else STORED[EP0:EP0 + len(episodes)]
    match = None if stored is None else int(sum(abs(a["ret"] - b) < 1e-9 for a, b in zip(episodes, stored)))
    json.dump({"experiment": "visited_states_v24", "rule": "rebuttal/README.md R6", "arch": ARCH, "game": GAME, "seed": SEED, "K": K, "H": H, "T": T,
               "episode_range": [EP0, EP1], "complete": done, "n_episodes_done": len(episodes),
               "measure": dict(window_start=KW, duration=DUR, horizon=HM, delays=DELAYS_M, every=EVERY, offset=OFFSET, context=CONTEXT),
               "scorer": SCK.name, "aggregate": aggregate(all_pairs, DELAYS_M, Guard()) if all_pairs else None, "n_pairs": len(all_pairs),
               "stored_returns": stored, "returns": [x["ret"] for x in episodes], "determinism_match": match, "episodes": episodes,
               "seconds": time.time() - t0, "device": DEVICE_NAME}, open(WOUT / f"{NAME}.json", "w"))
    return match
for ep in range(EP0, EP1):
    rec, pairs = run_episode(ep); episodes.append(rec); all_pairs += pairs
    m = save(False)
    print(f"{ARCH} {G} ep {ep}: return {rec['ret']} over {rec['decisions']} decisions, {len(pairs)} pairs, {len(rec['skipped'])} skipped, match so far {m}, {time.time() - t0:.0f}s", flush=True)
m = save(True)
print(f"wrote {NAME}.json: {len(all_pairs)} pairs, determinism match {m}/{len(episodes)}, {time.time() - t0:.0f}s", flush=True)
