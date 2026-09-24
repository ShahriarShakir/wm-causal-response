"""OBJECT-LEVEL validation of the causal-blindness label.
The actuated object is defined by the emulator, not by us: its pixels are those where the two TRUE arms differ at the first delay
with a difference (the intervention's immediate effect). Its colour signature (the set of RGB values inside that difference, in the
arm-b true frame) then locates the object in ANY frame, true or generated, as the centroid of the pixels carrying that signature within
a window around the previous position (tracking). Per episode and delay we record, for the true arms and for the model's arms:
    displacement vector of the actuated object between arms (b minus a), in raw-frame pixels;
    presence: whether the object is found in each generated frame (signature pixel count >= half the true count).
Object-level errors:  direction error = angle between model and true displacement (deg); magnitude ratio; and a TOLERANT hit:
    the model's displacement is within tol pixels of the true one (tol in {2, 4, 8}) AND the object is present in both generated arms.
Also the SWAP statistic from 2 <p_b - p_c, t_b - t_c> / ||t_b - t_c||^2 (both true consequences, shared baseline cancels).
Usage: v19_object_level.py {diamond|iris|storm} ALE/Game-v5 [--dry]"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
import numpy as np, torch, cv2, gymnasium as gym, ale_py
from ccaus.joint import bootstrap_median
ARCH, GAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv; G = GAME.split("/")[-1].replace("-v5", ""); OUT = ROOT / "outputs"
S = json.load(open(OUT / "emulator_intervention_search.json")); OFFSET, WARMUP = int(S[G]["best_offset"]), int(S[G]["best_warmup"])
CONTEXT, HORIZON, K, DUR, IMG = 4, 24, 3, 8, 64; DELAYS = [1, 3, 8, 20]; N_EP = 2 if DRY else 40; DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOLS = (2, 4, 8)
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
def to_u8_64(x): return np.clip((x + 1) / 2 * 255, 0, 255).astype(np.uint8)
def signature(frame, mask):
    """mean colour of the actuated object on the TRUE frame (excluding the frame's background colour); None if the difference is background-only."""
    vals, cnt = np.unique(frame.reshape(-1, 3), axis=0, return_counts=True); bg = vals[np.argmax(cnt)].astype(int)
    cols = frame[mask].reshape(-1, 3).astype(int); cols = cols[np.abs(cols - bg).sum(1) > 30]
    if len(cols) < 4: return None
    v2, c2 = np.unique(cols, axis=0, return_counts=True); return v2[np.argmax(c2)].astype(int)     # the object's dominant colour
def locate(frame, sig, tol=40, region=None):
    """centroid and pixel count of pixels within colour distance tol of the signature, optionally within a region mask (the actuated object's
    lane, e.g. Pong's paddle column), found anywhere in the frame: no cross-arm window, so large true displacements are measured correctly."""
    m = np.abs(frame.astype(int) - sig).sum(-1) <= tol
    if region is not None: m &= region
    ys, xs = np.nonzero(m)
    return (None, 0) if len(ys) == 0 else ((float(ys.mean()), float(xs.mean())), int(len(ys)))
def up(frame64): return cv2.resize(frame64, (160, 210), interpolation=cv2.INTER_NEAREST)
raw = []
for ep in range(N_EP):
    rng = np.random.default_rng(42 * 1000 + ep); a = rng.integers(0, n_act, CONTEXT + HORIZON); b = a.copy()
    for t in range(CONTEXT + K, CONTEXT + K + DUR): b[t] = (a[t] + OFFSET) % n_act
    c = None
    for k2 in range(1, n_act):
        if k2 == OFFSET: continue
        cc = a.copy()
        for t in range(CONTEXT + K, CONTEXT + K + DUR): cc[t] = (a[t] + k2) % n_act
        c = cc; break
    ctx = emu(ep, a[:CONTEXT - 1])[:CONTEXT]
    RA, RB, RC = [emu(ep, x)[CONTEXT:CONTEXT + HORIZON] for x in (a, b, c)]                           # full-resolution truth
    TA, TB, TC = [to_u8_64(to64(x)) for x in (RA, RB, RC)]
    PA, PB, PC = [to_u8_64(predict(ctx, x, ep)) for x in (a, b, c)]
    # actuated object = first delay with a true difference
    d0 = next((d for d in range(1, HORIZON - K) if (np.abs(RB[K + d].astype(int) - RA[K + d].astype(int)).sum(-1) > 30).any()), None)
    if d0 is None: continue
    mask = (np.abs(RB[K + d0].astype(int) - RA[K + d0].astype(int)).sum(-1) > 30); sig = signature(RB[K + d0], mask)
    if sig is None: continue
    region = np.zeros(RA.shape[1:3], bool)                                                              # the object's lane: union of true differences over the horizon, dilated
    for d in range(1, HORIZON - K): region |= (np.abs(RB[K + d].astype(int) - RA[K + d].astype(int)).sum(-1) > 30)
    region = cv2.dilate(region.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    rec = {"d0": d0, "true_disp": {}, "model_disp": {}, "present": {}, "dir_err_deg": {}, "mag_ratio": {}, "hit": {str(t): {} for t in TOLS}, "swap": {}, "true_count": {}}
    for d in DELAYS:
        i = K + d
        pos = {}; cnt = {}
        for k, F in (("ta", RA[i]), ("tb", RB[i]), ("pa", up(PA[i])), ("pb", up(PB[i]))):
            pos[k], cnt[k] = locate(F, sig, region=region)
        ds = str(d)
        if pos["ta"] is None or pos["tb"] is None: continue
        td = np.array(pos["tb"]) - np.array(pos["ta"]); rec["true_disp"][ds] = td.tolist(); rec["true_count"][ds] = min(cnt["ta"], cnt["tb"])
        present = pos["pa"] is not None and pos["pb"] is not None and cnt["pa"] >= 0.25 * cnt["ta"] and cnt["pb"] >= 0.25 * cnt["tb"]
        rec.setdefault("count_ratio", {})[ds] = [float(cnt["pa"] / max(cnt["ta"], 1)), float(cnt["pb"] / max(cnt["tb"], 1))]
        rec["present"][ds] = bool(present)
        if present:
            md = np.array(pos["pb"]) - np.array(pos["pa"]); rec["model_disp"][ds] = md.tolist()
            nt, nm = np.linalg.norm(td), np.linalg.norm(md)
            rec["mag_ratio"][ds] = float(nm / nt) if nt > 0 else None
            rec["dir_err_deg"][ds] = float(np.degrees(np.arccos(np.clip(td @ md / (nt * nm), -1, 1)))) if nt > 0 and nm > 0 else None
            for t in TOLS: rec["hit"][str(t)][ds] = bool(np.linalg.norm(md - td) <= t)
        else:
            for t in TOLS: rec["hit"][str(t)][ds] = False
        # swap statistic in pixel space (64x64, [-1,1] scale)
        pb_, pc_, pa_ = PB[i].astype(float), PC[i].astype(float), PA[i].astype(float); tb_, tc_, ta_ = TB[i].astype(float), TC[i].astype(float), TA[i].astype(float)
        dt = (tb_ - tc_).reshape(-1); dp = (pb_ - pc_).reshape(-1); n2 = float(dt @ dt)
        rec["swap"][ds] = float(2 * (dp @ dt) / n2) if n2 > 0 else None
    raw.append(rec)
def med(key, ds, sub=None):
    v = [(r[key][sub][ds] if sub else r[key][ds]) for r in raw if (r[key][sub].get(ds) if sub else r[key].get(ds)) is not None]
    return (bootstrap_median([float(x) for x in v]).as_dict() if v else None)
agg = {}
for d in DELAYS:
    ds = str(d); agg[ds] = dict(n=sum(1 for r in raw if ds in r["true_disp"]), present=float(np.mean([r["present"][ds] for r in raw if ds in r["present"]])) if any(ds in r["present"] for r in raw) else None,
                         dir_err_deg=med("dir_err_deg", ds), mag_ratio=med("mag_ratio", ds), swap=med("swap", ds),
                         hit={str(t): float(np.mean([r["hit"][str(t)][ds] for r in raw if ds in r["hit"][str(t)]])) if any(ds in r["hit"][str(t)] for r in raw) else None for t in TOLS},
                         true_disp_norm=float(np.median([np.linalg.norm(r["true_disp"][ds]) for r in raw if ds in r["true_disp"]])) if any(ds in r["true_disp"] for r in raw) else None)
    x = agg[ds]; print(f"{ARCH} {G} d{d}: n {x['n']} present {x['present']} | true disp {x['true_disp_norm']} px | dir err {x['dir_err_deg']['median'] if x['dir_err_deg'] else None} deg | mag ratio {x['mag_ratio']['median'] if x['mag_ratio'] else None} | hit@2/4/8 {x['hit']} | swap {x['swap']['median'] if x['swap'] else None}", flush=True)
json.dump({"experiment": "object_level_v19", "arch": ARCH, "game": GAME, "n_episodes": N_EP, "delays": DELAYS, "tolerances": TOLS, "aggregate": agg, "raw": raw}, open(OUT / f"obj_{ARCH}_{G}{'_DRY' if DRY else ''}.json", "w")); print("wrote")
