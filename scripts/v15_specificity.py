"""Consequence-specificity of the PIXEL response (audit 2026-09-08). Three arms per episode under the emulator-chosen intervention:
A (base), B (offset k), C (a different offset whose true future differs from both). Per delay we record
  align_B   = <dp_B, dt_B>/||dt_B||^2   (the probe),  align_CB = <dp_C, dt_B>/||dt_B||^2  (the model's response to the WRONG action, projected on B's truth),
  cos_pBpC  = cos(dp_B, dp_C)           (are the model's responses to two different actions the same change?),
  offsupport = 1 - ||dp_B restricted to support(dt_B)||^2 / ||dp_B||^2   (fraction of the model's change energy that falls where the world did not change),
  cos_tBtC  = cos(dt_B, dt_C)           (how different the two true responses are), plus PSNR of arm B against its own truth.
Usage: v15_specificity.py {diamond|iris} ALE/Game-v5 [--dry]"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch, cv2, gymnasium as gym, ale_py
from ccaus.joint import bootstrap_median, prediction_psnr
ARCH, GAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv; G = GAME.split("/")[-1].replace("-v5", ""); OUT = ROOT / "outputs"
S = json.load(open(OUT / "emulator_intervention_search.json")); OFFSET, WARMUP = int(S[G]["best_offset"]), int(S[G]["best_warmup"])
CONTEXT, HORIZON, K, DUR, IMG = 4, 24, 3, 8, 64; DELAYS = [1, 2, 3, 5, 8, 12, 16, 20]
N_EP = 2 if DRY else 40; SEEDS = [42] if DRY else [42, 7, 128]; DEV = "cuda" if torch.cuda.is_available() else "cpu"

if ARCH == "storm":                                                  # STORM's observation footing (max of the last two of 4 repeated frames), sticky actions off
    sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
    import env_wrapper as _sw
    def _make_env(seed):
        e = gym.make(GAME, frameskip=1, repeat_action_probability=0.0, full_action_space=False); e = _sw.MaxLast2FrameSkipWrapper(e, skip=4); return e
else:
    def _make_env(seed): return gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
def emu(seed, actions):
    e = _make_env(seed); obs, _ = e.reset(seed=seed); wr = np.random.default_rng(seed + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, e.action_space.n)))
    fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)
to64 = lambda fr: np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1
flat = lambda x: x.reshape(x.shape[0], -1)
env = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); n_act = env.action_space.n; env.close()
if ARCH == "diamond":
    from ccaus.diamond_harness import load_diamond, imagine
    sampler = load_diamond(str(ROOT / f"models/{G}.pt"), num_actions=n_act, device=DEV)
    predict = lambda ctx, acts, seed: imagine(sampler, ctx, acts, HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
elif ARCH == "storm":
    from ccaus.storm_harness import load_storm, imagine as storm_imagine, CONTEXT as SCTX
    swm = load_storm(f"<STORM_SRC>/ckpt/{G}-life_done-wm_2L512D8H-100k-seed1/world_model_100000.pth", num_actions=n_act, device=DEV)
    CONTEXT = SCTX; WARMUP = max(0, WARMUP - (SCTX - 4))   # same absolute emulator time for the intervention as with a 4-frame context
    predict = lambda ctx, acts, seed: storm_imagine(swm, ctx, acts, HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
else:
    from ccaus.iris_harness import load_iris, imagine
    tok, wm = load_iris(str(ROOT / f"models_iris/{G}.pt"), num_actions=n_act, device=DEV)
    predict = lambda ctx, acts, seed: imagine(tok, wm, ctx[-1], acts[CONTEXT - 1:], HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
def cos(u, v):
    n = float(np.linalg.norm(u) * np.linalg.norm(v)); return float(u @ v / n) if n > 0 else None
res = {}
for seed in SEEDS:
    raw = []
    for ep in range(N_EP):
        rng = np.random.default_rng(seed * 1000 + ep); a = rng.integers(0, n_act, CONTEXT + HORIZON); b = a.copy()
        for t in range(CONTEXT + K, CONTEXT + K + DUR): b[t] = (a[t] + OFFSET) % n_act
        ctx = emu(ep, a[:CONTEXT - 1])[:CONTEXT]; ta = flat(to64(emu(ep, a)[CONTEXT:CONTEXT + HORIZON])); tb = flat(to64(emu(ep, b)[CONTEXT:CONTEXT + HORIZON]))
        i20 = K + 20
        if np.linalg.norm(tb[i20] - ta[i20]) < 1e-6: continue
        tc = None
        for k2 in range(1, n_act):
            if k2 == OFFSET: continue
            c = a.copy()
            for t in range(CONTEXT + K, CONTEXT + K + DUR): c[t] = (a[t] + k2) % n_act
            t_ = flat(to64(emu(ep, c)[CONTEXT:CONTEXT + HORIZON]))
            if np.linalg.norm(t_[i20] - tb[i20]) > 1e-6 and np.linalg.norm(t_[i20] - ta[i20]) > 1e-6: tc = t_; break
        if tc is None: continue
        pa = flat(predict(ctx, a, ep)); pb = flat(predict(ctx, b, ep)); pc = flat(predict(ctx, c, ep))
        rec = {k: {} for k in ("align_B", "align_CB", "cos_pBpC", "offsupport", "cos_tBtC", "true_div_B", "psnr_B")}
        for d in DELAYS:
            i = K + d; dtB, dtC, dpB, dpC = tb[i] - ta[i], tc[i] - ta[i], pb[i] - pa[i], pc[i] - pa[i]
            nB = float(dtB @ dtB)
            rec["true_div_B"][str(d)] = float(np.sqrt(nB)); rec["psnr_B"][str(d)] = prediction_psnr(pb[i], tb[i])
            if nB > 0:
                rec["align_B"][str(d)] = float(dpB @ dtB / nB); rec["align_CB"][str(d)] = float(dpC @ dtB / nB)
                sup = np.abs(dtB) > 0.01 * np.abs(dtB).max(); e_all = float(dpB @ dpB)
                rec["offsupport"][str(d)] = float(1 - (dpB[sup] @ dpB[sup]) / e_all) if e_all > 0 else None
            rec["cos_pBpC"][str(d)] = cos(dpB, dpC); rec["cos_tBtC"][str(d)] = cos(dtB, dtC)
        raw.append(rec)
    agg = {k: {str(d): bootstrap_median([x[k][str(d)] for x in raw if x[k].get(str(d)) is not None and x[k][str(d)] == x[k][str(d)]]).as_dict() for d in DELAYS if any(x[k].get(str(d)) is not None for x in raw)} for k in raw[0]} if raw else {}
    res[str(seed)] = {"aggregate": agg, "raw": raw, "n_used": len(raw)}
    if agg: print(f"seed {seed}: n {len(raw)} | d20: align_B {agg['align_B'].get('20', {}).get('median', float('nan')):.2f} align_CB {agg['align_CB'].get('20', {}).get('median', float('nan')):.2f} cos(pB,pC) {agg['cos_pBpC'].get('20', {}).get('median', float('nan')):.2f} offsupport {agg['offsupport'].get('20', {}).get('median', float('nan')):.2f} cos(tB,tC) {agg['cos_tBtC'].get('20', {}).get('median', float('nan')):.2f}", flush=True)
name = f"spec_{ARCH}_{G}" + ("_DRY" if DRY else "")
json.dump({"experiment": "pixel_specificity_v15", "arch": ARCH, "game": GAME, "seeds": SEEDS, "n_episodes": N_EP, "delays": DELAYS, "offset": OFFSET, "warmup": WARMUP, "results": res}, open(OUT / f"{name}.json", "w")); print(f"wrote outputs/{name}.json")
