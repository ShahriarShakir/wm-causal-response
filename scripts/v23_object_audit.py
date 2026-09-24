"""Object-readout audit (ext/object_audit/README.md). Same object definition, harnesses and episodes as v19; adds a template locator, the
zero-displacement baseline, and blinded panels for human annotation. Usage: v23_object_audit.py {diamond|iris|storm} ALE/Game-v5 [--dry]"""
import os, json, pathlib, sys, hashlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
import numpy as np, torch, cv2, gymnasium as gym, ale_py
ARCH, GAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv; G = GAME.split("/")[-1].replace("-v5", ""); OUT = ROOT / "ext" / "outputs"; PAN = ROOT / "ext" / "object_audit" / ("panels" + os.environ.get("OA_TAG", ""))
S = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json")); OFFSET, WARMUP = int(S[G]["best_offset"]), int(S[G]["best_warmup"])
CONTEXT, HORIZON, K, DUR, IMG = 4, 24, 3, 8, 64; DELAYS = [8, 20]; N_EP = 1 if DRY else int(os.environ.get("OA_NEP", "6")); TAG = os.environ.get("OA_TAG", ""); DEV = "cuda" if torch.cuda.is_available() else "cpu"; TOLS = (2, 4, 8); TM_THR = 0.60
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
    vals, cnt = np.unique(frame.reshape(-1, 3), axis=0, return_counts=True); bg = vals[np.argmax(cnt)].astype(int)
    cols = frame[mask].reshape(-1, 3).astype(int); cols = cols[np.abs(cols - bg).sum(1) > 30]
    if len(cols) < 4: return None
    v2, c2 = np.unique(cols, axis=0, return_counts=True); return v2[np.argmax(c2)].astype(int)
def locate(frame, sig, tol=40, region=None):
    m = np.abs(frame.astype(int) - sig).sum(-1) <= tol
    if region is not None: m &= region
    ys, xs = np.nonzero(m); return (None, 0) if len(ys) == 0 else ((float(ys.mean()), float(xs.mean())), int(len(ys)))
def up(frame64): return cv2.resize(frame64, (160, 210), interpolation=cv2.INTER_NEAREST)
def template_locate(frame, tmpl, region):
    """normalised cross-correlation of the true object's patch inside the lane; returns (centre, peak score)."""
    if tmpl.shape[0] >= frame.shape[0] or tmpl.shape[1] >= frame.shape[1] or tmpl.shape[0] < 3 or tmpl.shape[1] < 3: return None, 0.0
    res = cv2.matchTemplate(frame.astype(np.uint8), tmpl.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
    th, tw = tmpl.shape[:2]; reg = region[th // 2: region.shape[0] - (th - th // 2) + 1, tw // 2: region.shape[1] - (tw - tw // 2) + 1]
    reg = reg[:res.shape[0], :res.shape[1]]; res = np.where(reg, res, -1.0)
    y, x = np.unravel_index(int(np.argmax(res)), res.shape); return (float(y + th / 2), float(x + tw / 2)), float(res.max())
def patch(frame, mask, pad=2):
    """patch around the LARGEST connected component of the mask (the object), so stray pixels of the same colour elsewhere in the lane do not
    inflate the template; returns None if the mask is empty."""
    if not mask.any(): return None
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1: return None
    sizes = [(lab == c).sum() for c in range(1, n)]; mask = (lab == (1 + int(np.argmax(sizes))))
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(frame.shape[0], ys.max() + pad + 1), max(0, xs.min() - pad), min(frame.shape[1], xs.max() + pad + 1); return frame[y0:y1, x0:x1]
raw, key = [], {}
for ep in range(N_EP):
    rng = np.random.default_rng(42 * 1000 + ep); a = rng.integers(0, n_act, CONTEXT + HORIZON); b = a.copy()
    for t in range(CONTEXT + K, CONTEXT + K + DUR): b[t] = (a[t] + OFFSET) % n_act
    ctx = emu(ep, a[:CONTEXT - 1])[:CONTEXT]; RA, RB = [emu(ep, x)[CONTEXT:CONTEXT + HORIZON] for x in (a, b)]
    PA, PB = [to_u8_64(predict(ctx, x, ep)) for x in (a, b)]
    d0 = next((d for d in range(1, HORIZON - K) if (np.abs(RB[K + d].astype(int) - RA[K + d].astype(int)).sum(-1) > 30).any()), None)
    if d0 is None: continue
    mask0 = (np.abs(RB[K + d0].astype(int) - RA[K + d0].astype(int)).sum(-1) > 30); sig = signature(RB[K + d0], mask0)
    if sig is None: continue
    region = np.zeros(RA.shape[1:3], bool)
    for d in range(1, HORIZON - K): region |= (np.abs(RB[K + d].astype(int) - RA[K + d].astype(int)).sum(-1) > 30)
    region = cv2.dilate(region.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    rec = {"episode": ep, "d0": d0, "delays": {}}
    for d in DELAYS:
        i = K + d; r = {}
        pos = {}; cnt = {}
        for k, F in (("ta", RA[i]), ("tb", RB[i]), ("pa", up(PA[i])), ("pb", up(PB[i]))): pos[k], cnt[k] = locate(F, sig, region=region)
        if pos["ta"] is None or pos["tb"] is None: continue
        td = np.array(pos["tb"]) - np.array(pos["ta"]); r["true_disp"] = td.tolist(); r["true_disp_norm"] = float(np.linalg.norm(td))
        r["zero_disp_passes"] = {str(t): bool(np.linalg.norm(td) <= t) for t in TOLS}
        # colour locator (v19 rule)
        present_c = pos["pa"] is not None and pos["pb"] is not None and cnt["pa"] >= 0.25 * cnt["ta"] and cnt["pb"] >= 0.25 * cnt["tb"]
        r["colour"] = {"present": bool(present_c), "count_ratio": [float(cnt["pa"] / max(cnt["ta"], 1)), float(cnt["pb"] / max(cnt["tb"], 1))]}
        if present_c:
            md = np.array(pos["pb"]) - np.array(pos["pa"]); r["colour"]["model_disp"] = md.tolist(); r["colour"]["hit"] = {str(t): bool(np.linalg.norm(md - td) <= t) for t in TOLS}
            nt, nm = np.linalg.norm(td), np.linalg.norm(md); r["colour"]["mag_ratio"] = float(nm / nt) if nt > 0 else None
            r["colour"]["dir_err_deg"] = float(np.degrees(np.arccos(np.clip(td @ md / (nt * nm), -1, 1)))) if nt > 0 and nm > 0 else None
        # template locator: the object's patch from the TRUE frame of the same arm at the same delay
        mA = (np.abs(RA[i].astype(int) - sig).sum(-1) <= 40) & region; mB = (np.abs(RB[i].astype(int) - sig).sum(-1) <= 40) & region
        tA, tB = patch(RA[i], mA), patch(RB[i], mB); tm = {}
        if tA is not None and tB is not None:
            (ca, sa), (cb, sb) = template_locate(up(PA[i]), tA, region), template_locate(up(PB[i]), tB, region)
            tm = {"score_a": sa, "score_b": sb, "present": bool(ca is not None and cb is not None and sa >= TM_THR and sb >= TM_THR), "template_shape": [list(tA.shape[:2]), list(tB.shape[:2])]}
            if tm["present"]:
                md = np.array(cb) - np.array(ca); tm["model_disp"] = md.tolist(); tm["hit"] = {str(t): bool(np.linalg.norm(md - td) <= t) for t in TOLS}
                nt, nm = np.linalg.norm(td), np.linalg.norm(md); tm["mag_ratio"] = float(nm / nt) if nt > 0 else None
                tm["dir_err_deg"] = float(np.degrees(np.arccos(np.clip(td @ md / (nt * nm), -1, 1)))) if nt > 0 and nm > 0 else None
            # self-check: the template must find the object in its own true frame (sanity of the locator itself)
            (cs, ss) = template_locate(RB[i], tB, region); tm["self_score"] = ss
        r["template"] = tm
        rec["delays"][str(d)] = r
        if d == 20:                                                                                   # blinded panel
            pid = hashlib.md5(f"{ARCH}-{G}-{ep}".encode()).hexdigest()[:10]
            tiles = [RA[i], RB[i], up(PA[i]), up(PB[i])]; ref = RB[K + d0].copy(); ys, xs = np.nonzero(mask0)
            if len(ys): cv2.rectangle(ref, (int(xs.min()) - 2, int(ys.min()) - 2), (int(xs.max()) + 2, int(ys.max()) + 2), (255, 255, 0), 1)
            tiles.append(ref); canvas = np.concatenate([cv2.resize(t_, (160 * 3, 210 * 3), interpolation=cv2.INTER_NEAREST) for t_ in tiles], axis=1)
            cv2.imwrite(str(PAN / f"{pid}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)); key[pid] = {"arch": ARCH, "game": G, "episode": ep, "d0": d0, "tiles": "true a | true b | generated a | generated b | true frame at first difference with the actuated object boxed"}
    raw.append(rec)
name = f"objaudit{TAG}_{ARCH}_{G}{'_DRY' if DRY else ''}"
json.dump({"experiment": "object_audit_v23", "arch": ARCH, "game": GAME, "n_episodes": N_EP, "delays": DELAYS, "template_threshold": TM_THR, "raw": raw, "panel_key": key}, open(OUT / f"{name}.json", "w"), indent=1)
pc = [r["delays"]["20"]["colour"]["present"] for r in raw if "20" in r["delays"]]; pt = [r["delays"]["20"]["template"].get("present") for r in raw if "20" in r["delays"] and r["delays"]["20"]["template"]]
print(f"{ARCH} {G}: {len(raw)} episodes | d20 colour present {np.mean(pc) if pc else float('nan'):.2f} | template present {np.mean([x for x in pt if x is not None]) if pt else float('nan'):.2f} | self-score median {np.median([r['delays']['20']['template'].get('self_score', np.nan) for r in raw if '20' in r['delays'] and r['delays']['20']['template']]):.2f}")
print("wrote", OUT / f"{name}.json")
