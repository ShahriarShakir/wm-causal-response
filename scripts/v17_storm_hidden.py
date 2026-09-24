"""STORM latent probe WITH the frames-only control built in. Hidden state = the transformer feature dist_feat
(512-d) at each imagined step; the reference is the same feature computed by teacher-forcing the TRUE frames' posterior latents with the
same actions (full), with the actions of arm a (frames-only), and with the true frames of arm a but the actions of arm b (actions-only).
Alignment of the imagined hidden change with each reference; share = ||frames-only|| / ||full||. Usage: v17_storm_hidden.py ALE/Game-v5 k w [--dry]"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
import numpy as np, torch, gymnasium as gym, ale_py
from ccaus.storm_harness import load_storm, to_model, _reset_cache, CONTEXT
from ccaus.joint import measure_pair, bootstrap_median
from ccaus.repr_probe import repr_alignment
import env_wrapper as _sw
GAME, OFFSET, WARMUP = sys.argv[1], int(sys.argv[2]), max(0, int(sys.argv[3]) - (CONTEXT - 4)); DRY = "--dry" in sys.argv
G = GAME.split("/")[-1].replace("-v5", ""); OUT = ROOT / "outputs"; DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_EP = 2 if DRY else 30; SEEDS = [42] if DRY else [42, 7, 128, 256, 512, 1024, 2024]; HORIZON, K, DUR = 24, 3, 8; DELAYS = [1, 2, 3, 5, 8, 12, 16, 20]
def emu(seed, actions):
    e = _sw.MaxLast2FrameSkipWrapper(gym.make(GAME, frameskip=1, repeat_action_probability=0.0, full_action_space=False), skip=4); obs, _ = e.reset(seed=seed); wr = np.random.default_rng(seed + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, e.action_space.n)))
    fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)
import cv2
to64 = lambda fr: np.stack([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1
flat = lambda x: x.reshape(x.shape[0], -1)
n_act = gym.make(GAME).action_space.n; wm = load_storm(f"<STORM_SRC>/ckpt/{G}-life_done-wm_2L512D8H-100k-seed1/world_model_100000.pth", num_actions=n_act, device=DEV)
def imagine_hidden(ctx, actions, seed):
    """imagined frames [H,64,64,3] in [-1,1] and hidden features [H,512] under CRN."""
    with torch.no_grad():
        obs = to_model(ctx, DEV); _reset_cache(wm, DEV); torch.manual_seed(seed * 100003 - 1); lat = wm.encode_obs(obs); last = None
        for i in range(obs.shape[1]): _, _, _, last, feat = wm.predict_next(lat[:, i:i + 1], torch.as_tensor([[int(actions[i])]], device=DEV), log_video=False)
        fr, hs = [], []
        for t in range(HORIZON):                                      # registration fixed 2026-09-12: fr[t] is frame CONTEXT+t (see ccaus/storm_harness.py)
            if t > 0: torch.manual_seed(seed * 100003 + t); _, _, _, last, feat = wm.predict_next(last, torch.as_tensor([[int(actions[CONTEXT + t - 1])]], device=DEV), log_video=False)
            o = wm.image_decoder(last); fr.append(o[0, 0].float().cpu().numpy().transpose(1, 2, 0) * 2 - 1); hs.append(feat[0, 0].float().cpu().numpy())
    return np.stack(fr), np.stack(hs)
def true_hidden(frames_uint8, actions, seed):
    """teacher-forced: posterior latents of the TRUE frames (context + horizon), transformer features with the given actions -> [H,512]."""
    with torch.no_grad():
        obs = to_model(frames_uint8, DEV); _reset_cache(wm, DEV); torch.manual_seed(seed * 100003 - 1); lat = wm.encode_obs(obs); hs = []
        for i in range(obs.shape[1] - 1):
            _, _, _, _, feat = wm.predict_next(lat[:, i:i + 1], torch.as_tensor([[int(actions[i])]], device=DEV), log_video=False)
            if i >= CONTEXT - 1: hs.append(feat[0, 0].float().cpu().numpy())
    return np.stack(hs)[:HORIZON]
res = {}
for seed in SEEDS:
    raw = []
    for ep in range(N_EP):
        r = np.random.default_rng(seed * 1000 + ep); a = r.integers(0, n_act, CONTEXT + HORIZON); b = a.copy()
        for t in range(CONTEXT + K, CONTEXT + K + DUR):
            b[t] = (a[t] + OFFSET) % n_act
            if b[t] == a[t]: b[t] = (a[t] + 1) % n_act
        ctx = emu(ep, a[:CONTEXT - 1])[:CONTEXT]; TA = emu(ep, a)[:CONTEXT + HORIZON]; TB = emu(ep, b)[:CONTEXT + HORIZON]
        ta, tb = to64(TA[CONTEXT:]), to64(TB[CONTEXT:])
        pa, za = imagine_hidden(ctx, a, ep); pb, zb = imagine_hidden(ctx, b, ep)
        zta, ztb = true_hidden(TA, a, ep), true_hidden(TB, b, ep); ztb_A = true_hidden(TB, a, ep); zta_B = true_hidden(TA, b, ep)
        pm = measure_pair(flat(pa), flat(pb), flat(ta), flat(tb), flat(to64(ctx)), K, DELAYS)
        rec = {"pixel_align": {}, "full_align": {}, "frames_only_align": {}, "actions_only_align": {}, "share": {}, "true_div": {}}
        for d in DELAYS:
            i = K + d; rec["pixel_align"][str(d)] = pm.causal_alignment.get(d); rec["true_div"][str(d)] = pm.true_divergence.get(d)
            rec["full_align"][str(d)] = repr_alignment(za[i], zb[i], zta[i], ztb[i]); rec["frames_only_align"][str(d)] = repr_alignment(za[i], zb[i], zta[i], ztb_A[i]); rec["actions_only_align"][str(d)] = repr_alignment(za[i], zb[i], zta[i], zta_B[i])
            nu = float(np.linalg.norm(ztb[i] - zta[i])); rec["share"][str(d)] = float(np.linalg.norm(ztb_A[i] - zta[i]) / nu) if nu > 0 else None
        raw.append(rec)
    agg = {k: {str(d): bootstrap_median([x[k][str(d)] for x in raw if x[k].get(str(d)) is not None and x[k][str(d)] == x[k][str(d)]]).as_dict() for d in DELAYS} for k in raw[0]}
    res[str(seed)] = {"aggregate": agg, "raw": raw}
    print(f"seed {seed}: " + " ".join(f"d{d}: pix {agg['pixel_align'][str(d)]['median']:.2f} full {agg['full_align'][str(d)]['median']:.2f} fo {agg['frames_only_align'][str(d)]['median']:.2f} ao {agg['actions_only_align'][str(d)]['median']:.2f} share {agg['share'][str(d)]['median']:.2f}" for d in (1, 3, 8, 20)), flush=True)
name = f"stormhidden_{G}_k{OFFSET}_w{WARMUP}" + ("_DRY" if DRY else "")
json.dump({"experiment": "storm_hidden_probe_with_controls", "game": GAME, "seeds": SEEDS, "n_episodes": N_EP, "delays": DELAYS, "results": res}, open(OUT / f"{name}.json", "w")); print(f"wrote outputs/{name}.json")
