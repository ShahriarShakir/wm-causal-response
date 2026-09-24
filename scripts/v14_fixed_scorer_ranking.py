"""A controlled design: planner, reward information and evaluation states held FIXED.
A reward scorer is trained on EMULATOR frame pairs (independent of every model), then applied to each model's
predicted frames under K candidate action sequences; candidates are ranked by scored predicted return and compared
with the emulator's replayed return. A ceiling (the scorer applied to the TRUE frames) says how much of any failure
is the scorer's rather than the model's. Ranking quality then depends only on whether the predicted frames respond to
the action the way the world does, which is what causal alignment measures and PSNR / dPSNR do not.
Usage: v14_fixed_scorer_ranking.py {diamond|iris} ALE/Alien-v5 [--dry]"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch, torch.nn as nn, cv2, gymnasium as gym, ale_py
from ccaus.joint import bootstrap_median
from ccaus.validate import spearman
ARCH, GAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv
G = GAME.split("/")[-1].replace("-v5", ""); OUT = ROOT / "outputs"
SEARCH = json.load(open(OUT / "emulator_intervention_search.json")); WARMUP = int(SEARCH[G]["best_warmup"]) if G in SEARCH else 0
CONTEXT, IMG = 4, 64; DEV = "cuda" if torch.cuda.is_available() else "cpu"
DENSE = "--densedata" in sys.argv                                                    # reward-sparse games: 3x more scorer data, all episodes hold an action, 60-step episodes
HORIZON, K, N_CTX, N_DATA = (6, 3, 2, 6) if DRY else (40, 8, 20, (600 if DENSE else 200)); SEEDS = [42] if DRY else [42, 7, 128]
torch.manual_seed(0); np.random.seed(0)

if ARCH == "storm":                                                  # STORM's observation footing (max of the last two of 4 repeated frames), sticky actions off
    sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
    import env_wrapper as _sw
    def _make_env(seed):
        e = gym.make(GAME, frameskip=1, repeat_action_probability=0.0, full_action_space=False); e = _sw.MaxLast2FrameSkipWrapper(e, skip=4); return e
else:
    def _make_env(seed): return gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
def emu_rew(seed, actions):
    e = _make_env(seed); obs, _ = e.reset(seed=seed); wr = np.random.default_rng(seed + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, e.action_space.n)))
    fr, rw = [np.asarray(obs)], []
    for a in actions: obs, r, *_ = e.step(int(a)); fr.append(np.asarray(obs)); rw.append(float(r))
    e.close(); return np.stack(fr), np.asarray(rw)
def to64(fr): return (np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2.0 - 1.0)
env = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); n_act = env.action_space.n; env.close()
# ---------------- 1. a model-independent reward scorer on emulator frame pairs ----------------
class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(6, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(), nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.SiLU(), nn.Linear(256, 3))
    def forward(self, prev, nxt): return self.net(torch.cat([prev, nxt], 1))
X_prev, X_next, Y = [], [], []
rng = np.random.default_rng(12345)
for ep in range(N_DATA):
    acts = rng.integers(0, n_act, (60 if DENSE else HORIZON) + CONTEXT)
    if DENSE or ep % 2: acts[CONTEXT:] = rng.integers(0, n_act)                     # held actions, like the candidates (all episodes when --densedata)
    fr, rw = emu_rew(100000 + ep, acts); f = to64(fr)
    X_prev.append(f[:-1]); X_next.append(f[1:]); Y.append(np.sign(rw).astype(np.int64) + 1)
X_prev, X_next, Y = np.concatenate(X_prev), np.concatenate(X_next), np.concatenate(Y)
counts = np.bincount(Y, minlength=3); majority = counts.max() / counts.sum()
perm = rng.permutation(len(Y)); n_tr = int(0.8 * len(Y)); tr, te = perm[:n_tr], perm[n_tr:]
SCORER_DIR = OUT / "scorers"; SCORER_DIR.mkdir(exist_ok=True); SCK = SCORER_DIR / f"{G}{'_dense' if DENSE else ''}.pt"
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True, warn_only=True)
scorer = Scorer().to(DEV); opt = torch.optim.Adam(scorer.parameters(), 1e-3)
if SCK.exists() and not DRY:                                            # FROZEN per-game scorer: identical instrument for every architecture
    scorer.load_state_dict(torch.load(SCK, map_location=DEV)); print(f"{G}: loaded frozen scorer {SCK.name}", flush=True); FROZEN = True
else: FROZEN = False
w = torch.tensor(counts.sum() / (3.0 * np.maximum(counts, 1)), dtype=torch.float32, device=DEV)
tp, tn, ty = (torch.as_tensor(X_prev).permute(0, 3, 1, 2), torch.as_tensor(X_next).permute(0, 3, 1, 2), torch.as_tensor(Y))
for epoch in range(0 if FROZEN else (2 if DRY else 8)):
    scorer.train(); order = torch.as_tensor(rng.permutation(tr))
    for i in range(0, len(order), 128):
        idx = order[i:i + 128]; loss = nn.functional.cross_entropy(scorer(tp[idx].to(DEV), tn[idx].to(DEV)), ty[idx].to(DEV), weight=w); opt.zero_grad(); loss.backward(); opt.step()
scorer.eval()
if not FROZEN and not DRY: torch.save(scorer.state_dict(), SCK)
with torch.no_grad():
    pred = torch.cat([scorer(tp[te[i:i + 256]].to(DEV), tn[te[i:i + 256]].to(DEV)).argmax(1).cpu() for i in range(0, len(te), 256)])
acc = float((pred == ty[te]).float().mean()); bal = float(np.mean([float((pred[ty[te] == c] == c).float().mean()) for c in range(3) if (ty[te] == c).any()]))
gate = bool(bal >= 0.5 and counts[[0, 2]].sum() >= 50) or ("--force-gate" in sys.argv)   # scorer must beat chance on balanced accuracy and rewards must exist; --force-gate only for DRY runs
print(f"{G}: scorer data {len(Y)} pairs, reward classes {counts.tolist()}, held-out acc {acc:.3f} (majority {majority:.3f}), balanced acc {bal:.3f} -> gate {'PASS' if gate else 'FAIL'}", flush=True)
def scored_return(frames):                                                        # frames: [1+H, 64, 64, 3] in [-1,1] (context last frame + H predicted)
    with torch.no_grad():
        f = torch.as_tensor(np.asarray(frames, np.float32)).permute(0, 3, 1, 2).to(DEV)
        e = torch.softmax(scorer(f[:-1], f[1:]), -1) @ torch.tensor([-1.0, 0.0, 1.0], device=DEV)
    return float(e.sum().cpu())
# ---------------- 2. the models ----------------
if ARCH == "diamond":
    from ccaus.diamond_harness import load_diamond, imagine
    sampler = load_diamond(str(ROOT / f"models/{G}.pt"), num_actions=n_act, device=DEV)
    def predict(ctx, acts_full, seed): return imagine(sampler, ctx, acts_full, HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
elif ARCH == "storm":
    from ccaus.storm_harness import load_storm, imagine as storm_imagine, CONTEXT as SCTX
    swm = load_storm(f"<STORM_SRC>/ckpt/{G}-life_done-wm_2L512D8H-100k-seed1/world_model_100000.pth", num_actions=n_act, device=DEV)
    CONTEXT = SCTX; WARMUP = max(0, WARMUP - (SCTX - 4))   # same absolute emulator time for the intervention as with a 4-frame context
    def predict(ctx, acts_full, seed): return storm_imagine(swm, ctx, acts_full, HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
else:
    from ccaus.iris_harness import load_iris, imagine
    tok, wm = load_iris(str(ROOT / f"models_iris/{G}.pt"), num_actions=n_act, device=DEV)
    def predict(ctx, acts_full, seed): return imagine(tok, wm, ctx[-1], acts_full[CONTEXT - 1:], HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
def regret(gm, ge):
    ge = np.asarray(ge, float); top = ge[int(np.argmax(gm))]; return float((ge.max() - top) / (ge.max() - ge.mean())) if ge.max() > ge.mean() else None
results = {}
if gate:
    for seed in SEEDS:
        per_ctx = []
        for ep in range(N_CTX):
            r = np.random.default_rng(seed * 1000 + ep); ctx_acts = r.integers(0, n_act, CONTEXT - 1)
            ctx_raw = emu_rew(ep, ctx_acts)[0][:CONTEXT]; ctx64 = to64(ctx_raw)
            cands = r.integers(0, n_act, (K, HORIZON + 1))
            for k in range(K // 2): cands[k, :] = r.integers(0, n_act)                  # half held, half random
            gm, ge, gc = [], [], []
            for k in range(K):
                acts_full = np.concatenate([ctx_acts, cands[k]])
                fr, rw = emu_rew(ep, acts_full); ge.append(float(np.sign(rw[CONTEXT - 1:CONTEXT - 1 + HORIZON]).sum()))
                true64 = to64(fr[CONTEXT - 1:CONTEXT + HORIZON]); gc.append(scored_return(true64))                      # ceiling: scorer on TRUE frames
                p = predict(ctx_raw, acts_full, seed=ep); gm.append(scored_return(np.concatenate([ctx64[-1:], p])))       # model: scorer on PREDICTED frames
            inf = len(set(ge)) > 1
            per_ctx.append({"model_return": gm, "emu_return": ge, "ceiling_return": gc, "informative": inf,
                            "spearman": spearman(gm, ge) if inf else None, "spearman_ceiling": spearman(gc, ge) if inf else None,
                            "regret": regret(gm, ge) if inf else None, "regret_ceiling": regret(gc, ge) if inf else None,
                            "hit2": (bool(ge[int(np.argmax(gm))] >= sorted(ge)[-2]) if inf else None)})
            if (ep + 1) % 5 == 0 or DRY: print(f"  seed {seed}: {ep+1}/{N_CTX}", flush=True)
        inf = [c for c in per_ctx if c["informative"]]
        agg = {"n_informative": len(inf), "n_ctx": len(per_ctx),
               "spearman": bootstrap_median([c["spearman"] for c in inf]).as_dict() if inf else None,
               "spearman_ceiling": bootstrap_median([c["spearman_ceiling"] for c in inf]).as_dict() if inf else None,
               "regret": bootstrap_median([c["regret"] for c in inf if c["regret"] is not None]).as_dict() if inf else None,
               "hit2": float(np.mean([c["hit2"] for c in inf])) if inf else None}
        results[str(seed)] = {"aggregate": agg, "raw": per_ctx}
        print(f"{ARCH} {G} seed {seed}: informative {agg['n_informative']}/{agg['n_ctx']}  rho model {agg['spearman']['median'] if agg['spearman'] else None}  rho ceiling {agg['spearman_ceiling']['median'] if agg['spearman_ceiling'] else None}  regret {agg['regret']['median'] if agg['regret'] else None}", flush=True)
name = f"fsrank2_{ARCH}_{G}" + ("_dense" if DENSE else "") + ("_DRY" if DRY else "")
json.dump({"experiment": "fixed_scorer_ranking_v14", "frozen_scorer": SCK.name, "arch": ARCH, "game": GAME, "seeds": SEEDS, "n_ctx": N_CTX, "K": K, "horizon": HORIZON, "warmup": WARMUP,
           "scorer": {"n_pairs": int(len(Y)), "classes": counts.tolist(), "held_out_acc": acc, "majority": majority, "balanced_acc": bal, "gate": gate}, "results": results}, open(OUT / f"{name}.json", "w"))
print(f"wrote outputs/{name}.json")
