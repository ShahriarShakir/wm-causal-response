"""Does the world model rank candidate action sequences the way the emulator does?
Fixed contexts, K candidate action sequences per context, the SAME model seed for every candidate (common random
numbers), model-predicted return from the model's own reward head vs the emulator's replayed return (both sign-clipped,
which is what both reward heads predict). Per context: Spearman between the two returns across candidates, top-1
regret, hit@2. Per game the medians are later correlated with causal alignment, PSNR and dPSNR (scripts/ranking_analysis.py).
Usage: v11_action_ranking.py {diamond|iris} ALE/Alien-v5 [--dry]"""
import json, pathlib, sys, os
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch, cv2, gymnasium as gym, ale_py
from ccaus.joint import bootstrap_median
from ccaus.validate import spearman
ARCH, GAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv
G = GAME.split("/")[-1].replace("-v5", "")
SEARCH = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json"))
WARMUP = int(SEARCH[G]["best_warmup"]) if G in SEARCH else 0
CONTEXT, IMG = 4, 64
LONG = "--long" in sys.argv
HORIZON, K, N_CTX = (6, 3, 2) if DRY else ((40, 8, 20) if LONG else (20, 8, 20)); SEEDS = [42] if DRY else [42, 7, 128]
DEV = "cuda" if torch.cuda.is_available() else "cpu"; OUT = ROOT / "outputs"
def emu_rew(seed, actions):
    e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); obs, _ = e.reset(seed=seed)
    _wr = np.random.default_rng(seed + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(_wr.integers(0, e.action_space.n)))
    fr, rw = [np.asarray(obs)], []
    for a in actions:
        obs, r, *_ = e.step(int(a)); fr.append(np.asarray(obs)); rw.append(float(r))
    e.close(); return np.stack(fr), np.asarray(rw)
def to64(fr): return (np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2.0 - 1.0)
env = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); n_act = env.action_space.n; env.close()
if ARCH == "diamond":
    from ccaus.diamond_harness import load_diamond, imagine, to_model
    from models.rew_end_model import RewEndModel, RewEndModelConfig
    CKPT = ROOT / f"models/{G}.pt"; sampler = load_diamond(str(CKPT), num_actions=n_act, device=DEV)
    rew = RewEndModel(RewEndModelConfig(lstm_dim=512, img_channels=3, img_size=IMG, cond_channels=128, depths=[2, 2, 2, 2], channels=[32, 32, 32, 32], attn_depths=[0, 0, 0, 0], num_actions=n_act))
    sd = torch.load(str(CKPT), map_location=DEV, weights_only=False)
    missing, unexpected = rew.load_state_dict({k[len("rew_end_model."):]: v for k, v in sd.items() if k.startswith("rew_end_model.")}, strict=False)
    assert not missing and not unexpected, f"rew_end_model mismatch {list(missing)[:3]} {list(unexpected)[:3]}"
    rew.eval().to(DEV)
    def model_return(ctx, acts_full, seed):
        fr = imagine(sampler, ctx, acts_full, HORIZON, seed=seed, device=DEV)                     # [H,3,64,64] in [-1,1]
        with torch.no_grad():
            c = to_model(ctx)[None].to(DEV)                                                        # [1,CONTEXT,3,64,64]
            a_ctx = torch.as_tensor(np.asarray(acts_full[:CONTEXT - 1]), dtype=torch.long, device=DEV)[None]
            _, _, hx = rew.predict_rew_end(c[:, :-1], a_ctx, c[:, 1:])                            # burn-in on the real context
            f = torch.as_tensor(fr, device=DEV)[None]
            prev = torch.cat([c[:, -1:], f[:, :-1]], dim=1); a_h = torch.as_tensor(np.asarray(acts_full[CONTEXT - 1:CONTEXT - 1 + HORIZON]), dtype=torch.long, device=DEV)[None]
            lr, _, _ = rew.predict_rew_end(prev, a_h, f, hx)
            er = (torch.softmax(lr, -1) * torch.tensor([-1.0, 0.0, 1.0], device=DEV)).sum(-1)     # expected sign-clipped reward per step
        return float(er.sum().cpu())
else:
    from ccaus.iris_harness import load_iris, to_model
    from envs.world_model_env import WorldModelEnv
    CKPT = ROOT / f"models_iris/{G}.pt"; tok, wm = load_iris(str(CKPT), num_actions=n_act, device=DEV)
    def model_return(ctx, acts_full, seed):
        envw = WorldModelEnv(tokenizer=tok, world_model=wm, device=DEV); envw.reset_from_initial_observations(to_model(ctx[-1], DEV))
        tot = 0.0
        for t in range(HORIZON):
            torch.manual_seed(seed * 100003 + t)                                                    # common random numbers across candidates
            _, r, _, _ = envw.step(int(acts_full[CONTEXT - 1 + t]), should_predict_next_obs=True); tot += float(np.asarray(r).reshape(-1)[0])
        return tot
def regret(gm, ge):
    ge = np.asarray(ge, float); top = ge[int(np.argmax(gm))]; return float((ge.max() - top) / (ge.max() - ge.mean())) if ge.max() > ge.mean() else None
results = {}
for seed in SEEDS:
    per_ctx = []
    for ep in range(N_CTX):
        rng = np.random.default_rng(seed * 1000 + ep)
        ctx_acts = rng.integers(0, n_act, CONTEXT - 1)
        ctx = emu_rew(ep, ctx_acts)[0][:CONTEXT]
        cands = rng.integers(0, n_act, (K, HORIZON + 1))
        if LONG:                                                    # half the candidates HOLD one action for the whole horizon (purposeful sequences that reach rewards)
            for k in range(K // 2): cands[k, :] = rng.integers(0, n_act)
        gm, ge = [], []
        for k in range(K):
            acts_full = np.concatenate([ctx_acts, cands[k]])
            _, r = emu_rew(ep, acts_full); ge.append(float(np.sign(r[CONTEXT - 1:CONTEXT - 1 + HORIZON]).sum()))
            gm.append(model_return(ctx, acts_full, seed=ep))
        rho = spearman(gm, ge) if len(set(ge)) > 1 else None
        per_ctx.append({"model_return": gm, "emu_return": ge, "spearman": rho, "regret": regret(gm, ge), "hit2": (None if len(set(ge)) < 2 else bool(ge[int(np.argmax(gm))] >= sorted(ge)[-2])), "informative": len(set(ge)) > 1})
        if (ep + 1) % 5 == 0 or DRY: print(f"  seed {seed}: {ep+1}/{N_CTX} rho {rho}", flush=True)
    inf = [c for c in per_ctx if c["informative"]]
    agg = {"n_informative": len(inf), "n_ctx": len(per_ctx),
           "spearman": bootstrap_median([c["spearman"] for c in inf]).as_dict() if inf else None,
           "regret": bootstrap_median([c["regret"] for c in inf if c["regret"] is not None]).as_dict() if inf else None,
           "hit2": (float(np.mean([c["hit2"] for c in inf])) if inf else None)}
    results[str(seed)] = {"aggregate": agg, "raw": per_ctx}
    print(f"{ARCH} {G} seed {seed}: informative {agg['n_informative']}/{agg['n_ctx']}  rho {agg['spearman']['median'] if agg['spearman'] else None}  regret {agg['regret']['median'] if agg['regret'] else None}  hit2 {agg['hit2']}", flush=True)
name = f"ranking_{ARCH}_{G}" + ("_h40" if LONG else "") + ("_DRY" if DRY else "")
json.dump({"experiment": "action_ranking_v11", "arch": ARCH, "game": GAME, "seeds": SEEDS, "n_ctx": N_CTX, "K": K, "horizon": HORIZON, "warmup": WARMUP, "held_candidates": (K // 2 if LONG else 0), "results": results}, open(OUT / f"{name}.json", "w"))
print(f"wrote outputs/{name}.json")
