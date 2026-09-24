"""POLICY-CONTEXT CONTROL (added 2026-09-12 after the panel audit). Same paired measurement as v4/v5, but the warm-up, the context and
the base action sequence are produced by the released agent's OWN policy acting in the real emulator (epsilon-random with EPS), instead
of uniformly random actions. The intervention still replaces the window's actions by the per-game offset, both arms are replayed
exactly, and the model imagines both under common random numbers with the corrected registration. The question it answers: is
causal blindness an artefact of random-action contexts that lie outside the agent's state distribution?
Usage: v22_policy_context.py {diamond|iris} ALE/Game-v5 N_EP DUR OFFSET WARMUP [EPS]  -> outputs/policy_{arch}_{G}-v5_dur{DUR}_k{k}_w{w}_eps{EPS}.json"""
import json, os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / ".tools"))
import numpy as np, torch, cv2, gymnasium as gym, ale_py
from ccaus.joint import Guard, aggregate, decompose_response, measure_pair, verdict
gym.register_envs(ale_py)
ARCH, GAME = sys.argv[1], sys.argv[2]; G = GAME.split("/")[-1].replace("-v5", ""); DRY = "--dry" in sys.argv
N_EP = int(sys.argv[3]); DUR = int(sys.argv[4]); OFFSET = int(sys.argv[5]); WARMUP = int(sys.argv[6]); EPS = float(sys.argv[7]) if len(sys.argv) > 7 and not sys.argv[7].startswith("--") else 0.0
HORIZON, K, IMG = int(os.environ.get("HORIZON", 24)), 3, 64
SEEDS = [42] if DRY else [42, 7, 128, 256, 512, 1024, 2024]
if DRY: N_EP = 2
DEV = "cuda" if torch.cuda.is_available() else "cpu"; OUT = ROOT / "outputs"
DELAYS = [d for d in (1, 2, 3, 5, 8, 12, 16, 20, 28, 36, 44) if K + d < HORIZON]
FIELDS = ("actuated_mag", "actuated_align", "consequence_mag", "consequence_align", "consequence_fraction")
e0 = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); n_act = e0.action_space.n; e0.close()
if ARCH == "diamond":
    from ccaus.diamond_harness import load_diamond, imagine, CONTEXT; from ccaus.policies import DiamondPolicy
    sampler = load_diamond(str(ROOT / f"models/{G}.pt"), num_actions=n_act, device=DEV); pol = DiamondPolicy(str(ROOT / f"models/{G}.pt"), n_act, DEV)
    predict = lambda ctx, acts, seed: imagine(sampler, ctx, acts, HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
else:
    from ccaus.iris_harness import load_iris, imagine; from ccaus.policies import IrisPolicy
    CONTEXT = 4; tok, wm = load_iris(str(ROOT / f"models_iris/{G}.pt"), num_actions=n_act, device=DEV); pol = IrisPolicy(str(ROOT / f"models_iris/{G}.pt"), tok, n_act, DEV)
    predict = lambda ctx, acts, seed: imagine(tok, wm, ctx[-1], acts[CONTEXT - 1:], HORIZON, seed=seed, device=DEV).transpose(0, 2, 3, 1)
def to64(fr): return np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2.0 - 1.0
flat = lambda x: x.reshape(x.shape[0], -1)
def agg_dec(decs, delays):
    out = {}
    for d in delays:
        vals = [x[d] for x in decs if d in x]; out[str(d)] = {f: (float(np.nanmedian([v[f] for v in vals])) if vals else float("nan")) for f in FIELDS}
    return out
def policy_rollout(seed, ep):
    """Warm-up, context and horizon actions all from the policy (epsilon-random), acting on the real emulator. Returns the
    post-warm-up frames [CONTEXT+HORIZON+1] and the actions [CONTEXT+HORIZON] it took; the warm-up actions are stored for replay."""
    e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); obs, _ = e.reset(seed=ep); rng = np.random.default_rng(seed * 1000 + ep); pol.reset()
    warm = []
    for _ in range(WARMUP):
        a = pol.act(np.asarray(obs), rng, EPS); obs, *_ = e.step(a); warm.append(a)
    fr, acts = [np.asarray(obs)], []
    for _ in range(CONTEXT + HORIZON):
        a = pol.act(np.asarray(obs), rng, EPS); obs, r, term, trunc, _ = e.step(a); fr.append(np.asarray(obs)); acts.append(a)
        if term or trunc: e.close(); return None, None, None
    e.close(); return np.stack(fr), np.asarray(acts), np.asarray(warm)
def replay(ep, warm, actions):
    e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); obs, _ = e.reset(seed=ep)
    for a in warm: obs, *_ = e.step(int(a))
    fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)
per_seed, audit = {}, {}
for seed in SEEDS:
    pairs, decs, leak, skipped, agree = [], [], 0.0, 0, []
    for ep in range(N_EP):
        fr_a, acts, warm = policy_rollout(seed, ep)
        if fr_a is None: skipped += 1; continue
        b = acts.copy()
        for _t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(acts))):
            b[_t] = (acts[_t] + OFFSET) % n_act
            if b[_t] == acts[_t]: b[_t] = (acts[_t] + 1) % n_act
        assert np.array_equal(replay(ep, warm, acts), fr_a), "policy rollout not replayable"          # exact emulator replay of the policy's own trajectory
        fr_b = replay(ep, warm, b)
        ctx = fr_a[:CONTEXT]; ta = to64(fr_a[CONTEXT:CONTEXT + HORIZON]); tb = to64(fr_b[CONTEXT:CONTEXT + HORIZON])
        pa = predict(ctx, acts, seed=ep); pb = predict(ctx, b, seed=ep)
        leak = max(leak, float(np.abs(flat(pa)[:K] - flat(pb)[:K]).max()))
        pairs.append(measure_pair(flat(pa), flat(pb), flat(ta), flat(tb), flat(to64(ctx)), K, DELAYS))
        decs.append(decompose_response(flat(pa), flat(pb), flat(ta), flat(tb), K, DELAYS))
        agree.append(float(np.mean(acts == np.bincount(acts, minlength=n_act).argmax())))         # how peaked the policy's action use is
        if (ep + 1) % 10 == 0: print(f"  seed {seed}: {ep+1}/{N_EP}", flush=True)
    raw = [{k: {str(d): getattr(p, k).get(d) for d in DELAYS} for k in ("consistency", "causal_magnitude", "causal_alignment", "true_divergence", "prediction_psnr", "model_divergence")} for p in pairs]
    agg = aggregate(pairs, DELAYS, Guard()); v = verdict(agg, DELAYS)
    per_seed[seed] = {"aggregate": agg, "verdict": v, "raw": raw, "decomposition": agg_dec(decs, DELAYS), "n_skipped_terminal": skipped, "policy_mode_share": float(np.mean(agree)) if agree else None}
    audit[seed] = {"max_pre_intervention_diff": leak}
    print(f"seed {seed}: {len(pairs)} pairs, {skipped} skipped (episode ended), leak {leak:.3g}, guard kept {agg['guard']['n_kept']} dropped {agg['guard']['n_dropped']} -> {v['verdict']}", flush=True)
print("delay | " + " ".join(f"s{s}" for s in SEEDS) + " | true div")
for d in DELAYS:
    print(f"{d:>5} | " + " ".join(f"{per_seed[s]['aggregate']['rows'][str(d)]['causal_alignment']['median']:.3f}" for s in SEEDS) + f" | {np.mean([per_seed[s]['aggregate']['rows'][str(d)]['true_divergence']['median'] for s in SEEDS]):.3f}")
name = f"policy_{ARCH}_{G}-v5_dur{DUR}_k{OFFSET}_w{WARMUP}_eps{EPS:g}" + ("_DRY" if DRY else "")
json.dump({"experiment": "policy_context_control", "arch": ARCH, "game": GAME, "epsilon": EPS, "n_episodes": N_EP, "seeds": SEEDS, "horizon": HORIZON, "intervention_step": K, "delays": DELAYS, "intervention_duration": DUR, "action_offset": OFFSET, "warmup_steps": WARMUP, "audit": audit, "results": {str(s): per_seed[s] for s in SEEDS}}, open(OUT / f"{name}.json", "w"), indent=1)
print(f"wrote outputs/{name}.json")
