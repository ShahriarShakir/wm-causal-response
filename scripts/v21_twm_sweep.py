"""Core chosen-intervention sweep for TWM (fourth arm), same design as v4/v5/v16, in TWM's own observation space (64x64 grayscale, last frame
of the stack). Runs under the TWM venv. Truth = TWM's preprocessing pipeline replayed exactly (repeat_action_probability 0, no noop start).
Files: ext/outputs/twm_<Game>-v5_dur{DUR}_k{k}_w{w}.json with the same structure as the other arms (per-seed aggregate/verdict/raw + audit).
Usage: v21_twm_sweep.py ALE/Game-v5 N_EP DUR OFFSET WARMUP [--dry]"""
import json, pathlib, sys, os
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
from ccaus.twm_harness import make_env, load_twm, imagine, CONTEXT, IMG
from ccaus.joint import Guard, aggregate, measure_pair, verdict as verdict_fn, decompose_response
GAME = sys.argv[1]; G = GAME.split("/")[-1].replace("-v5", ""); N_EP = int(sys.argv[2]); DUR = int(sys.argv[3]); OFFSET = int(sys.argv[4]); WARMUP = int(sys.argv[5]); DRY = "--dry" in sys.argv
HORIZON, K = int(os.environ.get("HORIZON", 24)), 3; SEEDS = [42] if DRY else [42, 7, 128, 256, 512, 1024, 2024]
DELAYS = [d for d in (1, 2, 3, 5, 8, 12, 16, 20, 28, 36, 44) if K + d < HORIZON]; DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = pathlib.Path(os.environ.get("TWM_OUT") or (ROOT / "ext" / "outputs")); OUT.mkdir(exist_ok=True)
CKPT = os.environ.get("TWM_CKPT") or f"<SCRATCH>/twm_src/ckpt/{G}-seed1/agent_final.pt"
env0 = make_env(G, 0); n_act = env0.action_space.n; env0.close()
agent, cfg = load_twm(CKPT, n_act, device=DEV)
def emu(seed, actions):
    """replay: returns [len(actions)+1, 4, 64, 64] uint8 stacks after WARMUP random steps (same warm-up in both arms)."""
    e = make_env(G, seed); obs = e.reset(seed=seed); obs = obs[0] if isinstance(obs, tuple) else obs; wr = np.random.default_rng(seed + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, n_act)))
    st = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); st.append(np.asarray(obs))
    e.close(); return np.stack(st)
last = lambda stacks: stacks[:, -1].astype(np.float32) / 255.0 * 2.0 - 1.0          # last frame of each stack, [-1, 1]
flat = lambda x: x.reshape(x.shape[0], -1)
print(f"{GAME}: {n_act} actions, TWM ckpt {pathlib.Path(CKPT).parent.name}, device {DEV}, {N_EP} eps x {len(SEEDS)} seeds", flush=True)
per_seed, audit = {}, {}
for seed in SEEDS:
    pairs, decs, leak = [], [], 0.0
    for ep in range(N_EP):
        rng = np.random.default_rng(seed * 1000 + ep); acts = rng.integers(0, n_act, CONTEXT + HORIZON); b = acts.copy()
        for t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(acts))): b[t] = (acts[t] + OFFSET) % n_act
        stacks_a = emu(ep, acts); stacks_b = emu(ep, b)
        ctx = stacks_a[:CONTEXT]                                                     # CONTEXT stacked observations (identical in both arms)
        ta = last(stacks_a[CONTEXT:CONTEXT + HORIZON]); tb = last(stacks_b[CONTEXT:CONTEXT + HORIZON])
        pa = imagine(agent, cfg, ctx, acts, HORIZON, seed=ep, device=DEV); pb = imagine(agent, cfg, ctx, b, HORIZON, seed=ep, device=DEV)
        try:
            pm = measure_pair(flat(pa), flat(pb), flat(ta), flat(tb), flat(last(ctx)), K, DELAYS, atol=1e-5)
        except AssertionError as e:
            raise AssertionError(f"seed {seed} ep {ep}: {e}") from None
        leak = max(leak, float(np.abs(pa[:K] - pb[:K]).max())); pairs.append(pm)
        decs.append(decompose_response(flat(pa), flat(pb), flat(ta), flat(tb), K, DELAYS))
        if (ep + 1) % 10 == 0: print(f"  seed {seed}: {ep+1}/{N_EP}", flush=True)
    agg = aggregate(pairs, DELAYS, Guard()); v = verdict_fn(agg, DELAYS) if callable(verdict_fn) else None
    raw = [{"consistency": {str(d): p.consistency.get(d) for d in DELAYS}, "causal_magnitude": {str(d): p.causal_magnitude.get(d) for d in DELAYS}, "causal_alignment": {str(d): p.causal_alignment.get(d) for d in DELAYS}, "true_divergence": {str(d): p.true_divergence.get(d) for d in DELAYS}, "prediction_psnr": {str(d): p.prediction_psnr.get(d) for d in DELAYS}, "model_divergence": {str(d): p.model_divergence.get(d) for d in DELAYS}, "cross_psnr": {str(d): (p.cross_psnr.get(d) if p.cross_psnr else None) for d in DELAYS}, "intervened_psnr": {str(d): (p.intervened_psnr.get(d) if p.intervened_psnr else None) for d in DELAYS}} for p in pairs]
    per_seed[str(seed)] = {"aggregate": agg, "verdict": v, "raw": raw, "decomposition": {str(d): {f: float(np.nanmedian([x[d][f] for x in decs if d in x and x[d].get(f) is not None])) if any(d in x and x[d].get(f) is not None for x in decs) else float("nan") for f in ("actuated_mag", "actuated_align", "consequence_mag", "consequence_align", "consequence_fraction")} for d in DELAYS}}
    audit[str(seed)] = {"max_pre_intervention_diff": leak}
    print(f"seed {seed}: d20 A {agg['rows']['20']['causal_alignment']['median'] if '20' in agg['rows'] else float('nan'):.3f} PSNR {agg['rows']['20']['prediction_psnr']['median'] if '20' in agg['rows'] else float('nan'):.1f} | identity max diff {leak:.2e}", flush=True)
name = f"twm_{G}-v5_dur{DUR}{'_k' + str(OFFSET) if OFFSET != 3 else ''}{'_w' + str(WARMUP) if WARMUP else ''}{'_h' + str(HORIZON) if HORIZON != 24 else ''}{'_DRY' if DRY else ''}"
json.dump({"experiment": "twm", "game": GAME, "n_episodes": N_EP, "seeds": SEEDS, "horizon": HORIZON, "intervention_step": K, "delays": DELAYS, "intervention_duration": DUR, "action_offset": OFFSET, "warmup_steps": WARMUP, "obs_space": "64x64 grayscale, last frame of TWM's 4-stack", "audit": audit, "results": per_seed}, open(OUT / f"{name}.json", "w"))
print(f"wrote {OUT / name}.json")
