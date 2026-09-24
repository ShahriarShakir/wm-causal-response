"""IDM action-consistency (a one-rollout proxy) vs causal alignment, on the same rollouts.

Per game: (1) generate ~40k emulator transitions IN MEMORY (random policy, mixed warm-ups),
(2) train the IDM, (3) GATE: held-out accuracy on EMULATOR frames must be >= 2x the majority
baseline or the game is skipped as 'idm unusable' (a bad IDM proves nothing), (4) generate the
paired rollouts exactly as the sweeps do (same seeds, same emulator-chosen (k,w)), and score:
   idm_acc_A[d]   P(IDM reads the fed action from generated frames i-1 -> i)  -- 'video depicts the action'
   idm_delta[d]   P(IDM reads DIFFERENT actions from arm A and arm B at i)    -- 'video depicts the intervention'
   align[d]       causal alignment from measure_pair                          -- 'the change is the RIGHT one'
Files written: models_idm/idm_<game>.pt and outputs/idm_<arch>_<game>.json. Nothing else.
"""
import json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, torch, cv2, gymnasium as gym, ale_py
from ccaus.idm import train_idm, predict, IDM
from ccaus.joint import measure_pair, bootstrap_median
gym.register_envs(ale_py)
ARCH, GAME = sys.argv[1], sys.argv[2]; OFFSET, WARMUP = int(sys.argv[3]), int(sys.argv[4])
DRY = "--dry" in sys.argv
N_EP = 2 if DRY else 60; SEEDS = [42] if DRY else [42, 7, 128, 256, 512, 1024, 2024]
N_TRANS, EPOCHS = (2000, 1) if DRY else (40000, 4)
CONTEXT, HORIZON, K, DUR = 4, 24, 3, 8; DELAYS = [1, 2, 3, 5, 8, 12, 16, 20]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; MD = ROOT / "models_idm"; MD.mkdir(exist_ok=True)
G = GAME.split("/")[-1].replace("-v5", "")
to64 = lambda fr: np.stack([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in fr]).astype(np.float32) / 255.0 * 2 - 1
env = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); n_act = int(env.action_space.n)

# ---- (1) transitions in memory, random policy, mixed warm-ups (0/30/60/120) ----
rng = np.random.default_rng(123); f0, f1, acts = [], [], []
while len(acts) < N_TRANS:
    obs, _ = env.reset(seed=int(rng.integers(1e6))); w = int(rng.choice([0, 30, 60, 120]))
    for _ in range(w): obs, *_ = env.step(int(rng.integers(n_act)))
    prev = obs
    for _ in range(int(rng.integers(20, 80))):
        a = int(rng.integers(n_act)); obs, _, term, trunc, _ = env.step(a)
        f0.append(prev); f1.append(obs); acts.append(a); prev = obs
        if term or trunc or len(acts) >= N_TRANS: break
env.close()
f0, f1, acts = to64(np.stack(f0)), to64(np.stack(f1)), np.asarray(acts)
# ---- (2)+(3) train and GATE ----
idm, acc, chance = train_idm(f0, f1, acts, n_act, device=DEV, epochs=EPOCHS)
gate = acc >= 2 * chance or ("--force-gate" in sys.argv and DRY)   # --force-gate: DRY ONLY, to exercise the scoring path
print(f"IDM {G}: n_act={n_act} held-out emulator accuracy {acc:.3f} vs majority baseline {chance:.3f} -> {'PASS' if gate else 'FAIL (idm unusable)'}", flush=True)
torch.save(idm.state_dict(), MD / f"idm_{G}.pt")
if not gate:
    json.dump({"game": GAME, "arch": ARCH, "idm_acc": acc, "chance": chance, "gate": "FAIL"}, open(OUT / f"idm_{ARCH}_{G}{'_DRY' if DRY else ''}.json", "w"), indent=1); sys.exit(0)
del f0, f1

# ---- (4) paired rollouts, exactly as the sweeps ----
if ARCH == "DIAMOND":
    from ccaus.diamond_harness import load_diamond, imagine as im_d, CONTEXT as CTX
    sampler = load_diamond(f"models/{G}.pt", num_actions=n_act, device=DEV)
    gen = lambda ctx, a, ep: im_d(sampler, ctx, a, HORIZON, seed=ep, device=DEV).transpose(0, 2, 3, 1)
else:
    from ccaus.iris_harness import load_iris, imagine as im_i
    tok, wm = load_iris(f"models_iris/{G}.pt", num_actions=n_act, device=DEV)
    gen = lambda ctx, a, ep: im_i(tok, wm, ctx[-1], a[CONTEXT - 1:], HORIZON, seed=ep, device=DEV).transpose(0, 2, 3, 1)
def emu(seed, actions):
    e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); obs, _ = e.reset(seed=seed); wr = np.random.default_rng(seed + 7919)
    for _ in range(WARMUP): obs, *_ = e.step(int(wr.integers(0, e.action_space.n)))
    fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)
flat = lambda x: x.reshape(x.shape[0], -1)
res = {}
for seed in SEEDS:
    raw = []
    for ep in range(N_EP):
        r = np.random.default_rng(seed * 1000 + ep); a = r.integers(0, n_act, CONTEXT + HORIZON); b = a.copy()
        for t in range(CONTEXT + K, min(CONTEXT + K + DUR, len(a))):
            b[t] = (a[t] + OFFSET) % n_act
            if b[t] == a[t]: b[t] = (a[t] + 1) % n_act
        ctx = emu(ep, a[:CONTEXT - 1])[:CONTEXT]; ta = to64(emu(ep, a)[CONTEXT:CONTEXT + HORIZON]); tb = to64(emu(ep, b)[CONTEXT:CONTEXT + HORIZON])
        pa, pb = gen(ctx, a, ep), gen(ctx, b, ep)
        pm = measure_pair(flat(pa), flat(pb), flat(ta), flat(tb), flat(to64(ctx)), K, DELAYS)
        ra, rb = predict(idm, pa, DEV), predict(idm, pb, DEV)          # predicted action for transition i-1 -> i, index i-1
        rt = predict(idm, ta, DEV)                                        # IDM on TRUE frames (control)
        fed_a, fed_b = a[CONTEXT:CONTEXT + HORIZON], b[CONTEXT:CONTEXT + HORIZON]
        rec = {"align": {}, "idm_acc_A": {}, "idm_acc_true": {}, "idm_delta": {}, "in_window": {}}
        for d in DELAYS:
            i = K + d; j = i - 1                                            # transition j -> i is driven by fed action at j
            rec["align"][str(d)] = pm.causal_alignment.get(d)
            rec["idm_acc_A"][str(d)] = int(ra[j] == fed_a[j]); rec["idm_acc_true"][str(d)] = int(rt[j] == fed_a[j])
            rec["idm_delta"][str(d)] = int(ra[j] != rb[j]); rec["in_window"][str(d)] = int(fed_a[j] != fed_b[j])
        raw.append(rec)
    agg = {str(d): {"align": bootstrap_median([x["align"][str(d)] for x in raw]).as_dict(),
                    "idm_acc_A": float(np.mean([x["idm_acc_A"][str(d)] for x in raw])),
                    "idm_acc_true": float(np.mean([x["idm_acc_true"][str(d)] for x in raw])),
                    "idm_delta": float(np.mean([x["idm_delta"][str(d)] for x in raw])),
                    "in_window": float(np.mean([x["in_window"][str(d)] for x in raw]))} for d in DELAYS}
    res[str(seed)] = {"aggregate": agg, "raw": raw}
    print(f"  seed {seed}: d1 idm_acc_A {agg['1']['idm_acc_A']:.2f} (true frames {agg['1']['idm_acc_true']:.2f}) idm_delta {agg['1']['idm_delta']:.2f} | d20 align {agg['20']['align']['median']:.3f}", flush=True)
json.dump({"game": GAME, "arch": ARCH, "idm_acc": acc, "chance": chance, "gate": "PASS", "offset": OFFSET, "warmup": WARMUP,
           "seeds": SEEDS, "n_episodes": N_EP, "delays": DELAYS, "results": res}, open(OUT / f"idm_{ARCH}_{G}{'_DRY' if DRY else ''}.json", "w"), indent=1)
print(f"wrote outputs/idm_{ARCH}_{G}.json")
