"""Registration test (it exposed a one-step offset in an earlier version of the IRIS and STORM harnesses): intervene on ONE
action at index CONTEXT+K; the emulator's first differing frame is index K+1 and every harness must first differ at the same
index. Needs the model checkpoints and the emulator (PYTHONPATH with .tools); runs on CPU in about two minutes per architecture.
Usage: test_registration.py {diamond|iris|storm} ALE/Pong-v5   (exit 1 on failure)"""
import sys, pathlib, numpy as np, torch, cv2, gymnasium as gym, ale_py
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, "<STORM_SRC>"); sys.path.insert(0, "<STORM_PKGS>")
import os; os.chdir(ROOT)
ARCH, GAME = sys.argv[1], sys.argv[2]; G = GAME.split("/")[-1].replace("-v5", ""); DEV = "cpu"; H = 6; K = 3
def emu(seed, actions):
    e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); obs, _ = e.reset(seed=seed); fr = [np.asarray(obs)]
    for a in actions: obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)
e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0); n_act = e.action_space.n; e.close()
if ARCH == "iris":
    from ccaus.iris_harness import load_iris, imagine; tok, wm = load_iris(f"models_iris/{G}.pt", num_actions=n_act, device=DEV); C = 4
    pred = lambda ctx, acts, ep: imagine(tok, wm, ctx[-1], acts[C - 1:], H, seed=ep, device=DEV)
elif ARCH == "storm":
    from ccaus.storm_harness import load_storm, imagine; swm = load_storm(f"<STORM_SRC>/ckpt/{G}-life_done-wm_2L512D8H-100k-seed1/world_model_100000.pth", num_actions=n_act, device=DEV); C = 8
    pred = lambda ctx, acts, ep: imagine(swm, ctx, acts, H, seed=ep, device=DEV)
else:
    from ccaus.diamond_harness import load_diamond, imagine; sampler = load_diamond(f"models/{G}.pt", num_actions=n_act, device=DEV); C = 4
    pred = lambda ctx, acts, ep: imagine(sampler, ctx, acts, H, seed=ep, device=DEV)
fp, ft = [], []
for ep in range(3):
    rng = np.random.default_rng(1000 + ep); acts = rng.integers(0, n_act, C + H); b = acts.copy(); b[C + K] = (acts[C + K] + 1) % n_act
    ctx = emu(ep, acts[:C - 1])[:C]
    ta, tb = emu(ep, acts)[C:C + H], emu(ep, b)[C:C + H]
    pa, pb = pred(ctx, acts, ep), pred(ctx, b, ep)
    dp = [float(np.abs(pa[j] - pb[j]).max()) for j in range(H)]; dt = [float(np.abs(ta[j].astype(float) - tb[j]).max()) for j in range(H)]
    fp.append(next((j for j in range(H) if dp[j] > 1e-5), None)); ft.append(next((j for j in range(H) if dt[j] > 0), None))
ok = all(x == K + 1 for x in fp)
print(f"{ARCH} {G}: intervention at action index C+{K}; first differing PREDICTED index per episode {fp}; TRUE {ft}; {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
