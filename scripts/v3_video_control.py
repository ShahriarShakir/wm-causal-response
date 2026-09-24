"""VIDEO ARM, STEP 2, null control, then does DIAMOND respond to ONE flipped action?

Order is deliberate and non-negotiable: the null runs FIRST. On DMControl a stochastic model
produced a 15/255 "response" to an intervention that never happened, and had that gone
unchecked every downstream number would have been sampling noise reported as causality.
Diffusion is stochastic too, so the same trap is live here.

  NULL      : identical actions -> prediction difference must be EXACTLY 0
              (common random numbers make this achievable; without them it cannot be)
  RESPONSE  : one flipped action -> prediction must change, and change NOTHING before it
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, torch
import gymnasium as gym, ale_py
from ccaus.diamond_harness import load_diamond, imagine, CONTEXT

gym.register_envs(ale_py)
GAME, CKPT = "ALE/Pong-v5", "models/Pong.pt"
HORIZON, K, SEED = 12, 3, 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
n_act = env.action_space.n
print(f"{GAME}: {n_act} actions, device={DEV}")

def real_context(seed, actions):
    e = gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
    obs, _ = e.reset(seed=seed); fr = [np.asarray(obs)]
    for a in actions:
        obs, *_ = e.step(int(a)); fr.append(np.asarray(obs))
    e.close(); return np.stack(fr)

sampler = load_diamond(CKPT, num_actions=n_act, device=DEV)
rng = np.random.default_rng(0)
acts = rng.integers(0, n_act, CONTEXT + HORIZON)
ctx = real_context(SEED, acts[:CONTEXT - 1])[:CONTEXT]
print(f"context {ctx.shape}, horizon {HORIZON}")

print("\n1. NULL CONTROL (identical actions)")
pa = imagine(sampler, ctx, acts, HORIZON, seed=SEED, device=DEV)
pb = imagine(sampler, ctx, acts.copy(), HORIZON, seed=SEED, device=DEV)
nd = float(np.abs(pa - pb).max())
print(f"   max |A-B| = {nd:.6f}")
null_ok = nd == 0.0
print("   " + ("OK, no leakage" if null_ok else "** LEAKS: sampling noise would masquerade as causality **"))

print("\n2. RESPONSE to ONE flipped action")
b = acts.copy()
b[CONTEXT + K] = (acts[CONTEXT + K] + 3) % n_act
if b[CONTEXT + K] == acts[CONTEXT + K]: b[CONTEXT + K] = (acts[CONTEXT + K] + 1) % n_act
pc = imagine(sampler, ctx, b, HORIZON, seed=SEED, device=DEV)
per = np.abs(pa - pc).reshape(HORIZON, -1).mean(axis=1)
print(f"   action {acts[CONTEXT+K]} -> {b[CONTEXT+K]} at horizon step {K}")
print("   " + " ".join(f"{v:.4f}" for v in per))
pre_zero = float(per[:K + 1].max()) == 0.0
responds = float(per[K + 1:].max()) > 1e-6
print(f"   pre-intervention exactly zero : {'OK' if pre_zero else '** LEAKAGE **'}")
print(f"   responds after                : {'OK' if responds else '** NO RESPONSE: model ignores the action **'}")
print(f"\nVIDEO STEP 2 {'PASSED' if (null_ok and pre_zero and responds) else 'FAILED'}")
