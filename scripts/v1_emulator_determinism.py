"""VIDEO ARM, STEP 1, is the Atari emulator pixel-exact under replay?

This is the step-1 equivalent from the DMControl arm, and it is a gate, not a formality:
Atari world models are used because an emulator can generate the true counterfactual future. If
replay is not bit-identical there is no exact ground truth and the paired design fails, leaving only
a recorded clip or a learned judge to score against.

Checks, hardest first:
  1. same seed + same actions -> identical frames (exact ground truth exists)
  2. flipping ONE early action changes the future, and changes NOTHING before it
     (a cause cannot act backwards; any pre-intervention difference means a broken harness)
"""
import sys, pathlib
sys.path.insert(0, "<ROOT>/.tools")
import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
GAME = sys.argv[1] if len(sys.argv) > 1 else "ALE/Breakout-v5"
T, K, SEED = 24, 6, 0

def rollout(actions, seed=SEED):
    env = gym.make(GAME, frameskip=4, repeat_action_probability=0.0)  # deterministic ALE
    obs, _ = env.reset(seed=seed)
    frames = [np.asarray(obs)]
    for a in actions:
        obs, *_ = env.step(int(a))
        frames.append(np.asarray(obs))
    env.close()
    return np.stack(frames)

env = gym.make(GAME, frameskip=4, repeat_action_probability=0.0)
n_act = env.action_space.n
obs, _ = env.reset(seed=SEED)
print(f"{GAME}: obs {np.asarray(obs).shape} dtype={np.asarray(obs).dtype}, {n_act} actions")
env.close()

rng = np.random.default_rng(0)
A = rng.integers(0, n_act, T)

r1, r2 = rollout(A), rollout(A)
d = float(np.abs(r1.astype(float) - r2.astype(float)).max())
print(f"\n1. determinism: max |pixel diff| over {r1.shape} = {d}")
det = d == 0.0
print("   " + ("OK, exact ground truth available" if det else "** NOT DETERMINISTIC: design breaks **"))

B = A.copy()
B[K] = (A[K] + n_act // 2) % n_act      # a different action at one early step
if B[K] == A[K]: B[K] = (A[K] + 1) % n_act
r3 = rollout(B)
per_step = np.abs(r1.astype(float) - r3.astype(float)).mean(axis=(1, 2, 3))
print(f"\n2. single flipped action at k={K} (act {A[K]} -> {B[K]}):")
print("   " + " ".join(f"{v:.2f}" for v in per_step))
pre_zero = float(per_step[:K+1].max()) == 0.0
grows = per_step[-1] > 0
print(f"   pre-intervention divergence exactly zero : {'OK' if pre_zero else '** LEAKAGE **'}")
print(f"   divergence appears after                 : {'OK' if grows else '** no effect: pick a game/action with consequence **'}")
print(f"\nVIDEO STEP 1 {'PASSED' if (det and pre_zero and grows) else 'FAILED'}")
