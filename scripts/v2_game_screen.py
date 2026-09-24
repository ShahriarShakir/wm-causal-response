"""VIDEO ARM, STEP 1b, which games carry a SINGLE action's consequence forward?

Breakout failed the step-1 gate for a real reason: the paddle is position-bounded, so two
trajectories that differ by one action RECONVERGE and the divergence returns to exactly 0.
A game where the cause does not persist cannot test whether a model REMEMBERS the cause --
there would be nothing to remember. This screens for substrates where it does persist.

Screen on the EMULATOR only (no model): for each game, flip one early action and measure how
long the true divergence survives. Report the fraction of later steps that still differ.
"""
import sys
sys.path.insert(0, "<ROOT>/.tools")
import numpy as np, gymnasium as gym, ale_py
gym.register_envs(ale_py)

GAMES = ["ALE/Breakout-v5", "ALE/Pong-v5", "ALE/MsPacman-v5", "ALE/Freeway-v5",
         "ALE/Boxing-v5", "ALE/Seaquest-v5", "ALE/Qbert-v5", "ALE/SpaceInvaders-v5"]
T, K, N_EP = 40, 6, 3

def rollout(game, actions, seed):
    env = gym.make(game, frameskip=4, repeat_action_probability=0.0)
    obs, _ = env.reset(seed=seed)
    fr = [np.asarray(obs)]
    for a in actions:
        obs, *_ = env.step(int(a))
        fr.append(np.asarray(obs))
    env.close()
    return np.stack(fr)

print(f"{'game':<24} {'persist':>8} {'late div':>10} {'verdict'}")
best = []
for g in GAMES:
    try:
        env = gym.make(g, frameskip=4, repeat_action_probability=0.0)
        n_act = env.action_space.n; env.close()
        persist, lates = [], []
        for ep in range(N_EP):
            rng = np.random.default_rng(100 + ep)
            A = rng.integers(0, n_act, T)
            B = A.copy(); B[K] = (A[K] + 1) % n_act
            if B[K] == A[K]: B[K] = (A[K] + 2) % n_act
            ra, rb = rollout(g, A, ep), rollout(g, B, ep)
            d = np.abs(ra.astype(float) - rb.astype(float)).mean(axis=(1, 2, 3))
            after = d[K+1:]
            persist.append(float((after > 0).mean()))       # fraction of later steps still differing
            lates.append(float(after[-8:].mean()))          # divergence at the far end
        p, l = float(np.mean(persist)), float(np.mean(lates))
        ok = p > 0.8 and l > 0.1
        print(f"{g:<24} {p:>7.1%} {l:>10.3f}  {'GOOD, cause persists' if ok else 'reconverges / weak'}")
        if ok: best.append((g, p, l))
    except Exception as e:
        print(f"{g:<24} {'--':>8} {'--':>10}  skip ({type(e).__name__})")

print("\nUsable substrates:", [b[0] for b in best] or "NONE")
