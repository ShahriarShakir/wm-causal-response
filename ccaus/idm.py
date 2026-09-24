"""A small inverse-dynamics model (IDM): (frame_t, frame_{t+1}) -> action_t.

Why it is here. Some evaluations of action-conditioned video models score "action consistency"
with an IDM: read the action back from consecutive GENERATED frames and check it matches the
action that was fed. It is a ONE-rollout proxy -- it asks whether the video depicts an action,
not whether the depicted change is the one the world would have produced. The proposition
says no such score can measure causal response; this module lets the paper show it on the
same rollouts rather than argue it.

Trained per game on emulator transitions generated IN MEMORY (nothing written to disk except
the final weights), gated on held-out emulator accuracy before it is allowed to score anything.
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class IDM(nn.Module):
    def __init__(self, n_act: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1), nn.ReLU(),      # 64 -> 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),     # 32 -> 16
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),     # 16 -> 8
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),     # 8 -> 4
            nn.Flatten(), nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, n_act))
        self.n_act = n_act

    def forward(self, f0, f1):                          # [B,3,64,64] each in [-1,1]
        return self.net(torch.cat([f0, f1], 1))


def to_t(x: np.ndarray, device) -> torch.Tensor:
    """[N,64,64,3] float in [-1,1] -> [N,3,64,64] tensor."""
    return torch.as_tensor(np.asarray(x, np.float32), device=device).permute(0, 3, 1, 2)


def train_idm(f0, f1, acts, n_act, device="cpu", epochs=3, bs=256, lr=1e-3, seed=0):
    """Train on in-memory transitions. Returns (model, held-out accuracy, chance)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n = len(acts); idx = rng.permutation(n); cut = int(0.9 * n); tr, te = idx[:cut], idx[cut:]
    m = IDM(n_act).to(device); opt = torch.optim.Adam(m.parameters(), lr=lr)
    F0, F1, A = to_t(f0, device), to_t(f1, device), torch.as_tensor(acts, dtype=torch.long, device=device)
    for _ in range(epochs):
        m.train(); p = rng.permutation(tr)
        for i in range(0, len(p), bs):
            b = torch.as_tensor(p[i:i + bs], device=device)
            loss = F.cross_entropy(m(F0[b], F1[b]), A[b]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = float((m(F0[te], F1[te]).argmax(1) == A[te]).float().mean())
    chance = float(np.bincount(acts, minlength=n_act).max() / n)   # majority-class baseline, not 1/n_act
    return m, acc, chance


@torch.no_grad()
def predict(m: IDM, frames: np.ndarray, device="cpu") -> np.ndarray:
    """[T,64,64,3] frames -> [T-1] predicted actions for transitions t -> t+1."""
    x = to_t(frames, device)
    return m(x[:-1], x[1:]).argmax(1).cpu().numpy()
