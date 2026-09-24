"""Drive DIAMOND's diffusion world model directly, with OUR context and OUR actions.

DIAMOND's own `WorldModelEnv` seeds its buffers from a training data loader, which is right
for RL but wrong here: the paired-intervention design needs to inject a specific context and a
specific action sequence, identical between the two arms except at one step. So we drive the
`DiffusionSampler` ourselves.

DIAMOND conditions on `num_steps_conditioning = 4` previous frames and the 4 matching actions,
and diffuses the next frame. Frames are 64x64 RGB in [-1, 1].

The sampler is STOCHASTIC (diffusion). The DMControl arm showed what that costs: sampling
noise alone produced a 15/255 "response" to a NULL intervention, which reads as causality.
Here we fix the torch seed identically before each arm (common random numbers), so the two
rollouts see the SAME noise and the only difference is the action. The null control must
return exactly 0.
"""
from __future__ import annotations
import sys
import numpy as np
import torch

DIAMOND_SRC = "<DIAMOND_SRC>/src"
if DIAMOND_SRC not in sys.path:
    sys.path.insert(0, DIAMOND_SRC)

CONTEXT = 4          # num_steps_conditioning
IMG = 64


def load_diamond(ckpt_path, num_actions, num_steps_denoising=3, device="cpu"):
    """Build the denoiser from the released config and load the released weights."""
    from models.diffusion import Denoiser, DenoiserConfig, InnerModelConfig
    from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
    inner = InnerModelConfig(img_channels=3, num_steps_conditioning=CONTEXT,
                             cond_channels=256, depths=[2, 2, 2, 2],
                             channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
                             num_actions=num_actions)
    den = Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3))
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    dsd = {k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}
    missing, unexpected = den.load_state_dict(dsd, strict=False)
    assert not missing and not unexpected, f"checkpoint mismatch: {list(missing)[:3]} {list(unexpected)[:3]}"
    den.eval().to(device)
    sampler = DiffusionSampler(den, DiffusionSamplerConfig(num_steps_denoising=num_steps_denoising))
    return sampler


def to_model(frames_uint8):
    """[T,H,W,3] uint8 -> [T,3,64,64] float in [-1,1] (DIAMOND's convention)."""
    import cv2
    out = np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA)
                    for f in frames_uint8])
    x = out.astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def imagine(sampler, context_frames, actions, horizon, seed=0, device="cpu"):
    """Roll the world model forward under a given action sequence.

    `context_frames`: [CONTEXT,H,W,3] uint8 real frames.
    `actions`: length CONTEXT+horizon ints; the first CONTEXT align with the context frames.
    Returns [horizon, 3, 64, 64] float in [-1,1].
    """
    obs = to_model(context_frames)[None].to(device)                 # [1,CONTEXT,3,64,64]
    act = torch.as_tensor(np.asarray(actions[:CONTEXT]), dtype=torch.long, device=device)[None]
    out = []
    for t in range(horizon):
        # COMMON RANDOM NUMBERS: identical noise in both arms, so a difference can only come
        # from the action. Without this, diffusion sampling noise masquerades as causal response.
        torch.manual_seed(seed * 100003 + t)
        nxt, _ = sampler.sample(obs, act)
        out.append(nxt[0].detach().cpu())
        obs = torch.roll(obs, -1, dims=1); obs[:, -1] = nxt
        act = torch.roll(act, -1, dims=1)
        act[:, -1] = int(actions[CONTEXT + t]) if CONTEXT + t < len(actions) else int(actions[-1])
    return torch.stack(out).numpy()
