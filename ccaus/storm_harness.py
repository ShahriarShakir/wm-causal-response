"""STORM (Zhang et al., NeurIPS 2023) harness: a stochastic-latent transformer world model, our third Atari family.
Trained by us with the official config (scripts in <STORM_SRC>/pbs). Same footing as the DIAMOND
and IRIS harnesses: uint8 context frames in, [H, 3, 64, 64] float in [-1, 1] out, COMMON RANDOM NUMBERS by re-seeding torch
before every categorical draw with seed*100003 + t, so two rollouts under the same actions are identical and a difference can
only come from the action. Latent features (stochastic sample ++ transformer hidden) are returned for the representation probe."""
import sys, pathlib
import numpy as np, torch
STORM_SRC = "<STORM_SRC>"; STORM_PKGS = "<STORM_PKGS>"
for p in (STORM_PKGS, STORM_SRC):
    if p not in sys.path: sys.path.insert(0, p)
CONTEXT, IMG = 8, 64                     # STORM imagines from 8 context frames (ImagineContextLength)


def load_storm(ckpt_path, num_actions, device="cpu"):
    # bit-exact rollouts on GPU: the first GPU run differed by 5e-7 between arms before the intervention (cuDNN/TF32 nondeterminism, not a leak)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    from sub_models.world_models import WorldModel
    wm = WorldModel(in_channels=3, action_dim=num_actions, transformer_max_length=64, transformer_hidden_dim=512, transformer_num_layers=2, transformer_num_heads=8)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = wm.load_state_dict(sd, strict=True)
    wm.use_amp = False; wm.tensor_dtype = torch.float32           # float32, no autocast: exact reproducibility under common random numbers
    return wm.eval().to(device)


def to_model(frames_uint8, device):
    """[T, H, W, 3] uint8 (raw ALE or already 64x64) -> [1, T, 3, 64, 64] float in [0, 1]. Resized with INTER_AREA, which is what
    gymnasium's ResizeObservation (used in STORM's training) does, and what the DIAMOND/IRIS harnesses do."""
    import cv2
    fr = np.asarray(frames_uint8)
    if fr.shape[1] != IMG or fr.shape[2] != IMG:
        fr = np.stack([cv2.resize(f, (IMG, IMG), interpolation=cv2.INTER_AREA) for f in fr])
    x = torch.as_tensor(fr, dtype=torch.float32, device=device) / 255.0
    return x.permute(0, 3, 1, 2)[None]


def _reset_cache(wm, device):
    tr = wm.storm_transformer
    tr.kv_cache_list = [torch.zeros((1, 0, tr.feat_dim), dtype=torch.float32, device=device) for _ in tr.layer_stack]


def imagine(wm, context_frames, actions, horizon, seed=0, device="cpu", return_repr=False):
    """context_frames: [CONTEXT, 64, 64, 3] uint8 (already resized as in the other harnesses); actions: ints of length
    CONTEXT + horizon, the first CONTEXT aligned with the context frames. Returns [horizon, 3, 64, 64] in [-1, 1]
    (and [horizon, 1024 + 512] latent features if return_repr)."""
    with torch.no_grad():
        obs = to_model(context_frames, device); T = obs.shape[1]
        _reset_cache(wm, device)
        torch.manual_seed(seed * 100003 - 1)                          # the posterior draws of the context are seeded too
        ctx_latent = wm.encode_obs(obs)                                # [1, T, 1024]
        last_latent = None
        for i in range(T):
            a = torch.as_tensor([[int(actions[i])]], device=device)
            _, _, _, last_latent, last_feat = wm.predict_next(ctx_latent[:, i:i + 1], a, log_video=False)
        out, reps = [], []
        for t in range(horizon):
            # REGISTRATION (fixed 2026-09-12): after the context loop `last_latent` is the prior for frame T, so out[0] is its
            # decode; step t >= 1 applies actions[T + t - 1], the action taken AT frame T + t - 1, exactly as the emulator does.
            # The old loop stepped first and returned the frame after the next one, one step early relative to the true frames.
            if t > 0:
                torch.manual_seed(seed * 100003 + t)                    # COMMON RANDOM NUMBERS across arms
                j = T + t - 1; a = torch.as_tensor([[int(actions[j]) if j < len(actions) else int(actions[-1])]], device=device)
                _, _, _, last_latent, last_feat = wm.predict_next(last_latent, a, log_video=False)
            obs_hat = wm.image_decoder(last_latent)
            out.append(obs_hat[0, 0].float().cpu())
            if return_repr: reps.append(torch.cat([last_latent[0, 0], last_feat[0, 0]], -1).float().cpu())
    frames = torch.stack(out).numpy() * 2.0 - 1.0
    return (np.clip(frames, -1, 1), torch.stack(reps).numpy()) if return_repr else np.clip(frames, -1, 1)
