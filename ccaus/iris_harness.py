"""Drive IRIS (transformer + discrete VQ tokens) with OUR context and OUR actions.

Why a THIRD model family matters more than any other gap-closer here. With DIAMOND alone, a
game where the model is causally blind is ambiguous: is the TASK hard, or is that one released
checkpoint bad? IRIS is architecturally unrelated -- a VQ tokenizer plus an autoregressive
transformer over discrete tokens, not a diffusion model over pixels -- and it is trained on the
same Atari-100k budget. If the same games are causally hard for both, the property belongs to
the task. If not, the measurement discriminates architectures. Both answers are publishable.

IRIS samples observation tokens from a Categorical. Unpaired, that sampling noise would read
as a causal response exactly as the RSSM's `.rsample()` did. We seed identically per step in
both arms (common random numbers), so before the intervention the logits AND the draws match
and the two futures are bit-identical -- which `measure_pair` then asserts.

Conditioning asymmetry: DIAMOND conditions on 4 frames, while IRIS resets from one observation
and carries its context in the transformer cache. Absolute values are therefore not directly
comparable across the two families; the comparisons made with them are within a model across
games (for example BankHeist against Pong), which the asymmetry does not affect.
"""
from __future__ import annotations
import sys
import numpy as np
import torch

IRIS_SRC = "<IRIS_SRC>/src"
if IRIS_SRC not in sys.path:
    sys.path.insert(0, IRIS_SRC)

IMG = 64


def load_iris(ckpt_path, num_actions, device="cpu"):
    """Rebuild tokenizer + world model from the released config and load the weights."""
    from models.tokenizer import Encoder, Decoder, Tokenizer
    from models.tokenizer.nets import EncoderDecoderConfig
    from models.world_model import WorldModel
    from models.transformer import TransformerConfig

    enc_cfg = EncoderDecoderConfig(resolution=64, in_channels=3, z_channels=512, ch=64,
                                   ch_mult=[1, 1, 1, 1, 1], num_res_blocks=2,
                                   attn_resolutions=[8, 16], out_ch=3, dropout=0.0)
    tok = Tokenizer(vocab_size=512, embed_dim=512,
                    encoder=Encoder(enc_cfg), decoder=Decoder(enc_cfg))
    tcfg = TransformerConfig(tokens_per_block=17, max_blocks=20, attention="causal",
                             num_layers=10, num_heads=4, embed_dim=256,
                             embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1)
    wm = WorldModel(obs_vocab_size=512, act_vocab_size=num_actions, config=tcfg)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tsd = {k[len("tokenizer."):]: v for k, v in sd.items() if k.startswith("tokenizer.")}
    wsd = {k[len("world_model."):]: v for k, v in sd.items() if k.startswith("world_model.")}
    m1, u1 = tok.load_state_dict(tsd, strict=False)
    m2, u2 = wm.load_state_dict(wsd, strict=False)
    assert not m1 and not u1, f"tokenizer mismatch: missing {list(m1)[:3]} unexpected {list(u1)[:3]}"
    assert not m2 and not u2, f"world model mismatch: missing {list(m2)[:3]} unexpected {list(u2)[:3]}"
    return tok.to(device).eval(), wm.to(device).eval()


def to_model(frame_uint8, device):
    """[H,W,3] uint8 -> [1,3,64,64] float in [0,1] (IRIS's convention)."""
    import cv2
    x = cv2.resize(frame_uint8, (IMG, IMG), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1)[None].to(device)


@torch.no_grad()
def encode_repr(tok, frames_uint8, device="cpu"):
    """[T,H,W,3] uint8 TRUE frames -> [T, K*E] token embeddings via the model's OWN encoder (deterministic)."""
    x = torch.stack([to_model(f, device)[0] for f in frames_uint8])          # [T,3,64,64] in [0,1]
    toks = tok.encode(x, should_preprocess=True).tokens                       # [T,K]
    return tok.embedding(toks).reshape(len(frames_uint8), -1).cpu().numpy()  # [T, K*E]


@torch.no_grad()
def imagine(tok, wm, init_frame, actions, horizon, seed=0, device="cpu", return_repr=False):
    """Roll IRIS forward under a given action sequence. Returns [horizon, 3, 64, 64] in [-1,1].

    Output is rescaled to [-1,1] to match the DIAMOND harness, so both feed `measure_pair`
    on the same footing.
    """
    from envs.world_model_env import WorldModelEnv
    env = WorldModelEnv(tokenizer=tok, world_model=wm, device=device)
    env.reset_from_initial_observations(to_model(init_frame, device))
    out, reps = [], []
    for t in range(horizon):
        # COMMON RANDOM NUMBERS: identical draws in both arms, so a difference can only come
        # from the action. Without this, categorical token sampling masquerades as a response.
        torch.manual_seed(seed * 100003 + t)
        obs, _, _, _ = env.step(int(actions[t]), should_predict_next_obs=True)
        out.append(obs[0].detach().cpu())
        if return_repr:   # the model's representation of its OWN predicted frame: the sampled tokens' embeddings
            reps.append(tok.embedding(env.obs_tokens).reshape(-1).detach().cpu())
    frames = torch.stack(out).numpy() * 2.0 - 1.0
    return (frames, torch.stack(reps).numpy()) if return_repr else frames
