"""Layer-correct representation probe for IRIS: the transformer's PRE-SAMPLING hidden state.

The earlier IRIS probe used embeddings of the SAMPLED tokens, which feed a near-lossless VQ decoder, so a
'decoding failure' could not appear there by construction. The analogous layer to the RSSM's features is the
transformer output_sequence (B, 1+K, E) BEFORE token sampling. z for an imagined step = that state; z* for the
true step = the same transformer teacher-forced on the TRUE frames' tokens under the same actions.
"""
from __future__ import annotations
import sys, numpy as np, torch
IRIS_SRC = "<IRIS_SRC>/src"
if IRIS_SRC not in sys.path: sys.path.insert(0, IRIS_SRC)


class HiddenEnv:
    """Drives WorldModelEnv but records the pre-sampling hidden state at every step."""
    def __init__(self, tok, wm, device):
        from envs.world_model_env import WorldModelEnv
        self.env = WorldModelEnv(tokenizer=tok, world_model=wm, device=device); self.hidden = []
        self._orig = self.env.world_model.forward
        def hooked(*a, **k):
            out = self._orig(*a, **k); self._last = out.output_sequence.detach(); return out
        self.env.world_model.forward = hooked
    def reset(self, obs): self.hidden = []; return self.env.reset_from_initial_observations(obs)
    def step(self, a):
        seqs = []
        # WorldModelEnv.step runs 1 + K forward passes (action token, then K obs tokens); collect all K+1 outputs
        orig = self.env.world_model.forward
        def cap(*x, **k):
            out = orig(*x, **k); seqs.append(out.output_sequence.detach()); return out
        self.env.world_model.forward = cap
        r = self.env.step(a, should_predict_next_obs=True)
        self.env.world_model.forward = orig
        # when the KV cache is full, WorldModelEnv.step first re-runs the initial tokens (an extra forward pass);
        # keep only the LAST 1+K outputs (action token + K observation tokens) so every step has the same shape
        K = self.env.num_observations_tokens
        self.hidden.append(torch.cat(seqs[-(1 + K):], 1)[0].reshape(-1).cpu().numpy())   # (1+K)*E flattened
        return r


@torch.no_grad()
def imagine_hidden(tok, wm, init_frame, actions, horizon, seed, device):
    from ccaus.iris_harness import to_model
    env = HiddenEnv(tok, wm, device); env.reset(to_model(init_frame, device)); out = []
    for t in range(horizon):
        torch.manual_seed(seed * 100003 + t); obs, *_ = env.step(int(actions[t])); out.append(obs[0].detach().cpu())
    return torch.stack(out).numpy() * 2.0 - 1.0, np.stack(env.hidden)


@torch.no_grad()
def true_hidden(tok, wm, frames_uint8, actions, device):
    """Teacher-force the world model on the TRUE frames' tokens: z* per step, same layer, same actions."""
    from ccaus.iris_harness import to_model
    x = torch.stack([to_model(f, device)[0] for f in frames_uint8])           # [T+1,3,64,64]
    toks = tok.encode(x, should_preprocess=True).tokens                        # [T+1,K]
    kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=wm.config.max_tokens)
    out = wm(toks[:1], past_keys_values=kv); zs = []
    for t in range(len(actions)):
        if kv.size + 1 + toks.shape[1] > wm.config.max_tokens:
            kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=wm.config.max_tokens); wm(toks[t:t+1], past_keys_values=kv)
        a = torch.tensor([[int(actions[t])]], dtype=torch.long, device=device)
        o1 = wm(a, past_keys_values=kv); o2 = wm(toks[t+1:t+2], past_keys_values=kv)
        zs.append(torch.cat([o1.output_sequence, o2.output_sequence], 1)[0].reshape(-1).cpu().numpy())
    return np.stack(zs)
