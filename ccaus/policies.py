"""The released agents' own policies (DIAMOND and IRIS ship their actor-critic weights in the same checkpoint as the world
model). Used by the policy-context control (scripts/v22_policy_context.py, 2026-09-12): contexts and base actions drawn from the
trained policy instead of uniformly at random, to test whether causal blindness is an artefact of random-action contexts."""
import sys, numpy as np, torch
from torch.distributions import Categorical


class DiamondPolicy:
    def __init__(self, ckpt_path, num_actions, device="cpu"):
        sys.path.insert(0, "<DIAMOND_SRC>/src")
        from models.actor_critic import ActorCritic, ActorCriticConfig
        self.ac = ActorCritic(ActorCriticConfig(lstm_dim=512, img_channels=3, img_size=64, channels=[32, 32, 64, 64], down=[1, 1, 1, 1], num_actions=num_actions))
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        asd = {k[len("actor_critic."):]: v for k, v in sd.items() if k.startswith("actor_critic.")}
        m, u = self.ac.load_state_dict(asd, strict=True); self.ac.to(device).eval(); self.device = device; self.n = num_actions
    def reset(self):
        self.hx = torch.zeros(1, 512, device=self.device); self.cx = torch.zeros(1, 512, device=self.device)
    @torch.no_grad()
    def act(self, frame_uint8, rng, epsilon=0.0):
        """frame_uint8: [H, W, 3]; returns an int action sampled from the actor (DIAMOND's collector: Categorical, then epsilon-random)."""
        from ccaus.diamond_harness import to_model
        obs = to_model(frame_uint8[None]).to(self.device)                          # [1, 3, 64, 64] in [-1, 1]
        logits, _, (self.hx, self.cx) = self.ac.predict_act_value(obs, (self.hx, self.cx))
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        a = int(Categorical(logits=logits).sample())
        return int(rng.integers(0, self.n)) if rng.random() < epsilon else a


class IrisPolicy:
    def __init__(self, ckpt_path, tokenizer, num_actions, device="cpu"):
        sys.path.insert(0, "<IRIS_SRC>/src")
        from models.actor_critic import ActorCritic
        self.ac = ActorCritic(act_vocab_size=num_actions, use_original_obs=False)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        asd = {k[len("actor_critic."):]: v for k, v in sd.items() if k.startswith("actor_critic.")}
        self.ac.load_state_dict(asd, strict=True); self.ac.to(device).eval(); self.tok = tokenizer; self.device = device; self.n = num_actions
    def reset(self):
        self.ac.reset(1)
    @torch.no_grad()
    def act(self, frame_uint8, rng, epsilon=0.0):
        """IRIS's Agent.act: the actor sees the tokenizer's reconstruction of the frame (use_original_obs=False), samples a Categorical."""
        from ccaus.iris_harness import to_model
        obs = to_model(frame_uint8, self.device)                                     # [1, 3, 64, 64] in [0, 1]
        inp = torch.clamp(self.tok.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        logits = self.ac(inp).logits_actions[:, -1]
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        a = int(Categorical(logits=logits).sample())
        return int(rng.integers(0, self.n)) if rng.random() < epsilon else a
