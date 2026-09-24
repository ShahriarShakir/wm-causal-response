"""TWM (Robine et al., ICLR 2023) harness: Transformer-XL dynamics over discrete latents, our FOURTH Atari family, trained by us with the
authors' code. Runs under the TWM venv (torch 2.0.1, gym 0.26). TWM observes 4-stacked 64x64 GRAYSCALE frames (gym AtariPreprocessing), so
its pixel space is [64, 64] grayscale: we take the LAST frame of the decoded stack as the predicted frame, scaled to [-1, 1], and measure
everything in that space. Common random numbers: torch is re-seeded before every latent sample with seed*100003 + t, identically in both arms."""
import sys, pathlib, numpy as np, torch
TWM_SRC = "<SCRATCH>/twm_src/twm"
if TWM_SRC not in sys.path: sys.path.insert(0, TWM_SRC)
CONTEXT, IMG = 4, 64   # TWM's frame stack is 4: the context is the 4 stacked frames = 4 emulator steps


def make_env(game, seed):
    """TWM's own training pipeline, made exactly replayable: repeat_action_probability = 0, no noop start, no episodic lives."""
    import gym, utils as twm_utils
    from utils import AtariEnv, rom_name_to_id
    env = AtariEnv(rom_name_to_id(game), frameskip=1, repeat_action_probability=0.0)
    env.spec = gym.spec(game + "NoFrameskip-v4")
    env = gym.wrappers.AtariPreprocessing(env, noop_max=0, frame_skip=4, screen_size=IMG, terminal_on_life_loss=False, grayscale_obs=True)
    env = gym.wrappers.FrameStack(env, 4)
    return env


def load_twm(ckpt_path, num_actions, device="cpu"):
    import config as C
    from agent import Agent
    ck = torch.load(ckpt_path, map_location=device)
    cfg = dict(ck["config"]); cfg["model_device"] = device
    agent = Agent(cfg, num_actions); agent.load_state_dict(ck["state_dict"]); agent.eval().to(device)
    return agent, cfg


def to_model(stack_uint8, device):
    """[T, 4, 64, 64] uint8 stacks -> [1, T, 4, 64, 64] float in [0, 1] (TWM encodes the stack)."""
    x = torch.as_tensor(np.asarray(stack_uint8), dtype=torch.float32, device=device) / 255.0
    return x[None]


def imagine(agent, cfg, context_stacks, actions, horizon, seed=0, device="cpu", return_repr=False):
    """context_stacks: [CONTEXT, 4, 64, 64] uint8 (the last CONTEXT observations, each a 4-frame stack); actions: ints of length CONTEXT + horizon.
    Returns [horizon, 64, 64] float in [-1, 1] (last frame of each decoded stack) and optionally [horizon, z_dim + h_dim] latent features."""
    from agent import Dreamer
    with torch.no_grad():
        obs = to_model(context_stacks, device)                                     # [1, C, 4, 64, 64]
        a_ctx = torch.as_tensor(np.asarray(actions[:CONTEXT - 1]), dtype=torch.long, device=device)[None]   # actions between the context observations
        r_ctx = torch.zeros(1, CONTEXT - 1, device=device); term = torch.zeros(1, CONTEXT - 1, dtype=torch.bool, device=device); trunc = torch.zeros_like(term)
        # Evaluation copy of the config: TWM's imagination resets the latent to a fresh start state when the predicted cumulative
        # discount falls below wm_discount_threshold (a training-time device). A reset inside an evaluation rollout would replace
        # the context-conditioned state by an unrelated one, so the threshold is set to 0 here (d is then always False) and the
        # start-state sampler, which Dreamer insists on in imagine mode, is a stub that is never called.
        cfg_eval = dict(cfg); cfg_eval["wm_discount_threshold"] = 0
        dr = Dreamer(cfg_eval, agent.wm, mode="imagine", start_z_sampler=lambda n: (_ for _ in ()).throw(RuntimeError("imagined reset requested with threshold 0")))
        torch.manual_seed(seed * 100003 - 1)
        start_z = agent.wm.obs_model.encode_sample(obs, temperature=0)             # deterministic context encoding (argmax)
        z, h, _, _ = dr.imagine_reset(start_z, a_ctx, r_ctx, term, trunc)
        frames, reps = [], []
        for t in range(horizon):
            torch.manual_seed(seed * 100003 + t)                                      # COMMON RANDOM NUMBERS across arms
            a = torch.as_tensor([[int(actions[CONTEXT - 1 + t])]], dtype=torch.long, device=device)
            z, h, z_dist, r, g, d, w = dr.imagine_step(a, temperature=1)[:7]
            rec = agent.wm.obs_model.decode(z)                                          # [1, 1, 4, 64, 64] in [0, 1]
            frames.append(rec[0, 0, -1].float().cpu().numpy() * 2.0 - 1.0)
            if return_repr: reps.append(torch.cat([z[0, 0], h[0, 0]], -1).float().cpu().numpy())
    out = np.clip(np.stack(frames), -1, 1)
    return (out, np.stack(reps)) if return_repr else out
