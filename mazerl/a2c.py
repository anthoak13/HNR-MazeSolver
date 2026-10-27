"""Training: advantage actor-critic over whole attempts, so bypass attempts can be left out.

Each parallel env buffers its current attempt. When the attempt ends it is either added to the
training pool or, if the agent never entered the maze, dropped. Once the pool holds enough moves the
network takes one gradient step on it (A2C: one pass per move). Because a pooled attempt may have
started a few updates ago, the policy loss uses PPO's clipped probability ratio to stay stable.
"""
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F

from .agent import Agent
from .env import MazeEnv, Rewards, OBS_CHANNELS, CROP, N_COMPASS
from .maze import training_bank

STAT_KEYS = ("return", "success", "steps", "probes", "breaks", "outside", "new_cells", "bypass")


@dataclass
class Settings:
    episodes: int = 126_000       # attempts (the article: 126,000)
    layouts: int = 13_148         # distinct training layouts (the article: 13,148)
    locked_frac: float = 0.5      # share of layouts that require breaking an interior wall
    exclude_bypass: bool = True   # leave attempts that never entered the maze out of training
    learning_rate: float = 5e-4
    discount: float = 0.99
    exploration: float = 0.05     # entropy bonus: rewards keeping move choices varied
    exploration_final: float = 0.01
    max_steps: int = 100
    n_envs: int = 256
    batch_moves: int = 2048
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_weight: float = 0.25
    max_grad_norm: float = 40.0
    seed: int = 0


@dataclass
class Run:
    """The result of a training run: the trained agent plus a record of how training went."""
    agent: Agent
    rewards: Rewards
    settings: Settings
    history: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)
    seconds: float = 0.0

    def save(self, path):
        torch.save({"agent": self.agent.state_dict(), "rewards": self.rewards.as_dict(),
                    "settings": asdict(self.settings), "history": self.history,
                    "checkpoints": self.checkpoints, "seconds": self.seconds}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        d = torch.load(path, map_location=device, weights_only=False)
        agent = Agent()
        agent.load_state_dict(d["agent"])
        return cls(agent, Rewards(**d["rewards"]), Settings(**d["settings"]), d["history"],
                   d["checkpoints"], d["seconds"])

    def agent_at(self, episodes):
        """The agent as it was after ``episodes`` attempts (from the saved checkpoints)."""
        agent = Agent()
        agent.load_state_dict(self.checkpoints[episodes])
        return agent


class _Pool:
    """Finished attempts waiting to be trained on."""

    def __init__(self):
        self.parts = []
        self.moves = 0

    def add(self, **tensors):
        self.parts.append(tensors)
        self.moves += len(tensors["action"])

    def take(self):
        batch = {k: torch.cat([p[k] for p in self.parts]) for k in self.parts[0]}
        self.parts, self.moves = [], 0
        return batch


def train(rewards=None, settings=None, start_from=None, device=None, progress=True,
          checkpoint_at=(), bank=None, **overrides):
    """Train an agent. ``overrides`` set fields of Settings (e.g. episodes=5000, discount=0.95)."""
    rewards = rewards or Rewards()
    s = Settings(**{**asdict(settings or Settings()), **overrides})
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(s.seed)
    bank = bank or training_bank(s.layouts, seed=s.seed, locked_frac=s.locked_frac)

    agent = Agent().to(device)
    if start_from is not None:
        src = start_from.agent if isinstance(start_from, Run) else start_from
        agent.load_state_dict(src.state_dict())
    opt = torch.optim.Adam(agent.parameters(), lr=s.learning_rate)
    env = MazeEnv(bank, s.n_envs, rewards, s.max_steps, device, seed=s.seed)
    scale = max(abs(v) for v in rewards.as_dict().values()) or 1.0   # rewards -> [-1, 1]

    N, T = s.n_envs, s.max_steps
    buf_obs = torch.zeros((N, T, OBS_CHANNELS, CROP, CROP), dtype=torch.uint8, device=device)
    buf_comp = torch.zeros((N, T), dtype=torch.uint8, device=device)
    buf = {k: torch.zeros((N, T), device=device) for k in ("action", "logp", "value", "reward")}
    discounts = _gae_matrix(T, s.discount * s.gae_lambda, device)

    pool = _Pool()
    history = {k: [] for k in ("episodes", "seconds") + STAT_KEYS}
    window = {k: [] for k in STAT_KEYS}
    checkpoints = {}
    pending_ckpt = sorted(set(checkpoint_at) | {s.episodes})
    episodes = 0
    t0 = time.time()
    obs, comp = env.observe()
    bar = _progress(s.episodes) if progress else None

    while episodes < s.episodes:
        with torch.no_grad():
            action, logp, value = agent.act(obs, comp)
        t = env.t.clone()
        buf_obs[env.rows, t] = obs.to(torch.uint8)
        buf_comp[env.rows, t] = comp.argmax(1).to(torch.uint8)
        buf["action"][env.rows, t] = action.float()
        buf["logp"][env.rows, t] = logp
        buf["value"][env.rows, t] = value
        reward, done, info = env.step(action)
        buf["reward"][env.rows, t] = reward / scale
        obs, comp = env.observe()

        if done.any():
            ended = env.rows[done]
            keep = ended
            if s.exclude_bypass:
                keep = ended[~info["bypass"][ended]]
            if len(keep):
                _collect(pool, keep, info["final_steps"][keep], buf_obs, buf_comp, buf,
                         s.discount, discounts)
            for k, v in (("return", info["final_return"]), ("success", info["reached"]),
                         ("steps", info["final_steps"]), ("probes", info["final_probes"]),
                         ("breaks", info["final_breaks"]), ("outside", info["final_outside"]),
                         ("new_cells", info["final_new_cells"]), ("bypass", info["bypass"])):
                window[k].append(v[ended].float().cpu())
            episodes += len(ended)
            if bar:
                bar.update(len(ended))

        if pool.moves >= s.batch_moves:
            frac = min(episodes / s.episodes, 1.0)
            for g in opt.param_groups:
                g["lr"] = s.learning_rate * (1 - frac) + 1e-6
            beta = s.exploration + (s.exploration_final - s.exploration) * frac
            _update(agent, opt, pool.take(), s, beta)
            _log(history, window, episodes, time.time() - t0)

        while pending_ckpt and episodes >= pending_ckpt[0]:
            checkpoints[pending_ckpt.pop(0)] = {k: v.detach().cpu().clone()
                                                for k, v in agent.state_dict().items()}
    _log(history, window, episodes, time.time() - t0)
    if bar:
        bar.close()
    agent = agent.cpu().eval()
    return Run(agent, rewards, s, history, checkpoints, time.time() - t0)


def _gae_matrix(T, decay, device):
    """M[j, t] = decay**(j - t) for j >= t, so advantages = deltas @ M."""
    j = torch.arange(T, device=device)
    power = (j[:, None] - j[None, :]).float()
    return torch.where(power >= 0, decay ** power, torch.zeros_like(power))


def _collect(pool, envs, lengths, buf_obs, buf_comp, buf, discount, discounts):
    T = buf_obs.shape[1]
    steps = torch.arange(T, device=lengths.device)
    valid = steps[None, :] < lengths[:, None]                         # (k, T)
    r, v = buf["reward"][envs], buf["value"][envs]
    v_next = torch.cat([v[:, 1:], torch.zeros_like(v[:, :1])], 1)
    v_next = torch.where(steps[None, :] + 1 < lengths[:, None], v_next, torch.zeros_like(v_next))
    delta = torch.where(valid, r + discount * v_next - v, torch.zeros_like(r))
    adv = delta @ discounts
    m = valid.flatten()
    flat = lambda x: x.flatten(0, 1)[m]
    pool.add(obs=flat(buf_obs[envs]), comp=flat(buf_comp[envs]), action=flat(buf["action"][envs]),
             logp=flat(buf["logp"][envs]), adv=flat(adv), ret=flat(adv + v))


def _update(agent, opt, batch, s, beta):
    obs = batch["obs"].float()
    comp = F.one_hot(batch["comp"].long(), N_COMPASS).float()
    logits, value = agent(obs, comp)
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(batch["action"].long())
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = torch.exp(logp - batch["logp"])
    policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - s.clip, 1 + s.clip) * adv).mean()
    value_loss = F.mse_loss(value, batch["ret"])
    loss = policy_loss + s.value_weight * value_loss - beta * dist.entropy().mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), s.max_grad_norm)
    opt.step()


def _log(history, window, episodes, seconds):
    if not window["return"]:
        return
    history["episodes"].append(episodes)
    history["seconds"].append(seconds)
    for k in STAT_KEYS:
        history[k].append(float(torch.cat(window[k]).mean()))
        window[k].clear()


def _progress(total):
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, unit="attempt")
    except ImportError:
        return None
