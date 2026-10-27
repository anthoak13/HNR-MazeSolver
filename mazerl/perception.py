"""Reward-free pretraining of the agent's perception (vision + first reasoning layer).

Like pretraining a language model before RL: the network first learns to understand mazes by
predicting things about them, with no rewards involved. Two prediction tasks, on agents exploring the
training layouts:
  * map completion: what is in the cells of its 11 x 11 window that it hasn't seen yet?
    (open, outside, wall, or edge; weak spots count as wall, so nothing hidden leaks)
  * frontier direction: which move heads toward the nearest reachable cell it hasn't visited?
RL then trains only the deciding layers on top (a2c.train(perception=...)).
"""
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .agent import Agent
from .env import MazeEnv, PAD, CROP, SIZE, UNSEEN, M_FLOOR, M_OUTSIDE, M_FLAG
from .maze import MazeBank, training_bank

N_CLASSES = 4                                  # open inside, outside, wall (incl. weak spots), edge
TRUE_CLASS = torch.tensor([0, 2, 2, 1, 3])     # by true cell code: FLOOR, WALL, WEAK, OUTSIDE, EDGE
CLASS_NAMES = ("open", "outside", "wall", "edge")


def frontier_direction(env):
    """The move (0-3) heading toward the nearest known-open cell not yet visited, or -1 if none.
    Distances are along cells the agent already knows are open."""
    mem = env.memory
    passable = (mem == M_FLOOR) | (mem == M_OUTSIDE) | (mem == M_FLAG)
    far = 10_000
    dist = torch.where(passable & ~env.visited, 0, far)
    for _ in range(4 * SIZE):
        nb = torch.full_like(dist, far)
        nb[:, 1:] = torch.minimum(nb[:, 1:], dist[:, :-1])
        nb[:, :-1] = torch.minimum(nb[:, :-1], dist[:, 1:])
        nb[:, :, 1:] = torch.minimum(nb[:, :, 1:], dist[:, :, :-1])
        nb[:, :, :-1] = torch.minimum(nb[:, :, :-1], dist[:, :, 1:])
        new = torch.where(passable, torch.minimum(dist, nb + 1), dist)
        if torch.equal(new, dist):
            break
        dist = new
    nxt = env.pos[:, None, :] + env.moves[None]                       # (n, 4, 2)
    d = dist[env.rows[:, None], nxt[..., 0], nxt[..., 1]]
    best = d.min(1)
    return torch.where(best.values < far, best.indices, torch.full_like(best.indices, -1))


def _true_window(env):
    r = env.pos[:, 0, None] + env.crop_range
    c = env.pos[:, 1, None] + env.crop_range
    cells = env.grid[env.rows[:, None, None], r[:, :, None], c[:, None, :]].long()
    return TRUE_CLASS.to(cells.device)[cells]


def collect(bank, n_samples, n_envs=256, explore=0.3, max_steps=200, seed=0, device="cpu"):
    """Observations from agents that mostly head for unexplored ground (and move at random with
    probability ``explore``), with the prediction targets. No rewards are involved."""
    env = MazeEnv(bank, n_envs, max_steps=max_steps, device=device, seed=seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    keep = {k: [] for k in ("obs", "comp", "truth", "unseen", "frontier")}
    n = 0
    while n < n_samples:
        obs, comp = env.observe()
        front = frontier_direction(env)
        keep["obs"].append(obs.to(torch.uint8).cpu())
        keep["comp"].append(comp.argmax(1).to(torch.uint8).cpu())
        keep["truth"].append(_true_window(env).to(torch.int8).cpu())
        keep["unseen"].append((obs[:, UNSEEN] > 0).cpu())
        keep["frontier"].append(front.to(torch.int8).cpu())
        n += n_envs
        rand = torch.randint(0, 4, (n_envs,), generator=gen).to(env.device)
        wander = torch.rand(n_envs, generator=gen).to(env.device) < explore
        env.step(torch.where(wander | (front < 0), rand, front))
    return {k: torch.cat(v)[:n_samples] for k, v in keep.items()}


class PredictionHeads(nn.Module):
    """Read-outs used only during pretraining; RL never sees them."""

    def __init__(self, hidden=100):
        super().__init__()
        self.map = nn.Linear(hidden, CROP * CROP * N_CLASSES)
        self.frontier = nn.Linear(hidden, 4)

    def forward(self, z):
        return self.map(z).view(-1, N_CLASSES, CROP, CROP), self.frontier(z)


@dataclass
class Pretrained:
    """Result of perception pretraining: an Agent whose perceiving part is trained (deciding part
    untrained), plus how well it predicts on mazes it never saw."""
    agent: Agent
    history: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
    seconds: float = 0.0

    def save(self, path):
        torch.save({"perception": self.agent.perception_state(), "history": self.history,
                    "test": self.test, "seconds": self.seconds}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        agent = Agent()
        agent.load_perception(d["perception"])
        return cls(agent, d["history"], d["test"], d["seconds"])


def _losses(agent, heads, b, device):
    obs = b["obs"].to(device).float()
    comp = F.one_hot(b["comp"].to(device).long(), 8).float()
    map_logits, front_logits = heads(agent.perceive(obs, comp))
    truth = b["truth"].to(device).long()
    unseen = b["unseen"].to(device)
    per_cell = F.cross_entropy(map_logits, truth, reduction="none")
    weight = torch.where(unseen, 1.0, 0.1)
    map_loss = (per_cell * weight).sum() / weight.sum()
    front = b["frontier"].to(device).long()
    has = front >= 0
    front_loss = F.cross_entropy(front_logits[has], front[has]) if has.any() else map_loss * 0
    with torch.no_grad():
        pred = map_logits.argmax(1)
        map_acc = (pred == truth)[unseen].float().mean()
        front_acc = (front_logits.argmax(1) == front)[has].float().mean()
    return map_loss + front_loss, float(map_acc), float(front_acc)


def pretrain_perception(steps=3000, batch=256, n_samples=150_000, lr=1e-3, seed=0, device=None,
                        bank=None, progress=True, log_every=100):
    """Pretrain the perceiving part of a fresh Agent on the two prediction tasks."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    torch.manual_seed(seed)
    bank = bank or training_bank()
    data = collect(bank, n_samples, seed=seed, device=device)
    test_bank = MazeBank.generate(300, seed=10_000 + seed)        # layouts it never trains on
    test = collect(test_bank, 20_000, seed=seed + 1, device=device)
    t_data = time.time() - t0

    agent, heads = Agent().to(device), PredictionHeads().to(device)
    params = [p for n, p in agent.named_parameters() if n.startswith(("vision.", "reasoning.0."))]
    opt = torch.optim.Adam(params + list(heads.parameters()), lr=lr)
    history = {"step": [], "loss": [], "map_acc": [], "frontier_acc": []}
    gen = torch.Generator().manual_seed(seed)
    bar = _bar(steps) if progress else None
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data["obs"]), (batch,), generator=gen)
        loss, map_acc, front_acc = _losses(agent, heads, {k: v[idx] for k, v in data.items()}, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps:
            for k, v in zip(("step", "loss", "map_acc", "frontier_acc"), (step, loss.item(), map_acc, front_acc)):
                history[k].append(v)
        if bar:
            bar.update(1)
    if bar:
        bar.close()

    with torch.no_grad():
        accs = [_losses(agent, heads, {k: v[i:i + 2000] for k, v in test.items()}, device)[1:]
                for i in range(0, len(test["obs"]), 2000)]
    majority = _majority_baseline(data, test)
    result = {"map_acc_unseen": sum(a for a, _ in accs) / len(accs),
              "frontier_acc": sum(f for _, f in accs) / len(accs),
              "map_acc_majority_baseline": majority, "frontier_acc_chance": 0.25,
              "data_seconds": t_data}
    return Pretrained(agent.cpu().eval(), history, result, time.time() - t0)


def _majority_baseline(data, test):
    counts = torch.bincount(data["truth"].long()[data["unseen"]], minlength=N_CLASSES)
    return float((test["truth"].long()[test["unseen"]] == counts.argmax()).float().mean())


def _bar(total):
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, unit="step")
    except ImportError:
        return None
