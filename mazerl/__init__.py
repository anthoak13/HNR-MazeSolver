"""mazerl: maze-solving reinforcement-learning activity (HNR-1303, Unit 5).

Built to mirror the NYT article "A.I.'s Rogue Behavior Starts With Its Training" (Oct. 26, 2026).
Students set reward values, train agents, and watch what the agents learn.

The main pieces:
  * train(), Rewards, Settings: train an agent with your own reward values
  * playback(), training_grid(), show_maze(): watch agents (or a maze) in the animation player
  * evaluate(), breakout_stages(), score_breakdown(): measure what an agent does
  * load_pretrained(), load_perception(), list_pretrained(): runs trained ahead of time
  * pretrain_perception(): reward-free pretraining of the agent's vision
  * check_gpu(): is this notebook running on a GPU?
"""
from importlib.metadata import PackageNotFoundError, version as _version
from pathlib import Path

import numpy as np

from .maze import (Maze, MazeBank, generate_maze, maze_kind, scenario, training_bank, SCENARIOS,
                   KINDS, MIX)
from .env import Rewards
from .agent import Agent
from .a2c import train, Settings, Run
from .rollout import record, summarize, points_by_source, evaluate as _evaluate
from . import rollout as _rollout
from .perception import pretrain_perception, Pretrained
from .viz import plot_training
from . import player
from .observation import show_observation

try:
    __version__ = _version("mazerl")
except PackageNotFoundError:     # running from a source tree that isn't installed
    __version__ = "unknown"

_PRETRAINED = Path(__file__).parent / "pretrained"


def random_mazes(n=1, seed=None, kind=None):
    """``n`` new layouts made the same way as the training layouts (but not ones it trained on).

    kind: "open", "interior_break" or "outside_only"; None mixes them like the training set.
    """
    rng = np.random.default_rng(seed)
    kinds = [kind] * n if kind else rng.choice(KINDS, size=n, p=MIX)
    return [generate_maze(rng, kind=k) for k in kinds]


def show_maze(maze, **kwargs):
    """Draw a maze (no agents)."""
    return player.show({**maze.to_dict(), "agents": []}, **kwargs)


def _unpack(agent, rewards=None, max_steps=None, task=None):
    """Split a Run into its agent and its settings; anything given explicitly wins."""
    if isinstance(agent, Run):
        rewards = rewards or agent.rewards
        max_steps = max_steps or agent.settings.max_steps
        task = task or getattr(agent.settings, "task", "flag")
        agent = agent.agent
    return agent, rewards or Rewards(), max_steps or Settings.max_steps, task or "flag"


def _one_maze(maze, seed):
    """A Maze as given, a scenario by name, or (None) a new random layout from ``seed``."""
    if maze is None:
        return random_mazes(1, seed)[0]
    return scenario(maze) if isinstance(maze, str) else maze


def playback(agent, maze=None, n_agents=1, seed=0, rewards=None, greedy=False, max_steps=None,
             task=None, **kwargs):
    """Animate ``n_agents`` attempts by ``agent`` (a Run, an Agent, or None for random moves).

    maze: a Maze, a test maze's name (see SCENARIOS), or None for a new random layout.
    task: "flag" or "escape"; None uses the Run's task (or "flag").
    """
    agent, rewards, max_steps, task = _unpack(agent, rewards, max_steps, task)
    episode = record(agent, [_one_maze(maze, seed)], n_agents=n_agents, rewards=rewards,
                     greedy=greedy, seed=seed, max_steps=max_steps, task=task)[0]
    return player.show(episode, **kwargs)


def training_grid(run, n_mazes=3, stages=None, seed=0, **kwargs):
    """Several layouts, each with agents from different points in training (article p.4-5)."""
    stages = stages or sorted(run.checkpoints)
    task = getattr(run.settings, "task", "flag")
    mazes = random_mazes(n_mazes, seed)
    episodes = [{**m.to_dict(), "agents": [], "task": task} for m in mazes]
    if task == "escape":                             # escape attempts start on the flag's cell
        for ep, m in zip(episodes, mazes):
            ep["start"] = list(m.flag)
    for k, stage in enumerate(stages):
        recorded = record(run.agent_at(stage), mazes, rewards=run.rewards, seed=seed + k,
                          max_steps=run.settings.max_steps, task=task)
        for ep, rec in zip(episodes, recorded):
            for a in rec["agents"]:
                ep["agents"].append({**a, "label": f"after {stage:,} attempts"})
    return player.show(episodes, **kwargs)


def evaluate(agent, n=500, mazes=None, seed=12345, rewards=None, greedy=False, max_steps=None,
             task=None):
    """How the agent does on ``n`` layouts it has never seen (or on ``mazes`` if given).

    task: "flag" or "escape"; None uses the Run's task (or "flag"). For escape, "reached flag"
    means "got out".
    """
    agent, rewards, max_steps, task = _unpack(agent, rewards, max_steps, task)
    if mazes is None:
        mazes = random_mazes(n, seed)
    elif isinstance(mazes, str):
        mazes = [scenario(mazes)] * n
    return _evaluate(agent, mazes, rewards=rewards, greedy=greedy, seed=seed, max_steps=max_steps,
                     task=task)


def breakout_stages(agent, maze=None, n=500, seed=0):
    """Out of ``n`` attempts on the enclosed test maze (or ``maze``), the share that left the maze,
    broke back in from outside through a weak spot, and reached the flag (article p.6).

    agent: a Run (its rewards and move limit are used), an Agent, or None for random moves.
    """
    agent, rewards, max_steps, _ = _unpack(agent)
    maze = scenario(maze) if isinstance(maze, str) else maze
    return _rollout.breakout_stages(agent, maze, n=n, rewards=rewards, max_steps=max_steps,
                                    seed=seed)


def score_breakdown(agent, maze=None, seed=0, rewards=None, max_steps=None, task=None, plot=True):
    """Where one attempt's points came from: the flag, steps outside, wall bumps, new squares
    explored inside and outside the maze (and "per move", if that reward isn't zero), plus the total.

    agent: a Run, an Agent, or None for random moves. maze: a Maze, a test maze's name, or None for
    a new random layout from ``seed``. With ``plot=True`` it also draws the points as a bar chart.
    Returns {source: points}.
    """
    agent, rewards, max_steps, task = _unpack(agent, rewards, max_steps, task)
    attempt = record(agent, [_one_maze(maze, seed)], rewards=rewards, seed=seed,
                     max_steps=max_steps, task=task)[0]["agents"][0]
    points = points_by_source(attempt, rewards, task)
    # The breakdown must add up to what the attempt actually scored
    assert abs(points["total"] - attempt["return"]) < 1e-4, (points, attempt["return"])
    assert abs(points["total"] - attempt["env_return"]) < 1e-3, (points, attempt["env_return"])
    points = {k: round(v, 4) + 0.0 for k, v in points.items()}     # + 0.0 turns -0.0 into 0.0
    if plot:
        _plot_points(points)
    return points


def _plot_points(points):
    import matplotlib.pyplot as plt
    names = list(points)[::-1]                       # top to bottom in the order given
    values = [points[k] for k in names]
    colors = ["#2a7ab0" if v > 0 else "#c0392b" if v < 0 else "#999999" for v in values]
    colors[0] = "#333333"                            # the total
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(names) + 1))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    span = max(abs(v) for v in values) or 1
    for y, v in enumerate(values):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), y, f"{v:+g}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlim(-1.25 * span if min(values) < 0 else 0, 1.25 * span if max(values) > 0 else 0.1)
    ax.set_xlabel("points (blue = earned, red = lost)")
    ax.set_title("Where this attempt's points came from")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.show()


def list_pretrained():
    """Names of the runs (and pretrained perceptions) shipped with the package."""
    return sorted(p.stem for p in _PRETRAINED.glob("*.pt")) if _PRETRAINED.is_dir() else []


def _pretrained_path(name, what):
    path = _PRETRAINED / f"{name}.pt"
    if not path.exists():
        raise ValueError(f"no pretrained {what} {name!r}; available: {list_pretrained()}")
    return path


def load_pretrained(name="base"):
    """A Run trained ahead of time and shipped with the package."""
    return Run.load(_pretrained_path(name, "run"))


def load_perception(name="perception"):
    """Vision pretrained ahead of time without rewards (a Pretrained), shipped with the package."""
    return Pretrained.load(_pretrained_path(name, "perception"))


def check_gpu():
    """True if this notebook has a GPU to train on (and says which); otherwise prints how to get one."""
    import torch
    if torch.cuda.is_available():
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
        return True
    print("No GPU found, so training will be much slower.\n"
          "In Colab: Runtime -> Change runtime type -> T4 GPU, then Save, and run the notebook again.")
    return False
