"""mazerl: maze-solving reinforcement-learning activity (HNR-1303, Unit 5).

Built to mirror the NYT article "A.I.'s Rogue Behavior Starts With Its Training" (Oct. 26, 2026).
Students set reward values, train agents, and watch what the agents learn.
"""
from pathlib import Path

import numpy as np

from .maze import (Maze, MazeBank, generate_maze, maze_kind, scenario, training_bank, SCENARIOS,
                   KINDS, MIX)
from .env import Rewards
from .agent import Agent
from .a2c import train, Settings, Run
from .rollout import record, summarize, evaluate as _evaluate
from .viz import plot_training
from . import player

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


def playback(agent, maze=None, n_agents=1, seed=0, rewards=None, greedy=False, max_steps=None,
             **kwargs):
    """Animate ``n_agents`` attempts by ``agent`` (a Run, an Agent, or None for random moves)."""
    if isinstance(agent, Run):
        rewards = rewards or agent.rewards
        max_steps = max_steps or agent.settings.max_steps
        agent = agent.agent
    maze = maze if maze is not None else random_mazes(1, seed)[0]
    maze = scenario(maze) if isinstance(maze, str) else maze
    episode = record(agent, [maze], n_agents=n_agents, rewards=rewards, greedy=greedy, seed=seed,
                     max_steps=max_steps or Settings.max_steps)[0]
    return player.show(episode, **kwargs)


def training_grid(run, n_mazes=3, stages=None, seed=0, **kwargs):
    """Several layouts, each with agents from different points in training (article p.4-5)."""
    stages = stages or sorted(run.checkpoints)
    mazes = random_mazes(n_mazes, seed)
    episodes = [{**m.to_dict(), "agents": []} for m in mazes]
    for k, stage in enumerate(stages):
        recorded = record(run.agent_at(stage), mazes, rewards=run.rewards, seed=seed + k,
                          max_steps=run.settings.max_steps)
        for ep, rec in zip(episodes, recorded):
            for a in rec["agents"]:
                ep["agents"].append({**a, "label": f"after {stage:,} attempts"})
    return player.show(episodes, **kwargs)


def evaluate(agent, n=500, mazes=None, seed=12345, rewards=None, greedy=False, max_steps=None):
    """How the agent does on ``n`` layouts it has never seen (or on ``mazes`` if given)."""
    if isinstance(agent, Run):
        rewards = rewards or agent.rewards
        max_steps = max_steps or agent.settings.max_steps
        agent = agent.agent
    if mazes is None:
        mazes = random_mazes(n, seed)
    elif isinstance(mazes, str):
        mazes = [scenario(mazes)] * n
    return _evaluate(agent, mazes, rewards=rewards, greedy=greedy, seed=seed,
                     max_steps=max_steps or Settings.max_steps)


def load_pretrained(name="base"):
    """A Run trained ahead of time and shipped with the package."""
    path = _PRETRAINED / f"{name}.pt"
    if not path.exists():
        names = sorted(p.stem for p in _PRETRAINED.glob("*.pt"))
        raise ValueError(f"no pretrained run {name!r}; available: {names}")
    return Run.load(path)
