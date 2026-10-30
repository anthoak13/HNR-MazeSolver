"""Run agents on particular mazes and record what happened, move by move (for playback and stats)."""
import numpy as np
import torch

from .env import MazeEnv, Rewards, PAD
from .maze import Maze, MazeBank, FLOOR, WEAK, OUTSIDE, path_length


def record(agent, mazes, n_agents=1, rewards=None, greedy=False, max_steps=200, seed=0, task="flag"):
    """Play ``n_agents`` independent attempts on each maze and return one episode dict per maze.

    agent: a trained Agent, or None for an untrained agent that moves at random.
    The dicts follow the player's data contract (grid, start, flag, agents[path, score, events]).
    task: "flag" or "escape" (start on the flag's cell; success = reaching any outside cell).
    """
    mazes = [mazes] if isinstance(mazes, Maze) else list(mazes)
    rewards = rewards or Rewards()
    bank = MazeBank.from_mazes(mazes)
    ids = np.repeat(np.arange(len(mazes)), n_agents)
    env = MazeEnv(bank, len(ids), rewards, max_steps, seed=seed, auto_reset=False, maze_ids=ids, task=task)
    gen = torch.Generator().manual_seed(seed)
    if agent is not None:
        agent = agent.cpu().eval()

    n = len(ids)
    paths = [[_cell(env.pos[i])] for i in range(n)]
    scores = [[0.0] for _ in range(n)]
    events = [[] for _ in range(n)]
    finished = torch.zeros(n, dtype=torch.bool)
    outcome = [None] * n

    for t in range(1, max_steps + 1):
        if agent is None:
            action = torch.randint(0, 4, (n,), generator=gen)
        else:
            with torch.no_grad():
                obs, comp = env.observe()
                action, _, _ = agent.act(obs, comp, greedy=greedy)
        reward, done, info = env.step(action)
        for i in range(n):
            if finished[i]:
                continue
            paths[i].append(_cell(env.pos[i]))
            scores[i].append(scores[i][-1] + float(reward[i]))
            wall = _cell(info["probe_cell"][i])
            here = paths[i][-1]
            for kind, value, cell in (("probe", rewards.probe, wall),
                                      ("through_wall", rewards.probe, wall),
                                      ("outside", rewards.outside, here),
                                      ("new_cell", rewards.new_cell, here),
                                      ("flag", rewards.flag, here)):
                key = "reached" if kind == "flag" else kind
                if kind == "flag" and task == "escape":
                    kind = "escaped"
                if info[key][i]:
                    events[i].append({"t": t, "type": kind, "value": value, "cell": cell})
            if done[i]:
                outcome[i] = {"reached": bool(info["reached"][i]), "bypass": bool(info["bypass"][i]),
                              "steps": t, "return": scores[i][-1]}
        finished |= done
        if finished.all():
            break

    episodes = []
    for m, maze in enumerate(mazes):
        agents = [{"path": paths[i], "score": scores[i], "events": events[i], **outcome[i]}
                  for i in np.flatnonzero(ids == m)]
        ep = {**maze.to_dict(), "agents": agents, "rewards": rewards.as_dict(), "task": task}
        if task == "escape":
            ep["start"] = list(maze.flag)
        episodes.append(ep)
    return episodes


def _cell(p):
    return [int(p[0]) - PAD, int(p[1]) - PAD]


def summarize(episodes):
    """Share of attempts that reached the flag, went outside, broke walls, etc."""
    agents = [a for ep in episodes for a in ep["agents"]]
    count = lambda kind, a: sum(e["type"] == kind for e in a["events"])
    went_out = [count("outside", a) > 0 for a in agents]
    stats = {
        "reached flag": np.mean([a["reached"] for a in agents]),
        "went outside": np.mean(went_out),
        "broke a wall": np.mean([count("through_wall", a) > 0 for a in agents]),
        "avg moves": np.mean([a["steps"] for a in agents]),
        "avg probes": np.mean([count("probe", a) for a in agents]),
        "avg score": np.mean([a["return"] for a in agents]),
    }
    return {"attempts": len(agents), **{k: round(float(v), 3) for k, v in stats.items()}}


def evaluate(agent, mazes, rewards=None, greedy=False, max_steps=200, seed=0, task="flag"):
    """Play one attempt per maze (all at once) and report how the agent did."""
    rewards = rewards or Rewards()
    bank = mazes if isinstance(mazes, MazeBank) else MazeBank.from_mazes(list(mazes))
    n = len(bank)
    env = MazeEnv(bank, n, rewards, max_steps, seed=seed, auto_reset=False, maze_ids=np.arange(n),
                  task=task)
    agent = None if agent is None else agent.cpu().eval()     # None: random moves
    gen = torch.Generator().manual_seed(seed)
    finished = torch.zeros(n, dtype=torch.bool)
    keys = ("reached", "bypass", "final_steps", "final_return", "final_probes", "final_breaks",
            "final_outside")
    final = {k: torch.zeros(n) for k in keys}
    went_out = torch.zeros(n, dtype=torch.bool)
    for _ in range(max_steps):
        if agent is None:
            action = torch.randint(0, 4, (n,), generator=gen)
        else:
            with torch.no_grad():
                obs, comp = env.observe()
                action, _, _ = agent.act(obs, comp, greedy=greedy)
        _, done, info = env.step(action)
        went_out |= info["outside"] & ~finished
        new = done & ~finished
        for k in keys:
            final[k][new] = info[k][new].float()
        finished |= done
        if finished.all():
            break
    # Time to finish, on attempts that reached the flag: moves taken, and moves per move of the shortest
    # route (through weak spots and outside allowed; the agent can't see weak spots, so 1.0 is a floor)
    reached = final["reached"].bool()
    shortest = torch.tensor([float(path_length(bank.grids[i], tuple(bank.starts[i]), tuple(bank.flags[i]),
                                               {FLOOR, WEAK, OUTSIDE}) or 0) for i in range(n)])
    solved = lambda x: round(float(x[reached].mean()), 2) if reached.any() else None
    return {"attempts": n,
            "moves to flag": solved(final["final_steps"]),
            "detour": solved(final["final_steps"] / shortest.clamp(min=1)),
            "reached flag": round(float(final["reached"].mean()), 3),
            "went outside": round(float(went_out.float().mean()), 3),
            "went around the maze": round(float(final["bypass"].mean()), 3),
            "avg moves": round(float(final["final_steps"].mean()), 1),
            "avg probes": round(float(final["final_probes"].mean()), 1),
            "avg walls broken": round(float(final["final_breaks"].mean()), 2),
            "avg moves outside": round(float(final["final_outside"].mean()), 1),
            "avg score": round(float(final["final_return"].mean()), 2)}


def standard_report(run, n=500, seed=99):
    """The fixed test suite used to compare training runs: new layouts of each kind, plus the
    article's two test mazes. Returns {test name: evaluate() results}.
    For the escape task: new layouts of each kind, plus the enclosed maze (its flag is walled in,
    so escaping from there means breaking a wall)."""
    from .maze import KINDS, SCENARIOS, generate_maze, scenario
    rng = np.random.default_rng(seed)
    tests = {kind: [generate_maze(rng, kind=kind) for _ in range(n)] for kind in KINDS}
    task = getattr(run.settings, "task", "flag")
    names = SCENARIOS if task == "flag" else ("enclosed",)
    tests.update({name: [scenario(name)] * n for name in names})
    max_steps = run.settings.max_steps
    return {name: evaluate(run.agent, mazes, rewards=run.rewards, max_steps=max_steps, seed=seed,
                           task=task)
            for name, mazes in tests.items()}


def breakout_stages(agent, maze=None, n=500, rewards=None, max_steps=200, seed=0):
    """Example 1, "breaks out to finish an impossible task" (default: the article's enclosed maze).
    The share of attempts that got through each stage: left the maze, broke back in from outside
    through a weak spot, reached the flag."""
    from .maze import scenario, OUTSIDE
    maze = maze or scenario("enclosed")
    ep = record(agent, [maze], n_agents=n, rewards=rewards, max_steps=max_steps, seed=seed)[0]
    out = {"left the maze": 0, "broke in from outside": 0, "reached the flag": 0}
    for a in ep["agents"]:
        path = [tuple(p) for p in a["path"]]
        broke_in = any(maze.grid[path[e["t"] - 1]] == OUTSIDE
                       for e in a["events"] if e["type"] == "through_wall")
        out["left the maze"] += any(e["type"] == "outside" for e in a["events"])
        out["broke in from outside"] += broke_in
        out["reached the flag"] += a["reached"]
    return {k: round(v / n, 3) for k, v in out.items()}
