"""One picture of exactly what the network receives at one moment of an attempt (TEACHING.md Part 1,
step 1, "What the agent sees").

Everything in panels 2-4 is read straight from the tensors ``MazeEnv.observe()`` hands the network, so
the figure cannot drift from what the agent really gets.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

from .env import (MazeEnv, Rewards, PAD, VIEW, CROP, N_MEMORY, N_COMPASS, EDGE,
                  UNSEEN, M_FLOOR, M_OUTSIDE, M_WALL, M_FLAG, M_EDGE)
from .maze import Maze, MazeBank, FLOOR, WALL, WEAK, OUTSIDE, ARENA, KINDS, MIX, generate_maze, scenario

# The player's palette (mazerl/player/player.js), plus a few colors it doesn't need
RED = "#d9472b"          # the agent
WALL_GRAY = "#dcdcdc"
WEAK_GRAY = "#f3f3f3"
TINT = "#f6dfd9"         # outside
FLAG_INK = "#262626"
INK, INK_2 = "#121212", "#727272"
UNSEEN_C = "#4b4f57"     # not seen yet: dark, like fog
EDGE_C = "#a9a9a9"       # beyond the arena
FLAG_C = "#eda100"       # the flag, in memory (gold, so it doesn't read as "not seen")
SEE_C = "#2a78d6"        # visible right now

# memory code -> (color, plain-words label), in the order of env.py's codes
MEMORY = {UNSEEN: (UNSEEN_C, "not seen yet"), M_FLOOR: ("#ffffff", "open floor"),
          M_OUTSIDE: (TINT, "outside the maze"), M_WALL: (WALL_GRAY, "wall"),
          M_FLAG: (FLAG_C, "flag"), M_EDGE: (EDGE_C, "edge of the arena (can't pass)")}
_MEM_CMAP = ListedColormap([MEMORY[k][0] for k in range(N_MEMORY)])

# compass sector k points k * 45 degrees counter-clockwise from "right" (env.compass)
DIRECTIONS = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]


def _resolve(agent, maze, seed, task):
    from .a2c import Run, Settings
    rewards, max_steps = None, Settings.max_steps
    if isinstance(agent, Run):
        rewards, max_steps = agent.rewards, agent.settings.max_steps
        task = task or getattr(agent.settings, "task", "flag")
        agent = agent.agent
    task = task or "flag"
    if maze is None:                       # same as mazerl.random_mazes(1, seed)[0]
        rng = np.random.default_rng(seed)
        maze = generate_maze(rng, kind=rng.choice(KINDS, size=1, p=MIX)[0])
    elif isinstance(maze, str):
        maze = scenario(maze)
    return agent, maze, task, rewards or Rewards(), max_steps


def _run_to(agent, maze, step, seed, task, rewards, max_steps):
    """Play one attempt for up to ``step`` moves; returns (env, observation, compass, moves, path, ended)."""
    env = MazeEnv(MazeBank.from_mazes([maze]), 1, rewards, max_steps, seed=seed, auto_reset=False,
                  maze_ids=[0], task=task)
    gen = torch.Generator().manual_seed(seed)
    if agent is not None:
        agent = agent.cpu().eval()
    path = [env.pos[0].tolist()]
    moves, ended = 0, False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        while moves < min(step, max_steps):
            if agent is None:
                action = torch.randint(0, 4, (1,), generator=gen)
            else:
                with torch.no_grad():
                    obs, comp = env.observe()
                    action, _, _ = agent.act(obs, comp)
            _, done, _ = env.step(action)
            moves += 1
            path.append(env.pos[0].tolist())
            if bool(done[0]):
                ended = True
                break
    obs, comp = env.observe()
    return env, obs[0].cpu().numpy(), comp[0].cpu().numpy(), moves, path, ended


def _cells(ax, n_rows, n_cols, x0=0, y0=0, color="#ffffff", lw=0.6):
    """Thin white grid lines between squares."""
    for r in range(n_rows + 1):
        ax.plot([x0 - 0.5, x0 + n_cols - 0.5], [y0 + r - 0.5] * 2, color=color, lw=lw, zorder=2)
    for c in range(n_cols + 1):
        ax.plot([x0 + c - 0.5] * 2, [y0 - 0.5, y0 + n_rows - 0.5], color=color, lw=lw, zorder=2)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _legend(ax, handles, ncol=2):
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=ncol,
              frameon=False, fontsize=9, handlelength=1.3, columnspacing=1.2)


def show_observation(agent=None, maze=None, step=10, seed=0, task=None):
    """Draw what the network receives at move ``step`` of one attempt.

    agent: a Run, an Agent, or None (random moves).
    maze: a Maze, a scenario name such as "enclosed", or None for a random layout made from ``seed``.
    task: "flag" or "escape"; defaults to the Run's task, else "flag".
    Returns the matplotlib Figure: the real maze, the agent's memory window, the extra clues on that
    window, and the compass. If the attempt ends before ``step``, the last moment is shown.
    """
    agent, maze, task, rewards, max_steps = _resolve(agent, maze, seed, task)
    env, obs, comp, moves, path, ended = _run_to(agent, maze, step, seed, task, rewards, max_steps)

    memory = obs[:N_MEMORY].argmax(0)              # (11, 11) memory code per square
    stood, now, firm = obs[N_MEMORY] > 0, obs[N_MEMORY + 1] > 0, obs[N_MEMORY + 2] > 0
    pr, pc = (int(v) - PAD for v in env.pos[0])   # agent, arena coordinates
    grid = env.grid[0].cpu().numpy()                # true layout now (broken weak spots are floor)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1.25, 1, 1, 0.8]})
    fig.patch.set_facecolor("white")
    when = f"after move {moves}" + (" (the attempt ended here)" if ended else "")
    goal = "escape: get out of the maze" if task == "escape" else "reach the flag"
    fig.suptitle(f"What the network sees {when}   ·   task: {goal}", fontsize=14, fontweight="bold",
                 x=0.02, ha="left", color=INK)

    # ---- (a) the real maze -----------------------------------------------------------------
    ax = axes[0]
    lo_r, hi_r = min(0, pr - PAD), max(ARENA - 1, pr + PAD)
    lo_c, hi_c = min(0, pc - PAD), max(ARENA - 1, pc + PAD)
    true_colors = {FLOOR: "#ffffff", WALL: WALL_GRAY, WEAK: WEAK_GRAY, OUTSIDE: TINT, EDGE: EDGE_C}
    rgb = np.array([[_hex(true_colors[int(grid[r + PAD, c + PAD])]) for c in range(lo_c, hi_c + 1)]
                    for r in range(lo_r, hi_r + 1)])
    ext = (lo_c - 0.5, hi_c + 0.5, hi_r + 0.5, lo_r - 0.5)
    ax.imshow(rgb, extent=ext, interpolation="nearest", zorder=0)
    for r, c in zip(*np.nonzero(grid == WEAK)):     # weak spots: wall with a pale middle, as in the player
        ax.add_patch(Rectangle((c - PAD - 0.5, r - PAD - 0.5), 1, 1, fc=WALL_GRAY, ec="none", zorder=1))
        ax.add_patch(Rectangle((c - PAD - 0.4, r - PAD - 0.4), 0.8, 0.8, fc=WEAK_GRAY, ec="none", zorder=1))
    vr, vc = np.nonzero(env.visible[0].cpu().numpy())
    for r, c in zip(vr, vc):
        ax.add_patch(Rectangle((pc - VIEW + c - 0.5, pr - VIEW + r - 0.5), 1, 1, fc=SEE_C, alpha=0.22,
                               ec="none", zorder=3))
    ax.add_patch(Rectangle((pc - VIEW - 0.5, pr - VIEW - 0.5), 2 * VIEW + 1, 2 * VIEW + 1, fill=False,
                           ec=SEE_C, lw=1.2, ls="--", zorder=4))
    ax.add_patch(Rectangle((pc - PAD - 0.5, pr - PAD - 0.5), CROP, CROP, fill=False, ec=INK, lw=2, zorder=4))
    trail = np.array(path) - PAD
    ax.plot(trail[:, 1], trail[:, 0], color=RED, alpha=0.55, lw=2.5, solid_capstyle="round", zorder=5)
    sr, sc = (int(v) - PAD for v in env.start[0])
    ax.add_patch(Circle((sc, sr), 0.3, fc="white", ec=RED, lw=1.2, zorder=5))
    if task != "escape":
        fr, fc = maze.flag
        ax.add_patch(Circle((fc, fr), 0.43, fc=FLAG_INK, ec="none", zorder=5))
        ax.plot([fc - 0.13, fc - 0.13], [fr + 0.25, fr - 0.22], color="white", lw=1, zorder=6)
        ax.fill([fc - 0.13, fc + 0.22, fc - 0.13], [fr - 0.23, fr - 0.09, fr + 0.06], color="white", zorder=6)
    ax.add_patch(Circle((pc, pr), 0.3, fc=RED, ec="none", zorder=7))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _bare(ax)
    ax.set_title("1. The real maze (the network never sees this)", fontsize=11, loc="left", color=INK)
    handles = [Line2D([], [], marker="o", ls="", color=RED, ms=8, label="the agent"),
               Patch(fc="none", ec=INK, lw=2, label="the 11×11 window it gets"),
               Patch(fc=SEE_C, alpha=0.3, ec=SEE_C, ls="--", label="what it can see right now (5×5)"),
               Line2D([], [], color=RED, alpha=0.55, lw=2.5, label="its path so far"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=RED, ms=8, label="where it started")]
    if task != "escape":
        handles.append(Line2D([], [], marker="o", ls="", color=FLAG_INK, ms=8, label="the flag"))
    handles += [Patch(fc=WEAK_GRAY, ec=WALL_GRAY, lw=2, label="weak spot (looks like wall to it)"),
                Patch(fc=EDGE_C, label="edge of the arena")]
    _legend(ax, handles)

    # ---- (b) memory window -------------------------------------------------------------------
    ax = axes[1]
    ax.imshow(memory, cmap=_MEM_CMAP, vmin=-0.5, vmax=N_MEMORY - 0.5, interpolation="nearest")
    _cells(ax, CROP, CROP, color="#bbbbbb", lw=0.4)
    ax.add_patch(Circle((PAD, PAD), 0.3, fc=RED, ec="none", zorder=5))
    ax.add_patch(Rectangle((-0.5, -0.5), CROP, CROP, fill=False, ec=INK, lw=2, zorder=6))
    _bare(ax)
    ax.set_title("2. Its memory of the 11×11 squares around it", fontsize=11, loc="left", color=INK)
    _legend(ax, [Patch(fc=MEMORY[k][0], ec="#bbbbbb", label=MEMORY[k][1]) for k in range(N_MEMORY)]
            + [Line2D([], [], marker="o", ls="", color=RED, ms=8, label="the agent (always in the middle)")])

    # ---- (c) extra clues on the same window ----------------------------------------------------
    ax = axes[2]
    ax.imshow(memory, cmap=_MEM_CMAP, vmin=-0.5, vmax=N_MEMORY - 0.5, interpolation="nearest", alpha=0.3)
    _cells(ax, CROP, CROP, color="#dddddd", lw=0.4)
    for r, c in zip(*np.nonzero(now)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=SEE_C, alpha=0.3, ec="none", zorder=3))
    sr_, sc_ = np.nonzero(stood)
    ax.scatter(sc_, sr_, s=28, color=INK_2, zorder=4)
    fr_, fc_ = np.nonzero(firm)
    ax.scatter(fc_, fr_, s=90, marker="X", color=RED, linewidths=0, zorder=5)
    ax.add_patch(Circle((PAD, PAD), 0.3, fc="none", ec=RED, lw=1.8, zorder=6))
    ax.add_patch(Rectangle((-0.5, -0.5), CROP, CROP, fill=False, ec=INK, lw=2, zorder=6))
    _bare(ax)
    ax.set_title("3. Three more layers on the same squares", fontsize=11, loc="left", color=INK)
    _legend(ax, [Patch(fc=SEE_C, alpha=0.3, label="can see right now"),
                 Line2D([], [], marker="o", ls="", color=INK_2, ms=6, label="has stood here"),
                 Line2D([], [], marker="X", ls="", color=RED, ms=9, label="bumped this wall; it held firm")],
            ncol=1)

    # ---- (d) compass ---------------------------------------------------------------------------
    ax = axes[3]
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.6, 1.45)
    _bare(ax)
    ax.add_patch(Circle((0, 0), 1, fc="#fafafa", ec="#cccccc", lw=1))
    for k in range(N_COMPASS):
        a = k * np.pi / 4
        ax.plot([0.88 * np.cos(a), np.cos(a)], [0.88 * np.sin(a), np.sin(a)], color="#bbbbbb", lw=1)
        ax.text(1.22 * np.cos(a), 1.22 * np.sin(a), DIRECTIONS[k], ha="center", va="center",
                fontsize=8, color=INK_2)
    if comp.sum() > 0:
        k = int(comp.argmax())
        ax.add_patch(Circle((0, 0), 0.07, fc=INK, zorder=6))
        a = k * np.pi / 4
        ax.add_patch(FancyArrowPatch((0, 0), (0.85 * np.cos(a), 0.85 * np.sin(a)), arrowstyle="-|>",
                                     mutation_scale=22, lw=3, color=RED, zorder=5))
        note = f"The flag is {DIRECTIONS[k]}.\n(One of 8 directions; no distance.)"
    else:
        ax.text(0, 0, "no compass", ha="center", va="center", fontsize=12, color=INK_2, fontweight="bold")
        note = ("Escape task: all 8 compass inputs\nare zero. There's no flag to point to."
                if task == "escape" else "All 8 compass inputs are zero.")
    ax.text(0, -1.42, note, ha="center", va="top", fontsize=9, color=INK)
    ax.set_title("4. The compass", fontsize=11, loc="left", color=INK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.2, wspace=0.08)
    return fig


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
