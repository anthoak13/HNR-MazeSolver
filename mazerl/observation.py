"""One picture of exactly what the network receives at one moment of an attempt (TEACHING.md Part 1,
step 1, "What the agent sees").

The memory window (all 9 layers) and the compass are read straight from the tensors
``MazeEnv.observe()`` hands the network, so the figure cannot drift from what the agent really gets.

``show_pretraining_quiz`` uses the same window drawing for Step 2: one moment of the scripted explorer's
walk, the answer key for the hidden squares, and the pretrained network's guesses.
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
    explorer = isinstance(agent, str) and agent == "explorer"
    if agent is not None and not explorer:
        agent = agent.cpu().eval()
    path = [env.pos[0].tolist()]
    moves, ended = 0, False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        while moves < min(step, max_steps):
            if agent is None:
                action = torch.randint(0, 4, (1,), generator=gen)
            elif explorer:
                from .perception import explorer_moves
                action = explorer_moves(env, gen)
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
    Returns the matplotlib Figure: the real maze, the agent's memory window, and the compass. If the attempt ends before ``step``, the last moment is shown.
    """
    agent, maze, task, rewards, max_steps = _resolve(agent, maze, seed, task)
    env, obs, comp, moves, path, ended = _run_to(agent, maze, step, seed, task, rewards, max_steps)

    memory = obs[:N_MEMORY].argmax(0)              # (11, 11) memory code per square
    stood, now, firm = obs[N_MEMORY] > 0, obs[N_MEMORY + 1] > 0, obs[N_MEMORY + 2] > 0
    pr, pc = (int(v) - PAD for v in env.pos[0])   # agent, arena coordinates
    grid = env.grid[0].cpu().numpy()                # true layout now (broken weak spots are floor)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), gridspec_kw={"width_ratios": [1.25, 1.3]})
    fig.patch.set_facecolor("white")
    when = f"after move {moves}" + (" (the attempt ended here)" if ended else "")
    goal = "escape: get out of the maze" if task == "escape" else "reach the flag"
    fig.suptitle(f"What the agent sees {when}   ·   task: {goal}", fontsize=13, fontweight="bold",
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
        _flag(ax, *maze.flag)
    ax.add_patch(Circle((pc, pr), 0.3, fc=RED, ec="none", zorder=7))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _bare(ax)
    ax.set_title("The real maze (the agent never sees this)", fontsize=11, loc="left", color=INK)

    # ---- (b) memory window -------------------------------------------------------------------
    ax = axes[1]
    _draw_memory(ax, memory, now, stood, firm)

    # the compass: a small dial beside the window (the network's other input)
    cx, cy, rad = CROP + 1.6, 2.0, 1.5
    ax.add_patch(Circle((cx, cy), rad, fc="#fafafa", ec="#cccccc", lw=1, zorder=5))
    for k in range(N_COMPASS):
        a = k * np.pi / 4                       # image y runs downward, so "up" is -y
        ax.plot([cx + 0.85 * rad * np.cos(a), cx + rad * np.cos(a)],
                [cy - 0.85 * rad * np.sin(a), cy - rad * np.sin(a)], color="#bbbbbb", lw=1, zorder=6)
    if comp.sum() > 0:
        k = int(comp.argmax())
        a = k * np.pi / 4
        ax.add_patch(FancyArrowPatch((cx, cy), (cx + 0.8 * rad * np.cos(a), cy - 0.8 * rad * np.sin(a)),
                                     arrowstyle="-|>", mutation_scale=16, lw=2.5, color=RED, zorder=7))
        ax.add_patch(Circle((cx, cy), 0.12, fc=INK, zorder=8))
        note = f"compass:\nflag is {DIRECTIONS[k]}"
    else:
        ax.text(cx, cy, "off", ha="center", va="center", fontsize=10, color=INK_2, fontweight="bold",
                zorder=7)
        note = "compass:\nnone (escape)" if task == "escape" else "compass:\noff"
    ax.text(cx, cy + rad + 0.4, note, ha="center", va="top", fontsize=9, color=INK)
    ax.set_xlim(-0.5, CROP + 3.4)
    ax.set_ylim(CROP - 0.5, -0.5)
    _bare(ax)
    ax.set_title("What the agent gets: its 11×11 memory window + compass", fontsize=11,
                 loc="left", color=INK)

    # ---- legends: one under each panel --------------------------------------------------------
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9,
              title_fontsize=9.5, handlelength=1.3, columnspacing=1.4, alignment="left")
    real = [Line2D([], [], marker="o", ls="", color=RED, ms=8, label="the agent"),
            Line2D([], [], marker="o", ls="", mfc="white", mec=RED, ms=8, label="where it started"),
            Line2D([], [], color=RED, alpha=0.55, lw=2.5, label="its path so far"),
            Patch(fc="none", ec=INK, lw=2, label="the 11×11 window it gets"),
            Patch(fc="none", ec=SEE_C, lw=1.2, ls="--", label="maximum vision range (5×5)"),
            Patch(fc=WEAK_GRAY, ec=WALL_GRAY, lw=2, label="weak spot (looks like wall to it)")]
    axes[0].legend(handles=real, ncol=2, title="The real maze", **kw)
    axes[1].legend(handles=_memory_handles(), ncol=3, title="Memory: 9 layers", **kw)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.24, wspace=0.06)
    plt.close(fig)       # Jupyter shows the returned figure; closing stops a second copy
    return fig


def _draw_memory(ax, memory, now=None, stood=None, firm=None):
    """An 11 x 11 window colored by memory code, with the flag symbol, the three extra layers (when
    given) and the agent in the middle. Shared by show_observation and show_pretraining_quiz."""
    shown = np.where(memory == M_FLAG, M_FLOOR, memory)       # the flag is drawn as a symbol instead
    ax.imshow(shown, cmap=_MEM_CMAP, vmin=-0.5, vmax=N_MEMORY - 0.5, interpolation="nearest")
    _cells(ax, CROP, CROP, color="#bbbbbb", lw=0.4)
    for r, c in zip(*np.nonzero(memory == M_FLAG)):
        _flag(ax, r, c)
    if now is not None:
        for (x0, y0), (x1, y1) in _outline(now):              # layer 8: what it can see now
            ax.plot([x0, x1], [y0, y1], color=SEE_C, lw=1.8, ls=(0, (3, 2)), zorder=8)
    if stood is not None:
        centre = (np.arange(CROP)[:, None] == PAD) & (np.arange(CROP) == PAD)
        stood_r, stood_c = np.nonzero(stood & ~centre)
        ax.scatter(stood_c, stood_r, s=22, color=INK_2, zorder=6)   # layer 7: stood here
    if firm is not None:
        firm_r, firm_c = np.nonzero(firm)
        ax.scatter(firm_c, firm_r, s=80, marker="X", color=RED, linewidths=0, zorder=6)  # layer 9
    ax.add_patch(Circle((PAD, PAD), 0.3, fc=RED, ec="none", zorder=7))
    ax.add_patch(Rectangle((-0.5, -0.5), CROP, CROP, fill=False, ec=INK, lw=2, zorder=6))
    ax.set_xlim(-0.5, CROP - 0.5)
    ax.set_ylim(CROP - 0.5, -0.5)
    _bare(ax)


def _memory_handles(extra=True):
    square = lambda code, label: Patch(fc=MEMORY[code][0], ec="#bbbbbb", label=label)
    handles = [square(UNSEEN, "not seen"), square(M_FLOOR, "open floor"), square(M_OUTSIDE, "outside"),
               square(M_WALL, "wall"),
               Line2D([], [], marker="o", ls="", color=FLAG_INK, ms=8, label="flag"),
               square(M_EDGE, "map edge")]
    if extra:
        handles += [Line2D([], [], marker="X", ls="", color=RED, ms=8, mew=0, label="have prodded (it held)"),
                    Line2D([], [], marker="o", ls="", color=INK_2, ms=5, label="have stood"),
                    Line2D([], [], color=SEE_C, lw=1.8, ls=(0, (3, 2)), label="can see now")]
    return handles


def _outline(mask):
    """Edges between squares in ``mask`` and squares outside it, merged into straight runs so the
    dashes flow along each side. Returns [((x0, y0), (x1, y1)), ...] in image coordinates."""
    n_r, n_c = mask.shape
    inside = lambda r, c: 0 <= r < n_r and 0 <= c < n_c and bool(mask[r, c])
    runs = []
    for r in range(n_r + 1):                    # horizontal edges, on the line y = r - 0.5
        c = 0
        while c < n_c:
            if inside(r - 1, c) != inside(r, c):
                start = c
                while c < n_c and inside(r - 1, c) != inside(r, c):
                    c += 1
                runs.append(((start - 0.5, r - 0.5), (c - 0.5, r - 0.5)))
            else:
                c += 1
    for c in range(n_c + 1):                    # vertical edges, on the line x = c - 0.5
        r = 0
        while r < n_r:
            if inside(r, c - 1) != inside(r, c):
                start = r
                while r < n_r and inside(r, c - 1) != inside(r, c):
                    r += 1
                runs.append(((c - 0.5, start - 0.5), (c - 0.5, r - 0.5)))
            else:
                r += 1
    return runs


def _flag(ax, r, c):
    """The player's flag symbol: a black disc with a white pennant."""
    ax.add_patch(Circle((c, r), 0.43, fc=FLAG_INK, ec="none", zorder=5))
    ax.plot([c - 0.13, c - 0.13], [r + 0.25, r - 0.22], color="white", lw=1, zorder=6)
    ax.fill([c - 0.13, c + 0.22, c - 0.13], [r - 0.23, r - 0.09, r + 0.06], color="white", zorder=6)


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


# prediction class (perception.CLASS_NAMES: open, outside, wall, edge) -> memory code, for coloring
_CLASS_TO_MEMORY = np.array([M_FLOOR, M_OUTSIDE, M_WALL, M_EDGE])


def show_pretraining_quiz(perception, maze=None, step=20, seed=0):
    """One question from the pretraining quiz, at move ``step`` of the scripted explorer's walk.

    perception: a Pretrained (``mz.load_perception()``) that kept its prediction read-outs.
    maze: a Maze, a scenario name, or None for a random layout made from ``seed``.
    Returns a Figure with three windows: what the network is given; the answers (hidden squares filled
    in); the network's guesses, with wrong guesses outlined. (Pretraining also has a second, easy
    question, the direction to unexplored ground; it isn't drawn.)
    """
    from .perception import true_window
    _, maze, _, rewards, max_steps = _resolve(None, maze, seed, "flag")
    env, obs, comp, moves, path, ended = _run_to("explorer", maze, step, seed, "flag", rewards, max_steps)

    memory = obs[:N_MEMORY].argmax(0)
    stood, now, firm = obs[N_MEMORY] > 0, obs[N_MEMORY + 1] > 0, obs[N_MEMORY + 2] > 0
    hidden = memory == UNSEEN
    truth = true_window(env)[0].cpu().numpy()
    o, c = env.observe()
    guess = perception.guess(o.cpu(), c.cpu())[0][0].numpy()

    answer = np.where(hidden, _CLASS_TO_MEMORY[truth], memory)
    guessed = np.where(hidden, _CLASS_TO_MEMORY[guess], memory)
    wrong = hidden & (guess != truth)
    n_hidden, n_right = int(hidden.sum()), int((hidden & ~wrong).sum())

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Pretraining quiz: the explorer after move {moves}", fontsize=13, fontweight="bold",
                 x=0.02, ha="left", color=INK)

    _draw_memory(axes[0], memory, now, stood, firm)
    axes[0].set_title("1. What the network is given", fontsize=11, loc="left", color=INK)

    _draw_memory(axes[1], answer)
    _seen_line(axes[1], hidden)
    axes[1].set_title("2. The answers", fontsize=11, loc="left", color=INK)

    _draw_memory(axes[2], guessed)
    _seen_line(axes[2], hidden)
    for r, cc in zip(*np.nonzero(wrong)):
        axes[2].add_patch(Rectangle((cc - 0.45, r - 0.45), 0.9, 0.9, fill=False, ec=RED, lw=1.8, zorder=9))
    axes[2].set_title(f"3. The network's guesses: {n_right} of {n_hidden} right", fontsize=11, loc="left",
                      color=INK)

    handles = _memory_handles() + [
        Line2D([], [], color=INK, lw=1.4, ls=(0, (1, 1.5)), label="edge of what it had seen"),
        Patch(fc="none", ec=RED, lw=1.8, label="wrong guess")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
               handlelength=1.4, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.2, wspace=0.06)
    plt.close(fig)       # Jupyter shows the returned figure; closing stops a second copy
    return fig


def _seen_line(ax, hidden):
    for (x0, y0), (x1, y1) in _outline(~hidden):
        ax.plot([x0, x1], [y0, y1], color=INK, lw=1.4, ls=(0, (1, 1.5)), zorder=8)
