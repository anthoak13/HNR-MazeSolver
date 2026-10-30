"""Figures that explain reward-free pretraining: one exploring walker on a real training maze, its
path, and snapshots along the path showing what the network is given and what it is asked to guess.

    python scripts/pretraining_figures.py --out reports/figures/pretraining
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle, FancyArrow

from mazerl.env import (MazeEnv, PAD, VIEW, CROP, UNSEEN, M_FLOOR, M_OUTSIDE, M_WALL, M_FLAG, M_EDGE, EDGE,
                        MOVES)
from mazerl.maze import ARENA, WALL, WEAK, OUTSIDE, training_bank
from mazerl.perception import frontier_direction

# The animation player's palette (mazerl/player/player.js)
RED, WALL_C, WEAK_C, TINT, FLAG_C, INK = "#d9472b", "#dcdcdc", "#f3f3f3", "#f6dfd9", "#262626", "#121212"
EDGE_C, UNSEEN_C, FLOOR_C = "#8c8c8c", "#ffffff", "#ffffff"
ARROWS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}       # move index -> (dx, dy) on screen


def walk(maze_id, seed, explore=0.3, max_steps=200):
    """One walker, as in perception.collect(): mostly toward the nearest unexplored open square,
    sometimes at random. Records everything needed for the figures after every move."""
    bank = training_bank(verbose=False)
    env = MazeEnv(bank, 1, max_steps=max_steps, seed=seed, auto_reset=False, maze_ids=[maze_id])
    gen = torch.Generator().manual_seed(seed)
    frames = []
    for t in range(max_steps):
        front = int(frontier_direction(env)[0])
        frames.append({"pos": env.pos[0].tolist(), "memory": env.memory[0].clone().numpy(),
                       "visited": env.visited[0].clone().numpy(), "probed": env.probed[0].clone().numpy(),
                       "grid": env.grid[0].clone().numpy(), "front": front,
                       "compass": int(env.compass()[0].argmax())})
        rand = int(torch.randint(0, 4, (1,), generator=gen))
        wander = float(torch.rand(1, generator=gen)) < explore
        action = rand if (wander or front < 0) else front
        frames[-1]["random"] = action != front
        _, done, _ = env.step(torch.tensor([action]))
        if done[0]:
            frames.append({"pos": env.pos[0].tolist(), "end": True})
            break
    return bank, frames


def _cell(ax, r, c, color, **kw):
    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor="none", **kw))


def draw_true(ax, grid, rows, cols, flag=None, start=None):
    """The real contents of grid[rows, cols] (padded coordinates), drawn at local coordinates."""
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = grid[r, c]
            if v == OUTSIDE:
                _cell(ax, i, j, TINT)
            elif v == WALL:
                _cell(ax, i, j, WALL_C)
            elif v == WEAK:
                _cell(ax, i, j, WALL_C)
                ax.add_patch(Rectangle((j - 0.32, i - 0.32), 0.64, 0.64, facecolor=WEAK_C, edgecolor="none"))
            elif v == EDGE:
                _cell(ax, i, j, EDGE_C)
            if flag is not None and (r, c) == tuple(flag):
                ax.plot(j, i, marker=(3, 0, 0), color=FLAG_C, markersize=9)
            if start is not None and (r, c) == tuple(start):
                ax.plot(j, i, "o", mfc="none", mec=RED, markersize=9, mew=1.5)


def _frame(ax, n, title):
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def path_figure(env_grid0, frames, flag, picks, out):
    rows = cols = range(PAD, PAD + ARENA)
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    draw_true(ax, env_grid0, rows, cols, flag=flag, start=frames[0]["pos"])
    last = max(picks)
    pts = np.array([f["pos"] for f in frames[:last + 1]]) - PAD
    # the path, faint early and darker late, with random moves marked
    for k in range(len(pts) - 1):
        (r0, c0), (r1, c1) = pts[k], pts[k + 1]
        ax.plot([c0, c1], [r0, r1], color=RED, lw=1.6, alpha=0.25 + 0.6 * k / max(1, len(pts) - 1),
                solid_capstyle="round")
    rnd = [k for k in range(last) if frames[k].get("random")]
    ax.plot(pts[rnd, 1], pts[rnd, 0], "o", color=RED, markersize=2.5, alpha=0.6, lw=0)
    for n, k in enumerate(picks, 1):
        r, c = pts[k]
        ax.plot(c, r, "o", color="white", mec=INK, markersize=15, mew=1.2, zorder=5)
        ax.text(c, r, str(n), ha="center", va="center", fontsize=8.5, color=INK, zorder=6, weight="bold")
    _frame(ax, ARENA, f"One exploring walker, first {last} moves")
    ax.text(0, ARENA + 0.2, "Pink: outside the maze.  Gray: walls (pale centre: a weak spot).  Triangle: flag.\n"
            "Red line: the walker's path (small dots: random moves).  Numbers: the snapshots shown next.",
            fontsize=8, color="#555555", va="top")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


MEM_COLOR = {M_FLOOR: FLOOR_C, M_OUTSIDE: TINT, M_WALL: WALL_C, M_EDGE: EDGE_C, M_FLAG: FLOOR_C}


def draw_input(ax, f):
    """What the network is given: its 11 x 11 memory window (plus the compass, drawn as a pointer)."""
    r0, c0 = f["pos"]
    rows, cols = range(r0 - PAD, r0 + PAD + 1), range(c0 - PAD, c0 + PAD + 1)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            m = f["memory"][r, c]
            if m == UNSEEN:
                _cell(ax, i, j, UNSEEN_C)
                ax.text(j, i, "?", ha="center", va="center", fontsize=8, color="#9a9a9a")
                continue
            _cell(ax, i, j, MEM_COLOR[m])
            if m == M_FLAG:
                ax.plot(j, i, marker=(3, 0, 0), color=FLAG_C, markersize=7)
            if f["visited"][r, c] and (r, c) != (r0, c0):
                ax.plot(j, i, ".", color=RED, markersize=3, alpha=0.7)
            if f["probed"][r, c]:
                ax.plot(j, i, "x", color=INK, markersize=5, mew=1)
    v = VIEW
    ax.add_patch(Rectangle((PAD - v - 0.5, PAD - v - 0.5), 2 * v + 1, 2 * v + 1, fill=False,
                           edgecolor=RED, lw=1, ls="--"))
    ax.plot(PAD, PAD, "o", color=RED, markersize=8)
    _grid_lines(ax)


def draw_answer(ax, f):
    """What the network is asked to guess: the real contents of the squares it hasn't seen
    (outlined), and the move toward the nearest unexplored open square (arrow)."""
    r0, c0 = f["pos"]
    rows, cols = range(r0 - PAD, r0 + PAD + 1), range(c0 - PAD, c0 + PAD + 1)
    draw_true(ax, f["grid"], rows, cols)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if f["memory"][r, c] == UNSEEN:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#2b6cb0", lw=1.1))
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", alpha=0.65,
                                       edgecolor="none"))
    ax.plot(PAD, PAD, "o", color=RED, markersize=8)
    if f["front"] >= 0:
        dx, dy = ARROWS[f["front"]]
        ax.add_patch(FancyArrow(PAD, PAD, 1.6 * dx, 1.6 * dy, width=0.12, head_width=0.5, head_length=0.45,
                                length_includes_head=True, color="#2b6cb0", zorder=6))
    _grid_lines(ax)


def _grid_lines(ax):
    n = 2 * PAD + 1
    for k in range(n + 1):
        ax.plot([k - 0.5, k - 0.5], [-0.5, n - 0.5], color="#eeeeee", lw=0.5, zorder=0.5)
        ax.plot([-0.5, n - 0.5], [k - 0.5, k - 0.5], color="#eeeeee", lw=0.5, zorder=0.5)


COMPASS = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]


def snapshots_figure(frames, picks, out):
    fig, axes = plt.subplots(len(picks), 2, figsize=(7.2, 3.5 * len(picks)))
    for n, (k, row) in enumerate(zip(picks, axes), 1):
        f = frames[k]
        unseen = int((f["memory"][f["pos"][0] - PAD:f["pos"][0] + PAD + 1,
                                  f["pos"][1] - PAD:f["pos"][1] + PAD + 1] == UNSEEN).sum())
        draw_input(row[0], f)
        _frame(row[0], 2 * PAD + 1, f"{n}. Move {k}: what the network is given")
        row[0].text(-0.5, 2 * PAD + 1, f"Compass input: flag to the {COMPASS[f['compass']]}",
                    fontsize=8, color="#555555", va="top")
        draw_answer(row[1], f)
        _frame(row[1], 2 * PAD + 1, f"What it must guess ({unseen} hidden squares)")
    fig.text(0.01, -0.01,
             "Left: the agent's 11×11 memory window, centred on the walker (red dot). Dashed box: what it can see "
             "right now (5×5).\n"
             "'?' = not seen yet. Pink: outside. Light gray: wall. Dark gray: beyond the arena edge. Triangle: flag.\n"
             "Small red dots: squares it has stood on. ×: a wall it walked into that held firm.\n"
             "Right: the answers. Blue-outlined squares are the hidden ones it must guess (open, outside, wall or edge; "
             "weak spots count as wall).\n"
             "Squares it has already seen are faded: they are in the input, so they count for little. Blue arrow: the move "
             "toward the nearest open square\nit knows about but hasn't stood on.",
             fontsize=8, color="#555555", va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


UNSEEN_FILL, GUESS = "#e8edf4", "#2b6cb0"


def _window(f):
    r0, c0 = f["pos"]
    return range(r0 - PAD, r0 + PAD + 1), range(c0 - PAD, c0 + PAD + 1)


def draw_memory(ax, f, rows, cols, unseen="fill"):
    """The agent's memory over rows x cols (padded coordinates), drawn at local coordinates."""
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            m = f["memory"][r, c]
            if m == UNSEEN:
                if unseen == "fill":
                    _cell(ax, i, j, UNSEEN_FILL)
                elif unseen == "?":
                    ax.text(j, i, "?", ha="center", va="center", fontsize=8, color="#9a9a9a")
                continue
            _cell(ax, i, j, MEM_COLOR[m])
            if m == M_FLAG:
                ax.plot(j, i, marker=(3, 0, 0), color=FLAG_C, markersize=6)
            if f["visited"][r, c]:
                ax.plot(j, i, ".", color=RED, markersize=2.5, alpha=0.7)
            if f["probed"][r, c]:
                ax.plot(j, i, "x", color=INK, markersize=4, mew=0.9)


def _crop_box(ax, f, offset):
    r0, c0 = f["pos"]
    r, c = r0 - offset, c0 - offset
    ax.add_patch(Rectangle((c - PAD - 0.5, r - PAD - 0.5), CROP, CROP, fill=False, edgecolor=INK, lw=1.6))
    ax.plot(c, r, "o", color=RED, markersize=6, zorder=5)


def draw_filled(ax, f):
    """The 11 x 11 window with its hidden squares filled in with what's really there (hatched blue):
    the answers to the first guess. Arrow: the answer to the second guess."""
    rows, cols = _window(f)
    draw_memory(ax, f, rows, cols, unseen="blank")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if f["memory"][r, c] != UNSEEN:
                continue
            v = f["grid"][r, c]
            color = {OUTSIDE: TINT, WALL: WALL_C, WEAK: WALL_C, EDGE: EDGE_C}.get(int(v), FLOOR_C)
            _cell(ax, i, j, UNSEEN_FILL)                      # a hidden square, with its answer inside
            ax.add_patch(Rectangle((j - 0.34, i - 0.34), 0.68, 0.68, facecolor=color, edgecolor=GUESS, lw=0.8))
    ax.plot(PAD, PAD, "o", color=RED, markersize=8, zorder=5)
    if f["front"] >= 0:
        dx, dy = ARROWS[f["front"]]
        ax.add_patch(FancyArrow(PAD, PAD, 1.6 * dx, 1.6 * dy, width=0.14, head_width=0.55, head_length=0.45,
                                length_includes_head=True, color=GUESS, zorder=6))
    _grid_lines(ax)


def draw_crop_input(ax, f):
    rows, cols = _window(f)
    draw_memory(ax, f, rows, cols, unseen="?")
    ax.add_patch(Rectangle((PAD - VIEW - 0.5, PAD - VIEW - 0.5), 2 * VIEW + 1, 2 * VIEW + 1, fill=False,
                           edgecolor=RED, lw=1, ls="--"))
    ax.plot(PAD, PAD, "o", color=RED, markersize=8, zorder=5)
    _grid_lines(ax)


FULL = range(PAD - 3, PAD + ARENA + 3)      # the arena plus a strip of edge, enough for the crop box


def _note(fig, text):
    fig.text(0.01, -0.005, text, fontsize=8, color="#555555", va="top")


def version_a(frames, picks, out):
    """Per snapshot: the whole memory with the window the network gets; that window with its hidden
    squares filled in (what it must guess)."""
    n_full = len(FULL)
    fig, axes = plt.subplots(len(picks), 2, figsize=(8.4, 4.3 * len(picks)),
                             gridspec_kw={"width_ratios": [1, 1]})
    for n, (k, row) in enumerate(zip(picks, axes), 1):
        f = frames[k]
        draw_memory(row[0], f, FULL, FULL)
        _crop_box(row[0], f, FULL[0])
        _frame(row[0], n_full, f"{n}. Move {k}: everything it remembers")
        draw_filled(row[1], f)
        _frame(row[1], 2 * PAD + 1, "The window it gets, with the answers")
    _note(fig, "Left: the agent's memory of the whole arena so far. Pale blue: not seen yet. Pink: outside. "
               "Light gray: wall. Dark gray: arena edge.\nRed dots: squares it has stood on. ×: a wall it walked "
               "into that held firm. Black box: the 11×11 window the network is given (plus the compass).\n"
               "Right: that window. Pale blue squares are the ones it hasn't seen; the small square inside each is "
               "what is really there,\nwhich the network must guess (open, outside, wall or edge; weak spots count "
               "as wall). "
               "Blue arrow: the move toward the nearest open square it hasn't stood on, its second guess.")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def version_b(frames, picks, out):
    """Per snapshot: the real maze with what it has seen highlighted, and the window the network gets.
    Then one example of the exercise: fill in the '?' squares."""
    n_full = len(FULL)
    rows_n = len(picks) + 1
    fig, axes = plt.subplots(rows_n, 2, figsize=(8.4, 4.3 * rows_n),
                             gridspec_kw={"width_ratios": [1, 1]})
    for n, (k, row) in enumerate(zip(picks, axes), 1):
        f = frames[k]
        draw_true(row[0], f["grid"], FULL, FULL)
        for i, r in enumerate(FULL):
            for j, c in enumerate(FULL):
                if f["memory"][r, c] == UNSEEN:
                    row[0].add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", alpha=0.6,
                                               edgecolor="none"))
        _crop_box(row[0], f, FULL[0])
        _frame(row[0], n_full, f"{n}. Move {k}: the maze (seen so far in full colour)")
        draw_crop_input(row[1], f)
        _frame(row[1], 2 * PAD + 1, "What the network is given")
    k = picks[1]
    draw_crop_input(axes[-1][0], frames[k])
    _frame(axes[-1][0], 2 * PAD + 1, f"The exercise (snapshot 2): fill in the ?")
    draw_filled(axes[-1][1], frames[k])
    _frame(axes[-1][1], 2 * PAD + 1, "The answers")
    _note(fig, "Top rows, left: the real maze; squares the agent hasn't seen yet are faded. Black box: the 11×11 "
               "window of its memory the network is given.\nRight: that window as the network gets it. '?': not "
               "seen yet. Dashed box: what it can see right now (5×5). Red dots: squares it has stood on.\n"
               "Bottom row: the exercise. Each '?' becomes a pale blue square with the answer inside: what is really "
               "there (weak spots count as wall);\nthe arrow is the move toward the nearest open square it hasn't stood on.")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/figures/pretraining"))
    ap.add_argument("--maze", type=int, default=6, help="training layout index")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--picks", default="4,20,38,60", help="moves at which to take snapshots")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bank = training_bank(verbose=False)
    maze_id = a.maze
    _, frames = walk(maze_id, a.seed)
    picks = [k for k in map(int, a.picks.split(",")) if k < len(frames) - 1]
    flag = np.array(bank.flags[maze_id]) + PAD
    path_figure(frames[0]["grid"], frames, flag, picks, a.out / "walker-path.png")
    version_a(frames, picks, a.out / "snapshots-a.png")
    version_b(frames, picks, a.out / "snapshots-b.png")
    print(f"maze {maze_id}, {len(frames) - 1} moves, snapshots at {picks} -> {a.out}")


if __name__ == "__main__":
    main()
