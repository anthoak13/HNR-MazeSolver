"""Figures that explain reward-free pretraining. No text inside the images: the surrounding text
explains them.

  walker-path.png  one exploring walker on a real training maze; numbered markers where the
                   top-row snapshots are taken
  snapshots.png    top row: the walker's memory growing (whole arena), with the 11x11 window the
                   network gets and the 5x5 it sees right now; bottom row, one snapshot up close:
                   what the network gets, the first guess (what is under the unseen squares), the
                   second guess (which move heads toward unexplored ground)

    python scripts/pretraining_figures.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle, FancyArrow, Circle

from mazerl.env import MazeEnv, PAD, VIEW, CROP, UNSEEN, M_FLOOR, M_OUTSIDE, M_WALL, M_FLAG, M_EDGE, EDGE
from mazerl.maze import ARENA, FLOOR, WALL, WEAK, OUTSIDE, training_bank
from mazerl.perception import frontier_direction

FLOOR_C, WALL_C, EDGE_C, TINT, FOG = "#ffffff", "#dcdcdc", "#4a4a4a", "#f6dfd9", "#b7c4d6"   # wall and outside as in the player
RED, BLUE, INK = "#d9472b", "#2b6cb0", "#121212"
MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}          # move index -> (d row, d col)
TRUE_COLOR = {OUTSIDE: TINT, WALL: WALL_C, WEAK: WALL_C, EDGE: EDGE_C}     # weak spots count as wall
MEM_COLOR = {M_FLOOR: FLOOR_C, M_OUTSIDE: TINT, M_WALL: WALL_C, M_EDGE: EDGE_C, M_FLAG: FLOOR_C}
MARGIN = PAD                                 # squares drawn beyond the arena in the memory panels


def walk(maze_id, seed, explore=0.3, max_steps=200):
    """One walker, as in perception.collect(): mostly toward the nearest unexplored open square,
    sometimes at random. Records what the figures need after every move."""
    bank = training_bank(verbose=False)
    env = MazeEnv(bank, 1, max_steps=max_steps, seed=seed, auto_reset=False, maze_ids=[maze_id])
    gen = torch.Generator().manual_seed(seed)
    frames = []
    for _ in range(max_steps):
        front = int(frontier_direction(env)[0])
        frames.append({"pos": tuple(env.pos[0].tolist()), "memory": env.memory[0].clone().numpy(),
                       "probed": env.probed[0].clone().numpy(), "grid": env.grid[0].clone().numpy(),
                       "front": front, "compass": int(env.compass()[0].argmax())})
        rand = int(torch.randint(0, 4, (1,), generator=gen))
        wander = float(torch.rand(1, generator=gen)) < explore
        _, done, _ = env.step(torch.tensor([rand if (wander or front < 0) else front]))
        if done[0]:
            break
    return bank, frames


# ---------------------------------------------------------------------------------------------
def _cell(ax, i, j, color):
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="none"))


def _bare(ax, n):
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def _held_firm(ax, i, j):
    for s in (1, -1):
        ax.plot([j - 0.22, j + 0.22], [i - 0.22 * s, i + 0.22 * s], color=INK, lw=1.1, zorder=5)


def _trail(ax, frames, k, r0, c0, lw=2.0):
    """The walker's path up to move k, in local coordinates (origin at padded square r0, c0)."""
    pts = np.array([f["pos"] for f in frames[:k + 1]], dtype=float)
    ax.plot(pts[:, 1] - c0, pts[:, 0] - r0, color=RED, lw=lw, alpha=0.8, solid_capstyle="round",
            solid_joinstyle="round", zorder=4)
    r, c = frames[k]["pos"]
    ax.plot(c - c0, r - r0, "o", color=RED, markersize=8 if lw > 1.5 else 5, zorder=6)


def _blend(a, b, t):
    a, b = (np.array(matplotlib.colors.to_rgb(x)) for x in (a, b))
    return tuple((1 - t) * a + t * b)


def draw_memory(ax, f, rows, cols, reveal=False):
    """The walker's memory over rows x cols (padded coordinates). Unseen squares are fog; with
    reveal, they show what is really there instead, tinted toward the fog."""
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            m = f["memory"][r, c]
            if m == UNSEEN:
                if reveal:
                    _cell(ax, i, j, _blend(TRUE_COLOR.get(int(f["grid"][r, c]), FLOOR_C), FOG, 0.3))
                else:
                    _cell(ax, i, j, FOG)
                continue
            _cell(ax, i, j, MEM_COLOR[m])
            if m == M_FLAG:
                ax.plot(j, i, marker=(3, 0, 0), color=INK, markersize=7, zorder=5)
            if m == M_WALL and f["probed"][r, c]:
                _held_firm(ax, i, j)


def _seen_boundary(ax, f, rows, cols):
    """A dashed line along the border between seen and unseen squares."""
    seen = np.array([[f["memory"][r, c] != UNSEEN for c in cols] for r in rows])
    n, m = seen.shape
    style = dict(color=BLUE, lw=2.2, ls=(0, (2, 1.5)), zorder=5)
    for i in range(n):
        for j in range(m):
            if j + 1 < m and seen[i, j] != seen[i, j + 1]:
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], **style)
            if i + 1 < n and seen[i, j] != seen[i + 1, j]:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], **style)


def _boxes(ax, pos, r0, c0, window=True):
    r, c = pos[0] - r0, pos[1] - c0
    if window:
        ax.add_patch(Rectangle((c - PAD - 0.5, r - PAD - 0.5), CROP, CROP, fill=False, edgecolor=INK, lw=1.8,
                               zorder=7))
    ax.add_patch(Rectangle((c - VIEW - 0.5, r - VIEW - 0.5), 2 * VIEW + 1, 2 * VIEW + 1, fill=False,
                           edgecolor=RED, lw=1.4, ls="--", zorder=7))


def _compass(fig, ax, index):
    """A small dial below ``ax`` with an arrow toward the flag (8 directions, 0 = east, CCW)."""
    box = ax.get_position()
    size = box.width * 0.2
    cax = fig.add_axes([box.x0, box.y0 - size * 1.1, size, size])
    cax.set_xlim(-1.2, 1.2)
    cax.set_ylim(-1.2, 1.2)
    cax.set_aspect("equal")
    cax.axis("off")
    cax.add_patch(Circle((0, 0), 1, facecolor="white", edgecolor="#999999", lw=1))
    for k in range(8):
        a = k * np.pi / 4
        cax.plot([0.82 * np.cos(a), np.cos(a)], [0.82 * np.sin(a), np.sin(a)], color="#999999", lw=1)
    a = index * np.pi / 4
    cax.add_patch(FancyArrow(0, 0, 0.8 * np.cos(a), 0.8 * np.sin(a), width=0.07, head_width=0.28,
                             head_length=0.25, length_includes_head=True, color=INK))


def _outline(ax, n):
    ax.add_patch(Rectangle((-0.5, -0.5), n, n, fill=False, edgecolor=INK, lw=1.8, zorder=7))


# ---------------------------------------------------------------------------------------------
# The animation player's look (mazerl/player/player.js), reproduced for a still image
PLAYER = {"red": "#d9472b", "start": "#e2694f", "wall": "#dcdcdc", "weak": "#f3f3f3", "dot_in": "#a6a6a6",
          "dot_out": "#d3d3d3", "tint": "#f6dfd9", "flag": "#262626", "crack": "#9a9a9a"}


def path_figure(frames, flag, picks, out, inches=6.0):
    """The real maze drawn as the animation player draws it, the walker's trail up to the last
    snapshot, and numbered markers where the snapshots are taken."""
    first, last = frames[0]["grid"], frames[max(picks)]["grid"]
    arena = range(PAD, PAD + ARENA)
    pt = inches * 72 / ARENA                                   # points per square
    fig = plt.figure(figsize=(inches, inches))
    ax = fig.add_axes([0, 0, 1, 1])
    P = PLAYER
    for i, r in enumerate(arena):
        for j, c in enumerate(arena):
            v, now = int(first[r, c]), int(last[r, c])
            if v == OUTSIDE:
                _cell(ax, i, j, P["tint"])
            if v in (WALL, WEAK) and now != FLOOR:
                _cell(ax, i, j, P["wall"])
                if v == WEAK:
                    ax.add_patch(Rectangle((j - 0.395, i - 0.395), 0.79, 0.79, facecolor=P["weak"], edgecolor="none"))
            else:
                ax.add_patch(Circle((j, i), 0.055, facecolor=P["dot_out"] if v == OUTSIDE else P["dot_in"],
                                    edgecolor="none"))
            if v == WEAK and now == FLOOR:                     # a weak spot the walker broke open
                x, y = j - 0.5, i - 0.5
                ax.plot([x + .22, x + .46, x + .41, x + .74], [y + .26, y + .48, y + .62, y + .78], color=P["crack"],
                        lw=0.045 * pt, solid_capstyle="round", solid_joinstyle="round")
                ax.plot([x + .46, x + .72], [y + .48, y + .30], color=P["crack"], lw=0.045 * pt,
                        solid_capstyle="round")
    pts = np.array([f["pos"] for f in frames[:max(picks) + 1]], dtype=float) - PAD
    ax.plot(pts[:, 1], pts[:, 0], color=P["red"], lw=0.26 * pt, alpha=0.7, solid_capstyle="round",
            solid_joinstyle="round", zorder=4)
    r, c = np.array(frames[0]["pos"]) - PAD
    ax.add_patch(Circle((c, r), 0.29, facecolor="white", edgecolor=P["start"], lw=0.065 * pt, zorder=5))
    fr, fc = np.array(flag) - PAD
    ax.add_patch(Circle((fc, fr), 0.43, facecolor=P["flag"], edgecolor="none", zorder=5))
    ax.plot([fc - 0.13, fc - 0.13], [fr - 0.22, fr + 0.25], color="white", lw=0.055 * pt, solid_capstyle="round",
            zorder=6)
    ax.add_patch(plt.Polygon([(fc - 0.13, fr - 0.23), (fc + 0.22, fr - 0.09), (fc - 0.13, fr + 0.06)],
                             facecolor="white", edgecolor="none", zorder=6))
    r, c = pts[-1]
    ax.add_patch(Circle((c, r), 0.28, facecolor=P["red"], edgecolor="none", zorder=7))
    for n, k in enumerate(picks, 1):
        r, c = pts[k]
        ax.add_patch(Circle((c + 0.6, r - 0.6), 0.36, facecolor="white", edgecolor=INK, lw=1.1, zorder=8))
        ax.text(c + 0.6, r - 0.6, str(n), ha="center", va="center", fontsize=8.5, color=INK, weight="bold",
                zorder=9)
    _bare(ax, ARENA)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def snapshots_figure(frames, picks, close, out):
    """Top: memory growing at the moves in ``picks``. Bottom: move ``close`` up close."""
    lo = PAD - MARGIN
    span = range(lo, PAD + ARENA + MARGIN)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.6), gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    for ax, k in zip(axes[0], picks):
        f = frames[k]
        draw_memory(ax, f, span, span)
        _trail(ax, frames, k, lo, lo, lw=1.6)
        _boxes(ax, f["pos"], lo, lo)
        ax.add_patch(Rectangle((MARGIN - 0.5, MARGIN - 0.5), ARENA, ARENA, fill=False, edgecolor="#7a8799",
                               lw=1, zorder=3))                   # the arena's outline
        _bare(ax, len(span))

    f = frames[close]
    r0, c0 = f["pos"][0] - PAD, f["pos"][1] - PAD
    rows, cols = range(r0, r0 + CROP), range(c0, c0 + CROP)
    a_in, a_guess, a_dir = axes[1]
    draw_memory(a_in, f, rows, cols)                              # what the network gets
    _trail(a_in, frames, close, r0, c0)
    _boxes(a_in, f["pos"], r0, c0, window=False)
    draw_memory(a_guess, f, rows, cols, reveal=True)              # first guess: under the fog
    _seen_boundary(a_guess, f, rows, cols)
    _trail(a_guess, frames, close, r0, c0)
    draw_memory(a_dir, f, rows, cols)                             # second guess: which way
    _trail(a_dir, frames, close, r0, c0)
    if f["front"] >= 0:
        dr, dc = MOVES[f["front"]]
        a_dir.add_patch(FancyArrow(PAD, PAD, 1.7 * dc, 1.7 * dr, width=0.16, head_width=0.6, head_length=0.5,
                                   length_includes_head=True, color=BLUE, zorder=8))
    for ax in axes[1]:
        _bare(ax, CROP)
        _outline(ax, CROP)
    fig.canvas.draw()
    _compass(fig, a_in, f["compass"])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/figures/pretraining"))
    ap.add_argument("--maze", type=int, default=6, help="training layout index")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--picks", default="4,20,60", help="moves shown in the top row")
    ap.add_argument("--close", type=int, default=20, help="move shown up close")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bank, frames = walk(a.maze, a.seed)
    picks = [int(k) for k in a.picks.split(",")]
    flag = np.array(bank.flags[a.maze]) + PAD
    path_figure(frames, flag, picks, a.out / "walker-path.png")
    snapshots_figure(frames, picks, a.close, a.out / "snapshots.png")
    print(f"maze {a.maze}, {len(frames)} moves, top row at {picks}, close-up at {a.close} -> {a.out}")


if __name__ == "__main__":
    main()
