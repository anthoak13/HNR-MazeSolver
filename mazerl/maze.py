"""Maze layouts: generation, the ASCII debug format, and the bank of training layouts.

Geometry follows the NYT figures: an 18 x 18 arena whose middle 14 x 14 block is the maze,
surrounded by a 2-cell band of "outside". The agent starts in the open top-left corner of the maze.
"""
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLOOR, WALL, WEAK, OUTSIDE = 0, 1, 2, 3

ARENA = 18          # arena is ARENA x ARENA cells
BAND = 2            # width of the outside band
MAZE = ARENA - 2 * BAND
N_LAYOUTS = 13_148  # number of distinct training layouts in the article
GENERATOR_VERSION = 3

_TO_CHAR = {FLOOR: ".", WALL: "#", WEAK: "x", OUTSIDE: ":"}
_FROM_CHAR = {".": FLOOR, "#": WALL, "x": WEAK, ":": OUTSIDE, "S": FLOOR, "F": FLOOR}
_STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class Maze:
    grid: np.ndarray    # (ARENA, ARENA) int8 cell codes
    start: tuple
    flag: tuple

    def __str__(self):
        rows = []
        for r in range(self.grid.shape[0]):
            row = [_TO_CHAR[int(v)] for v in self.grid[r]]
            if r == self.start[0]:
                row[self.start[1]] = "S"
            if r == self.flag[0]:
                row[self.flag[1]] = "F"
            rows.append("".join(row))
        return "\n".join(rows)

    @classmethod
    def from_ascii(cls, text):
        """Build a maze from the ASCII format printed by ``str(maze)``."""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        grid = np.array([[_FROM_CHAR[ch] for ch in ln] for ln in lines], dtype=np.int8)
        find = lambda ch: next((r, ln.index(ch)) for r, ln in enumerate(lines) if ch in ln)
        return cls(grid, find("S"), find("F"))

    def to_dict(self):
        return {"grid": self.grid.tolist(), "start": list(self.start), "flag": list(self.flag)}


def path_length(grid, start, goal, passable):
    """Fewest moves from start to goal over cells whose codes are in ``passable`` (None if unreachable)."""
    dist = _bfs(grid, start, passable)
    d = dist[goal]
    return None if d < 0 else int(d)


def _bfs(grid, start, passable):
    ok = np.isin(grid, list(passable))
    dist = np.full(grid.shape, -1, dtype=np.int32)
    dist[start] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and ok[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _divide(g, rng, r0, r1, c0, c1, doors, max_room):
    """Recursive division of the open chamber rows r0..r1, cols c0..c1: draw a wall across it with
    one doorway, then divide each side. Small chambers are left as open rooms."""
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 and w < 3:
        return
    if h * w <= max_room and rng.random() < 0.8:
        return
    horizontal = h > w if h != w else rng.random() < 0.5
    if (horizontal and h < 3) or (not horizontal and w < 3):
        horizontal = not horizontal
    if horizontal:
        # a wall row whose two ends meet walls (never a doorway in the bounding wall)
        rows = [y for y in range(r0 + 1, r1) if g[y, c0 - 1] == WALL and g[y, c1 + 1] == WALL]
        if not rows:
            return
        y = rows[rng.integers(len(rows))]
        g[y, c0:c1 + 1] = WALL
        x = int(rng.integers(c0, c1 + 1))
        g[y, x] = FLOOR
        doors.append((y, x))
        _divide(g, rng, r0, y - 1, c0, c1, doors, max_room)
        _divide(g, rng, y + 1, r1, c0, c1, doors, max_room)
    else:
        cols = [x for x in range(c0 + 1, c1) if g[r0 - 1, x] == WALL and g[r1 + 1, x] == WALL]
        if not cols:
            return
        x = cols[rng.integers(len(cols))]
        g[r0:r1 + 1, x] = WALL
        y = int(rng.integers(r0, r1 + 1))
        g[y, x] = FLOOR
        doors.append((y, x))
        _divide(g, rng, r0, r1, c0, x - 1, doors, max_room)
        _divide(g, rng, r0, r1, x + 1, c1, doors, max_room)


KINDS = ("open", "interior_break", "outside_only")
MIX = (1 / 3, 1 / 3, 1 / 3)     # equal shares of open / interior_break / outside_only
FLAG_ZONE = (10, 14)             # every figure puts the flag in rows and columns 10-14


def generate_maze(rng, kind="open", n_openings=(3, 5), n_weak=(2, 6), max_room=(6, 25),
                  min_flag_dist=12, no_outside_shortcuts=False):
    """Generate one layout in the style of the article's figures.

    kind (all three appear among the article's training mazes):
      "open"            the flag can be reached from inside without breaking anything
      "interior_break"  the flag's area is sealed off inside; the way in is an interior weak spot
      "outside_only"    the flag's area is sealed off inside; the way in is an opening in the
                        outer wall, so the agent has to leave the maze and come back in
    no_outside_shortcuts: close outer openings/weak spots until going around the outside never
        gives a shorter route to the flag (off by default; the article's layouts have such routes).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    for _ in range(500):
        maze = _try_generate(rng, kind, n_openings, n_weak, max_room, min_flag_dist,
                             no_outside_shortcuts)
        if maze is not None:
            return maze
    raise RuntimeError("maze generation failed repeatedly")


def _ring():
    ring = np.zeros((ARENA, ARENA), dtype=bool)
    ring[BAND:BAND + MAZE, BAND:BAND + MAZE] = True
    ring[BAND + 1:BAND + MAZE - 1, BAND + 1:BAND + MAZE - 1] = False
    return ring


def _inner(p):
    """The maze cell just inside outer-wall cell p (None for corners)."""
    r, c = p
    lo, hi = BAND, BAND + MAZE - 1
    if r in (lo, hi) and c in (lo, hi):
        return None
    if r == lo:
        return (r + 1, c)
    if r == hi:
        return (r - 1, c)
    if c == lo:
        return (r, c + 1)
    return (r, c - 1)


def _try_generate(rng, kind, n_openings, n_weak, max_room, min_flag_dist, no_outside_shortcuts):
    m = np.full((MAZE, MAZE), WALL, dtype=np.int8)
    m[1:-1, 1:-1] = FLOOR
    doors = []
    _divide(m, rng, 1, MAZE - 2, 1, MAZE - 2, doors, int(rng.integers(*max_room)))
    m[0, 0] = m[0, 1] = m[1, 0] = FLOOR                    # entrance in the top-left corner

    grid = np.full((ARENA, ARENA), OUTSIDE, dtype=np.int8)
    grid[BAND:BAND + MAZE, BAND:BAND + MAZE] = m
    start = (BAND, BAND)
    ring = _ring()
    walled = lambda g: np.where(g == OUTSIDE, WALL, g)      # the maze with the outside shut

    lo, hi = FLAG_ZONE
    zone = np.zeros_like(ring)
    zone[lo:hi + 1, lo:hi + 1] = True
    cells = np.argwhere(zone & (grid == FLOOR))
    if len(cells) == 0:
        return None
    flag = tuple(int(v) for v in cells[rng.integers(len(cells))])

    # Seal the flag's area off from the start by closing one doorway on the way to it.
    sealed = np.zeros_like(ring)
    if kind != "open":
        options = []
        for d in doors:
            d = (d[0] + BAND, d[1] + BAND)
            grid[d] = WALL
            if _bfs(walled(grid), start, {FLOOR})[flag] < 0:
                region = (grid == FLOOR) & (_bfs(walled(grid), flag, {FLOOR}) >= 0)
                touches_ring = any(region[q] for p in np.argwhere(ring & (grid == WALL))
                                   if (q := _inner(tuple(p))) is not None)
                if 6 <= region.sum() <= 60 and (kind == "interior_break" or touches_ring):
                    options.append((d, region))
            grid[d] = FLOOR
        if not options:
            return None
        door, sealed = options[rng.integers(len(options))]
        grid[door] = WEAK if kind == "interior_break" else WALL

    # Openings in the outer wall. Outside-only mazes get one into the sealed area; otherwise
    # openings never lead into it.
    cands = [tuple(p) for p in np.argwhere(ring & (grid == WALL))
             if (q := _inner(tuple(p))) is not None and grid[q] == FLOOR and max(p) > BAND + 2]
    into = [p for p in cands if sealed[_inner(p)]]
    elsewhere = [p for p in cands if not sealed[_inner(p)]]
    k = int(rng.integers(n_openings[0], n_openings[1] + 1))
    chosen = []
    if kind == "outside_only":
        if not into:
            return None
        chosen.append(into[rng.integers(len(into))])
        k -= 1
    chosen += [elsewhere[i] for i in rng.permutation(len(elsewhere))[:k]]
    for p in chosen:
        grid[p] = FLOOR

    # Weak spots: wall cells with open cells on two opposite sides, so they open into a passage.
    # In outside-only mazes no interior weak spot may lead into the sealed area.
    open_ = (grid == FLOOR) | (grid == OUTSIDE)
    cands = []
    for r in range(BAND, BAND + MAZE):
        for c in range(BAND, BAND + MAZE):
            if grid[r, c] != WALL:
                continue
            for a, b in (((r - 1, c), (r + 1, c)), ((r, c - 1), (r, c + 1))):
                if open_[a] and open_[b]:
                    outer = grid[a] == OUTSIDE or grid[b] == OUTSIDE
                    if not (kind == "outside_only" and not outer and sealed[a] != sealed[b]):
                        cands.append((r, c))
                    break
    k = int(rng.integers(n_weak[0], n_weak[1] + 1)) - (kind == "interior_break")
    for i in rng.permutation(len(cands))[:max(k, 0)]:
        grid[cands[i]] = WEAK

    if no_outside_shortcuts:
        _remove_outside_shortcuts(grid, start, flag)
    if not _is_kind(grid, start, flag, kind):
        return None
    if path_length(grid, start, flag, {FLOOR, WEAK, OUTSIDE}) < min_flag_dist:
        return None
    return Maze(grid, start, flag)


def maze_kind(maze):
    """Which of the three kinds a layout is (see generate_maze)."""
    for kind in KINDS:
        if _is_kind(maze.grid, maze.start, maze.flag, kind):
            return kind
    return None


def _is_kind(grid, start, flag, kind):
    inside = np.where(grid == OUTSIDE, WALL, grid)
    no_break = path_length(inside, start, flag, {FLOOR}) is not None
    if kind == "open":
        return no_break
    if no_break:
        return False
    inside_break = path_length(inside, start, flag, {FLOOR, WEAK}) is not None
    via_opening = path_length(grid, start, flag, {FLOOR, OUTSIDE}) is not None
    if kind == "interior_break":
        return inside_break and not via_opening
    return via_opening and not inside_break


def _remove_outside_shortcuts(grid, start, flag):
    inside_only = grid.copy()
    inside_only[inside_only == OUTSIDE] = WALL
    inside = path_length(inside_only, start, flag, {FLOOR, WEAK})
    while True:
        via_outside = path_length(grid, start, flag, {FLOOR, WEAK, OUTSIDE})
        if inside is not None and via_outside >= inside:
            return
        # Close the outer opening / weak spot nearest the flag and try again.
        dist = _bfs(grid, flag, {FLOOR, WEAK})
        outer = [tuple(p) for p in np.argwhere((grid == WEAK) | (grid == FLOOR))
                 if tuple(p) != start and any(grid[p[0] + dr, p[1] + dc] == OUTSIDE for dr, dc in _STEPS)
                 and tuple(p) not in ((BAND, BAND + 1), (BAND + 1, BAND))]
        if not outer:
            return
        outer.sort(key=lambda p: dist[p] if dist[p] >= 0 else 10**6)
        grid[outer[0]] = WALL


class MazeBank:
    """A fixed set of layouts (the article trained on 13,148 of them)."""

    def __init__(self, grids, starts, flags, kinds=None):
        self.kinds = None if kinds is None else np.asarray(kinds)
        self.grids = np.asarray(grids, dtype=np.int8)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.flags = np.asarray(flags, dtype=np.int64)

    @classmethod
    def generate(cls, n=N_LAYOUTS, seed=0, mix=MIX, **kwargs):
        rng = np.random.default_rng(seed)
        kinds = rng.choice(KINDS, size=n, p=np.asarray(mix) / np.sum(mix))
        mazes = [generate_maze(rng, kind=k, **kwargs) for k in kinds]
        return cls.from_mazes(mazes, kinds)

    @classmethod
    def from_mazes(cls, mazes, kinds=None):
        return cls([m.grid for m in mazes], [m.start for m in mazes], [m.flag for m in mazes], kinds)

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, i):
        return Maze(self.grids[i].copy(), tuple(self.starts[i]), tuple(self.flags[i]))

    def save(self, path):
        extra = {} if self.kinds is None else {"kinds": self.kinds}
        np.savez_compressed(path, grids=self.grids, starts=self.starts, flags=self.flags, **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["grids"], d["starts"], d["flags"], d["kinds"] if "kinds" in d else None)


_CACHE = Path.home() / ".cache" / "mazerl"


def training_bank(n=N_LAYOUTS, seed=0, mix=MIX):
    """The standard training layouts, generated once and cached on disk."""
    tag = "-".join(f"{x:.3f}" for x in mix)
    path = _CACHE / f"bank_v{GENERATOR_VERSION}_n{n}_seed{seed}_mix{tag}.npz"
    if path.exists():
        return MazeBank.load(path)
    bank = MazeBank.generate(n, seed, mix)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    bank.save(tmp)
    os.replace(tmp, path)            # atomic, so parallel jobs never read a half-written file
    return bank


SCENARIOS = ("enclosed", "outside_shortcut")


def scenario(name):
    """A test layout transcribed from the article's figures.

    "enclosed" (p.6, "Agent facing a hard problem"): the flag is walled in; the start pocket opens
        only to the outside.
    "outside_shortcut" (p.7, "Agent taking outside path although inside path is available").
    """
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose from {SCENARIOS}")
    text = (Path(__file__).parent / "scenarios" / f"{name}.txt").read_text()
    return Maze.from_ascii(text)
