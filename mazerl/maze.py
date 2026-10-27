"""Maze layouts: generation, the ASCII debug format, and the bank of training layouts.

Geometry follows the NYT figures: an 18 x 18 arena whose middle 14 x 14 block is the maze,
surrounded by a 2-cell band of "outside". The agent starts in the open top-left corner of the maze.
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLOOR, WALL, WEAK, OUTSIDE = 0, 1, 2, 3

ARENA = 18          # arena is ARENA x ARENA cells
BAND = 2            # width of the outside band
MAZE = ARENA - 2 * BAND
N_LAYOUTS = 13_148  # number of distinct training layouts in the article
GENERATOR_VERSION = 2

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


def generate_maze(rng, locked=False, n_openings=(3, 6), n_weak=(2, 6), max_room=(6, 25),
                  min_flag_dist=12, no_outside_shortcuts=False):
    """Generate one layout in the style of the article's figures.

    locked: seal the flag's region so it can only be reached by breaking an interior weak spot
        (and never from outside).
    no_outside_shortcuts: close outer openings/weak spots until going around the outside never
        gives a shorter route to the flag. Off by default: the article's training layouts have
        such routes, and attempts that use them are left out of training instead (see MazeEnv).
    """
    for _ in range(200):
        maze = _try_generate(rng, locked, n_openings, n_weak, max_room, min_flag_dist,
                             no_outside_shortcuts)
        if maze is not None:
            return maze
    raise RuntimeError("maze generation failed repeatedly")


def _try_generate(rng, locked, n_openings, n_weak, max_room, min_flag_dist, no_outside_shortcuts):
    m = np.full((MAZE, MAZE), WALL, dtype=np.int8)
    m[1:-1, 1:-1] = FLOOR
    doors = []
    _divide(m, rng, 1, MAZE - 2, 1, MAZE - 2, doors, int(rng.integers(*max_room)))
    m[0, 0] = m[0, 1] = m[1, 0] = FLOOR                    # entrance in the top-left corner

    grid = np.full((ARENA, ARENA), OUTSIDE, dtype=np.int8)
    grid[BAND:BAND + MAZE, BAND:BAND + MAZE] = m
    start = (BAND, BAND)
    shift = lambda p: (p[0] + BAND, p[1] + BAND)

    sealed = np.zeros_like(grid, dtype=bool)
    if locked:
        door = shift(doors[rng.integers(len(doors))]) if doors else None
        if door is None:
            return None
        grid[door] = WALL
        sealed = (grid == FLOOR) & (_bfs(grid, start, {FLOOR}) < 0)
        if not 6 <= sealed.sum() <= 70:
            return None
        grid[door] = WEAK

    ring = np.zeros_like(sealed)
    ring[BAND:BAND + MAZE, BAND:BAND + MAZE] = True
    ring[BAND + 1:BAND + MAZE - 1, BAND + 1:BAND + MAZE - 1] = False

    def inner(p):
        """The maze cell just inside boundary cell p (None for corners)."""
        r, c = p
        lo, hi = BAND, BAND + MAZE - 1
        if r in (lo, hi) and c in (lo, hi):
            return None
        if r == lo: return (r + 1, c)
        if r == hi: return (r - 1, c)
        if c == lo: return (r, c + 1)
        return (r, c - 1)

    # Openings in the outer wall (never into the sealed region, never next to the entrance).
    cands = [tuple(p) for p in np.argwhere(ring & (grid == WALL))
             if inner(tuple(p)) is not None and grid[inner(tuple(p))] == FLOOR
             and not sealed[inner(tuple(p))] and max(p[0], p[1]) > BAND + 2]
    for i in rng.permutation(len(cands))[:rng.integers(n_openings[0], n_openings[1] + 1)]:
        grid[cands[i]] = FLOOR

    # Weak spots: wall cells with open cells on two opposite sides, so they open into a passage.
    open_ = (grid == FLOOR) | (grid == OUTSIDE)
    cands = []
    for r in range(BAND, BAND + MAZE):
        for c in range(BAND, BAND + MAZE):
            if grid[r, c] != WALL:
                continue
            for a, b in (((r - 1, c), (r + 1, c)), ((r, c - 1), (r, c + 1))):
                if open_[a] and open_[b]:
                    outer = grid[a] == OUTSIDE or grid[b] == OUTSIDE
                    if not (outer and (sealed[a] or sealed[b])):
                        cands.append((r, c))
                    break
    for i in rng.permutation(len(cands))[:rng.integers(n_weak[0], n_weak[1] + 1)]:
        grid[cands[i]] = WEAK

    # Flag: far enough from the start, inside the sealed region when locked.
    dist = _bfs(grid, start, {FLOOR, WEAK})
    ok = (grid == FLOOR) & ~ring & (dist >= min_flag_dist)
    if locked:
        ok &= sealed
    cells = np.argwhere(ok)
    if len(cells) == 0:
        return None
    flag = tuple(int(v) for v in cells[rng.integers(len(cells))])

    if no_outside_shortcuts:
        _remove_outside_shortcuts(grid, start, flag)
    if locked and path_length(grid, start, flag, {FLOOR, OUTSIDE}) is not None:
        return None
    return Maze(grid, start, flag)


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

    def __init__(self, grids, starts, flags):
        self.grids = np.asarray(grids, dtype=np.int8)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.flags = np.asarray(flags, dtype=np.int64)

    @classmethod
    def generate(cls, n=N_LAYOUTS, seed=0, locked_frac=0.5, **kwargs):
        rng = np.random.default_rng(seed)
        mazes = [generate_maze(rng, locked=rng.random() < locked_frac, **kwargs) for _ in range(n)]
        return cls.from_mazes(mazes)

    @classmethod
    def from_mazes(cls, mazes):
        return cls([m.grid for m in mazes], [m.start for m in mazes], [m.flag for m in mazes])

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, i):
        return Maze(self.grids[i].copy(), tuple(self.starts[i]), tuple(self.flags[i]))

    def save(self, path):
        np.savez_compressed(path, grids=self.grids, starts=self.starts, flags=self.flags)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["grids"], d["starts"], d["flags"])


_CACHE = Path.home() / ".cache" / "mazerl"


def training_bank(n=N_LAYOUTS, seed=0, locked_frac=0.5):
    """The standard training layouts, generated once and cached on disk."""
    path = _CACHE / f"bank_v{GENERATOR_VERSION}_n{n}_seed{seed}_locked{locked_frac}.npz"
    if path.exists():
        return MazeBank.load(path)
    bank = MazeBank.generate(n, seed, locked_frac)
    path.parent.mkdir(parents=True, exist_ok=True)
    bank.save(path)
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
