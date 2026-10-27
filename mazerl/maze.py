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
ROOMS = 6           # the maze is carved as a ROOMS x ROOMS room lattice, then stretched to MAZE x MAZE
N_LAYOUTS = 13_148  # number of distinct training layouts in the article

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


def _carve_rooms(rng):
    """Depth-first maze on a ROOMS x ROOMS room lattice (rooms at odd coordinates)."""
    n = 2 * ROOMS + 1
    lat = np.full((n, n), WALL, dtype=np.int8)
    seen = np.zeros((ROOMS, ROOMS), dtype=bool)
    seen[0, 0] = True
    lat[1, 1] = FLOOR
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in _STEPS
                if 0 <= r + dr < ROOMS and 0 <= c + dc < ROOMS and not seen[r + dr, c + dc]]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = nbrs[rng.integers(len(nbrs))]
        lat[r + nr + 1, c + nc + 1] = FLOOR      # the gap between the two rooms
        lat[2 * nr + 1, 2 * nc + 1] = FLOOR
        seen[nr, nc] = True
        stack.append((nr, nc))
    return lat


def _room_side(lat, cut):
    """Rooms reachable from the start room once lattice cell ``cut`` is walled off."""
    lat = lat.copy()
    lat[cut] = WALL
    return _bfs(lat, (1, 1), {FLOOR}) >= 0


def _add_loops(rng, lat, count, region=None):
    """Knock out ``count`` walls between rooms, never joining the two sides of ``region``."""
    n = lat.shape[0]
    cands = []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if lat[i, j] != WALL or (i + j) % 2 == 0:
                continue
            a, b = ((i - 1, j), (i + 1, j)) if i % 2 == 0 else ((i, j - 1), (i, j + 1))
            if region is None or region[a] == region[b]:
                cands.append((i, j))
    for k in rng.permutation(len(cands))[:count]:
        lat[cands[k]] = FLOOR


def _stretch(rng, lat, extra=None):
    """Duplicate one interior row and column (13 -> 14) for the mixed 1-2 cell walls and corridors."""
    n = lat.shape[0]
    ri, ci = rng.integers(1, n - 1, size=2)
    lat = np.insert(lat, ri, lat[ri], axis=0)
    lat = np.insert(lat, ci, lat[:, ci], axis=1)
    if extra is not None:
        extra = np.insert(extra, ri, extra[ri], axis=0)
        extra = np.insert(extra, ci, extra[:, ci], axis=1)
    return lat, extra


def generate_maze(rng, locked=False, loops=(0, 3), n_weak=(4, 8), min_flag_dist=12,
                  no_outside_shortcuts=True):
    """Generate one layout.

    locked: seal the flag's region off so it can only be reached by breaking an interior weak spot.
    no_outside_shortcuts: remove outer-wall weak spots until going outside never gives a shorter
        route to the flag than staying inside (training layouts never reward bypassing the maze).
    """
    for _ in range(100):
        maze = _try_generate(rng, locked, loops, n_weak, min_flag_dist, no_outside_shortcuts)
        if maze is not None:
            return maze
    raise RuntimeError("maze generation failed repeatedly")


def _try_generate(rng, locked, loops, n_weak, min_flag_dist, no_outside_shortcuts):
    lat = _carve_rooms(rng)
    sealed = None
    if locked:
        gaps = [(i, j) for i in range(1, lat.shape[0] - 1) for j in range(1, lat.shape[1] - 1)
                if (i + j) % 2 == 1 and lat[i, j] == FLOOR]
        cut = gaps[rng.integers(len(gaps))]
        side = _room_side(lat, cut)
        far = ~side & (lat == FLOOR)
        far[cut] = False
        if not 3 <= far[1::2, 1::2].sum() <= 15:
            return None
        lat[cut] = WEAK
        region = np.where(side, 1, 2)
        region[cut] = 0
        _add_loops(rng, lat, rng.integers(loops[0], loops[1] + 1), region)
        sealed = far
    else:
        _add_loops(rng, lat, rng.integers(loops[0], loops[1] + 1))

    lat, sealed = _stretch(rng, lat, sealed)
    lat[0, 0] = lat[0, 1] = FLOOR          # entrance in the top-left corner

    grid = np.full((ARENA, ARENA), OUTSIDE, dtype=np.int8)
    grid[BAND:BAND + MAZE, BAND:BAND + MAZE] = lat
    in_sealed = np.zeros_like(grid, dtype=bool)
    if sealed is not None:
        in_sealed[BAND:BAND + MAZE, BAND:BAND + MAZE] = sealed
    start = (BAND, BAND)

    # Flag: far enough from the start, inside the sealed region when locked.
    dist = _bfs(grid, start, {FLOOR, WEAK})
    ok = (grid == FLOOR) & (dist >= min_flag_dist)
    if locked:
        ok &= in_sealed
    cells = np.argwhere(ok)
    if len(cells) == 0:
        return None
    flag = tuple(int(v) for v in cells[rng.integers(len(cells))])

    # Weak spots: wall cells with open cells on two opposite sides, so they open into a passage.
    open_ = (grid == FLOOR) | (grid == OUTSIDE)
    cands = []
    for r in range(BAND, BAND + MAZE):
        for c in range(BAND, BAND + MAZE):
            if grid[r, c] != WALL:
                continue
            ns = open_[r - 1, c] and open_[r + 1, c]
            ew = open_[r, c - 1] and open_[r, c + 1]
            if not (ns or ew):
                continue
            touches = [(r - 1, c), (r + 1, c)] if ns else [(r, c - 1), (r, c + 1)]
            outer = any(grid[p] == OUTSIDE for p in touches)
            if outer and any(in_sealed[p] for p in touches):
                continue                      # the sealed region never touches an outer weak spot
            cands.append((r, c))
    k = min(len(cands), rng.integers(n_weak[0], n_weak[1] + 1))
    for i in rng.permutation(len(cands))[:k]:
        grid[cands[i]] = WEAK

    if no_outside_shortcuts:
        _remove_outside_shortcuts(grid, start, flag)
    if locked and path_length(grid, start, flag, {FLOOR}) is not None:
        return None
    return Maze(grid, start, flag)


def _remove_outside_shortcuts(grid, start, flag):
    inside = path_length(grid, start, flag, {FLOOR, WEAK})
    while True:
        via_outside = path_length(grid, start, flag, {FLOOR, WEAK, OUTSIDE})
        if via_outside >= inside:
            return
        # Demote the outer weak spot nearest the flag and try again.
        dist = _bfs(grid, flag, {FLOOR, WEAK})
        outer = [tuple(p) for p in np.argwhere(grid == WEAK)
                 if any(grid[p[0] + dr, p[1] + dc] == OUTSIDE for dr, dc in _STEPS)]
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
    path = _CACHE / f"bank_n{n}_seed{seed}_locked{locked_frac}.npz"
    if path.exists():
        return MazeBank.load(path)
    bank = MazeBank.generate(n, seed, locked_frac)
    path.parent.mkdir(parents=True, exist_ok=True)
    bank.save(path)
    return bank
