"""Batched maze environment: many independent attempts stepped together with tensor operations.

Rules (see PLAN.md for which come from the article):
  * moves are up / right / down / left
  * walking into a wall is a "probe" and costs the probe penalty; a weak spot gives way (the agent
    moves in), a solid wall doesn't (the agent stays put)
  * the agent sees a 5 x 5 window (walls block line of sight) and remembers everything it has seen,
    including which walls it has already walked into and found solid
  * it knows the compass direction (8-way) of the flag
"""
from dataclasses import dataclass, asdict
from fractions import Fraction
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from .maze import ARENA, BAND, MAZE, FLOOR, WALL, WEAK, OUTSIDE

EDGE = 4                 # beyond the arena: behaves like a wall that never breaks
PAD = 5                  # arena is padded by this much so every crop stays in bounds
SIZE = ARENA + 2 * PAD
VIEW = 2                 # the agent sees VIEW cells in every direction (5 x 5)
CROP = 2 * PAD + 1       # the network sees an 11 x 11 window of its memory

# What the agent remembers about a cell
UNSEEN, M_FLOOR, M_OUTSIDE, M_WALL, M_FLAG, M_EDGE = range(6)
N_MEMORY = 6
OBS_CHANNELS = N_MEMORY + 3      # + visited-by-me, + visible-right-now, + probed-and-held-firm
N_COMPASS = 8

MOVES = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])
ACTION_NAMES = ["up", "right", "down", "left"]


@dataclass
class Rewards:
    """Points the agent gets for each kind of event (negative = penalty)."""
    flag: float = 5.0            # reaching the flag
    outside: float = -0.01       # every move that ends on an outside cell
    probe: float = -0.01         # walking into a wall (weak spots give way, solid ones don't)
    new_cell: float = 0.005      # first visit to a cell (less than the outside penalty, so a
                                 #   step outside is always a net penalty)
    explore_outside: bool = True # whether new cells outside the maze also earn new_cell
    step: float = 0.0            # every move

    def as_dict(self):
        return asdict(self)


def _line_of_sight_groups():
    """For each cell in the 5 x 5 window, the groups of cells the sight line passes through.

    The line from the agent's cell center to the target's center is blocked if, at any point,
    every cell containing that point is opaque (grazing a corner between a wall and an open
    cell does not block). Groups are lists of window indices.
    """
    offsets = [(dr, dc) for dr in range(-VIEW, VIEW + 1) for dc in range(-VIEW, VIEW + 1)]
    index = {o: k for k, o in enumerate(offsets)}

    def cells(x):
        if x - Fraction(1, 2) == (x - Fraction(1, 2)).__floor__():   # exactly on a cell boundary
            return [int(x - Fraction(1, 2)), int(x + Fraction(1, 2))]
        return [round(x)]

    groups = []
    for dr, dc in offsets:
        target_groups = set()
        for i in range(1, 120):
            s = Fraction(i, 120)
            g = tuple(sorted(index[(r, c)] for r, c in product(cells(s * dr), cells(s * dc))))
            if index[(0, 0)] in g or index[(dr, dc)] in g:
                continue
            target_groups.add(g)
        groups.append(sorted(target_groups))
    return offsets, groups


def _los_tables():
    offsets, groups = _line_of_sight_groups()
    n = len(offsets)
    true_col, false_col = n, n + 1
    g_max = max(len(g) for g in groups)
    table = torch.full((n, g_max, 4), true_col, dtype=torch.long)
    for k, gs in enumerate(groups):
        for j in range(g_max):
            if j < len(gs):
                table[k, j, :len(gs[j])] = torch.tensor(gs[j])
            else:
                table[k, j, 0] = false_col       # padding group: never blocks
    return torch.tensor(offsets), table


class MazeEnv:
    """``n_envs`` attempts running in parallel; finished attempts restart on a new layout.

    bank: a MazeBank of layouts to sample from.

    An attempt counts as a *bypass* (training leaves these out, as the article's did) if the agent
    left the maze and never came back in: the attempt ends with the agent outside.
    """

    def __init__(self, bank, n_envs, rewards=None, max_steps=100, device="cpu", seed=0,
                 auto_reset=True, maze_ids=None, bypass_rule="left"):
        self.n = n_envs
        self.rewards = rewards or Rewards()
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.auto_reset = auto_reset
        self.bypass_rule = bypass_rule      # "left" (default), "around", "break_in", "never_entered"
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        dev = self.device
        grids = torch.full((len(bank), SIZE, SIZE), EDGE, dtype=torch.int8)
        grids[:, PAD:PAD + ARENA, PAD:PAD + ARENA] = torch.as_tensor(bank.grids)
        self.bank_grids = grids.to(dev)
        self.bank_starts = (torch.as_tensor(bank.starts) + PAD).to(dev)
        self.bank_flags = (torch.as_tensor(bank.flags) + PAD).to(dev)
        self.fixed_ids = None if maze_ids is None else torch.as_tensor(maze_ids).to(dev)

        offsets, self.los_table = _los_tables()
        self.view_offsets = offsets.to(dev)
        self.los_table = self.los_table.to(dev)
        self.moves = MOVES.to(dev)
        self.memory_code = torch.tensor([M_FLOOR, M_WALL, M_WALL, M_OUTSIDE, M_EDGE], device=dev)
        self.crop_range = torch.arange(-PAD, PAD + 1, device=dev)
        inside = torch.zeros(SIZE, SIZE, dtype=torch.bool)
        inside[PAD + BAND:PAD + BAND + MAZE, PAD + BAND:PAD + BAND + MAZE] = True
        self.inside = inside.to(dev)
        self.rows = torch.arange(n_envs, device=dev)

        self.grid = torch.empty((n_envs, SIZE, SIZE), dtype=torch.int8, device=dev)
        self.memory = torch.zeros((n_envs, SIZE, SIZE), dtype=torch.int8, device=dev)
        self.visited = torch.zeros((n_envs, SIZE, SIZE), dtype=torch.bool, device=dev)
        self.probed = torch.zeros((n_envs, SIZE, SIZE), dtype=torch.bool, device=dev)
        self.visible = torch.zeros((n_envs, 2 * VIEW + 1, 2 * VIEW + 1), dtype=torch.bool, device=dev)
        self.pos = torch.zeros((n_envs, 2), dtype=torch.long, device=dev)
        self.start = torch.zeros_like(self.pos)
        self.flag = torch.zeros_like(self.pos)
        self.maze_id = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.t = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.entered = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        self.went_around = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        self.broke_in = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        self.last_exit = torch.zeros((n_envs, 2), dtype=torch.long, device=dev)
        self.done = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        self.stats = {k: torch.zeros(n_envs, device=dev)
                      for k in ("return", "probes", "breaks", "outside", "new_cells")}
        self.reset()

    # ----------------------------------------------------------------------------------------
    def reset(self, which=None):
        """Start new attempts in the envs selected by boolean mask ``which`` (all if None)."""
        idx = self.rows if which is None else self.rows[which]
        if len(idx) == 0:
            return
        if self.fixed_ids is not None:
            ids = self.fixed_ids[idx % len(self.fixed_ids)]
        else:
            ids = torch.randint(len(self.bank_grids), (len(idx),), generator=self.gen).to(self.device)
        self.maze_id[idx] = ids
        self.grid[idx] = self.bank_grids[ids]
        self.start[idx] = self.bank_starts[ids]
        self.pos[idx] = self.bank_starts[ids]
        self.flag[idx] = self.bank_flags[ids]
        self.memory[idx] = UNSEEN
        self.visited[idx] = False
        self.probed[idx] = False
        self.visited[idx, self.pos[idx, 0], self.pos[idx, 1]] = True
        self.t[idx] = 0
        self.entered[idx] = False
        self.went_around[idx] = False
        self.broke_in[idx] = False
        self.last_exit[idx] = self.start[idx]
        self.done[idx] = False
        for v in self.stats.values():
            v[idx] = 0
        self._look(idx)

    def _look(self, idx):
        """Update what the agents in ``idx`` can see right now and add it to their memory."""
        cells = self.pos[idx, None, :] + self.view_offsets[None]                  # (b, 25, 2)
        b = idx[:, None]
        true = self.grid[b, cells[..., 0], cells[..., 1]].long()                  # (b, 25)
        opaque = (true == WALL) | (true == WEAK) | (true == EDGE)
        pad = torch.stack([torch.ones_like(opaque[:, 0]), torch.zeros_like(opaque[:, 0])], 1)
        opaque = torch.cat([opaque, pad], 1)                                      # (b, 27)
        blocked = opaque[:, self.los_table].all(-1).any(-1)                       # (b, 25)
        visible = ~blocked
        code = self.memory_code[true]
        is_flag = (cells == self.flag[idx, None, :]).all(-1)
        code = torch.where(is_flag, torch.full_like(code, M_FLAG), code)
        old = self.memory[b, cells[..., 0], cells[..., 1]]
        self.memory[b, cells[..., 0], cells[..., 1]] = torch.where(visible, code.to(old.dtype), old)
        self.visible[idx] = visible.view(-1, 2 * VIEW + 1, 2 * VIEW + 1)

    # ----------------------------------------------------------------------------------------
    def observe(self):
        """Network input: (n, 9, 11, 11) memory window around the agent, and (n, 8) compass."""
        r = self.pos[:, 0, None] + self.crop_range
        c = self.pos[:, 1, None] + self.crop_range
        b = self.rows[:, None, None]
        mem = self.memory[b, r[:, :, None], c[:, None, :]].long()
        vis = self.visited[b, r[:, :, None], c[:, None, :]]
        tried = self.probed[b, r[:, :, None], c[:, None, :]]
        now = torch.zeros((self.n, CROP, CROP), dtype=torch.bool, device=self.device)
        now[:, PAD - VIEW:PAD + VIEW + 1, PAD - VIEW:PAD + VIEW + 1] = self.visible
        x = torch.cat([F.one_hot(mem, N_MEMORY).permute(0, 3, 1, 2).float(),
                       vis[:, None].float(), now[:, None].float(), tried[:, None].float()], 1)
        return x, self.compass()

    def compass(self):
        d = (self.flag - self.pos).float()
        angle = torch.atan2(-d[:, 0], d[:, 1])                    # 0 = east, counter-clockwise
        sector = torch.round(angle / (np.pi / 4)).long() % N_COMPASS
        return F.one_hot(sector, N_COMPASS).float()

    # ----------------------------------------------------------------------------------------
    def step(self, actions):
        """Apply one move per env. Returns (reward, done, info); info holds per-move events and,
        for attempts that just ended, their totals (``final_*``) and whether they were bypasses."""
        R = self.rewards
        target = self.pos + self.moves[actions]
        cell = self.grid[self.rows, target[:, 0], target[:, 1]]
        passable = (cell == FLOOR) | (cell == OUTSIDE)
        through = cell == WEAK
        probe = ~passable & ~through
        self.probed[self.rows[probe], target[probe, 0], target[probe, 1]] = True
        self.grid[self.rows[through], target[through, 0], target[through, 1]] = FLOOR
        moved = passable | through
        prev = self.pos
        self.pos = torch.where(moved[:, None], target, self.pos)
        was_in = self.inside[prev[:, 0], prev[:, 1]]
        now_in = self.inside[self.pos[:, 0], self.pos[:, 1]]
        left = was_in & ~now_in
        self.last_exit = torch.where(left[:, None], prev, self.last_exit)
        came_back = ~was_in & now_in
        self.went_around |= came_back & (self.pos != self.last_exit).any(1)
        self.broke_in |= came_back & through

        r, c = self.pos[:, 0], self.pos[:, 1]
        new = ~self.visited[self.rows, r, c]
        self.visited[self.rows, r, c] = True
        outside = self.grid[self.rows, r, c] == OUTSIDE
        reached = (self.pos == self.flag).all(1)
        self.entered |= self.inside[r, c] & ((self.pos - self.start).abs().max(1).values >= 2)
        self.t += 1

        explored = new & (self.inside[r, c] | R.explore_outside)
        reward = (R.step + R.probe * (probe | through) + R.new_cell * explored
                  + R.outside * outside + R.flag * reached).float()
        s = self.stats
        s["return"] += reward
        s["probes"] += probe
        s["breaks"] += through
        s["outside"] += outside
        s["new_cells"] += new

        timeout = ~reached & (self.t >= self.max_steps)
        done = reached | timeout
        info = {"probe": probe, "through_wall": through, "new_cell": new, "outside": outside,
                "reached": reached, "timeout": timeout, "probe_cell": target, "done": done}
        # "left": left the maze and never came back in (ends outside)  <- default
        # alternatives kept for experiments:
        # "around": re-entered somewhere other than where it left, or never entered
        # "break_in": broke in through an outer weak spot, or never entered
        # "never_entered": never got more than one cell into the maze
        if self.bypass_rule == "left":
            bypass = ~self.inside[self.pos[:, 0], self.pos[:, 1]]
        elif self.bypass_rule == "around":
            bypass = ~self.entered | self.went_around
        elif self.bypass_rule == "break_in":
            bypass = ~self.entered | self.broke_in
        else:
            bypass = ~self.entered
        info["bypass"] = done & bypass
        info["final_steps"] = self.t.clone()
        info.update({f"final_{k}": v.clone() for k, v in s.items()})

        self.done = done.clone()
        if self.auto_reset:
            self.reset(done)
            self._look(self.rows[~done])
        else:
            self._look(self.rows)
        return reward, done, info
