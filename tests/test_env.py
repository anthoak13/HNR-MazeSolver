import numpy as np
import pytest
import torch

from mazerl.env import (MazeEnv, Rewards, PAD, VIEW, M_WALL, M_FLOOR, M_OUTSIDE, M_FLAG, M_EDGE,
                        UNSEEN, N_MEMORY, OBS_CHANNELS)
from mazerl.maze import Maze, MazeBank, FLOOR, WALL, WEAK, OUTSIDE

UP, RIGHT, DOWN, LEFT = range(4)

# Start at the maze corner (2, 2); a weak spot at (2, 5) in the top wall and one at (5, 4) inside.
SIMPLE = """
::::::::::::::::::
::::::::::::::::::
::S..x##########::
::#.#..........#::
::#.#.########.#::
::#...x.......F#::
::#.############::
::#............#::
::#.##########.#::
::#............#::
::##############::
::#............#::
::#............#::
::#............#::
::#............#::
::##############::
::::::::::::::::::
::::::::::::::::::
"""


def make_env(text=SIMPLE, rewards=None, n=1, **kw):
    maze = Maze.from_ascii(text)
    bank = MazeBank.from_mazes([maze])
    return MazeEnv(bank, n, rewards or Rewards(), auto_reset=False, **kw), maze


def arena(env, i=0):
    return tuple(int(v) - PAD for v in env.pos[i])


def act(env, a):
    return env.step(torch.tensor([a]))


def test_ascii_round_trip():
    maze = Maze.from_ascii(SIMPLE)
    assert maze.start == (2, 2) and maze.flag == (5, 14)
    assert str(Maze.from_ascii(str(maze))) == str(maze)


def test_move_and_new_cell_reward():
    env, _ = make_env()
    r, done, info = act(env, RIGHT)
    assert arena(env) == (2, 3)
    assert info["new_cell"].item() and not info["probe"].item()
    assert r.item() == pytest.approx(Rewards().new_cell)
    act(env, LEFT)
    r, _, info = act(env, RIGHT)                 # revisiting earns nothing
    assert not info["new_cell"].item() and r.item() == pytest.approx(0)


def test_probe_solid_wall_stays_put():
    env, _ = make_env()
    act(env, DOWN)                               # (3, 2) is wall
    assert arena(env) == (2, 2)
    r, _, info = act(env, DOWN)
    assert info["probe"].item() and r.item() == pytest.approx(Rewards().probe)


def test_break_through_weak_spot_opens_it():
    env, _ = make_env()
    for a in (RIGHT, RIGHT):
        act(env, a)
    r, _, info = act(env, RIGHT)                 # (2, 5) is weak
    assert info["through_wall"].item() and arena(env) == (2, 5)
    assert r.item() == pytest.approx(Rewards().probe + Rewards().new_cell)   # one wall penalty
    assert env.grid[0, 2 + PAD, 5 + PAD] == FLOOR


def test_outside_penalty_and_bypass():
    env, _ = make_env(max_steps=4)
    r, _, info = act(env, LEFT)                  # step out of the entrance
    assert arena(env) == (2, 1) and info["outside"].item()
    assert r.item() == pytest.approx(Rewards().outside + Rewards().new_cell)
    for _ in range(3):
        _, done, info = act(env, UP)
    assert done.item() and info["timeout"].item() and info["bypass"].item()


def test_entering_maze_is_not_bypass():
    env, _ = make_env(max_steps=4)
    for a in (RIGHT, DOWN, DOWN, DOWN):
        _, done, info = act(env, a)
    assert done.item() and not info["bypass"].item()


def test_breaking_back_in_is_not_bypass():
    # leave by the entrance, walk outside along the top, break in through the weak spot at (2, 5)
    env, _ = make_env(max_steps=12)
    for a in (UP, RIGHT, RIGHT, RIGHT, DOWN):
        act(env, a)
    assert arena(env) == (2, 5) and env.went_around[0] and env.broke_in[0]
    for _ in range(7):
        _, done, info = act(env, DOWN)
    assert done.item() and not info["bypass"].item()


def test_ending_outside_is_bypass():
    env, _ = make_env(max_steps=6)
    for a in (RIGHT, DOWN, LEFT, UP, UP, LEFT):     # into the maze, back out, stays out
        _, done, info = act(env, a)
    assert done.item() and info["bypass"].item()


def test_stepping_out_and_back_is_not_bypass():
    env, _ = make_env(max_steps=8)
    for a in (LEFT, RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT, RIGHT):
        _, done, info = act(env, a)
    assert done.item() and not info["bypass"].item()


def test_reaching_flag_ends_attempt():
    env, maze = make_env()
    path = [RIGHT, DOWN, DOWN, DOWN, RIGHT, RIGHT] + [UP, UP, RIGHT] + [RIGHT] * 8 + [DOWN, DOWN, RIGHT]
    for a in path:
        r, done, info = act(env, a)
        if done.item():
            break
    assert info["reached"].item() and arena(env) == maze.flag
    assert info["final_return"].item() > 4


def test_vision_blocked_by_walls():
    env, _ = make_env()
    mem = env.memory[0]
    at = lambda r, c: int(mem[r + PAD, c + PAD])
    assert at(2, 3) == M_FLOOR and at(2, 4) == M_FLOOR
    assert at(3, 2) == M_WALL and at(2, 1) == M_OUTSIDE
    assert at(4, 2) == UNSEEN                    # behind the wall at (3, 2)
    assert env.visible[0].sum() < (2 * VIEW + 1) ** 2
    act(env, RIGHT)
    assert int(env.memory[0, 2 + PAD, 5 + PAD]) == M_WALL   # weak spots look like walls


def test_flag_marked_in_memory_when_seen():
    env, maze = make_env()
    env.pos[0] = torch.tensor([5 + PAD, 12 + PAD])
    env._look(env.rows)
    assert int(env.memory[0, 5 + PAD, 14 + PAD]) == M_FLAG


def test_compass_points_to_flag():
    env, _ = make_env()
    _, compass = env.observe()
    # flag (5, 14) from (2, 2): mostly east, slightly south -> sector 0 (east)
    assert compass[0].argmax().item() == 0
    env.pos[0] = torch.tensor([12 + PAD, 14 + PAD])     # directly south of the flag
    assert env.compass()[0].argmax().item() == 2        # north


def test_observation_shapes():
    env, _ = make_env(n=3)
    x, compass = env.observe()
    assert x.shape == (3, OBS_CHANNELS, 11, 11) and compass.shape == (3, 8)
    assert torch.allclose(x[:, :N_MEMORY].sum(1), torch.ones(3, 11, 11))


def test_auto_reset_samples_new_attempts():
    bank = MazeBank.generate(20, seed=0)
    env = MazeEnv(bank, 64, max_steps=5)
    for _ in range(5):
        _, done, info = env.step(torch.randint(0, 4, (64,)))
    assert done.all() and (info["final_steps"] == 5).all()
    assert (env.t == 0).all()


def test_remembers_walls_it_probed():
    env, _ = make_env()
    act(env, DOWN)                               # (3, 2) is solid
    x, _ = env.observe()
    assert env.probed[0, 3 + PAD, 2 + PAD]
    assert x[0, -1, 6, 5] == 1                   # probed channel, one row below the agent
    assert x[0, -1].sum() == 1


def test_arena_edge_is_its_own_kind_of_cell():
    env, _ = make_env()
    act(env, UP)                                 # step into the outside band at (1, 2)
    act(env, UP)                                 # (0, 2): the arena edge is now in view
    assert int(env.memory[0, -1 + PAD, 2 + PAD]) == M_EDGE


def test_explore_outside_off():
    env, _ = make_env(rewards=Rewards(explore_outside=False))
    r, _, _ = act(env, LEFT)                     # new outside cell: penalty only
    assert r.item() == pytest.approx(Rewards().outside)


# ---- escape task: start on the flag's cell, success = reaching any outside cell -------------------
# The flag cell (5, 14) sits in the corridor of row 5; the nearest way out is the weak spot at (2, 5)
# or the entrance at (2, 2), so walk back along the corridor.

def test_escape_starts_on_flag_cell_with_no_flag_or_compass():
    env, maze = make_env(task="escape")
    assert arena(env) == maze.flag
    _, comp = env.observe()
    assert comp.sum().item() == 0
    assert not (env.memory == M_FLAG).any()


def test_escape_succeeds_on_first_outside_cell_without_outside_penalty():
    env, _ = make_env(task="escape")
    # row 5 corridor west to (5, 3), up column 3 to (2, 3), west to (2, 2), out at (2, 1)
    path = [LEFT] * 9 + [LEFT, LEFT] + [UP, UP, UP, LEFT, LEFT]
    for a in path:
        r, done, info = act(env, a)
        if done.item():
            break
    assert done.item() and info["reached"].item() and arena(env) == (2, 1)
    assert not info["bypass"].item()
    assert not info["outside"].item()                              # no outside penalty on the way out
    assert r.item() == pytest.approx(Rewards().flag + Rewards().new_cell)


def test_escape_through_outer_weak_spot():
    # put the escape start just below the weak spot in the top wall
    env, _ = make_env(task="escape")
    env.pos[0] = torch.tensor([3 + PAD, 5 + PAD])
    r, done, info = act(env, UP)                                   # (2, 5) is weak: breaks into the wall cell
    assert info["through_wall"].item() and not done.item()
    r, done, info = act(env, UP)                                   # (1, 5) is outside
    assert done.item() and info["reached"].item()


def test_escape_compass_is_stored_as_none_for_training():
    from mazerl.env import compass_index, compass_from_index
    env, _ = make_env(task="escape")
    _, comp = env.observe()
    assert torch.equal(compass_from_index(compass_index(comp)), comp)
    env, _ = make_env()
    _, comp = env.observe()
    assert torch.equal(compass_from_index(compass_index(comp)), comp)


def test_escape_training_runs():
    from mazerl.a2c import train
    bank = MazeBank.from_mazes([Maze.from_ascii(SIMPLE)])
    run = train(task="escape", episodes=300, n_envs=16, batch_moves=256, bank=bank, device="cpu",
                progress=False)
    assert run.settings.task == "escape" and run.history["bypass"][-1] == 0
