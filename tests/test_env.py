import numpy as np
import pytest
import torch

from mazerl.env import (MazeEnv, Rewards, PAD, VIEW, M_WALL, M_FLOOR, M_OUTSIDE, M_FLAG, UNSEEN)
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
    assert r.item() == pytest.approx(Rewards().through_wall + Rewards().new_cell)
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


def test_breaking_in_from_outside_is_bypass():
    # leave by the entrance, walk outside along the top, break in through the weak spot at (2, 5)
    env, _ = make_env(max_steps=12)
    for a in (UP, RIGHT, RIGHT, RIGHT, DOWN):
        act(env, a)
    assert arena(env) == (2, 5) and env.went_around[0]
    for _ in range(7):
        _, done, info = act(env, DOWN)
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
    assert x.shape == (3, 7, 11, 11) and compass.shape == (3, 8)
    assert torch.allclose(x[:, :5].sum(1), torch.ones(3, 11, 11))


def test_auto_reset_samples_new_attempts():
    bank = MazeBank.generate(20, seed=0)
    env = MazeEnv(bank, 64, max_steps=5)
    for _ in range(5):
        _, done, info = env.step(torch.randint(0, 4, (64,)))
    assert done.all() and (info["final_steps"] == 5).all()
    assert (env.t == 0).all()
