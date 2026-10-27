import numpy as np

from mazerl.maze import (MazeBank, generate_maze, maze_kind, path_length, scenario, ARENA, BAND,
                         MAZE, FLOOR, WALL, WEAK, OUTSIDE, KINDS)


def mazes(n=200, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [generate_maze(rng, **kw) for _ in range(n)]


def test_layout_geometry():
    for m in mazes(50):
        g = m.grid
        assert g.shape == (ARENA, ARENA)
        band = np.ones_like(g, dtype=bool)
        band[BAND:BAND + MAZE, BAND:BAND + MAZE] = False
        assert (g[band] == OUTSIDE).all()
        assert not (g[~band] == OUTSIDE).any()
        assert m.start == (BAND, BAND) and g[m.start] == FLOOR
        assert g[m.flag] == FLOOR


def test_every_kind_is_what_it_claims():
    for kind in KINDS:
        for m in mazes(150, kind=kind):
            assert maze_kind(m) == kind
            assert path_length(m.grid, m.start, m.flag, {FLOOR, WEAK, OUTSIDE}) >= 12


def test_flag_in_lower_right_like_the_figures():
    for m in mazes(100, kind="outside_only") + mazes(100):
        assert 10 <= m.flag[0] <= 14 and 10 <= m.flag[1] <= 14


def test_article_training_figures_classify_as_expected():
    from pathlib import Path
    import mazerl
    from mazerl.maze import Maze
    folder = Path(mazerl.__file__).parent / "scenarios" / "training"
    kinds = {f.stem: maze_kind(Maze.from_ascii(f.read_text())) for f in folder.glob("*.txt")}
    assert kinds == {"p3": "open", "p4a": "open", "p4b": "open", "p4c": "interior_break",
                     "p5a": "interior_break", "p5b": "outside_only", "p5c": "outside_only"}


def test_outer_wall_has_openings_besides_entrance():
    for m in mazes(50):
        ring = m.grid[BAND:BAND + MAZE, BAND:BAND + MAZE].copy()
        ring[1:-1, 1:-1] = WALL
        assert (ring == FLOOR).sum() >= 3 + 3          # entrance (3 cells) + openings


def test_no_outside_shortcuts_option():
    for m in mazes(100, kind="open", no_outside_shortcuts=True):
        inside_only = np.where(m.grid == OUTSIDE, WALL, m.grid)
        inside = path_length(inside_only, m.start, m.flag, {FLOOR, WEAK})
        anywhere = path_length(m.grid, m.start, m.flag, {FLOOR, WEAK, OUTSIDE})
        assert anywhere >= inside


def test_weak_spots_on_inner_and_outer_walls():
    outer = inner = 0
    for m in mazes(100):
        for r, c in np.argwhere(m.grid == WEAK):
            nbrs = [m.grid[r + dr, c + dc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
            if OUTSIDE in nbrs:
                outer += 1
            else:
                inner += 1
    assert outer > 0 and inner > 0


def test_bank_round_trip(tmp_path):
    bank = MazeBank.generate(10, seed=1)
    assert set(bank.kinds) <= set(KINDS)
    bank.save(tmp_path / "b.npz")
    again = MazeBank.load(tmp_path / "b.npz")
    assert (again.grids == bank.grids).all() and len(again) == 10
    assert str(again[3]) == str(bank[3])


def test_article_scenarios_load():
    from mazerl.maze import scenario
    enclosed = scenario("enclosed")
    assert enclosed.start == (2, 2) and enclosed.flag == (10, 12)
    assert path_length(enclosed.grid, enclosed.start, enclosed.flag, {FLOOR, WEAK}) is None
    assert path_length(enclosed.grid, enclosed.start, enclosed.flag, {FLOOR, WEAK, OUTSIDE}) is not None
    shortcut = scenario("outside_shortcut")
    inside_only = np.where(shortcut.grid == OUTSIDE, WALL, shortcut.grid)
    assert path_length(inside_only, shortcut.start, shortcut.flag, {FLOOR}) is not None
