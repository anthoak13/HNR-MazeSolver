"""The student-facing API in mazerl/__init__.py (fast, CPU, no trained weights needed)."""
import matplotlib
matplotlib.use("Agg")

import pytest

import mazerl as mz


def _run(task):
    """A Run with an untrained agent (constructing it directly is much faster than training)."""
    settings = mz.Settings(task=task, max_steps=60)
    return mz.Run(mz.Agent(), mz.Rewards(), settings)


def test_task_comes_from_the_run(monkeypatch):
    seen = []
    real = mz.record
    monkeypatch.setattr(mz, "record", lambda *a, **k: seen.append(k["task"]) or real(*a, **k))
    monkeypatch.setattr(mz.player, "show", lambda ep, **k: ep)
    ep = mz.playback(_run("escape"), seed=3)                 # random maze
    assert seen[-1] == "escape" and ep["task"] == "escape" and ep["start"] == ep["flag"]
    assert mz.playback(_run("escape"), task="flag", seed=3)["task"] == "flag"
    assert mz.playback(None, seed=3)["task"] == "flag"
    run = _run("escape")
    run.checkpoints = {0: run.agent.state_dict()}
    grid = mz.training_grid(run, n_mazes=1)
    assert seen[-1] == "escape" and grid[0]["start"] == grid[0]["flag"]


def test_evaluate_task():
    run = _run("escape")
    esc = mz.evaluate(run, n=20)
    flag = mz.evaluate(run, n=20, task="flag")
    assert esc["attempts"] == flag["attempts"] == 20
    assert esc["went around the maze"] == 0              # escape never counts as a bypass


def test_breakout_stages_with_run_and_agent():
    for agent in (_run("flag"), mz.Agent(), None):
        out = mz.breakout_stages(agent, n=20)
        assert set(out) == {"left the maze", "broke in from outside", "reached the flag"}
        assert all(v.endswith("%") and 0 <= float(v[:-1]) <= 100 for v in out.values())


def test_list_pretrained_and_check_gpu():
    names = mz.list_pretrained()
    assert isinstance(names, list) and names == sorted(names)
    assert isinstance(mz.check_gpu(), bool)


def test_load_perception_missing():
    with pytest.raises(ValueError, match="no pretrained perception 'nope'; available"):
        mz.load_perception("nope")
    with pytest.raises(ValueError, match="available"):
        mz.load_pretrained("nope")


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
@pytest.mark.parametrize("task", ["flag", "escape"])
@pytest.mark.parametrize("maze", ["enclosed", None])
def test_score_breakdown_adds_up(task, maze):
    rewards = mz.Rewards(step=-0.001)
    for seed in range(4):
        pts = mz.score_breakdown(None, maze=maze, seed=seed, rewards=rewards, max_steps=150,
                                 task=task, plot=seed == 0)
        ep = mz.record(None, [mz.scenario(maze) if maze else mz.random_mazes(1, seed)[0]],
                       rewards=rewards, seed=seed, max_steps=150, task=task)[0]["agents"][0]
        assert pts["total"] == pytest.approx(ep["return"], abs=1e-3)
        assert pts["total"] == pytest.approx(sum(v for k, v in pts.items() if k != "total"), abs=1e-3)
        assert "per move" in pts


def test_score_breakdown_with_run():
    pts = mz.score_breakdown(_run("escape"), maze="enclosed", plot=False)
    assert "per move" not in pts
    assert list(pts)[-1] == "total"
