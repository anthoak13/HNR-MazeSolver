import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

import mazerl as mz
from mazerl.perception import Pretrained, PredictionHeads, CLASS_NAMES
from mazerl.agent import Agent
from mazerl.rollout import record


@pytest.fixture(scope="module")
def tiny():
    """A barely-trained Pretrained with read-outs (fast; accuracy doesn't matter here)."""
    return mz.pretrain_perception(steps=5, batch=32, n_samples=512, device="cpu", progress=False,
                                  bank=mz.MazeBank.generate(8, seed=1))


def test_heads_survive_save_and_load(tiny, tmp_path):
    path = tmp_path / "p.pt"
    tiny.save(path)
    back = Pretrained.load(path)
    assert back.heads is not None
    for k, v in tiny.heads.state_dict().items():
        assert torch.equal(v, back.heads.state_dict()[k])


def test_old_files_without_heads_still_load(tiny, tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"perception": tiny.agent.perception_state(), "history": {}, "test": {}, "seconds": 0.0}, path)
    old = Pretrained.load(path)
    assert old.heads is None
    with pytest.raises(ValueError):
        old.guess(torch.zeros(1, 9, 11, 11), torch.zeros(1, 8))


def test_guess_shapes(tiny):
    classes, move = tiny.guess(torch.zeros(2, 9, 11, 11), torch.zeros(2, 8))
    assert classes.shape == (2, 11, 11) and move.shape == (2,)
    assert int(classes.max()) < len(CLASS_NAMES)


def test_explorer_is_recorded_and_reproducible():
    maze = mz.random_mazes(1, seed=0)[0]
    a = record("explorer", [maze], n_agents=2, seed=3)[0]["agents"]
    b = record("explorer", [maze], n_agents=2, seed=3)[0]["agents"]
    assert [x["path"] for x in a] == [x["path"] for x in b]
    assert len(set(map(tuple, map(tuple, a[0]["path"])))) > 5     # it actually explores


@pytest.mark.parametrize("maze", [None, "enclosed"])
def test_quiz_figure(tiny, maze):
    fig = mz.show_pretraining_quiz(tiny, maze=maze, step=15, seed=0)
    assert isinstance(fig, plt.Figure) and len(fig.axes) == 3
    assert "right" in fig.axes[2].get_title(loc="left")
