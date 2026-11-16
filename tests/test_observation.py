import json
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import mazerl as mz
from mazerl import player
from mazerl.env import N_MEMORY, PAD
from mazerl.observation import show_observation, _run_to, _resolve


@pytest.mark.parametrize("kwargs", [
    dict(),                                              # random moves, random maze, flag task
    dict(task="escape"),
    dict(maze="enclosed", step=5),
    dict(maze="enclosed", task="escape", step=0),
    dict(agent="fresh", step=12, seed=3),
    dict(agent="fresh", task="escape"),
    dict(step=10_000),                                   # past the end of the attempt
])
def test_show_observation_returns_figure(kwargs):
    if kwargs.get("agent") == "fresh":
        torch.manual_seed(0)
        kwargs = {**kwargs, "agent": mz.Agent()}
    fig = show_observation(**kwargs)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_show_observation_accepts_a_maze_object():
    fig = show_observation(maze=mz.random_mazes(1, 5)[0], step=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_observation_matches_a_fresh_env():
    """The figure's data is the env's own observation: replaying the same moves gives the same tensors."""
    agent, maze, task, rewards, max_steps = _resolve(None, "enclosed", 0, None)
    env, obs, comp, moves, path, _ = _run_to(agent, maze, 6, 0, task, rewards, max_steps)
    assert obs.shape == (N_MEMORY + 3, 2 * PAD + 1, 2 * PAD + 1) and comp.sum() == 1
    assert np.allclose(obs[:N_MEMORY].sum(0), 1)          # one memory code per square
    assert obs[N_MEMORY, PAD, PAD] == 1                   # it is standing in the middle
    _, _, comp_e, *_ = _run_to(agent, maze, 6, 0, "escape", rewards, max_steps)
    assert comp_e.sum() == 0                              # escape: no compass


# ---- the player and escape episodes -------------------------------------------------------------
GRID = [[3, 3, 3, 3], [3, 0, 0, 3], [3, 0, 0, 3], [3, 3, 3, 3]]


def _episode(task):
    goal = "escaped" if task == "escape" else "flag"
    end = [2, 3] if task == "escape" else [2, 2]
    return {"grid": GRID, "start": [1, 1], "flag": [1, 1] if task == "escape" else [2, 2], "task": task,
            "agents": [{"path": [[1, 1], [1, 2], [2, 2]] + ([end] if task == "escape" else []),
                        "score": [0, 0, 0, 5] if task == "escape" else [0, 0, 5],
                        "events": [{"t": 3 if task == "escape" else 2, "type": goal, "value": 5.0,
                                    "cell": end}]}]}


def test_record_marks_escape_episodes_for_the_player():
    ep = mz.record(None, [mz.scenario("enclosed")], max_steps=5, task="escape")[0]
    assert ep["task"] == "escape"
    html = player.render(ep)
    assert '"task":"escape"' in html
    assert "'escaped': return ['Escaped'" in html


# Mount the player in node with a tiny DOM stub, jump to the end, and report what was drawn.
DOM_STUB = r"""
const fs = require('fs');
function mk(tag) { return { tag, attrs: {}, children: [], style: {}, className: '', textContent: '',
  setAttribute(k, v) { this.attrs[k] = String(v); }, appendChild(c) { this.children.push(c); return c; },
  removeChild(c) { this.children.splice(this.children.indexOf(c), 1); }, insertBefore(c) { return this.appendChild(c); },
  querySelector() { return null; }, addEventListener() {}, getBoundingClientRect() { return { width: 520 }; },
  isConnected: true }; }
global.document = { createElementNS: (ns, t) => mk(t), createElement: t => mk(t) };
global.performance = { now: () => 0 }; global.requestAnimationFrame = () => {};
const MZP = eval(fs.readFileSync(process.argv[1], 'utf8'));
const root = mk('div');
MZP.mount(root, { mode: 'single', width: 520, episodes: [JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))] });
root.__mzp.seek(root.__mzp.T);
const all = []; (function walk(e) { all.push(e); e.children.forEach(walk); })(root);
const text = e => (e.textContent || '') + e.children.map(text).join('');
const co = all.find(e => e.tag === 'g' && e.attrs['class'] === 'mzp-callouts');
console.log(JSON.stringify({
  flags: all.filter(e => e.tag === 'circle' && e.attrs.fill === '#262626').length,
  labels: all.filter(e => e.tag === 'text' && e.attrs['font-weight'] === '600').map(e => e.textContent),
  callout: text(co) }));
"""


def _drawn(ep, tmp_path):
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    from importlib import resources
    js = tmp_path / "player.js"
    js.write_text(resources.files("mazerl.player").joinpath("player.js").read_text())
    data = tmp_path / "ep.json"
    data.write_text(json.dumps(ep))
    out = subprocess.run(["node", "-e", DOM_STUB, str(js), str(data)], capture_output=True, text=True,
                         check=True)
    return json.loads(out.stdout)


def test_player_escape_episode(tmp_path):
    d = _drawn(_episode("escape"), tmp_path)
    assert d["flags"] == 0                 # no flag to reach
    assert "GOAL" not in d["labels"]
    assert d["callout"] == "Escaped +5"    # the success callout stays up at the end


def test_player_flag_episode_unchanged(tmp_path):
    for ep in (_episode("flag"), {k: v for k, v in _episode("flag").items() if k != "task"}):
        d = _drawn(ep, tmp_path)
        assert d["flags"] == 1
        assert d["labels"] == ["OUTSIDE", "GOAL", "AGENT"]
        assert d["callout"] == "Flag +5"
