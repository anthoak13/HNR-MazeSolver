"""Train the agents that ship with the package (mazerl/pretrained/*.pt).

    python scripts/pretrain.py base            # the article's setup
    python scripts/pretrain.py sweep --jobs 3  # reward-variation agents for class
    python scripts/pretrain.py all

Each run uses the article's scale (126,000 attempts over 13,148 layouts) unless --episodes is given.
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import torch

from mazerl.a2c import train, Settings
from mazerl.env import Rewards

OUT = Path(__file__).resolve().parents[1] / "mazerl" / "pretrained"
STAGES = (1_000, 5_000, 20_000, 60_000)

BASE_SETTINGS = {}
BASE = Rewards()

# name -> (reward changes, settings changes); each differs from the base in one way
SWEEP = {
    "outside_free": ({"outside": 0.0}, {}),
    "outside_costly": ({"outside": -0.1}, {}),
    "explore_bonus_big": ({"new_cell": 0.1}, {}),
    "explore_bonus_none": ({"new_cell": 0.0}, {}),
    "probing_costly": ({"probe": -0.1, "through_wall": -0.25}, {}),
    "no_bypass_filter": ({}, {"exclude_bypass": False}),
}


def job(name, reward_changes, settings_changes, episodes, threads):
    torch.set_num_threads(threads)
    rewards = Rewards(**{**BASE.as_dict(), **reward_changes})
    overrides = {**BASE_SETTINGS, **settings_changes}
    if episodes:
        overrides["episodes"] = episodes
    run = train(rewards, progress=False, checkpoint_at=STAGES if name == "base" else (), **overrides)
    OUT.mkdir(exist_ok=True)
    run.save(OUT / f"{name}.pt")
    h = run.history
    print(f"{name}: {run.seconds / 60:.1f} min, final success {h['success'][-1]:.2f}, "
          f"outside {h['outside'][-1]:.1f}, bypass {h['bypass'][-1]:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["base", "sweep", "all"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    jobs = []
    if a.what in ("base", "all"):
        jobs.append(("base", {}, {}))
    if a.what in ("sweep", "all"):
        jobs += [(n, r, s) for n, (r, s) in SWEEP.items()]
    with mp.Pool(a.jobs) as pool:
        pool.starmap(job, [(n, r, s, a.episodes, a.threads) for n, r, s in jobs])


if __name__ == "__main__":
    main()
