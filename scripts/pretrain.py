"""Train the agents that ship with the package (mazerl/pretrained/*.pt).

    python scripts/pretrain.py --list                    # names of all runs
    python scripts/pretrain.py base                      # one run (what each SLURM array task does)
    python scripts/pretrain.py base --episodes 2000      # quick smoke test
    python scripts/pretrain.py all --jobs 3              # everything, 3 at a time on one machine
    python scripts/pretrain.py base --name expA --reward explore_outside=False \
        --setting maze_mix=0.29,0.43,0.29               # one-off experiment

Each run uses the article's scale (126,000 attempts over 13,148 layouts) unless --episodes is given.
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import torch

from mazerl.a2c import train
from mazerl.env import Rewards

OUT = Path(__file__).resolve().parents[1] / "mazerl" / "pretrained"
STAGES = (1_000, 5_000, 20_000, 60_000)

# name -> (reward changes, settings changes); each variant differs from "base" in one way
RUNS = {
    "base": ({}, {}),
    "outside_free": ({"outside": 0.0}, {}),
    "outside_costly": ({"outside": -0.1}, {}),
    "explore_bonus_big": ({"new_cell": 0.1}, {}),
    "explore_bonus_none": ({"new_cell": 0.0}, {}),
    "probing_costly": ({"probe": -0.1}, {}),
    "no_bypass_filter": ({}, {"exclude_bypass": False}),
}


def job(name, episodes, threads, device, out, seed, extra_rewards=None, extra_settings=None, save_as=None):
    torch.set_num_threads(threads)
    reward_changes, overrides = RUNS[name]
    rewards = Rewards(**{**reward_changes, **(extra_rewards or {})})
    overrides = {**overrides, **(extra_settings or {}), "seed": seed}
    if episodes:
        overrides["episodes"] = episodes
    run = train(rewards, device=device, progress=False, checkpoint_at=STAGES, **overrides)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{save_as or name}.pt"
    run.save(path)
    h = run.history
    print(f"{save_as or name}: rewards={rewards.as_dict()} settings={overrides}", flush=True)
    print(f"{save_as or name}: {run.seconds / 60:.1f} min on {device or 'auto'}, final success {h['success'][-1]:.2f}, "
          f"outside {h['outside'][-1]:.1f}, bypass {h['bypass'][-1]:.2f} -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", help="a run name (see --list) or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1, help="parallel runs when what='all'")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward", action="append", default=[], help="override a reward, e.g. probe=-0.02")
    ap.add_argument("--setting", action="append", default=[], help="override a setting, e.g. maze_mix=0.3,0.4,0.3")
    ap.add_argument("--name", default=None, help="file name for a one-off experiment")
    a = ap.parse_args()
    if a.list or not a.what:
        print("\n".join(RUNS))
        return
    names = list(RUNS) if a.what == "all" else [a.what]
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        ap.error(f"unknown run {unknown[0]!r}; use --list")
    extra_r, extra_s = dict(map(_parse, a.reward)), dict(map(_parse, a.setting))
    if (extra_r or extra_s or a.name) and len(names) != 1:
        ap.error("--reward/--setting/--name apply to a single run")
    args = [(n, a.episodes, a.threads, a.device, a.out, a.seed, extra_r, extra_s, a.name) for n in names]
    if len(args) == 1:
        job(*args[0])
    else:
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            pool.starmap(job, args)


def _parse(item):
    key, value = item.split("=", 1)
    if value in ("True", "False"):
        return key, value == "True"
    if "," in value:
        return key, tuple(float(v) for v in value.split(","))
    try:
        return key, int(value)
    except ValueError:
        return key, float(value)


if __name__ == "__main__":
    main()
