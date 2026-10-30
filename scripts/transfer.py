"""Pretraining-and-transfer experiments (reports/pretraining-transfer.md): RL on one task, starting
from scratch, from reward-free pretrained perception, or from another run's perception.

    python -m scripts.transfer --task escape --start scratch --seed 0 --episodes 20000 --out runs/local/t1
    python -m scripts.transfer --task escape --start runs/local/perception_s0.pt --seed 0 ...
    python -m scripts.transfer --task escape --start runs/local/perception_s0.pt --freeze ...
    python -m scripts.transfer --task escape --start runs/local/t1/flag_scratch_s0.pt ...   # an RL run's perception

Saves <name>.pt (the Run) and <name>.json (settings, learning curve, standard test suite).
Run it with ``python -m`` from the repo root so it imports this checkout's mazerl.
"""
import argparse
import json
from pathlib import Path

import torch

from mazerl.a2c import train, Run
from mazerl.perception import Pretrained
from mazerl.rollout import standard_report


def load_start(path):
    """Perception weights from a pretraining file or from a trained Run."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "perception" in d:
        return Pretrained.load(path).agent, "reward-free pretraining"
    return Run.load(path).agent, f"perception of RL run ({d['settings'].get('task', 'flag')} task)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("flag", "escape"), required=True)
    ap.add_argument("--start", default="scratch", help="'scratch', or a .pt file whose perception to start from")
    ap.add_argument("--freeze", action="store_true", help="keep the starting perception fixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--evaluate", type=int, default=300, help="attempts per test in the test suite")
    ap.add_argument("--policy-init-scale", type=float, default=1.0,
                    help="shrink the fresh move-choice layer at the start (0.01: even starting policy)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    perception, source = (None, "none") if a.start == "scratch" else load_start(a.start)
    run = train(task=a.task, seed=a.seed, episodes=a.episodes, device="cpu", progress=False,
                perception=perception, freeze_perception=a.freeze, policy_init_scale=a.policy_init_scale,
                checkpoint_at=(1_000, 2_000, 5_000, 10_000))
    a.out.mkdir(parents=True, exist_ok=True)
    run.save(a.out / f"{a.name}.pt")
    report = standard_report(run, n=a.evaluate)
    summary = {"name": a.name, "task": a.task, "seed": a.seed, "episodes": a.episodes,
               "start": a.start, "start_source": source, "frozen": a.freeze,
               "policy_init_scale": a.policy_init_scale,
               "minutes": run.seconds / 60, "history": run.history, "tests": report}
    (a.out / f"{a.name}.json").write_text(json.dumps(summary, indent=1, default=str))
    print(a.name, f"{run.seconds / 60:.1f} min", {k: v["reached flag"] for k, v in report.items()}, flush=True)


if __name__ == "__main__":
    main()
