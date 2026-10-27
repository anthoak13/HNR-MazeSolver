"""Reward-free perception pretraining (see mazerl/perception.py), timed.

    python scripts/pretrain_perception.py --threads 2 --out mazerl/pretrained/perception.pt
"""
import argparse
import json

import torch

from mazerl.perception import pretrain_perception


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=150_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    p = pretrain_perception(steps=a.steps, n_samples=a.samples, seed=a.seed, device=a.device, progress=False)
    p.save(a.out)
    print(json.dumps({"seconds": round(p.seconds, 1), **{k: round(v, 3) for k, v in p.test.items()}}))


if __name__ == "__main__":
    main()
