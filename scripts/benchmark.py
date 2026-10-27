"""Speed check on this machine (see mazerl/benchmark.py).

    python scripts/benchmark.py --threads 4
"""
import argparse

from mazerl.benchmark import run_benchmark

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--pretrain-steps", type=int, default=300)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    run_benchmark(a.episodes, a.pretrain_steps, threads=a.threads)
