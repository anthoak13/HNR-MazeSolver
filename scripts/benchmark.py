"""Speed check: environment alone, then short training runs, on each available device.

    python scripts/benchmark.py --episodes 2000 --threads 4
"""
import argparse
import platform
import time

import torch

from mazerl.a2c import train
from mazerl.env import MazeEnv
from mazerl.maze import training_bank


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def env_rate(bank, device, n=512, steps=200):
    env = MazeEnv(bank, n, device=device)
    for _ in range(10):
        env.step(torch.randint(0, 4, (n,), device=device)); env.observe()
    sync(device)
    t = time.time()
    for _ in range(steps):
        env.step(torch.randint(0, 4, (n,), device=device)); env.observe()
    sync(device)
    return n * steps / (time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"host {platform.node()}  torch {torch.__version__}  threads {a.threads}")
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
        print(f"gpu {torch.cuda.get_device_name(0)}")
    else:
        print("gpu: none visible")

    t = time.time()
    bank = training_bank()
    print(f"layout bank: {len(bank)} layouts in {time.time() - t:.0f}s")

    for device in devices:
        print(f"\n[{device}] environment alone: {env_rate(bank, device):,.0f} moves/s")
        run = train(episodes=a.episodes, device=device, progress=False, bank=bank)
        h = run.history
        attempts = h["episodes"][-1]
        avg_moves = sum(h["steps"]) / len(h["steps"])
        rate = attempts * avg_moves / run.seconds
        print(f"[{device}] training: {attempts} attempts in {run.seconds:.0f}s "
              f"(~{rate:,.0f} moves/s, {attempts / run.seconds:.1f} attempts/s)")
        print(f"[{device}] projected 126k-attempt run: "
              f"{126_000 / (attempts / run.seconds) / 60:.0f} min at this early-training pace "
              f"(later attempts are shorter, so the real run is faster)")


if __name__ == "__main__":
    main()
