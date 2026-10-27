"""Timing on whatever machine this runs on (Colab CPU, Colab T4, a laptop, ...).

    from mazerl.benchmark import run_benchmark
    run_benchmark()
"""
import platform
import time

import torch

from .a2c import train
from .env import MazeEnv
from .maze import training_bank
from .perception import pretrain_perception


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _env_rate(bank, device, n=512, steps=200):
    env = MazeEnv(bank, n, device=device)
    for _ in range(10):
        env.step(torch.randint(0, 4, (n,), device=device)); env.observe()
    _sync(device)
    t = time.time()
    for _ in range(steps):
        env.step(torch.randint(0, 4, (n,), device=device)); env.observe()
    _sync(device)
    return n * steps / (time.time() - t)


def run_benchmark(episodes=2000, pretrain_steps=300, devices=None, threads=None):
    """Time the pieces students would run, and project class-sized budgets.

    The RL rates come from early training, when attempts are long, so the projections are upper bounds.
    """
    if threads:
        torch.set_num_threads(threads)
    if devices is None:
        devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    print(f"python {platform.python_version()}  torch {torch.__version__}  cpu threads {torch.get_num_threads()}"
          + (f"  gpu {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  (no gpu)"))
    t = time.time()
    bank = training_bank()
    print(f"training layouts ready in {time.time() - t:.0f}s (generated once, then cached)\n")

    results = {}
    for device in devices:
        print(f"===== {device.upper()} =====", flush=True)
        print("1/4  Environment alone (random moves)...", flush=True)
        r = {"env_moves_per_s": _env_rate(bank, device)}
        print(f"2/4  RL from scratch: {episodes:,} attempts...", flush=True)
        run = train(episodes=episodes, device=device, progress=True, bank=bank)
        r["scratch_attempts_per_s"] = run.history["episodes"][-1] / run.seconds
        print(f"3/4  Perception pretraining (no rewards): {pretrain_steps} steps...", flush=True)
        p = pretrain_perception(steps=pretrain_steps, n_samples=pretrain_steps * 100, device=device,
                                bank=bank, progress=True)
        r["pretrain_s_per_step"] = (p.seconds - p.test["data_seconds"]) / pretrain_steps
        r["pretrain_data_s"] = p.test["data_seconds"]
        print(f"4/4  RL on top of the pretrained perception: {episodes:,} attempts...", flush=True)
        run = train(episodes=episodes, device=device, progress=True, bank=bank, perception=p.agent,
                    freeze_perception=True)
        r["on_pretrained_attempts_per_s"] = run.history["episodes"][-1] / run.seconds
        results[device] = r
        print(f"\n[{device}] summary")
        print(f"  environment alone:               {r['env_moves_per_s']:,.0f} moves/s")
        print(f"  RL from scratch:                 {r['scratch_attempts_per_s']:.0f} attempts/s"
              f"  -> 20k attempts in ~{20_000 / r['scratch_attempts_per_s'] / 60:.0f} min,"
              f" 126k in ~{126_000 / r['scratch_attempts_per_s'] / 60:.0f} min")
        print(f"  perception pretraining:          {r['pretrain_s_per_step']:.3f} s/step"
              f"  -> 3000 steps in ~{3000 * r['pretrain_s_per_step'] / 60:.0f} min (+ data collection)")
        print(f"  RL on pretrained perception:     {r['on_pretrained_attempts_per_s']:.0f} attempts/s"
              f"  -> 20k attempts in ~{20_000 / r['on_pretrained_attempts_per_s'] / 60:.0f} min\n")
    return results
