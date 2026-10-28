"""Re-run the standard test suite on saved runs and update their .json files (e.g. after a new measure
is added to the suite). The .json must sit next to the .pt, as pretrain.py --evaluate writes it.

    python scripts/rescore.py runs/r11/*.pt
"""
import json
import sys
from pathlib import Path

from mazerl.a2c import Run
from mazerl.rollout import standard_report


def main(paths, n=500):
    for p in map(Path, paths):
        summary = json.loads(p.with_suffix(".json").read_text())
        summary["tests"] = standard_report(Run.load(p), n=n)
        p.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
        print(f"rescored {p}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
