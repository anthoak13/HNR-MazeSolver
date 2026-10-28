"""Tabulate experiment results (the .json files written by pretrain.py --evaluate).

Columns: success on new open / interior_break / outside_only mazes; "impossible" = breaks out to finish
an impossible task (reached the flag on the p.6 enclosed maze) and walls broken there; "needless out" =
breaks out when it doesn't need to (left the maze on the p.7 maze, which has an inside route).

    python scripts/summarize_runs.py runs/r1/*.json
"""
import json
import sys

COLUMNS = [("open", "reached flag"), ("interior_break", "reached flag"), ("outside_only", "reached flag"),
           ("enclosed", "reached flag"), ("enclosed", "avg walls broken"),
           ("outside_shortcut", "went outside")]
HEAD = ["open", "interior", "outside_only", "impossible", "impos. breaks", "needless out"]


def main(paths):
    rows = [json.load(open(p)) for p in paths]
    print(f"{'run':24s}" + "".join(f"{h:>14s}" for h in HEAD))
    for r in rows:
        cells = []
        for test, key in COLUMNS:
            v = r["tests"][test][key]
            cells.append(f"{v:>14.2f}" if key.startswith("avg") else f"{v:>13.1%} ")
        print(f"{r['name']:24s}" + "".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
