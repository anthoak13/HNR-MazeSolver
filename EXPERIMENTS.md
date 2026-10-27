# Experiments

Lab notebook for choosing the values that aren't fixed by the article. Every round follows the same
three steps: write the prediction down, run the smallest set of runs that tests it, then compare.

- **Held fixed (the author's numbers):** flag +5, outside −0.01 per move.
- **Free to tune:** the wall penalty (`probe`), the new-cell reward (`new_cell`, plus `explore_outside`),
  and the training mix of maze kinds (`maze_mix`, as open : interior_break : outside_only).

## What the demo needs to show (pedagogical goals)
1. **G1, competence:** the agent reliably solves new mazes of the kinds it trained on.
2. **G2, emergence:** on the article's `enclosed` maze, never seen in training, the agent sometimes
   leaves, breaks back in, and reaches the flag. It should happen often enough to show in class (≳10%).
3. **G3, legible cause and effect:** changing a reward or the training mix changes behavior in a way
   students could predict and explain.

## Method
- Every run is full scale: 126k attempts on 13,148 layouts, 1 A30 GPU, about 5 minutes.
- Scoring uses the standard suite (`mazerl.rollout.standard_report`), 500 attempts per test:
  - new mazes of each kind (open, interior_break, outside_only);
  - `enclosed` (p.6) and `outside_shortcut` (p.7).
- Commands: `sbatch --array=1-N%2 scripts/sweep.sbatch sweeps/<round>.txt`. Results are
  `runs/<round>/*.json` on the cluster.

## Reference run: A (seed 0)
Settings: memory of probed walls, probe −0.01, new cell +0.005 (inside and outside), mix 3:2:2.

| open | interior_break | outside_only | enclosed | shortcut: went outside |
|---|---|---|---|---|
| 99.0% | 89.6% | 97.2% | 15.0% | 8% |

---

## Round 1: How noisy is a single run, and does the breakout need both component skills?

**Question 1: seed noise.** Re-run A's settings with seeds 1 and 2. Any effect smaller than the spread
across these three runs can't be trusted.
- *Prediction:* open 98–100%, interior_break 85–94%, outside_only 95–99%, enclosed **8–20%**.

**Question 2: emergence needs the parts.**
- *Hypothesis:* the article says the agent "combined these separately learned behaviors."
  - The breakout chains two skills: leaving the maze and travelling outside (learned from outside_only
    mazes), and breaking through walls (learned from interior_break mazes).
  - Remove either kind from training and the breakout should mostly disappear, even though nothing else
    changes.
- *Runs:*
  - `mix_no_outside_only` (1:1:0)
  - `mix_no_interior_break` (1:0:1)
- *Predictions:*
  - **No outside_only:**
    - outside_only success falls well below the baseline, under 60%.
    - On `enclosed`, the agent still steps out, because the start pocket only opens outward. But it
      rarely travels and breaks in: **enclosed < 5%**.
  - **No interior_break:**
    - interior_break success falls below 50%.
    - It probes less.
    - **enclosed < 3%**, since that route needs 4 breaks in a row.
  - Open and outside_only (or open alone) stay near the baseline.
- *Why it matters for teaching:* if this holds, it's a clean classroom experiment. "Remove one kind of
  training maze; predict what happens to the breakout."

**Status:** submitted.
