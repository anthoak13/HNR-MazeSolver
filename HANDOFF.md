# Handoff: where things stand and how to work here

Read these first:
- `TEACHING.md`: what the activity is for, and the deliverables still to build.
- `PLAN.md`: the current design.
- `EXPERIMENTS.md`: every experiment, with its prediction and its result. The user's working preferences
  are in Claude's project memory, which loads automatically.

## In flight
- **Round 11 is done** (`EXPERIMENTS.md`): each reward varied on its own. The re-scored `.json` files
  (with time to finish) are in `runs/r11/rescored/` and `runs/r4/rescored/` on the cluster.
- **Pretraining and transfer:** a separate agent is testing the escape task locally on CPU, in a git
  worktree, and writing a report. Its runs are well under 126k attempts.
- The model-size experiments are done (`EXPERIMENTS.md`, round 10): trained to
  convergence, size doesn't change the behavior, so everything uses the article's default size from now
  on.
- **Scope** (from the user): the lesson asks one question, how rewards affect behavior. Every experiment
  keeps the 1:1:1 maze mix (equal shares of the three kinds of training maze) and the default size; only
  rewards change.
- **My role** (from the user): map the space by experiment. The user designs and builds the student
  notebook; `TEACHING.md` is a focus document for me, not the notebook's plan.
- **Training length:** every run is 126k attempts, the article's number. The user ruled out longer runs
  (no gain, harder story).
- **Open questions about the space:** whether pretraining is reused on a second task (the escape task
  below), and whatever the user's notebook design needs next.

## Agreed but not built: the escape task (Part 1 of `TEACHING.md`)
Same agent, mazes, rewards and pretrained perception, with a different goal:
- **Start** where the flag would be: the generated flag cell, in the lower right inside the maze.
- **Success** is stepping onto any outside cell. That ends the attempt and earns the +5. The wall penalty
  and the exploration reward are unchanged; no outside penalty on that final step.
- **No compass** (the compass input is all zeros).
- **Attempts that end outside are not excluded from training for this task** (`exclude_bypass=False`).
  Every success ends outside.
- **Then test:** RL on top of pretrained perception against RL from scratch, on both tasks, with 20k- and
  40k-attempt runs on a GPU (the lengths that fit in class). The perception code is
  `mazerl/perception.py`; `train(perception=..., freeze_perception=False)` is the version that worked
  (`EXPERIMENTS.md`, round 6: with the pretrained layers frozen, the agent didn't learn the task).

## Next after that
The deliverables list at the end of `TEACHING.md`:
1. Train the shipped agents. Use several seeds of `base` and ship a typical one, not the best one.
2. Build the missing code.
3. Write the student notebook and instructor key.

## Results (not in git)
- **On the cluster:** `~/HNR-MazeSolver/runs/<round>/<name>.pt` (trained runs) and `.json` (standard test
  suite: 500 attempts per test).
  - `runs/exp`: rounds A–C.
  - `runs/r1` … `runs/r10`: rounds 1–10.
  - `runs/r6_local`: round 6, the pretraining runs, done on the desktop CPU.
- To look at results: `scp` the `.json` files back and run `python scripts/summarize_runs.py <files>`.
  `mazerl.rollout.breakout_stages(agent)` gives, for the article's p.6 maze (flag walled off from the
  start), the share of attempts that leave, break back in from outside, and reach the flag.

## Cluster (`ssh hpc`; the user turns on the VPN)
- **Location:** repo at `~/HNR-MazeSolver`, branch `maze-rl`. Run `git pull` there after pushing.
- **Python env:** `~/pytorch/.venv` (torch 2.9, CUDA 12.8), with the repo installed in it as editable.
- **GPUs:** partition `gpus`, node `thor` (2× A30). Exclude `odin`: its P100 isn't supported by this
  torch build, and `/dev/nvidia-uvm` is missing there. `loki` is down.
- **Experiments:**
  1. Write the prediction in `EXPERIMENTS.md` *first*.
  2. List the runs in `sweeps/<round>.txt`.
  3. Submit: `sbatch --array=1-N%2 scripts/sweep.sbatch sweeps/<round>.txt`. Each run trains and then
     scores itself.
- **CPU partition** (`cs`): `scripts/sweep_cpu.sbatch`. Only useful when the run time itself is what's
  being measured.
- **Final shipped agents:** `scripts/pretrain.sbatch` / `scripts/pretrain.py` (not the experiment
  framework).
- **Timing:** a full 126k-attempt run takes about 5 min on an A30. A Colab T4 is about 2.4× slower.

## Gotchas
- **`pkill -f <pattern>` can kill the shell running it,** when the pattern appears in that shell's own
  command line. Find the PIDs first, then `kill` them.
- **The permission check sometimes blocks long `sbatch` submissions.** The user has approved cluster
  submissions; ask them if one is blocked.
- **Training anneals the learning rate to zero by the end of the budget,** so a run levels off at the end
  whether or not it has really converged. Check the learning curve itself.
- **Success on the p.6 maze varies a lot between training seeds** (35–75% for the same settings), while re-measuring
  one model varies by about 1 point. Use several seeds when comparing settings.
