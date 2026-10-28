# Handoff: where things stand and how to work here

Read these first:
- `TEACHING.md`: what the activity is for, and the deliverables still to build.
- `PLAN.md`: the current design.
- `EXPERIMENTS.md`: every experiment, with its prediction and its result. The user's working preferences
  are in Claude's project memory, which loads automatically.

## In flight
- Nothing on the cluster. Round 10 is done: model size doesn't matter once trained, so everything uses
  the article's default size from now on.
- **My role** (from the user): map the space by experiment. The user designs and builds the student
  notebook; `TEACHING.md` is a focus document for me, not the notebook's plan.
- **Convergence gotcha:** a flat final 20% can be the learning-rate anneal, not convergence (round 10,
  small). When convergence matters, check that a longer run doesn't change the result.
- **Open questions about the space:** whether pretraining is reused on a second task (the escape task
  below), and what causes Example 2 under the 1:1:1 mix.

## Agreed but not built: the escape task (Part 1 of `TEACHING.md`)
Same agent, mazes, rewards and pretrained perception, with a different goal:
- **Start** where the flag would be: the generated flag cell, in the lower right inside the maze.
- **Success** is stepping onto any outside cell. That ends the attempt and earns the +5. The wall and
  new-cell rewards are unchanged; no outside penalty on that final step.
- **No compass** (the compass input is all zeros).
- **The bypass filter is off for this task.** Every success ends outside.
- **Then test:** RL on top of pretrained perception against RL from scratch, on both tasks, at class
  budgets (20k and 40k attempts on a GPU). The perception code is `mazerl/perception.py`;
  `train(perception=..., freeze_perception=False)` is the version that worked (round 6).

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
  `mazerl.rollout.breakout_stages(agent)` gives Example 1's stages.

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
- **Example 1 varies a lot between training seeds** (35–75% for the same settings), while re-measuring
  one model varies by about 1 point. Use several seeds when comparing settings.
