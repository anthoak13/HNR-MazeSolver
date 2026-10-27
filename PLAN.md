# Maze RL activity — plan

An HNR-1303 (Unit 5) activity built on the NYT article *"A.I.'s Rogue Behavior Starts With Its Training"*
(Saha & Migliozzi, Oct. 26 2026; PDF in the repo root). Students set reward values, train or load agents,
and watch what the agents learn. That includes the article's central example: an agent that leaves the
maze and breaks back in to reach the flag, even though it was never trained to do so. Students don't
touch the training code. They change numbers and look at animations and curves.

## Sources
Every rule is tagged with where it comes from.

- **[A] The article:** its text, the 9 maze figures, and the training animation. It is the basis, and it
  wins any conflict.
- **[C] The author's comments:**
  - "reaching the flag could fetch the agent +5 points, while stepping outside could result in a penalty
    of −0.01"
  - "it took less than an hour to train the model on my laptop"
- **[P] The paper the article cites**, *AI Safety Gridworlds* (Leike et al. 2017, arXiv:1711.09883). Used
  only where the article is silent.
- **[D] Our design decision.** Every [D] is a setting, not hard-wired.

---

## 1. The world

| Item | Rule | Tag |
|---|---|---|
| Arena | 18 × 18 cells. The maze is the middle 14 × 14 block, with a 2-cell **outside** band around it. Walls fill whole cells | [A] figures |
| Start | Top-left corner of the maze. The entrance is 3 open cells touching the outside | [A] all figures |
| Flag | A floor cell in rows/cols 10–14 (lower right) | [A] all 9 figures |
| Openings | 3–5 gaps in the outer wall besides the entrance | [A] counted in the training figures |
| Weak spots | 2–6 wall cells per maze, inner or outer. Each has open cells on two opposite sides, so it "opens into a passage." They look like ordinary walls to the agent | [A] counts and "open into passages"; [D] agent can't tell them apart (it "must probe walls to discover" them) |
| Layout style | Recursive division: long walls with single doorways, some open rooms | [D], matched to the figures |
| Layout bank | 13,148 fixed layouts. Each attempt picks one at random | [A] the number; [D] uniform sampling |
| Arena edge | Impassable. The agent sees it as its own kind of cell, not a wall | [D] |

**Three kinds of training maze** [A]. These come from the article's 7 training figures, transcribed cell
by cell in `mazerl/scenarios/training/`:

| Kind | Figures | How the flag is reached |
|---|---|---|
| open | p3, p4a, p4b | inside, without breaking anything |
| interior_break | p4c, p5a | inside, through an interior weak spot |
| outside_only | p5b, p5c | out the entrance and back in through an outer opening |

- **Mix [D]:** equal shares, 1 : 1 : 1 (`maze_mix`). The figures don't imply a split. In experiments,
  equal shares gave the strongest version of both of the article's behaviors (`EXPERIMENTS.md`, round 4).
- **Outer weak spots:** allowed anywhere, including next to the flag's area. The training animation shows
  an agent breaking in from outside.

**The article's two test mazes**, transcribed from pp. 6–7 into `mazerl/scenarios/`:

- **`enclosed` (p.6, "Agent facing a hard problem"):**
  - The start pocket opens only to the outside.
  - The shortest route goes out, then in through 4 weak spots: 30 moves.
- **`outside_shortcut` (p.7):**
  - Inside route: 27 moves. Outside route: 25 moves, through plain openings.
  - The text says the agent exploited "a vulnerability in the wall." The figure shows openings instead,
    and we use the figure as drawn.

---

## 2. The agent

| Item | Rule | Tag |
|---|---|---|
| Moves | Up / down / left / right | [A] |
| Walking into a wall (a **probe**) | One penalty per attempt. A weak spot gives way on the first bump and the agent moves in; a solid wall or the arena edge doesn't, and the agent stays put | [A] "each probe or move through a wall carries a small penalty"; [D] first-bump |
| Vision | 5 × 5 window centered on the agent. Walls block line of sight; grazing a corner doesn't | [A] window and blocking; [D] corner rule |
| Memory | Everything it has seen this attempt: floor, outside, wall, edge, flag. Also which cells it has visited, and **which walls it walked into that held firm** | [A] "remember past observations" |
| Flag direction | Compass direction, one of 8 | [A] "compass direction"; [D] 8-way |
| Attempt length | Ends at the flag or after **200 moves** | [D]. [P] uses 100 for grids ≤10×10; our arena is 18×18 |
| Multiple agents | Several can share a maze. They move independently and never block each other | [A] figures; [D] no interaction |

**Network: "vision" + "reasoning"** [A], **229,553 connections** (article: "about 230,000") [A].
- **Input [D]:** an 11 × 11 window of the agent's memory, centered on it, with 9 channels:
  - unseen, floor, outside, wall, flag, edge;
  - visited, visible now, probed-and-held-firm.

  The 8-way compass comes in separately.
- **Vision [D]:** conv 3×3 (9→32) → conv 3×3 (32→64) → 2×2 max-pool → conv 3×3 (64→64).
- **Reasoning:** two hidden layers of 100 [P] → move probabilities (4) and a value estimate (1).

---

## 3. Rewards (student-tunable)

| Event | Points | Fires when | Why this value |
|---|---|---|---|
| Reach the flag | **+5** | landing on the flag; the attempt ends | [C]; "much larger" than the rest [A] |
| Outside | **−0.01** | every move that ends on an outside cell | [C]. Per move because p.6 says "penalties," plural, for one trip out |
| Walk into a wall | **−0.01** | any probe (weak spots give way, solid walls don't) | "small penalty" [A]. Uses the author's one small magnitude [C] |
| New cell | **+0.005** | first visit to a cell this attempt | "small reward" [A]. Smaller than 0.01 so a step outside is always a net penalty ("stepped … outside … penalized," p.3) |
| New cell outside | same as new cell (`explore_outside=True`) | — | [D]. **Open decision**, being tested |
| Per move | 0 | — | none mentioned [A] |

**Checks these values pass:**
- **Outside always costs.** Every move outside nets at least −0.005.
- **Exploring can't rival the flag.** Exploring the whole maze interior earns at most about +0.5.
  The flag, discounted from 200 moves away, is worth 0.68.
- **The flag outweighs the penalties on the article's routes (p.6):**
  - `enclosed`: −0.12 in penalties along the route, +3.76 net after discounting.
  - `outside_shortcut`: +3.98 net after discounting.

---

## 4. Training

**Algorithm.** A2C-style advantage actor-critic [P], with one gradient pass per move.
- Attempts are trained on whole, so excluded attempts can be dropped before any learning happens.
- Pooled attempts can lag the current policy by a few updates. The policy loss uses PPO's clipped ratio
  to stay stable [D].

| Setting | Value | Tag |
|---|---|---|
| Discount | 0.99 | [P] |
| Value-loss weight / gradient clip | 0.25 / 40 | [P] |
| Rewards | divided by the largest (5) | [P] |
| Optimizer | Adam, learning rate 1e-3, annealed to 0 | [D] |
| Entropy bonus ("exploration") | 0.05 → 0.01 | [P] range |
| Parallel attempts / moves per update | 256 / 2,048 | [D] |
| Scale | **126,000 attempts over 13,148 layouts** | [A] |

**"Training excluded scenarios where the agent bypassed the maze entirely"** [A].
- **Rule:** an attempt that ends with the agent outside the maze is left out of training. The agent left
  and never came back in.
  - Excluded attempts still count toward the 126,000, and they're logged.
  - `bypass_rule="never_entered"` is the one remaining alternative reading.
- **Why this reading:**
  - Two training mazes can only be solved by going outside and coming back in elsewhere, which rules out
    readings that exclude that.
  - The training animation shows agents breaking in from outside, which rules out excluding break-ins.
  - "Where the agent bypassed" describes something the agent did, not a property of the layout.

**Student-facing knobs:** reward values, `learning_rate`, `discount`, `exploration`, `episodes`,
`exclude_bypass`, `maze_mix`. Everything else is hidden.

**Engine [D]:** our own batched PyTorch environment, which runs on CPU or GPU. pycolab, the engine the
paper's environments use, doesn't run on numpy 2 and steps one game at a time. Mazes are written in ASCII
for definitions and debugging only.

---

## 5. Compute

| Where | Full 126k-attempt run | Use |
|---|---|---|
| Campus cluster, `thor` (NVIDIA A30), via SLURM | **≈5 min** (measured) | all pretraining and experiments |
| Desktop CPU (8 threads) | ≈50 min (measured) | quick checks and evaluation |
| Colab CPU / T4 | not measured yet (`notebooks/00-benchmark.ipynb`) | class |

**Cluster setup** (`ssh hpc`):
- Repo at `~/HNR-MazeSolver`, installed into `~/pytorch/.venv` (torch 2.9, CUDA 12.8). Jobs use partition
  `gpus` and exclude `odin`: its P100 isn't supported by this torch build, and `/dev/nvidia-uvm` is
  missing on that node.
- `scripts/pretrain.sbatch` runs one array task per pretrained agent. `scripts/pretrain.py` also takes
  `--reward`, `--setting` and `--name` for one-off experiments.
- Results are copied back here and committed.

---

## 6. Pretrained agents (shipped in `mazerl/pretrained/`)
- **`base`:** the article's setup, at full scale. It saves snapshots at 1k, 5k, 20k, 60k and 126k attempts
  for the "agents at different stages of training" view.
- **Variants,** each differing from `base` in one way:

  | Variant | Change |
  |---|---|
  | `outside_free` | outside 0 |
  | `outside_costly` | outside −0.1 |
  | `explore_bonus_big` | new cell +0.1 |
  | `explore_bonus_none` | new cell 0 |
  | `probing_costly` | wall −0.1 |
  | `no_bypass_filter` | excluded attempts are kept |

These get trained once the base settings are final (§9).

---

## 7. What students see
Every maze view is an **animation in the NYT style**:
- **Look:** dotted grid, light-gray walls, and pale squares for weak spots (visible to the viewer, not the
  agent). Pink tint outside, a hollow start circle, a red agent with a fading trail, and a black flag.
- **Motion:** agents glide between cells. Halos and callouts appear when a reward or penalty fires.
  Broken walls crack open. Several agents can share a maze.
- **Controls:** play, pause, scrub, speed, and a running score.

It's a self-contained HTML/SVG/JS player (`mazerl/player/`) that runs inline in Colab/Jupyter. It can also
save standalone `.html` files. Training curves use matplotlib.

| Function | Shows | Article figure | Status |
|---|---|---|---|
| `mz.show_maze(maze)` | a layout | pp. 1–2 | built |
| `mz.playback(agent, maze, n_agents=…)` | animated attempts with callouts | p.3 | built |
| `mz.training_grid(run)` | several layouts, agents from several training stages | pp. 4–5 | built |
| `mz.scenario("enclosed" / "outside_shortcut")` | the article's test mazes | pp. 6–7 | built |
| `mz.evaluate(agent, n=…, mazes=…)` | results on unseen layouts or a scenario | training vs. validation, as in 05-02 | built |
| `run.plot()` | curves: reward, % reaching the flag, moves, probes, breaks, time outside, % bypass | mirrors 05-01/05-02 loss curves | built |
| `mz.train(rewards, start_from=…, episodes=…)` | train or retrain an agent | — | built (a `minutes=` budget is planned) |
| `mz.load_pretrained(name)` | shipped agents | — | built (no agents shipped yet) |
| `mz.agent_view(agent, maze, step)` | what the agent knows: memory, view, compass | Methodology | planned |
| `mz.reward_breakdown(agent, maze)` | where an attempt's points came from | p.6 "outweighed the penalties" | planned |

```python
!pip install -q git+https://github.com/anthoak13/HNR-MazeSolver@maze-rl
import mazerl as mz
base = mz.load_pretrained("base")
mz.playback(base, mz.scenario("enclosed"), n_agents=3)
run = mz.train(mz.Rewards(new_cell=0.1), start_from=base, episodes=20_000)
run.plot()
mz.evaluate(run, n=500)
```

**Student notebook (planned):** `notebooks/01-maze-rl-in-class.ipynb` plus an instructor key, in the
05-0x style, with bold questions throughout. It follows the article:
1. the maze and an untrained agent;
2. rewards and penalties;
3. watching training;
4. the enclosed maze;
5. changing rewards, and reward hacking;
6. new layouts vs. training layouts;
7. discussion.

---

## 8. Repo layout
```
mazerl/
  __init__.py      public API
  maze.py          generator, maze kinds, layout bank, scenarios
  env.py           batched environment, rewards, bypass rule
  agent.py         vision + reasoning network
  a2c.py           training, Settings, Run
  rollout.py       recording attempts, evaluation
  viz.py           training curves
  player/          NYT-style animation player (HTML/SVG/JS)
  scenarios/       article test mazes; training/ = the 7 training figures
  pretrained/      shipped agents (pending)
scripts/           pretrain.py, pretrain.sbatch, benchmark.py
notebooks/         00-benchmark.ipynb (student notebook pending)
tests/             pytest
```

---

## 9. Status and next steps

**Current results** (old memory, before the probed-walls fix; 3 seeds, 500 attempts each):

| Maze | Reached the flag |
|---|---|
| open | 95–97% |
| outside_only | 94–96% |
| interior_break | 39–46% |
| `enclosed` (the breakout) | 0–1.2% |
| `outside_shortcut` | ~100%, always by the inside route |

The cause for interior_break: the agent bumped the same walls over and over (34 of 45 bumps were
repeats). It also left the maze instead of probing inside.

**In progress:** three full-scale runs on `thor`, each with the probed-walls memory:

| Run | What else changes |
|---|---|
| A | nothing; the memory fix only |
| B | `explore_outside=False` |
| C | `maze_mix` 2 : 3 : 2 (more interior_break mazes) |

**Next:**
1. Compare A–C and settle the two open decisions: `explore_outside` and `maze_mix`.
2. Measure how often the breakout happens (on `enclosed`), and whether the agent takes the outside
   shortcut. Report it as a frequency, never cherry-picked, and never by scripting the behavior.
3. Train `base` + the six variants on `thor`, then commit them to `mazerl/pretrained/`.
4. Build `agent_view` and `reward_breakdown`, plus a `minutes=` option for `train`.
5. Write the student notebook and instructor key.
6. Run the Colab benchmark, and set the in-class training budgets from it.
7. Test end to end from a fresh Colab runtime. Then merge to `main`.

## 10. Possible extension: one pretrained model, several RL tasks

This is the modern-LLM pattern: pretrain a model once without rewards, then use RL to make it good at
different jobs.

- **Pretraining** (built: `mazerl/perception.py`). The perceiving part of the network (vision plus the
  first reasoning layer) learns two prediction tasks from agents exploring the training mazes, with no
  rewards:
  - fill in the cells of its view it hasn't seen yet;
  - pick the move toward the nearest unexplored ground.

  Neither task mentions flags, so the result isn't tied to one goal. RL then trains only the deciding
  layers: `train(perception=..., freeze_perception=...)`.
- **What it could show students:**
  1. **A speed-up** on the main task, if it's large enough to matter in class (`EXPERIMENTS.md`,
     round 6).
  2. **Reuse:** the same pretrained model learns a *second* task with RL. That's a stronger argument for
     pretraining than a speed-up alone. Candidate second tasks, each a small change to the reward or the
     ending:
     - **Escape:** start deep inside and get out of the maze. It reuses probing and finding openings.
     - **Explore and map:** no flag; points only for new cells, within a time limit.
     - **Collect several flags** in one attempt, which needs planning and memory.
- **Gate:** keep pretraining in the activity only if it helps within class-sized budgets on 2 CPU
  threads. The test would be *H11: the same pretrained perception speeds up RL on a task it was never
  pretrained for, by about as much as on the main task.* Escape is the cheapest to try first.
  - **If it helps:** pretraining earns a section in the notebook.
  - **If not:** at most a short discussion of how LLMs are built.

## 11. Risks
- **The breakout may stay rare.** That would be reported honestly. The article shows one replay, so how
  often it happened for them is unknown.
- **A2C can be unstable.** Mitigations: fixed seeds for shipped agents, and comparing several seeds before
  accepting a result.
- **Colab speed for in-class training is unknown** until the benchmark runs. Pretrained agents mean
  every activity also works without training.
- **Our [D] choices aren't the article's.** The notebook says which rules come from the article and which
  are ours.

## Git
Work happens on `maze-rl` and is pushed often. It merges to `main` after the final review.
