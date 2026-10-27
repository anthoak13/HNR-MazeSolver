# Maze RL activity — build plan

An HNR-1303 (Unit 5) activity built on the NYT article *"A.I.'s Rogue Behavior Starts With Its Training"*
(Saha & Migliozzi, Oct. 26 2026; PDF in repo root). Students train and watch maze-solving agents and see
how reward values shape behavior, including the "go outside and break back in" behavior from the article.
Students don't touch the training code. They change reward values and a few training settings, then look
at plots and animations.

## Sources, in priority order
Every rule below is tagged with where it comes from:

- **[A] The article** — text and figures. This is the basis, and it wins any conflict.
- **[C] The author's comments** on the article:
  - "reaching the flag could fetch the agent +5 points, while stepping outside could result in a penalty
    of −0.01"
  - "it took less than an hour to train the model on my laptop. Once trained, running it takes less than a
    second to compute the solution paths."
- **[P] The paper the article cites**, *AI Safety Gridworlds* (Leike et al. 2017, arXiv:1711.09883). Used
  only to fill gaps the article leaves, for the parts of the method that are its conventions.
- **[D] Our design decision**, where none of the above says anything. Each [D] is a setting in the package,
  not hard-wired.

---

## 1. The world

### Layout (counted from the article's figures, pp. 1–7)
| Item | Rule | Tag |
|---|---|---|
| Arena | 18 × 18 grid of cells | [A] |
| Maze | Middle 14 × 14 block (rows/cols 2–15), with a 2-cell **outside** band all around | [A] |
| Walls | Walls fill whole cells. Wall thickness is 1–2 cells and corridor width is 1–2 cells | [A] |
| Start | Open top-left corner cell of the maze, right next to the outside band | [A] every figure |
| Flag | One cell in rows/cols 10–14 (lower right) | [A] true of all 9 figures |
| Openings | Besides the corner entrance, the outer wall has **3–5 plain gaps** to the outside | [A] counted in the 7 training figures |
| Weak spots | **2–6** wall cells per maze, 0–3 of them in the **outer boundary wall**, including in training mazes | [A] pale squares, counted in the figures |
| Layouts | A fixed bank of **13,148** generated layouts | [A] the number, [D] the generator |

**Maze generator [D], matched to the figures:** *recursive division*. Draw a wall across the open
interior with one doorway, then split each side again, and stop early so some chambers stay open rooms.
This reproduces the figures' long walls with single doorways enclosing open rooms. Then:
- open the top-left entrance (3 cells, as in every figure);
- cut 3–6 openings in the outer wall;
- turn 2–6 wall cells with open cells on two opposite sides into weak spots, so each one "opens into a
  passage" [A];
- place the flag at least 12 moves from the start.

The generator was checked by classifying every cell of the 7 training-figure images from the PDF.

**Three kinds of training maze**, from the article's 7 training figures (pp. 2–5, transcribed in
`mazerl/scenarios/training/`):

| Kind | Figures | How the flag is reached |
|---|---|---|
| open | p3, p4a, p4b | inside, without breaking anything |
| interior_break | p4c, p5a | inside, through an interior weak spot ("they *must* probe walls…") |
| outside_only | p5b, p5c | out the entrance and back in through an outer *opening*; no weak spot needed |

- The generator makes them in the same 3 : 2 : 2 mix (`maze_mix`).
- In every figure the flag sits in rows/cols 10–14 (the lower right), so the generator puts it there.
- No training figure *requires* breaking in from outside. But outer weak spots are allowed anywhere,
  including next to the flag's area, because the article's training animation shows an agent breaking
  in from outside.

Layouts are *stored* as ASCII art, the same convention as the paper's environments [P]. That covers the
two scenario mazes from pp. 6–7, transcribed from the PDF (§4). ASCII is for definitions and
debugging only (`print(maze)`, test failures). Students never see it. Everything they look at is the
animated article-style view (§4).

### Agent mechanics
| Item | Rule | Tag |
|---|---|---|
| Moves | Up / down / left / right | [A], also [P] |
| Walking into a wall (a **probe**) | One penalty per attempt. A weak spot gives way and the agent moves in; a solid wall doesn't, and the agent stays put. The arena edge behaves like a solid wall | [A] "each probe or move through a wall carries a small penalty," "stepped into a wall … penalized"; [D] a weak spot opens on the first bump |
| What the agent can tell apart | Weak spots look exactly like walls to the agent. Only the human viewer sees them as pale squares | [D] (the article says agents must *probe* to discover them) |
| Vision | 5 × 5 window centered on the agent. Walls block line of sight | [A] |
| Memory | Remembers every cell it has seen during the episode | [A] "remember past observations" |
| Flag direction | Compass direction to the flag, one of 8 (N, NE, …) | [A] "compass direction," [D] 8-way |
| Episode end | Agent reaches the flag, or a **200-step** cap | [P] uses 100 steps for grids ≤10×10; ours is 18×18, and at 100 steps every failure was a timeout (success 70% vs 91–94% at 200 in full-scale runs) [D] |
| Multiple agents | "One or more agents" share a maze, move independently and never block each other. Mostly used for visuals | [A] figures show several, [D] no interaction |

### Rewards (student-tunable)
| Event | Points | Fires when | Why this value |
|---|---|---|---|
| Reach the flag | **+5** | landing on the flag; the attempt ends | author's example [C]; "much larger" than the rest [A] |
| Outside | **−0.01** | every move that ends on an outside cell | author's example [C]. Charged per move because p.6 says "penalties," plural, for one trip out |
| Walk into a wall | **−0.01** | any move into a wall (a weak spot gives way, a solid wall doesn't) | "small penalty" [A]. The author gives only one small magnitude (0.01), so it's reused |
| New cell | **+0.005** | first visit to a cell this attempt, inside or outside | "small reward" [A]. It must be smaller than 0.01 so a step outside is still a net penalty ("stepped … outside … penalized," p.3). Half is the simplest value below that |
| Per move | 0 | — | none mentioned [A] |

Checks the values pass:
- **Outside always costs.** Every move outside nets at least −0.005, so wandering the outside band never
  pays.
- **Exploring can't compete with the flag.** Visiting the whole maze interior (about 100 cells) earns at
  most +0.5. Even 200 moves away, the discounted flag is worth 0.68.
- **The flag outweighs the route's penalties (p.6).**
  - Enclosed maze: the shortest route (30 moves: 8 outside, 4 weak spots) costs −0.12 in penalties and
    nets +3.76 after discounting.
  - Outside-shortcut maze: the shortest route (25 moves, 6 outside) nets +3.98.
  - An agent would need about 370 extra wall bumps before the route stopped paying.
- **Open choice, my decision:** new cells outside also earn the +0.005. One rule is simpler, and outside
  still nets negative.

### "Training excluded scenarios where the agent bypassed the maze entirely" [A]
Readings ruled out by the source:
- **Came back in somewhere else**, or **reached the flag via the outside:** ruled out, because two
  training mazes (p5b, p5c) can *only* be solved that way.
- **Broke in from outside:** ruled out, because the article's training animation shows it happening.
- **Layout-level readings** (mazes with an outside route were excluded): ruled out, because "where the
  agent bypassed" describes something the agent did, and p5b/p5c are outside-only training mazes.

Remaining readings, both available as `bypass_rule`:
- `"left"` (default): the agent left the maze and never came back in, so the attempt ends outside.
- `"never_entered"`: the agent never got more than one cell into the maze.

Bypass attempts are dropped from the training batch. They still count toward the 126,000 attempts, and
they're logged. `exclude_bypass=False` turns the filter off (classroom experiment).

---

## 2. The agent

### Observation [D]
An 11 × 11 window of the agent's **memory**, centered on the agent, with one channel per cell type: unseen,
open-inside, outside, wall, flag, visited-by-me, and visible-right-now. Plus the 8-way compass one-hot.
Centering the window on the agent means it doesn't know its absolute position, so what it learns has to
transfer across layouts.

### Network — "vision" + "reasoning" [A], ≈230,000 connections [A]
- **Vision** [A] says it exists, [D] its design: conv 3×3 (7→32) → conv 3×3 (32→64) → 2×2 max-pool →
  conv 3×3 (64→64) → 1,600 features.
- **Reasoning:** concat compass → **two hidden layers of 100 nodes** [P] → move probabilities (4) and
  value estimate (1).
- Measured parameter count of this shape: ≈221–225k. Final channel counts get adjusted to land at ≈230k.

### Learning algorithm: A2C (advantage actor-critic) [P]
The article doesn't name an algorithm. The cited paper's baselines are A2C and Rainbow. A2C is the simpler
of the two and the cheaper per step: one gradient pass per step, against three or more for PPO. That
cheapness is what makes the author's "under an hour on a laptop" realistic (§3).

Settings, with their sources:
- discount 0.99 [P]
- value-loss weight 0.25 [P]
- gradient clip 40 [P]
- rewards scaled to [−1, 1] by dividing by the largest one [P]
- Adam, learning rate 1e-3, annealed to 0 [D]. In full-scale tests this beat 5e-4: 94% vs 91% success.
- entropy bonus 0.05, annealed to 0.01 [P range 0.01–0.1]
- 256 parallel attempts, one update per 2,048 pooled moves [D]
- the 200-move cap is treated as the end of the attempt [D]

Adaptation for the bypass rule [D]:
- Each environment buffers its current attempt (≤200 moves). When the attempt ends, it either goes into
  the training pool or is dropped as a bypass.
- A pooled attempt may have started a few updates earlier. The policy loss therefore uses PPO's clipped
  probability ratio to stay stable.
- It's still one gradient pass per move, the same cost as A2C.

- **Student-facing knobs:** reward values, `learning_rate`, `discount` (how far ahead the agent "cares"),
  `exploration` (entropy bonus), `episodes`, `exclude_bypass`, `maze_mix`.
- **Everything else is hidden.**

### Engine [D]: our own batched environment, not pycolab
The paper's environments (`google-deepmind/ai-safety-gridworlds`) run on `pycolab`. We tested it, and
neither is usable as a foundation:

- `pycolab` (last commit 2019) crashes on numpy 2.x, which Colab ships. We tested it: `ascii_art_to_game`
  fails in `np.vstack`.
- `ai-safety-gridworlds` is archived (2022) and pins numpy 1.14.
- Both step one game at a time in Python, which is too slow for 126k attempts.

Instead, the environment is batched PyTorch tensor operations: hundreds of mazes step at once, on CPU or
GPU. From pycolab we keep the ASCII-art way of writing mazes.

---

## 3. How long will training take?

**Baseline [C]:** the author trained the full model in **under an hour on a laptop**, and a trained agent
runs in under a second. Our target is to match that: a full article-scale run in **≤1 hour on an ordinary
laptop CPU, with no GPU required**.

### Work in an article-scale run
- **Steps.** 126,000 attempts × at most 100 steps [P cap] gives ≤12.6 M steps. Early attempts hit the cap.
  Trained attempts are much shorter: the shortest paths in the article's figures are ~25 moves. So the
  realistic total is **≈5–9 M steps**.
- **Compute per step (A2C).** One forward pass while playing (≈7 MFLOP) plus one training pass (≈21 MFLOP)
  ≈ 28 MFLOP. A full run is therefore **≈0.15–0.25 PFLOP**.
- **What the author's figure implies.** "Under an hour" means ≥1.5–2.5k steps/s, which takes ≈40–70
  GFLOP/s. A modern laptop CPU can sustain that for a network this size, so the baseline and this design
  are consistent.

**Measured:** a full 126k-attempt run (200-move cap) took **≈55 min on 8 CPU threads** of the desktop
(i5-13400F). Pretraining runs on your campus GPU cluster via SLURM (`scripts/pretrain.sbatch`). I don't
have access to it, so you'll run it there.

### By platform (estimates, until Milestone 2 replaces them with measurements)
| Platform | Full 126k-attempt run | 5-minute class run |
|---|---|---|
| Author's laptop [C] | < 1 h | — |
| Colab **CPU** (2 vCPU, slower cores than a current laptop) | ≈1–3 h | ≈5–10k attempts: enough to watch early learning, not to finish |
| Colab **T4 GPU** | ≈10–20 min | ≈30–60k attempts: enough to retrain the base agent with new rewards |

The Colab numbers are scaled estimates, not measurements. **Milestone 2** is a 2-minute benchmark notebook
that you run once on a Colab CPU runtime and once on a T4. I'll put the measured numbers here and set the
class time budgets from them.

### What this means for class
- **Recommended:** a T4 runtime, as 05-02 already uses. Every activity also works on CPU through the
  pretrained agents.
- Students don't train from scratch at full scale in class. They:
  1. **Retrain from the pretrained base agent** with their own rewards for a few minutes, to watch behavior
     shift. Example: the agent learns to lap the outside when new-cell reward > outside penalty and
     `exclude_bypass` is off.
  2. **Train from scratch for a few minutes**, to watch the start of the learning curve.
  3. **Load precomputed agents** from a reward sweep, so nobody is stuck waiting.
- Pretraining is done once, on the campus SLURM GPU cluster (one array task per agent).

### Pretrained artifacts (shipped in the package, ~1 MB each)
- `base` — the article's setup: default rewards, 126k attempts, 13,148 layouts, bypass exclusion on.
  Saved partway through at ≈1k, 5k, 20k, 60k and 126k attempts, to reproduce the article's "agents at
  different stages of training" grid.
- **Reward variants**, each differing from `base` in one way (`scripts/pretrain.py --list`):
  - `outside_free`: outside 0
  - `outside_costly`: outside −0.1
  - `explore_bonus_big`: new cell +0.1
  - `explore_bonus_none`: new cell 0
  - `probing_costly`: wall −0.1
  - `no_bypass_filter`: bypass attempts kept in training

---

## 4. What students see (mirrors the article's figures)
Every maze view is an **animation in the NYT style**, not a static plot or text.

**Look**, copied from the figures:
- Faint dotted grid on white, light-gray wall blocks, and pale squares for weak spots (the viewer sees
  them, the agent doesn't).
- Pink tint on the outside band.
- Hollow coral circle for the start, solid red dot for the agent, and a fading red trail behind it.
- Black round flag marker.
- Sparse sans-serif labels (AGENT, GOAL, WALL, OUTSIDE).

**Motion:**
- The agent glides smoothly between cells rather than jumping frame to frame.
- A pulsing halo plus a short callout ("Penalty for stepping outside", "Probe −0.01", "Broke through
  wall") appears when a reward or penalty fires, as on p.3.
- A broken weak spot visibly opens.
- Several agents can move in the same maze at once (pp. 4–5).
- Play / pause / scrub controls and a running score.

**How it's built:**
- The package records each episode as data (positions, events, rewards).
- A small self-contained HTML/SVG/JavaScript player draws it inline in the notebook via
  `IPython.display.HTML`.
- This works in Colab and Jupyter with no external libraries, and SVG stays crisp at any size. matplotlib
  frame-by-frame animation was the alternative; it looks choppy and is heavy.
- The player can also save an episode as a standalone `.html` file, so it can be shared or shown on a
  projector.

**matplotlib is used only for the training curves** (`run.plot()`), which match the 05-01/05-02 plots
students already know.

| Function | Shows | Article figure |
|---|---|---|
| `mz.show_maze(maze)` | One layout | p.1–2 "Agent tasked with finding a flag" |
| `mz.playback(agent, maze)` | Animated attempt with a running score and pop-up labels for penalties/rewards ("Penalty for stepping outside") | p.3 "Playback of a training run" |
| `mz.training_grid(run)` | Several layouts with agents at several training stages | p.4–5 "Agents training on different maze layouts" |
| `mz.scenario("enclosed")` | Flag fully walled in (layout transcribed from p.6) | p.6 "Agent facing a hard problem" |
| `mz.scenario("outside_shortcut")` | An inside path exists, but the outside route is shorter (layout transcribed from p.7) | p.7 "Agent taking outside path…" |
| `mz.agent_view(agent, maze, step)` | What the agent knows: memory, current 5×5 view, compass | Methodology |
| `run.plot()` | Curves over training: average total reward, % reaching flag, steps per attempt, probes, breaks, time outside, % bypass | not in the article. Mirrors the 05-01/05-02 loss/accuracy curves and the paper's "episode return" plots [P] |
| `mz.reward_breakdown(agent, maze)` | Bar chart of where an attempt's points came from | "reward … outweighed the penalties" (p.6) |
| `mz.evaluate(agent, n=500)` | Table of results on *unseen* layouts | training vs. validation, as in 05-02 |

### Student-facing API sketch
```python
!pip install -q git+https://github.com/anthoak13/HNR-MazeSolver@maze-rl
import mazerl as mz

rewards = mz.Rewards(flag=5, outside=-0.01, probe=-0.01, new_cell=0.005)
base = mz.load_pretrained("base")
run = mz.train(rewards, start_from=base, minutes=5)   # or episodes=…
run.plot()
mz.playback(run.agent, mz.scenario("enclosed"))
mz.evaluate(run.agent, n=500)
```

---

### The article's two test mazes (transcribed)
Both layouts were read cell by cell from the embedded figure images. Every cell's shade falls into a few
clean values, so no cell was ambiguous. They live in `mazerl/scenarios/*.txt`. Two details in the figures
don't match the article's text exactly:

- **p.6 "enclosed":**
  - The start pocket opens *only* to the outside, so the agent must leave the maze first.
  - The flag's room has one weak spot, which isn't directly reachable from outside. The shortest route
    goes outside, then through 4 weak spots.
- **p.7 "outside_shortcut":**
  - The inside path takes 27 moves and the outside route 25.
  - The outside route uses plain *openings*, not a weak spot. The text says the agent exploited "a
    vulnerability in the wall."

We use the figures as drawn.

## 5. Repo layout
```
mazerl/
  __init__.py        public API only
  maze.py            generator, layout bank, ASCII scenarios
  env.py             batched tensor environment (CPU or GPU)
  agent.py           vision + reasoning network
  a2c.py             A2C training, bypass filter, logging
  viz.py             episode recording, training curves (matplotlib)
  player/            self-contained HTML/SVG/JS animation player (NYT style)
  pretrained/        *.pt checkpoints + metadata (rewards, settings, attempt counts)
notebooks/
  00-benchmark.ipynb         Colab speed check (Milestone 2)
  01-maze-rl-in-class.ipynb  student notebook, course style (bold questions)
  01-maze-rl-solution.ipynb  instructor key / notes
scripts/pretrain.py          full-scale runs + sweep
tests/                       pytest
pyproject.toml               pip-installable from GitHub
```

---

## 6. Milestones (I build and debug; you review at each ✔)
1. **World + player:** generator, environment mechanics, and the NYT-style animated player (random agents to start). Tests for: probe/break
   rules, outside penalty, line-of-sight occlusion, compass, bypass detection, every maze solvable.
   ✔ You compare rendered mazes with the PDF.
2. **Benchmark notebook.** ✔ You run it on Colab CPU and on a T4. I update §3 with real numbers and set
   the class time budgets.
3. **A2C training + curves.** Short runs to tune the [D] reward values until the base agent solves unseen
   layouts reliably. Final values recorded in §1.
4. **Full pretraining** (`base` + stage checkpoints). Measure the real run time against the ≤1 h laptop
   target.
5. **Article scenarios.** Evaluate `base` on the pp. 6–7 layouts. ✔ I report honestly whether the
   emergent behaviors show up and how often (over many attempts). If they don't, adjust the *training
   distribution or rewards*, never script the behavior, and document what changed.
6. **Reward sweep** pretrained agents.
7. **Student notebook + instructor key**, in the 05-0x style.
8. **End-to-end Colab test** from a fresh runtime. ✔ Final review.

---

## 7. Risks
- **The emergent "break back in" may not reproduce.** Nothing in training rewards that exact sequence;
  the article's point is that it emerges. Emergence can fail for concrete reasons:
  1. **It's three skills chained together:** leave the maze, travel outside, and break a wall *to get back
     in*. Each piece is practiced in training: the start borders the outside, outer weak spots exist, and
     `locked_frac` mazes require breaking interior walls. Only the combination is new at test time, which
     is the article's point. The risk is that the combination doesn't transfer, not that a piece is missing.
  2. **The view from outside looks different.** In training, the agent probes walls from inside the maze.
     Seen from outside, the same wall sits next to outside cells, a combination it has rarely seen. The
     cited paper's lava-world experiment [P] shows agents like this often fail under exactly this kind of
     shift.
  3. **Reward arithmetic.** The outside route has to fit inside the 100-step cap. And after discounting,
     the +5 flag has to outweigh the outside and wall penalties along that route ("the
     reward for completing the task outweighed the penalties," p.6). If the [D] penalty values are too
     harsh, the agent never commits to the long outside route.
  4. **Giving up.** In the enclosed-flag case the agent may keep probing the enclosure from inside, or
     wander once it runs out of new cells. Persistence comes from the new-cell reward, so its value matters.
  5. **Training luck.** A2C can collapse into a near-deterministic policy that never tries leaving. One
     unlucky seed can lose the behavior.
  6. **We don't know how often it happened for the NYT.** The article shows one replay. It may have been
     occasional for them as well.

  Milestone 5 measures it. It's reported with a frequency, for example "in 37% of attempts," not
  cherry-picked. If it's rare or absent, the legitimate levers are the reward values (2–4) and how often
  training mazes have reachable outer weak spots and require wall breaking (1–2). We never script the behavior. A model that only
  sometimes breaks in is also a good discussion point.
- **A2C can be unstable.** The paper reports occasional "entropy collapse" and drops its worst 25% of
  seeds [P]. Mitigations: the entropy floor from [P], fixed seeds for the pretrained agents, and a
  `run.plot()` that makes collapse visible, which is itself a teaching point. If A2C won't train reliably,
  the fallback is PPO, which costs ≈2–3× more compute per step.
- **Colab T4 availability and quotas.** The pretrained agents mean every activity also works on CPU.
- **Our [D] choices aren't the article's.** The student notebook says plainly which rules come from the
  article and which are ours.

## Git
All work goes on the `maze-rl` branch, pushed to `origin` after each milestone (and more often mid-milestone)
so you can switch machines. Merge to `main` after the final review.
