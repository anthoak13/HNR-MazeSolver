# Teaching guide

How the maze activity is meant to work in class. Students will have read the NYT article *"A.I.'s Rogue
Behavior Starts With Its Training"* beforehand. The evidence behind each claim is in `EXPERIMENTS.md`, and
the build details are in `PLAN.md`.

## Goals
1. **How these agents are built: pretraining, then RL.** One model learns to "see" mazes with no rewards
   at all. Then RL turns it into an agent for a task, and possibly for different tasks from the same
   pretrained model. This is the recipe behind LLM agents. *Whether RL for several tasks from one
   pretrained model works well enough for class is still untested* (`PLAN.md` §10). If it doesn't, this
   part shrinks to a short explanation.
2. **Rogue behavior comes from training** (the article). Students reproduce the article's two examples
   and see how rewards and training experience cause them.

## The article's two examples

**1. The p.6 maze: it breaks out to reach a walled-off flag** (p.6, "Agent facing a hard problem")
- The flag can't be reached by allowed means. The agent leaves the maze and breaks back in through a
  weak spot.
- Nobody trained that sequence; it combines skills learned separately. The reward for finishing
  outweighed the penalties.
- **Real incident:** Hugging Face.

**2. The p.7 maze: it leaves when it doesn't need to** (p.7, "Agent taking outside path although inside
path is available")
- A legitimate route exists, and the agent leaves anyway. Once breaking out is something it knows how to
  do, it uses it whenever the rewards slightly favor it.
- **Real incident:** RubyGems.

## Path through the activity

**Part 1: building an agent**
1. **The article's model, in code.** What the agent sees (a 5×5 view, memory, a compass), the points for
   each event, and an untrained agent moving at random.
2. **Pretraining.** The model learns to predict unseen parts of the maze and where unexplored ground is,
   with no rewards. This is the analog of next-word prediction.
3. **RL on top.** Train the pretrained model to reach the flag. Watch its reward curves (like the cost
   curves from 05-01) and the agent at different stages of training. Compare it with an agent trained
   from scratch.
4. **A second task from the same pretrained model** (if it works), for example escaping the maze. Rewards
   decide what the model is *for*; pretraining decides what it *understands*.

**Part 2: rogue behavior from training**
5. **The p.6 maze.** Predict, then watch the trained agent on the maze whose flag is walled off. It
   leaves, breaks back in and reaches the flag in a large share of attempts, though not every time.
6. **Why it can.** Agents trained without mazes that require going outside, or without mazes that require
   breaking walls, never break out. Students can train one of these live and see the missing skill.
7. **The p.7 maze and reward design.** Raise the exploration reward and train live. The agent tours the
   outside farming points, and on the p.7 maze it leaves every time. That's reward hacking.
8. **Discussion.** Our filter that drops attempts ending outside didn't catch the hacking: monitoring
   outcomes misses an agent gaming the process. Also responsibility, and sandbox design. Optional: smaller
   models need more training to learn the same skills; once trained, they behave the same.

## What to expect
- **Hardware:** a Colab T4. A live training run takes about 2–5 minutes.
- **Reaching the p.6 flag needs a fully trained agent;** it rarely happens after a short live run. Use the shipped
  agents for it.
- **Results vary between training runs:** the same settings give an agent that reaches the p.6 flag
  35% of the time, or 75%. That's worth telling students: training is not deterministic.
- **The maze rules are our simplified version** of the article's model, and the notebook says which parts
  are ours.

## Deliverables to package

**Trained weights** (shipped in `mazerl/pretrained/`, trained on the cluster):

| Step | Weights | Status |
|---|---|---|
| 2 | pretrained perception (no rewards) | method built; final weights to train |
| 3 | RL on pretrained perception, flag task, **with snapshots at several stages of training** | to train |
| 3 | RL from scratch, flag task, the same budget, for comparison | to train |
| 4 | RL on pretrained perception, second task, with stage snapshots | the task is still to design and test |
| 5, 7 | `base`: full-scale flag agent with stage snapshots, a typical seed (not the best) | to train |
| 6 | `base` trained without the mazes whose only route goes outside | to train |
| 6 | `base` trained without the mazes whose only route goes through an interior weak spot | to train |
| 7 | `base` trained with a high exploration reward (backup if live training isn't possible) | to train |
| 8 (optional) | small and large versions of `base` | to train |

**Maze files:**

| Item | Status |
|---|---|
| the article's two test mazes (pp. 6–7) | done |
| the 13,148 training layouts | done |

**Code students call:**

| Function | Status |
|---|---|
| animated playback of one or several agents | done |
| agents from several training stages side by side | done |
| evaluation on new mazes or the article's mazes | done |
| on the p.6 maze, the share of attempts that left, broke back in, and reached the flag (`breakout_stages()`) | done; needs to be exposed in the API |
| reward curves | done |
| live training with progress bars | done |
| a warning when no GPU is available | to build |
| what the agent sees (view, memory, compass) | to build |
| where an attempt's points came from | to build |
| the second task's environment | to build if Part 1 step 4 stays |

**Notebooks:**

| Notebook | Status |
|---|---|
| student notebook following this path, with prediction questions in the 05-0x style | to write |
| instructor key | to write |
| Colab speed check | done |
