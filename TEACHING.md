# Teaching guide

What the maze activity is for, and the path through it that the experiments support. This is a focus
document for the work; the user designs the student notebook. Students will have read the NYT article
*"A.I.'s Rogue Behavior Starts With Its Training"* beforehand. What has been learned is in `FINDINGS.md`,
the evidence in `EXPERIMENTS.md` and `reports/pretraining-transfer.md`, and the build details in `PLAN.md`.

## Goals
1. **General first, then specialized.** Pretraining with no rewards teaches the network general knowledge
   of mazes: what hidden squares probably hold, and where unexplored ground is. RL then specializes that
   network into an agent for one task, and a different reward specializes it into a different agent. This
   is the recipe behind LLMs.
2. **Rogue behavior comes from the rewards** (the article). Students change rewards and see how they, and
   the training experience, produce the article's two behaviors and reward hacking.

## The article's two examples

**1. The p.6 maze: it breaks out to reach a walled-off flag** (p.6, "Agent facing a hard problem")
- The flag can't be reached by allowed means. The agent leaves the maze and breaks back in through a
  weak spot.
- Nobody trained that sequence; it combines skills learned separately. The reward for finishing
  outweighed the penalties.
- **Real incident:** Hugging Face.

**2. The p.7 maze: it leaves when it doesn't need to** (p.7, "Agent taking outside path although inside
path is available")
- A legitimate route exists, and the agent leaves anyway, whenever the rewards slightly favor the outside
  route.
- **Real incident:** RubyGems.

## Path through the activity

**Part 1: building an agent**

Pretraining doubles as the introduction to the model: explaining what the network is trained to guess
explains what it sees.

1. **What the agent sees.** Each move, the network gets an 11×11 window of its memory centered on itself:
   every square marked as not seen yet, open floor, outside, wall, flag or arena edge, plus which squares
   it has stood on, which it can see right now (a 5×5 view; walls block sight), and which walls it walked
   into that held firm. It also gets an 8-way compass pointing to the flag. Then the points for each
   event, and an untrained agent moving at random.
2. **Pretraining: learning about mazes with no rewards.** The analog of next-token prediction.
   - **The data.** Rule-following walkers (simple programs, not networks) explore the 13,148 training
     mazes in the same environment: 70% of the time they step toward the nearest open square they've seen
     but not stood on, 30% of the time they move at random. After every move, the agent's input is
     recorded: about 150,000 snapshots. The walkers ignore the flag and earn nothing (figure:
     `reports/figures/pretraining/walker-path.png`).
   - **The exercise.** For each snapshot, the network guesses (a) what's really in every square of the
     window it hasn't seen yet: open, outside, wall or edge, with weak spots counting as wall so it never
     learns where they are; and (b) which move heads toward the nearest open square it hasn't stood on.
     To fill in hidden squares it has to learn how these mazes are built: walls in long lines with single
     gaps, an outside band beyond the outer wall, the edge beyond that (figure:
     `reports/figures/pretraining/snapshots.png`).
   - **What it learns.** On mazes it never saw, it guesses hidden squares right 86% of the time, against
     53–55% for always guessing the most common kind, and the direction 99% of the time. Both overstate it
     somewhat: the edge and the outside band are easy guesses, and the direction can mostly be worked out
     from the input.
   - **The compass during pretraining** still points to each maze's flag, which the walkers ignore.
     Neither guess depends on it.
   - **What the pretraining images must do for the story:**
     1. Connect to what students already know: recognizably the article's animated maze world, so
        pretraining is anchored in something familiar rather than a new diagram to decode.
     2. Put the reader in the agent's position: what it knows at one moment is limited, partial and
        local, and the reader should feel the gap between that and the whole maze.
     3. Let the reader try the task: looking at the partial view, the reader should find themselves
        guessing what's hidden before seeing the answer. If they can guess, they believe a network can
        learn to.
     4. Make the absence of a goal noticeable: nothing is chased or rewarded; it should feel like
        observation, not a game.
3. **Specializing it for the flag.** Keep the pretrained layers fixed and let RL train only the top layers
   (about 10k of 230k parameters). The agent learns to reach the flag: 62–75% on open mazes. Do the same on
   top of random layers and it stays at chance (about 9%). The general knowledge is in the pretrained
   layers; RL only teaches the agent what to want. The specialization is partial: on weak-spot and
   outside-route mazes the frozen agent reaches only 22–54%, while an agent whose whole network trains
   gets above 90%. The harder skills need the lower layers to keep learning.
4. **A second specialization from the same base: escaping the maze.** The same fixed pretrained layers, a
   different reward (start where the flag would be, no compass, the first step outside wins): 97% on open
   mazes. Same knowledge, different reward, different agent.
   - **Not to claim:** that pretraining makes RL faster (the gain is small and inconsistent), or that
     reward-free pretraining is the only route to general knowledge. Layers trained by RL on the flag task
     also carried over to escape, which can be a discussion point: specialized skills transfer to related
     tasks too.

**Part 2: rogue behavior from the rewards**
5. **The p.6 maze.** Predict, then watch the trained agent on the maze whose flag is walled off. It
   leaves, breaks back in and reaches the flag in a large share of attempts, though not every time.
6. **The outside penalty and the p.7 maze.** The cost of a step outside sets how often the agent leaves
   when it doesn't need to: 0–6% of attempts at −0.05, 28–51% at the default −0.01, up to 99% when free.
   That's the article's "when the rewards slightly favor it," shown directly.
7. **Reward hacking.** Wherever a step outside nets positive reward, the agent tours the outside farming
   points before heading to the flag.
   - A large exploration reward (+0.05) does it. So does the small default one (+0.005) once outside moves
     are free, about as much as a reward ten times larger. Whether there's a net-positive loop matters, not
     the size of the reward.
   - The farming agent's score goes up while it takes about 1.5× as long to finish weak-spot mazes, and it
     almost never reaches the p.6 flag.
   - Success on ordinary mazes barely changes under any reward tested. The rewards shape the route more
     than the result.
8. **Discussion.**
   - Our rule that excludes attempts ending outside didn't catch the hacking: checking outcomes misses an
     agent gaming the process.
   - The same rule starved some pretrained agents of training data: a safety rule can have side effects
     on learning.
   - Also responsibility, and sandbox design.

## What to expect
- **Hardware:** a Colab T4. A live training run takes about 2–5 minutes.
- **Reaching the p.6 flag needs a fully trained agent;** it rarely happens after a short live run. Use the
  shipped agents for it.
- **Results vary between training runs:** the same settings give an agent that reaches the p.6 flag 35% of
  the time, or 75%. Whether an agent discovers a loophole is partly luck too: in each farming setting, one
  seed in three never found it. That's worth telling students: training is not deterministic.
- **The maze rules are our simplified version** of the article's model, and the notebook says which parts
  are ours.

## Deliverables to package

**Trained weights** (shipped in `mazerl/pretrained/`):

| Step | Weights | Status |
|---|---|---|
| 2 | pretrained perception (no rewards) | method built; final weights to train |
| 3 | flag agent on fixed pretrained layers, with snapshots at several stages of training | to train |
| 3 | flag agent on fixed random layers, the same length, for comparison | to train |
| 4 | escape agent on the same fixed pretrained layers, with stage snapshots | to train |
| 5 | `base`: full-scale flag agent with stage snapshots, a typical seed (not the best) | to train |
| 6 | `base` with outside moves free, and with outside moves at −0.05 | to train |
| 7 | `base` with a high exploration reward (backup if live training isn't possible) | to train |

**Maze files:**

| Item | Status |
|---|---|
| the article's two test mazes (pp. 6–7) | done |
| the 13,148 training layouts | done |

**Code students call:**

| Function | Status |
|---|---|
| animated playback of one or several agents | done; can't show escape attempts yet |
| agents from several training stages side by side | done |
| evaluation on new mazes or the article's mazes | done |
| on the p.6 maze, the share of attempts that left, broke back in, and reached the flag (`breakout_stages()`) | done; needs to be exposed in the API |
| reward curves | done |
| live training with progress bars | done |
| the escape task's environment | done (`MazeEnv(task="escape")`, `Settings.task`) |
| a warning when no GPU is available | to build |
| what the agent sees (view, memory, compass) | to build |
| where an attempt's points came from | to build |

**Notebooks:**

| Notebook | Status |
|---|---|
| student notebook, with prediction questions in the 05-0x style | the user writes it |
| instructor key | the user writes it |
| Colab speed check | done |
