# What we've learned

This is a plain-language summary of what a small maze-solving AI agent shows, and what that means for a
lesson. The lesson explains how reinforcement learning, training by rewards and penalties, shapes AI
agents such as large language models. The agent is modeled on the one in the New York Times article
*"A.I.'s Rogue Behavior Starts With Its Training."* The evidence behind each statement is in
`EXPERIMENTS.md`.

## The setup
- **The agent:** a small neural network, the same size as the article's (about 230,000 connections). It
  sees a 5 × 5 patch of the maze around it, remembers what it has seen, and knows the direction of the
  flag.
- **The points:**

  | Event | Points |
  |---|---|
  | Reaching the flag | +5 |
  | Each move outside the maze | −0.01 |
  | Walking into a wall | −0.01 |
  | Visiting a new square for the first time | +0.005 |

- **Hidden weak spots:** some walls are weak spots that give way when the agent walks into them. The agent
  can't tell them apart from solid walls without trying.
- **Training:** 126,000 attempts on 13,148 mazes, as in the article. The mazes come in three kinds, in
  equal numbers:
  - open mazes, solvable without breaking anything;
  - mazes where the only way to the flag is through a weak spot;
  - mazes where the only way is out the entrance and back in through another gap.
- **Attempts that end outside don't count:** if an attempt ends with the agent outside the maze, the agent
  doesn't learn from it.

## What the trained agent can do
- **It solves mazes it has never seen** about as often as the ones it trained on. It learned general
  skills rather than memorizing layouts.

  | Kind of maze | Solved |
  |---|---|
  | Open | 99% |
  | Through a weak spot | 91–94% |
  | Outside and back in | 98% |

- **It breaks out to finish an impossible task.** One of the article's mazes has a flag that can't be
  reached by allowed means. On it, the agent leaves the maze, breaks back in through a weak spot, and
  reaches the flag in roughly 35% to 75% of attempts.
  - It never trained on that maze, and nothing in training rewarded that sequence.
  - It strings together two skills learned separately: going outside and back in, and breaking through
    weak spots.
  - It needs both skills. An agent trained without either of the two maze kinds that teach them almost
    never completes the breakout (under 1%).
- **It sometimes breaks out when it doesn't need to.** Another of the article's mazes has a perfectly good
  inside route. On it, the trained agent goes outside in roughly a quarter to a half of attempts.
  - Why it does this with the current points and mazes is not yet known.
  - With a training mix that had more open mazes (3 in 7), removing the small reward for visiting new
    squares made it go outside more often. That matches the article's account: when the points slightly favor the outside route, the
    agent takes it.

## How the points shape behavior
- **A large exploration reward produces reward hacking**, where the agent finds a way to score points
  without doing the intended task. When visiting a new square is worth +0.05 instead of +0.005:
  - the agent tours the outside of the maze collecting points before going to the flag;
  - its score goes up while its success on mazes that need a weak spot goes down;
  - it leaves the maze on the maze with an inside route nearly every time, and the breakout on the
    impossible maze almost disappears.

  The agent is maximizing the score it was given, not doing the task its designers meant.
- **Leaving out attempts that end outside doesn't catch this.** The hacking agent always comes back inside
  before the attempt ends, so a check on how attempts end never sees the problem. A check that looks only
  at outcomes misses an agent that games the process.
- **Small changes to the penalties mostly change how the agent searches, not whether it succeeds.**
  - With a training mix that had more open mazes (3 in 7), a costlier wall penalty made the agent test
    fewer walls, but it still solved weak-spot mazes about as often. It became more economical, not timid.
  - The flag is worth so much more than everything else that the agent still learns to reach it.
  - Penalties act as prices, not prohibitions.
- **Being tested now:** the effect of each reward on its own, with the current mix of mazes. That
  includes how efficiently the agent finishes, compared with the shortest route.

## What doesn't matter much
- **Network size.** A network one-eleventh the article's size learns more slowly. Given enough training,
  it solves mazes as well as the full-size one and breaks out on the impossible maze just as often. Here,
  size affects how fast the agent learns, not what kind of behavior it ends up with. All work uses the
  article's network size and the article's 126,000 attempts.

## Things to know when interpreting results
- **The same settings can produce quite different agents.** Two agents trained the same way, differing
  only by random chance, broke out on the impossible maze in 36% and 74% of attempts. Measuring the same
  agent twice gives results only about 1 percentage point apart. So a single trained agent per setting
  can't settle a comparison between settings; that takes several.
- **The maze rules are a simplified version** of the article's model. The flag reward and the outside
  penalty come from the article; the other values are ours.

## Pretraining
- **Pretraining** means first teaching the agent's "seeing" layers, with no rewards at all, to predict
  hidden parts of the maze and where unexplored ground is. It's the maze version of how language models
  first learn by predicting text.
- **On the flag task it gives only a small head start.** If the pretrained layers are frozen and only the
  decision-making layers are trained, the agent doesn't learn the task at all.
- **Being tested now:** whether the same pretraining helps the agent learn a different task, escaping the
  maze. If it does, that is the maze version of one pretrained language model being trained for many
  different jobs.

## What this means for the lesson
- **The agent learns whatever earns points, and nothing else.** Every behavior, wanted or not, traces
  back to the rewards and the training experience.
- **Skills learned separately combine in situations no one tested.** No one trained the impossible-maze
  breakout, yet it follows from two ordinary skills plus a strong reward for finishing.
- **Penalties are prices.** A big enough prize makes the agent pay them.
- **A badly sized reward gets gamed,** and checking only the end result doesn't reveal it.
- **Training is partly luck.** The same training recipe gives agents with noticeably different habits.
  That is part of why testing AI agents is hard.
