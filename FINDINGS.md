# What we've learned

What a small maze-solving RL agent, modeled on the one in the New York Times article *"A.I.'s Rogue
Behavior Starts With Its Training,"* shows, and what that means for a lesson on how RL shapes agents such
as LLMs. The evidence behind each statement is in `EXPERIMENTS.md`.

## The setup
- **The agent:** a CNN plus fully connected layers, the article's size (about 230,000 parameters). It sees
  a 5 × 5 window around itself, remembers everything it has seen during the attempt, and gets an 8-way
  compass direction to the flag.
- **Rewards:**

  | Event | Reward |
  |---|---|
  | Reaching the flag (ends the attempt) | +5 |
  | Each move that ends outside the maze | −0.01 |
  | Walking into a wall | −0.01 |
  | Visiting a square for the first time | +0.005 |

- **Weak spots:** some walls give way when the agent walks into them. They look like solid walls, so the
  agent can only find them by trying.
- **Training:** actor-critic, 126,000 attempts over 13,148 layouts, as in the article. Three kinds of
  layout, in equal numbers:
  - open mazes, solvable without breaking anything;
  - mazes where the only route to the flag goes through a weak spot;
  - mazes where the only route goes out the entrance and back in through another gap.
- **Attempts that end outside the maze are excluded from training,** following the article.

## What the trained agent does
- **It generalizes.** Success on new layouts matches success on training layouts:

  | Kind of maze | Solved |
  |---|---|
  | Open | 99% |
  | Through a weak spot | 91–94% |
  | Outside and back in | 98% |

- **On the article's p.6 maze, where the flag is walled off from the start, it breaks out to finish.** It
  leaves the maze, breaks back in through weak spots, and reaches the flag in 35–75% of attempts.
  - No training layout requires that sequence. It combines two separately learned skills: leaving and
    re-entering the maze, and breaking through weak spots.
  - Both skills are necessary. Training without the layouts that teach either one drops the success rate
    on this maze below 1%.
- **On the article's p.7 maze, which has an inside route, it still leaves the maze in a quarter to a half
  of attempts.**
  - What drives this under the current settings is not yet known.
  - With a training mix weighted toward open mazes (3 in 7), removing the reward for new squares raised
    the rate. That matches the article's account: when the rewards slightly favor the outside route, the
    agent takes it.

## How the rewards shape behavior
- **A large exploration reward produces reward hacking.** At +0.05 per new square (10× the default):
  - the agent tours the outside of the maze collecting reward before heading to the flag;
  - its return goes up while its success on weak-spot mazes goes down;
  - it leaves the p.7 maze nearly every time, and almost never reaches the p.6 flag.
- **Excluding attempts that end outside doesn't catch it.** The hacking agent always returns inside before
  the attempt ends. A check on outcomes misses an agent that games the process.
- **Moderate penalty changes alter how the agent searches more than whether it succeeds.**
  - With a training mix weighted toward open mazes (3 in 7), raising the wall penalty to −0.05 cut wall
    bumps by a third, but success on weak-spot mazes barely changed.
  - The flag reward dominates every other term, so the agent still learns to reach it. Penalties act as
    prices, not prohibitions.
- **Being tested now:** each reward varied on its own under the current equal mix, including how many
  moves the agent takes compared with the shortest route.

## What doesn't matter much
- **Network size.** A network with one-eleventh the parameters learns more slowly. Given enough training
  it matches the full-size network, both on ordinary mazes and on the p.6 maze. Size changes learning
  speed, not the behavior learned. All work uses the article's size and its 126,000 attempts.

## Variability
- **Identical settings give quite different agents.** Two seeds broke out on the p.6 maze in 36% and 74%
  of attempts. Re-measuring one agent varies by about 1 percentage point. Comparisons between settings
  need several seeds.
- **The rules are a simplification** of the article's model. The flag reward and the outside penalty come
  from the article's author; the other values are ours.

## Pretraining
- **Reward-free pretraining** of the vision layers, predicting hidden parts of the maze and the direction
  to unexplored ground, is the maze analog of next-token prediction.
- **On the flag task it gives only a small head start.** With the pretrained layers frozen, the agent
  doesn't learn the task.
- **Being tested now:** whether the same pretraining speeds up a different task, escaping the maze. If it
  does, that is the maze analog of one pretrained LLM fine-tuned for many jobs.

## What this means for the lesson
- **The agent learns whatever the rewards pay for.** Every behavior, wanted or not, traces back to the
  rewards and the training experience.
- **Separately learned skills combine in situations no one tested.** No one trained the p.6 behavior; it
  follows from two ordinary skills plus a large reward for finishing.
- **Penalties are prices.** A big enough reward makes the agent pay them.
- **A mis-sized reward gets gamed,** and checking only outcomes doesn't reveal it.
- **Training outcomes vary.** The same recipe yields agents with noticeably different habits, which is
  part of why testing agents is hard.
