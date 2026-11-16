#!/bin/bash
# Train the Part 1 agents that ship with the package (TEACHING.md, steps 2-4), all seed 0.
#
#   sbatch scripts/part1.sbatch                                   # task 0: perception pretraining
#   sbatch --dependency=afterok:<id> --array=1-4 scripts/part1.sbatch   # the four frozen RL runs
#
# Submit from the repo root. Results: runs/part1/<name>.pt and .json; logs in logs/.
#SBATCH --job-name=mazerl-part1
#SBATCH --partition=gpus
#SBATCH --exclude=odin   # P100 unsupported by this torch build; /dev/nvidia-uvm missing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0
#SBATCH --output=logs/part1_%A_%a.out

set -euo pipefail
mkdir -p logs "${OUT:-runs/part1}"
source "${VENV:-$HOME/pytorch/.venv}/bin/activate"
OUT=${OUT:-runs/part1}
P=$OUT/perception.pt
COMMON="--seed 0 --episodes 20000 --freeze --device cuda --threads $SLURM_CPUS_PER_TASK --evaluate 500 --out $OUT"
case $SLURM_ARRAY_TASK_ID in
  0) python -m scripts.pretrain_perception --seed 0 --device cuda --threads "$SLURM_CPUS_PER_TASK" --out $P ;;
  1) python -m scripts.transfer --task flag   --start $P       --name flag_frozen_pretrained   $COMMON ;;
  2) python -m scripts.transfer --task flag   --start scratch  --name flag_frozen_random       $COMMON ;;
  3) python -m scripts.transfer --task escape --start $P       --name escape_frozen_pretrained $COMMON ;;
  4) python -m scripts.transfer --task escape --start scratch  --name escape_frozen_random     $COMMON ;;
esac
