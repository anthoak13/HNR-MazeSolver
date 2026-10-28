#!/bin/bash
# Run a sweep of training experiments, one GPU each, at most 2 at a time.
#
#   sbatch --array=1-$(wc -l < sweeps/NAME.txt)%2 scripts/sweep.sbatch sweeps/NAME.txt
#
# Each line of the sweep file is: <run name> [pretrain.py options...]
# Results: runs/<sweep>/<run name>.pt and .json (standard test suite); logs in logs/.
#
#SBATCH --job-name=mazerl-sweep
#SBATCH --partition=gpus
#SBATCH --exclude=odin   # P100 unsupported by this torch build; /dev/nvidia-uvm missing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_%A_%a.out

set -euo pipefail
source "${VENV:-$HOME/pytorch/.venv}/bin/activate"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
read -r NAME ARGS <<< "$LINE"
OUT=runs/$(basename "$1" .txt)
echo "task=$SLURM_ARRAY_TASK_ID run=$NAME args=${ARGS:-} host=$(hostname)"
python scripts/pretrain.py base --device cuda --threads "$SLURM_CPUS_PER_TASK" --out "$OUT" \
    --name "$NAME" --evaluate 500 ${ARGS:-}
