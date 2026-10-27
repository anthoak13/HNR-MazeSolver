#!/bin/bash
# Pretrain every shipped agent as a SLURM array job (one GPU per run).
#
#   sbatch scripts/pretrain.sbatch                       # all runs, full scale
#   sbatch --array=0 scripts/pretrain.sbatch             # just "base"
#   EPISODES=2000 sbatch --array=0 scripts/pretrain.sbatch   # smoke test first
#
# Submit from the repo root. Results land in mazerl/pretrained/<name>.pt; logs in logs/.
# Array index -> run name follows `python scripts/pretrain.py --list` (0 = base).
#
#SBATCH --job-name=mazerl-pretrain
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%a.out
# ---- cluster-specific: fill in or delete --------------------------------------------------
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
# -------------------------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Environment: a venv with this repo installed (pip install -e .). Adjust to your cluster,
# e.g. `module load python cuda` before activating.
source .venv/bin/activate

NAMES=($(python scripts/pretrain.py --list))
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
echo "run=$NAME host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python scripts/pretrain.py "$NAME" --device cuda --threads "$SLURM_CPUS_PER_TASK" \
    ${EPISODES:+--episodes $EPISODES}
