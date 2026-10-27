#!/bin/bash
# Like sweep.sbatch, but on the CPU partition with 2 threads per run, the class-time stand-in for a Colab
# CPU runtime (2 vCPUs). Use it when the run time itself is the measurement.
#
#   sbatch --array=1-$(wc -l < sweeps/NAME.txt) scripts/sweep_cpu.sbatch sweeps/NAME.txt
#
#SBATCH --job-name=mazerl-cpu
#SBATCH --partition=cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweepcpu_%A_%a.out

set -euo pipefail
source "${VENV:-$HOME/pytorch/.venv}/bin/activate"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
read -r NAME ARGS <<< "$LINE"
OUT=runs/$(basename "$1" .txt)
echo "task=$SLURM_ARRAY_TASK_ID run=$NAME args=${ARGS:-} host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
lscpu | grep "Model name"
python scripts/pretrain.py base --device cpu --threads "$SLURM_CPUS_PER_TASK" --out "$OUT" \
    --name "$NAME" --evaluate 500 ${ARGS:-}
