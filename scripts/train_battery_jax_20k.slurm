#!/bin/bash
#SBATCH --job-name=pi05-jax-batt
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-preempt
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --exclude=spartan-gpgpu098
#SBATCH --output=logs/pi05-jax-train-%j.out
#SBATCH --error=logs/pi05-jax-train-%j.err
set -euo pipefail
repo=/data/gpfs/projects/punim2341/siyuanluo/openpi/openpi
cd "$repo"
test -f "$HOME/.openpi_env" && source "$HOME/.openpi_env"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=/data/gpfs/projects/punim2341/siyuanluo/jax_cache
export WANDB_DIR=/data/gpfs/projects/punim2341/siyuanluo/wandb
export PYTHONUNBUFFERED=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$WANDB_DIR"
args=()
if [[ "${OPENPI_RESUME:-0}" == 1 ]]; then
  checkpoint_dir=/data/gpfs/projects/punim2341/siyuanluo/openpi_checkpoints/pi05_so101_battery/battery_insertion_v2_jax_20k
  if ! find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
      -regex '.*/[0-9]+' -print -quit 2>/dev/null | grep -q .; then
    echo "No completed checkpoint is available; recovery has nothing to resume."
    exit 0
  fi
  args+=(--resume)
fi
.venv/bin/python scripts/train.py pi05_so101_battery \
  --exp-name=battery_insertion_v2_jax_20k \
  --checkpoint-base-dir=/data/gpfs/projects/punim2341/siyuanluo/openpi_checkpoints \
  --batch-size=32 \
  --num-train-steps=20000 \
  --save-interval=500 \
  "${args[@]}"
