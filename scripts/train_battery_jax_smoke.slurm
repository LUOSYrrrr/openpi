#!/bin/bash
#SBATCH --job-name=pi05-jax-smoke
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-preempt
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --exclude=spartan-gpgpu098
#SBATCH --output=logs/pi05-jax-smoke-%j.out
#SBATCH --error=logs/pi05-jax-smoke-%j.err
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
.venv/bin/python - <<'PY'
import jax
devices = jax.devices()
print("JAX devices:", devices, flush=True)
assert len(devices) == 1 and devices[0].platform == "gpu"
PY
.venv/bin/python scripts/train.py pi05_so101_battery \
  --exp-name=battery_jax_smoke_20kprep \
  --checkpoint-base-dir=/data/gpfs/projects/punim2341/siyuanluo/openpi_checkpoints \
  --batch-size=32 \
  --num-train-steps=2 \
  --save-interval=1 \
  --overwrite \
  --no-wandb-enabled
