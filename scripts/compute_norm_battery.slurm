#!/bin/bash
#SBATCH --job-name=pi05-jax-norm
#SBATCH --account=punim2341
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pi05-jax-norm-%j.out
#SBATCH --error=logs/pi05-jax-norm-%j.err
set -euo pipefail
repo=/data/gpfs/projects/punim2341/siyuanluo/openpi/openpi
cd "$repo"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_so101_battery
test -s assets/pi05_so101_battery/LUOSYrrrrr/so101_battery_insertion_v2_openpi_v21/norm_stats.json || \
  test -s assets/pi05_so101_battery/norm_stats.json || \
  test -s assets/LUOSYrrrrr/so101_battery_insertion_v2_openpi_v21/norm_stats.json
