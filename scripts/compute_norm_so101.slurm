#!/bin/bash
# Compute norm_stats for pi05_so101 on a CPU compute node.
# Login node kills long-running tasks; use this instead.
# Estimated time: 30-40 min on 16 CPUs (97 batches × 256 frames × 2 cameras of video decode).

#SBATCH --job-name=norm_so101
#SBATCH --account=punim2341
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-01:00:00
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

module load Anaconda3/2024.02-1
module load GCCcore/13.3.0
module load FFmpeg/7.0.2
conda activate openpi

export HF_LEROBOT_HOME=/data/gpfs/projects/punim2341/siyuanluo/lerobot_cache

cd /data/gpfs/projects/punim2341/siyuanluo/openpi/openpi

uv run scripts/compute_norm_stats.py --config-name pi05_so101

echo "==> norm_stats.json:"
ls -la assets/pi05_so101/LUOSYrrrrr/so101_yellow_tape_v1_v21/norm_stats.json
