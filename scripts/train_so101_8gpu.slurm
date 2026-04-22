#!/bin/bash
# Train pi05_so101 on 8× A100 80G — matches the original π0.5 fine-tuning recipe (batch=256).
# Estimated wall time: 5-7h for 30k steps.
#
# WARNING: Spartan A100 nodes typically have 4 GPUs each. If `--gres=gpu:8` on
# 1 node fails to schedule, switch to multi-node:
#   #SBATCH --nodes=2 --ntasks-per-node=1 --gres=gpu:4
# (Multi-node JAX needs extra setup; ask if you hit this.)

#SBATCH --job-name=pi05_so101_8g
#SBATCH --account=punim2341
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=0-10:00:00
#SBATCH --mem=480G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-user=siyuan.luo.1@student.unimelb.edu.au
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module load Anaconda3/2024.02-1
conda activate openpi

export HF_LEROBOT_HOME=/data/gpfs/projects/punim2341/siyuanluo/lerobot_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

REPO_DIR=/data/gpfs/projects/punim2341/siyuanluo/openpi/openpi
CKPT_BASE=/data/gpfs/projects/punim2341/siyuanluo/checkpoints
EXP_NAME=so101_yellow_tape_v1_8g
NORM_FILE=$REPO_DIR/assets/pi05_so101/LUOSYrrrrr/so101_yellow_tape_v1/norm_stats.json

cd $REPO_DIR
git pull --ff-only

if [ ! -f "$NORM_FILE" ]; then
  echo "==> Computing norm stats..."
  uv run scripts/compute_norm_stats.py pi05_so101
fi

echo "==> Training pi05_so101 (8 GPUs, batch=256 — full π0.5 recipe)..."
uv run scripts/train.py pi05_so101 \
  --exp-name="$EXP_NAME" \
  --checkpoint-base-dir="$CKPT_BASE"

CKPT_DIR="$CKPT_BASE/pi05_so101/$EXP_NAME/30000"
if [ -d "$CKPT_DIR" ]; then
  echo "==> Uploading ckpt to HF Hub..."
  hf upload LUOSYrrrrr/pi05_so101_ckpts \
    "$CKPT_DIR" "$EXP_NAME/30000" \
    --repo-type=model \
    --commit-message="$EXP_NAME step 30000"
fi

my-job-stats -a -n -s
