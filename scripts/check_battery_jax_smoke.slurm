#!/bin/bash
#SBATCH --job-name=pi05-jax-gate
#SBATCH --account=punim2341
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/pi05-jax-gate-%j.out
#SBATCH --error=logs/pi05-jax-gate-%j.err
set -euo pipefail
repo=/data/gpfs/projects/punim2341/siyuanluo/openpi/openpi
smoke=/data/gpfs/projects/punim2341/siyuanluo/openpi_checkpoints/pi05_so101_battery/battery_jax_smoke_20kprep
test -f "$smoke/wandb_id.txt" || true
latest=$(find "$smoke" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n | tail -n 1)
test -n "$latest"
test -d "$smoke/$latest/params"
checkpoint_bytes=$(du -sb "$smoke/$latest" | cut -f1)
free_bytes=$(df -B1 --output=avail /data/gpfs/projects/punim2341 | tail -n 1 | tr -d ' ')
required_bytes=$((checkpoint_bytes * 22 / 10 + 10 * 1024 * 1024 * 1024))
python - "$latest" "$checkpoint_bytes" "$free_bytes" "$required_bytes" <<'PY'
import json,sys
from pathlib import Path
step,size,free,required=map(int,sys.argv[1:])
record={"smoke_step":step,"checkpoint_bytes":size,"free_bytes":free,"required_bytes":required,"passed":free>=required}
path=Path("/data/gpfs/projects/punim2341/siyuanluo/so_arm101/outputs/setup/openpi_jax_smoke_gate.json")
path.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
if not record["passed"]: raise SystemExit("Insufficient space for safe rolling checkpoints")
PY
rm -rf "$smoke"
