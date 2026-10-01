# SO-101 battery insertion with native OpenPI pi0.5

This configuration fine-tunes the native JAX pi0.5 model on the 100 manually
labelled successful demonstrations from
[`LUOSYrrrrr/so101_battery_insertion_v2`](https://huggingface.co/datasets/LUOSYrrrrr/so101_battery_insertion_v2).
The training configuration is `pi05_so101_battery` in
`src/openpi/training/config.py`.

## Checkpoint

The deployable step-3500 checkpoint is stored in the private model repository
[`LUOSYrrrrr/so101-battery-insertion-models`](https://huggingface.co/LUOSYrrrrr/so101-battery-insertion-models)
under `openpi-jax/step-3500`. It contains the EMA inference parameters and the
matching normalization statistics. It does not contain the 31 GB AdamW state
needed to resume training.

Download it with an authenticated Hugging Face account:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LUOSYrrrrr/so101-battery-insertion-models",
    allow_patterns="openpi-jax/step-3500/**",
    local_dir="checkpoints/so101-battery-pi05",
)
PY

export PI05_CHECKPOINT="$PWD/checkpoints/so101-battery-pi05/openpi-jax/step-3500"
```

This is an intermediate checkpoint. Step 3500 corresponds to 112,000 sampled
frames, or about 2.96 passes over the 37,857-frame training set. Physical robot
evaluation is still required.

## Start the policy server

From this OpenPI checkout, install the environment as described in the main
README and run:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  --default-prompt="Pick up the black battery, insert it into the charger slot until seated, and release the gripper." \
  policy:checkpoint \
  --policy.config=pi05_so101_battery \
  --policy.dir="$PI05_CHECKPOINT"
```

The server loads `params/` as a JAX Orbax checkpoint and loads the state/action
normalization statistics from `assets/`. A `model.safetensors` file is not
required for native JAX inference.

## Query it from the robot

Install the lightweight client in the robot environment:

```bash
pip install -e packages/openpi-client
```

The client observation must use the same six joint values, units, ordering and
camera orientation as the demonstrations:

```python
import numpy as np
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(
    host="POLICY_SERVER_HOST",
    port=8000,
)

observation = {
    # float32 [6]: shoulder_pan, shoulder_lift, elbow_flex,
    # wrist_flex, wrist_roll, gripper
    "observation/state": np.asarray(joint_positions, dtype=np.float32),
    # uint8 HWC images. The server resizes them to the model input size.
    "observation/image.top": np.asarray(third_person_rgb, dtype=np.uint8),
    "observation/image.wrist": np.asarray(wrist_rgb, dtype=np.uint8),
    "prompt": (
        "Pick up the black battery, insert it into the charger slot until "
        "seated, and release the gripper."
    ),
}

action_chunk = client.infer(observation)["actions"]
assert action_chunk.shape == (10, 6)
```

The returned values are ten absolute six-joint targets. Send them through the
same joint ordering and units used while recording the dataset. On the first
hardware test, clamp targets to the robot limits and use reduced motion speed.

## Reproduce training

`scripts/prepare_battery_v21.py` creates a local LeRobot v2.1 compatibility
view without modifying the source recordings. Then compute normalization
statistics and submit training:

```bash
uv run scripts/prepare_battery_v21.py
sbatch scripts/compute_norm_battery.slurm
sbatch scripts/train_battery_jax_20k.slurm
```

The active recipe uses one A100 80 GB, global batch size 32, full-model JAX
fine-tuning, `discrete_state_input=True`, AdamW, EMA decay 0.999, 20,000 steps,
and a rolling checkpoint every 500 steps. Checkpoint serialization requires
substantially more host RAM than inference; the supplied Slurm script requests
200 GB.
