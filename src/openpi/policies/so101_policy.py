"""SO-101 single-arm policy transforms (top + wrist cameras, 6-DoF + gripper)."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so101_example() -> dict:
    """Random observation example used by clients for warmup."""
    return {
        "observation/state": np.random.rand(6),
        "observation/image.top": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/image.wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "Pick up the yellow tape and place it in the white box",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """SO-101 has 2 real cameras (top + wrist). Right wrist is zero-padded
    so the model's fixed 3-camera architecture is satisfied."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        top_image = _parse_image(data["observation/image.top"])
        wrist_image = _parse_image(data["observation/image.wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(top_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Strip 32-dim padded model output back to SO-101's 6 joint commands."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}
