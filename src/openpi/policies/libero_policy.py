import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """
    图像格式标准化：把各种格式的图像统一成 uint8 (H, W, C)。

    LeRobot parquet 里存的是 float32 (C, H, W)（归一化到 [0, 1]），
    但 π0 模型的视觉编码器（PaliGemma / SigLIP）期望 uint8 (H, W, C)（0-255）。

    处理两件事：
      1. dtype：float → uint8（×255 转回像素范围）
      2. 维度：CHW → HWC（einops rearrange）

    推理时如果环境已经传 uint8 HWC，两个 if 都不触发，直接返回。
    """
    image = np.asarray(image)
    # 如果是浮点数（LeRobot 训练数据），乘 255 转回 0-255 整数
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # 如果第 0 维是 3（通道），转成 HWC 排列（PaliGemma 期望）
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    LIBERO 数据集特有的"格式转换"transform。

    职责：把 LIBERO 的字段结构"翻译"成 π0 模型期望的统一格式。
    训练和推理都会用到（数据集原始字段 ↔ 模型输入格式的适配层）。

    具体做 3 件事：
      1. 图像格式修正：LeRobot float32 CHW → 模型需要的 uint8 HWC
      2. 补齐三路相机：LIBERO 只有 2 路（主+左腕），补零数组凑成 π0 的 3 路（主+左腕+右腕）
      3. 字段重组：扁平的 observation/xxx → 嵌套的 state / image / image_mask

    要自定义数据集只需复制这个类，修改 data 字典里的 key 即可。
    """

    # 模型类型（PI0 / PI05 / PI0_FAST），只影响右腕相机的 mask 策略（见下方）
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        """
        这个方法不是手动调用的，而是在 TransformedDataset.__getitem__ 触发 transform 链时
        由 CompositeTransform 按顺序执行到这一步时自动调用（Python 的 callable 协议：
        `obj(args)` 等价于 `obj.__call__(args)`）。

        输入 data：RepackTransform 处理后的字典，键名已标准化为 "observation/image" 等
        输出 inputs：π0 模型期望的嵌套字典结构
        """
        # ==================== 1. 图像格式修正 ====================
        # LIBERO 的图像字段：observation/image（主视角）、observation/wrist_image（左腕）
        # 都要通过 _parse_image 转成 uint8 HWC 格式（π0 视觉编码器的输入规范）
        #
        # 如果你的数据集键名不同（比如叫 "cam_high" / "cam_low"），这里要改对应名字。
        # 如果没有某种相机（如没有腕部相机），可以注释掉并补零（参考下方右腕处理）
        base_image = _parse_image(data["observation/image"])          # 主视角（桌面视角）
        wrist_image = _parse_image(data["observation/wrist_image"])   # 左腕视角

        # ==================== 2. 构造模型期望的嵌套结构 ====================
        # π0 模型架构固定支持 3 路相机：base_0_rgb（主视角）+ left_wrist_0_rgb + right_wrist_0_rgb
        # 字典里的键名是模型硬编码的，不要改！
        inputs = {
            # state：机器人低维状态（关节角、夹爪开合度等），直接透传
            "state": data["observation/state"],

            # image：字典结构，三路相机各一个 tensor
            "image": {
                "base_0_rgb": base_image,              # LIBERO 有主视角
                "left_wrist_0_rgb": wrist_image,       # LIBERO 有左腕相机
                # LIBERO 没有右腕相机，用零张量占位凑齐三路
                # （保证模型架构不变，所有数据集共享同一套权重）
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },

            # image_mask：告诉模型哪些相机是"真实有效的"，哪些是"填充的零数组"
            # True = 真实图像（要参与 attention），False = 填充（屏蔽掉）
            "image_mask": {
                "base_0_rgb": np.True_,         # 主视角真实存在
                "left_wrist_0_rgb": np.True_,   # 左腕真实存在
                # 右腕的 mask 策略因模型版本而异：
                # - π0 / π0.5：False，告诉模型这是填充，会屏蔽这路图像的 attention
                # - π0-FAST：True，即使是填充也让模型"看"（实现细节差异，不要改）
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # ==================== 3. 透传 actions（训练时才有）====================
        # 训练时 data 里有 actions（未来 action_horizon 帧的动作 chunk）
        # 推理时没有 actions（模型要预测动作，不是从数据集读）
        # 注意：这里只是透传，真正的 pad 到 32 维在后续 PadStatesAndActions transform 完成
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # ==================== 4. 透传 prompt ====================
        # 语言指令（任务描述），模型的语言输入
        # 如果你的数据集 prompt 字段名不同，只改这里的 key 读取，
        # 输出字典的 key 必须是 "prompt"（TokenizePrompt transform 会来取这个 key）
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    LIBERO 的"反向 transform" —— 把模型输出的动作转回 LIBERO 仿真器期望的格式。

    只在推理时使用（训练时不需要 —— 训练的监督信号已经是正确格式）。
    作用：π0 输出的 actions 是 pad 到 32 维的，但 LIBERO 实际只用前 7 维（7-DoF 机器人），
         需要截掉后 25 维的 padding 再发给仿真器执行。

    如果你的数据集维度不同，把下方的 `7` 改成你的真实 action 维度。
    """

    def __call__(self, data: dict) -> dict:
        """
        推理时机：policy 拿到模型输出 → 这个 transform 反向处理 → 发给仿真器/真机。

        data["actions"] shape: [action_horizon, 32]（pad 到 32 维）
        返回 shape:             [action_horizon, 7]（截回 LIBERO 实际维度）
        """
        # 只取前 7 维（LIBERO 是 7-DoF：6 关节 + 1 夹爪）
        # 后面的 32-7=25 维都是 PadStatesAndActions 训练时 pad 的零填充，丢掉
        return {"actions": np.asarray(data["actions"][:, :7])}
