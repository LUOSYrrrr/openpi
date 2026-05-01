"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

================================================================================
中文说明：
本脚本用于在训练前**离线**扫一遍数据集，统计 state / action 的归一化参数
（mean、std、q01、q99），并保存到
    <assets_dirs>/<repo_id>/norm_stats.json

为什么必须提前算：
1. 训练和推理（真机/仿真）必须用**同一份**统计量，否则反归一化时关节角会乱跑；
2. q01/q99 需要扫全量数据才稳定（直方图法），训练 batch 是 shuffle 的局部样本，算不出全局分位数；
3. 数据准备 vs 训练解耦，同一个数据集多次实验只需算一次；
4. 部署到真机时，norm_stats.json 必须随 checkpoint 一起打包，离线统计量便于复用。

只统计两类向量：
- "state"   ：机器人本体感受（关节角 / 末端位姿 / 夹爪等），是模型输入
- "actions" ：动作 chunk，shape 为 [action_horizon, action_dim]，是模型输出
图像在模型内部归一化（除以 255 或缩到 [-1, 1]），这里**不**统计；
prompt 字符串在喂给 RunningStats 之前会被 RemoveStrings 剔除。
================================================================================
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    """剔除 batch 中的字符串字段（如 prompt / language_instruction）。

    JAX/numpy 不能对字符串做数值运算，且字符串也不需要归一化，
    所以在喂给 RunningStats 前直接丢掉。
    """

    def __call__(self, x: dict) -> dict:
        # 只保留非字符串类型的键值对
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """为 LeRobot / 普通 torch 数据集构建 DataLoader。

    关键点：归一化前必须先经过 repack_transforms 和 data_transforms 的 inputs，
    保证统计的是 “模型真正看到的那个张量”，而不是原始数据；
    否则后续训练时 transform 改了顺序/维度，统计量就失配了。
    """
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # 创建底层 torch Dataset（封装 LeRobotDataset 等）
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)

    # 套上变换链，保证统计目标 = 模型真正看到的输入分布
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,   # repack：调整字段名 / 维度顺序
            *data_config.data_transforms.inputs,     # data transform：抽取 state / actions / images
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),  # 去掉字符串字段
        ],
    )

    # 决定要跑多少个 batch
    if max_frames is not None and max_frames < len(dataset):
        # 用户指定了上限：只采样一部分（开 shuffle 让采样更均匀）
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        # 否则扫全量数据集，顺序读即可（更省内存）
        num_batches = len(dataset) // batch_size
        shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """为 RLDS 格式数据集（如 DROID）构建 DataLoader。

    RLDS 是 TensorFlow Dataset 风格的迭代式数据集，
    所以这里用 IterableTransformedDataset，不是 map-style。
    """
    # 注意 shuffle=False：归一化统计要扫全量、确定性遍历
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)

    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,  # RLDS 数据已经是 batch 维度，告知 transform
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        # 注意：当前 RLDS 的长度是为 DROID 数据集硬编码的
        num_batches = len(dataset) // batch_size

    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    """主流程：加载配置 → 构建 dataloader → 累计统计 → 落盘 JSON。

    Args:
        config_name: 训练配置名（与 train.py 用同一个 config，例如 pi05_so101_lora）。
                     必须保证此处的 transform 链与训练时一致，否则统计量错位。
        max_frames:  可选，最多扫描的样本数。None 表示扫全量。
                     大数据集（如 DROID）可以用此参数加速调试。
    """
    # 1) 拿到训练配置和数据配置；assets_dirs 决定 norm_stats.json 的输出位置
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 2) 根据数据格式选 dataloader：RLDS（DROID 等）走 TF 路径，其它走 torch 路径
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    # 3) 只对 state 和 actions 做归一化统计；图像/文本不在此处理
    keys = ["state", "actions"]
    # 每个 key 一份 RunningStats：在线累计 mean / mean_of_squares / 直方图（用于分位数）
    stats = {key: normalize.RunningStats() for key in keys}

    # 4) 流式扫描：一次一个 batch 喂进 RunningStats，避免把整个数据集加载到内存
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    # 5) 收集最终结果：mean、std、q01、q99
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # 6) 写到 <assets_dirs>/<repo_id>/norm_stats.json
    #    训练（Normalize transform）和推理（Unnormalize transform）都会从这里加载
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    # tyro 自动把 main() 的参数变成 CLI 参数：
    #   uv run scripts/compute_norm_stats.py --config-name pi05_so101_lora
    #   uv run scripts/compute_norm_stats.py --config-name pi05_droid --max-frames 100000
    tyro.cli(main)
