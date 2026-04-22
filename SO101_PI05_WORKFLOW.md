# SO-101 + π0.5 端到端训练部署流程

> **目标**：把 openpi 的 π0.5 模型在 SO-101 单臂机器人上跑通 pick-and-place
> **架构**：Spartan HPC 训练 + 本地 RTX 4070 Ti Super (16GB) 推理
> **数据**：HuggingFace Hub 托管，dataset = `LUOSYrrrrr/so101_yellow_tape_v1`

---

## 决策记录

| 项目 | 选定 | 原因 |
|------|------|------|
| 模型 | π0.5（从 `pi05_base` finetune） | 比 π0 泛化好，transform 更简单（无 extra_delta） |
| 训练硬件 | Spartan 1×H100 80G | 单卡 batch=128 比 2×A100 还快，少排队 |
| 推理硬件 | 本地 4070 Ti Super 16GB | 显存够（π0.5 推理 ~10-12GB），延迟 ~80-100ms |
| 网络拓扑 | 同机 localhost | 绕开 Spartan 计算节点没公网 IP 的问题 |
| 数据格式 | LeRobot v2.1 原生 | `lerobot-record` 直出，无需转换 |
| 数据托管 | HuggingFace Hub | Spartan 直接拉，本地不用 rsync 大文件 |
| 数据集名 | `LUOSYrrrrr/so101_yellow_tape_v1` | `v` 后缀方便迭代 |

---

## Phase 0 — 一次性准备

### 0.1 本地环境
```bash
# SO-101 leader + follower 接 USB
pip install lerobot
lerobot-calibrate --robot.type=so101 --robot.port=/dev/ttyACM0
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1

# openpi 本地装好（推理时用）
cd ~/code/openpi
uv sync
```

### 0.2 HuggingFace 账号 + Token

数据集和 ckpt 都走 HF Hub，所以**两台机器都要 write 权限**。在 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 创建两个独立 token（方便分别撤销）：

| Token name | Type | 用途 |
| --- | --- | --- |
| `local-laptop` | Write | 本地 push dataset、拉 ckpt |
| `spartan-hpc` | Write（推荐 fine-grained，限 ckpt repo） | Spartan push ckpt、拉 dataset |

**fine-grained token 配置**（更安全）：

- Repository permissions:
  - `LUOSYrrrrr/so101_yellow_tape_v1` (dataset) → Read
  - `LUOSYrrrrr/pi05_so101_ckpts` (model, 训练前先建空 repo) → Write

本地登录：
```bash
hf auth login   # 粘贴 local-laptop token（新版 CLI 用 hf，旧的 huggingface-cli 已废弃）
```

### 0.3 Spartan 环境
```bash
ssh siyuanluo@spartan.hpc.unimelb.edu.au

mkdir -p /data/gpfs/projects/punim2341/siyuanluo/{openpi,checkpoints,lerobot_cache}
cd /data/gpfs/projects/punim2341/siyuanluo/openpi
git clone <你 fork 的 openpi repo> .

module load Anaconda3/2024.02-1
conda create -n openpi python=3.11 -y
conda activate openpi
pip install uv && uv sync

hf auth login   # 粘贴 spartan-hpc token
```

### 0.4 提前建 ckpt 的空 Hub repo

```bash
hf repos create pi05_so101_ckpts --type model
# 或者去 https://huggingface.co/new 网页建
```

---

## Phase 1 — 本地采数据 + 直接推送到 HF Hub

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=R12252801 \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=R12252802 \
  --dataset.repo_id=LUOSYrrrrr/so101_yellow_tape_v1 \
  --dataset.root=$HOME/.cache/huggingface/lerobot/LUOSYrrrrr/so101_yellow_tape_v1 \
  --dataset.single_task="Pick up the yellow tape and place it in the white box" \
  --dataset.fps=30 \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=true \
  --display_data=true
```

**关键参数说明（你实际接线对应的）：**

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `robot.type` | `so101_follower` | 被遥操的从臂 |
| `robot.port` | `/dev/ttyACM1` | follower 的串口 |
| `robot.id` | `R12252801` | follower 序列号 |
| `teleop.type` | `so101_leader` | 你手抓的主臂 |
| `teleop.port` | `/dev/ttyACM0` | leader 的串口 |
| `teleop.id` | `R12252802` | leader 序列号 |
| `top` 相机 | index 2，MJPG | 俯视全局摄像头 |
| `wrist` 相机 | index 0，MJPG | 装在末端的腕部摄像头 |
| `episode_time_s` | 25 | 每条 25 秒（够完成 pick + place） |
| `reset_time_s` | 10 | 重置 10 秒（手动把胶带放回起点） |
| `display_data` | true | 录制时实时显示摄像头画面，方便检查 |

### 采集纪律
- ✅ 物体起始位置随机分布（不要总放同一处）
- ✅ 抓取下去和抬起的路径**不要重合**（避免相同 state 给不同 action — 模型会"学乱"）
- ✅ 轨迹平滑无停顿
- ❌ 失败的 episode 当场重录（按 ESC 重录）
- ❌ 不要混入"调整一下"的犹豫动作

数据落地：

- 本地缓存 `~/.cache/huggingface/lerobot/LUOSYrrrrr/so101_yellow_tape_v1/`
- Hub 仓库 `https://huggingface.co/datasets/LUOSYrrrrr/so101_yellow_tape_v1`

---

## Phase 2 — Spartan 拉取数据

```bash
# Spartan 上
export HF_LEROBOT_HOME=/data/gpfs/projects/punim2341/siyuanluo/lerobot_cache

# 方法 A：手动预下载（推荐，避免训练 job 里下数据浪费 GPU 时间）
hf download LUOSYrrrrr/so101_yellow_tape_v1 \
  --repo-type=dataset \
  --local-dir=$HF_LEROBOT_HOME/LUOSYrrrrr/so101_yellow_tape_v1

# 方法 B：什么都不做，第一次训练时 LeRobotDataset 会自动下到 $HF_LEROBOT_HOME
```

数据约 1~2 GB（50 条），登录节点拉 1~2 分钟。

---

## Phase 3 — openpi 加 SO-101 支持（代码工作）

### 3.1 新建 `src/openpi/policies/so101_policy.py`
照 `libero_policy.py` 模板，定义 3 个东西：

```python
def make_so101_example() -> dict:
    """假数据，client 端 warmup 用"""
    return {
        "observation/state": np.zeros(6),
        "observation/image.top": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/image.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        "prompt": "Pick up the yellow tape",
    }


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """LeRobot keys → 模型 keys；state/action pad 到 model.action_dim"""
    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation.state"], self.action_dim)
        images = {
            "base_0_rgb": data["observation.images.top"],
            "left_wrist_0_rgb": data["observation.images.wrist"],
            "right_wrist_0_rgb": np.zeros_like(data["observation.images.top"]),
        }
        image_masks = {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}

        out = {"state": state, "image": images, "image_mask": image_masks}
        if "actions" in data:
            out["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """模型输出 32 维 action → 切回 SO-101 的 6 维"""
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}
```

### 3.2 在 `src/openpi/training/config.py` 注册 `pi05_so101`

照 `pi05_libero` 复制一份，关键改动：

- `repack_transforms` 把 LeRobot 的 `observation.images.top` repack 成 `image.top`，`observation.state` → `state` 等
- `data_transforms` 用 `SO101Inputs(action_dim=model.action_dim)` / `SO101Outputs()`
- **不加** `extra_delta_transform`（π0.5 不用）
- `weight_loader = CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
- `repo_id = "LUOSYrrrrr/so101_yellow_tape_v1"`
- `batch_size = 128`（H100 单卡），`num_train_steps = 30_000`

### 3.3 算 norm stats（CPU 任务，几分钟）
```bash
# Spartan 上
sbatch scripts/compute_norm.slurm   # 内容：uv run scripts/compute_norm_stats.py pi05_so101
```
输出 `assets/pi05_so101/<repo_id>/norm_stats.json`，**git commit 进仓库**（推理时也要用）。

### 3.4 推送代码
```bash
git add src/openpi/policies/so101_policy.py \
        src/openpi/training/config.py \
        assets/pi05_so101/
git commit -m "add pi05_so101 config"
git push
```

---

## Phase 4 — Spartan 训练（约 5~7h on H100）

### 4.1 `scripts/train_so101.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=pi05_so101
#SBATCH --account=punim2341
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=80G
#SBATCH -o slurm-%j.out -e slurm-%j.err
#SBATCH --mail-user=siyuan.luo.1@student.unimelb.edu.au
#SBATCH --mail-type=END,FAIL

module load Anaconda3/2024.02-1
conda activate openpi

export HF_LEROBOT_HOME=/data/gpfs/projects/punim2341/siyuanluo/lerobot_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /data/gpfs/projects/punim2341/siyuanluo/openpi
git pull   # 同步本地最新代码

uv run scripts/train.py pi05_so101 \
  --exp-name=so101_yellow_tape_v1 \
  --checkpoint-base-dir=/data/gpfs/projects/punim2341/siyuanluo/checkpoints

my-job-stats -a -n -s
```

### 4.2 提交 + 监控
```bash
sbatch scripts/train_so101.slurm
squeue --me                       # 看是否在跑
tail -f slurm-<jobid>.out         # 看 loss
```

排队状态参考 spartan-hpc skill 表格。`(MaxGRESPerAccount)` 就等组员任务跑完。

### 4.3 wandb（可选）
登录节点 `wandb login` 一次，训练脚本里加 `--wandb-enabled=true`。

### 4.4 训练完自动推 ckpt 到 HF Hub

在 `train_so101.slurm` 末尾追加：

```bash
# 训练完成后推 ckpt 到 Hub（30k step）
CKPT_DIR=/data/gpfs/projects/punim2341/siyuanluo/checkpoints/pi05_so101/so101_yellow_tape_v1/30000

hf upload LUOSYrrrrr/pi05_so101_ckpts \
  $CKPT_DIR \
  so101_yellow_tape_v1/30000 \
  --repo-type=model \
  --commit-message="so101_yellow_tape_v1 step 30000"
```

> 也可以在 slurm 脚本里加上多个中间 step 的上传（比如每 10k 推一次），方便对比不同训练阶段的效果。

---

## Phase 5 — 本地从 HF Hub 拉 ckpt

```bash
# 本地电脑
hf download LUOSYrrrrr/pi05_so101_ckpts \
  --include="so101_yellow_tape_v1/30000/*" \
  --local-dir=~/code/openpi/checkpoints/pi05_so101
```

ckpt ~7-8 GB，HF CDN 拉取大致 5~15 分钟（看你家宽带）。

> **如果家里宽带慢/没法上 HF**，备用方案：从 Spartan rsync
> ```bash
> rsync -avzP siyuanluo@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2341/siyuanluo/checkpoints/pi05_so101/so101_yellow_tape_v1/30000/ \
>   ~/code/openpi/checkpoints/pi05_so101/30000/
> ```

---

## Phase 6 — 本地部署（同机 server + client + 真机）

### 6.1 Terminal 1：起 server
```bash
cd ~/code/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101 \
  --policy.dir=checkpoints/pi05_so101/30000 \
  --port=8000
```
首次 JAX JIT 编译 1~2 分钟，之后 ~80-100ms / infer。

### 6.2 Terminal 2：起 client
新建 `examples/so101_real/main.py`，主循环：
```python
robot = SO101Robot(port="/dev/ttyACM0")
cam_front = cv2.VideoCapture(0)
cam_wrist = cv2.VideoCapture(2)
policy = WebsocketClientPolicy(host="localhost", port=8000)

while True:
    obs = {
        "observation/state": robot.get_state(),                    # (6,)
        "observation/image.top": cam_front.read()[1],            # (H, W, 3) uint8
        "observation/image.wrist": cam_wrist.read()[1],
        "prompt": "Pick up the yellow tape and place it in the white box",
    }
    action_chunk = policy.infer(obs)["actions"]                    # (50, 6)
    for action in action_chunk[:5]:                                # replan_steps=5
        robot.send_action(action)
        time.sleep(1/30)
```

---

## Phase 7 — 调试

| 现象 | 大概率原因 | 怎么办 |
|------|----------|--------|
| 机械臂乱动 | norm stats 不对 / repack key 名不对 | 检查 `assets/pi05_so101/<repo_id>/norm_stats.json` 是否合理 |
| 抓不到 | 数据物体位置太单一 | 重采，物体随机分布在桌面 |
| 抓到不放 | "放置"动作样本太少 | 检查 episode 末尾是否完整 |
| 推理 > 200ms | JAX 没用 BF16 | 检查 dtype，确认在 GPU 上 |
| 真机抖动 | client 时间步不一致 | `time.sleep(1/30)` 严格执行；考虑用 high-precision sleep |
| Loss 不降 | 数据 fps 和 config 不匹配 | 检查 `dataset.fps=30` 和训练 config 是否一致 |

---

## 时间预算

| 阶段 | 预计 | 说明 |
|------|------|------|
| Phase 0 | 半天 | SO-101 校准要细心 |
| Phase 1 | 半天~1 天 | 50 episodes + HF 上传（约 25-40 分钟手柄不离手 + 间隔休息） |
| Phase 2 | 5 分钟 | HF 拉数据 |
| Phase 3 | 1~2 小时 | 写 policy + config |
| Phase 4 | 5~7h + 排队 | 训练 |
| Phase 5 | 5 分钟 | rsync ckpt |
| Phase 6 | 1 小时 | 第一次跑通 |
| Phase 7 | 1~3 天 | 调试 + 重采迭代 |

**总计：~1 周看到第一次成功的 pick-and-place。**

---

## 关键路径速查

| 资源 | 位置 |
|------|------|
| 本地数据 | `~/.cache/huggingface/lerobot/LUOSYrrrrr/so101_yellow_tape_v1/` |
| HF Hub | `https://huggingface.co/datasets/LUOSYrrrrr/so101_yellow_tape_v1` |
| Spartan 数据 | `/data/gpfs/projects/punim2341/siyuanluo/lerobot_cache/LUOSYrrrrr/so101_yellow_tape_v1/` |
| Spartan ckpt | `/data/gpfs/projects/punim2341/siyuanluo/checkpoints/pi05_so101/so101_yellow_tape_v1/` |
| 本地 ckpt | `~/code/openpi/checkpoints/pi05_so101/30000/` |
| Norm stats | `assets/pi05_so101/LUOSYrrrrr/so101_yellow_tape_v1/norm_stats.json` |

| 关键文件 | 作用 |
|---------|------|
| [src/openpi/policies/so101_policy.py](src/openpi/policies/so101_policy.py) | SO-101 input/output transforms |
| [src/openpi/training/config.py](src/openpi/training/config.py) | 注册 `pi05_so101` 训练 config |
| [examples/so101_real/main.py](examples/so101_real/main.py) | 真机 client |
| [scripts/train_so101.slurm](scripts/train_so101.slurm) | Spartan 训练脚本 |
