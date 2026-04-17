# openpi 学习指南：LIBERO 推理复现 + pi0 训推逻辑

> 目标：(1) 在本地用 8G+ 显卡跑通 LIBERO 仿真的 inference，看到成功/失败视频；(2) 读懂 `pi0_pytorch.py` 里 `forward`（训练）和 `sample_actions`（推理）两个函数。

---

## Part 1 · LIBERO 推理复现（Docker 路线，强烈推荐）

LIBERO 是个 Mujoco 仿真 benchmark。整个流程是"两端架构"：

- **Server（Policy Server）**：加载 `pi05_libero` checkpoint，收观测、吐 action。
- **Client（LIBERO sim）**：跑 Mujoco 环境，把图像/state 发给 server，拿回 action 执行，录视频。

Docker 方案把两端的依赖打包好了，不用折腾 Mujoco 的系统库。非 Docker 方案要 Python 3.8 + CUDA 11.3 的子环境，非常容易翻车，**第一次跑建议 Docker**。

### Step 1. 环境前置

**硬件**：NVIDIA GPU ≥ 8GB 显存（inference 够用）。显卡必须是 NVIDIA，**AMD/Apple Silicon 跑不了**（Mujoco EGL + CUDA）。

**系统**：Ubuntu 22.04 最稳。README 明说没测过其它系统。如果你现在是 Mac，只能远程连一台 Linux 机器（学校机房 / 云 GPU / WSL2 理论可行但没官方支持）。

**软件**：
```bash
# NVIDIA driver（系统自带或 apt 装）
nvidia-smi   # 能看到卡就行

# Docker + nvidia-container-toolkit
curl -fsSL https://get.docker.com | sh
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi  # 验证
```

### Step 2. 克隆仓库（注意 submodule）

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi
# 如果忘了 --recurse-submodules，补一步：
git submodule update --init --recursive
```

`third_party/libero` 是 submodule，漏了会在启动时报 import error。

### Step 3.（可选）允许 Docker 访问 X11

**大多数人跳过这一步**。LIBERO 用 Mujoco 仿真，默认走 EGL 模式做 headless GPU 渲染 —— **不需要显示器、不需要 X server**，纯 SSH 进服务器也能跑。仿真渲染的相机图像是喂给 policy 当输入的，不是给人看的；给人看的东西最后会存成 `.mp4` 文件，`scp` 拉回本地看即可。

只有当你**在本地 Linux 桌面上想看实时弹窗**（例如本机带显卡的 Ubuntu 工作站），才需要：

```bash
sudo xhost +local:docker
```

Headless 服务器 / 纯 SSH 环境：直接跳到 Step 4。

### Step 4. 一键启动

默认跑 `libero_spatial` 任务套件、用 `pi05_libero` checkpoint：

```bash
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

**第一次 build 会慢**（下载镜像 + 装 Mujoco），花 15-30 分钟正常。build 完之后 server 会自动去 `gs://openpi-assets/checkpoints/pi05_libero` 下权重到 `~/.cache/openpi/`（几个 GB，根据网络看）。

**常见坑 & 对策**：

| 现象 | 解决 |
|---|---|
| `EGL: Could not create context` | 换 GLX：`MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build` |
| checkpoint 下不下来 | 手动 `gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero ~/.cache/openpi/openpi-assets/checkpoints/`（需要装 `gcloud`） |
| OOM（显存不够） | 检查是不是同时跑着别的程序；`nvidia-smi` 看看实际占用。pi05 inference 理论 < 8GB |
| Docker build 超慢 | 给 Docker 配国内 registry mirror |

### Step 5. 看输出视频

任务跑完后 server 端会打印成功率，client 端把每集的视频存到 `data/libero/videos/` 之类的目录（具体看 `examples/libero/main.py` 里的 `video_out_path`）。拉回本地就能看到机械臂是成功抓起物体还是抽搐乱挥。

### Step 6. 换任务套件 / 换 checkpoint

```bash
# 跑 libero_10 的难题
export CLIENT_ARGS="--args.task-suite-name libero_10"
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# 换成 pi0 (非 pi05)
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_libero --policy.dir gs://openpi-assets/checkpoints/pi0_libero"
```

**Goal**：至少跑完 `libero_spatial`（10 集左右），看到 success rate ≈ 95+%，并能打开一两个 mp4 确认机械臂动作合理。这一步通过你就对 VLA 的"输入（图+语言+state）→ 输出（动作 chunk）"有直观感觉了。

---

## Part 2 · 读懂 pi0 训练 / 推理逻辑

文件：[`src/openpi/models_pytorch/pi0_pytorch.py`](src/openpi/models_pytorch/pi0_pytorch.py)

### 先建一个心智模型

pi0 本质是 **Flow Matching VLA**，网络有两个塔：

```
┌────────────────────────────┐      ┌────────────────────────────┐
│ Prefix: PaliGemma (大模型)  │ ───►│ Suffix: Action Expert (小)  │
│   输入：多视角图像 + 语言   │      │   输入：state + noisy_action│
│   冻/半冻权重                │      │          + timestep         │
└────────────────────────────┘      └────────────────────────────┘
      ↑ 图像靠 SigLIP 编码          ↑ action_in_proj 线性映射
      ↑ 语言靠 Gemma embedding      ↑ timestep 走 sinusoidal PE
                         共享一套 block-causal attention：
                         suffix 能看 prefix；prefix 看不到 suffix
                         ── 这正是推理时 KV cache 可行的基础
```

**Flow Matching 的核心公式**（放在脑子里，对着代码看）：

- 训练目标：学一个速度场 `v_θ(x, t)`，使它在任意 `t ∈ [0,1]` 的中间态 `x_t = t·noise + (1-t)·action` 上，输出 `≈ u_t = noise - action`。
- 推理：从 `x_1 = noise` 出发，按 Euler 法 `x ← x + dt · v_θ(x, t)` 积分到 `x_0 = action`。

### Step 1. 先通读一遍文件结构

用你的 IDE 打开 `pi0_pytorch.py`，只看以下锚点（别一行一行抠）：

| 行号 | 角色 |
|---|---|
| [L84 `class PI0Pytorch`](src/openpi/models_pytorch/pi0_pytorch.py#L84) | 模型主类 |
| [L187 `embed_prefix`](src/openpi/models_pytorch/pi0_pytorch.py#L187) | 把 images + lang tokens 变成 prefix embedding |
| [L238 `embed_suffix`](src/openpi/models_pytorch/pi0_pytorch.py#L238) | 把 state + noisy_actions + time 变成 suffix embedding |
| [L317 `forward`](src/openpi/models_pytorch/pi0_pytorch.py#L317) | **训练入口** |
| [L377 `sample_actions`](src/openpi/models_pytorch/pi0_pytorch.py#L377) | **推理入口** |
| [L422 `denoise_step`](src/openpi/models_pytorch/pi0_pytorch.py#L422) | 推理时的单步去噪 |

### Step 2. 训练逻辑（`forward`，6 行就懂）

```python
# 1. 每样本采 1 个随机 t（Beta 分布偏向小 t）
time = self.sample_time(bsize, device)   # L182-185: Beta(1.5, 1.0) * 0.999 + 0.001

# 2. 采高斯噪声，做线性插值
noise = self.sample_noise(actions.shape, device)
x_t = t * noise + (1 - t) * actions       # L328
u_t = noise - actions                     # L329 ── 这就是监督目标

# 3. prefix + suffix 一起过一次完整 attention（use_cache=False！）
prefix_embs = embed_prefix(images, lang)
suffix_embs = embed_suffix(state, x_t, time)
v_t = paligemma_with_expert(prefix_embs ⊕ suffix_embs)[-action_horizon:]

# 4. MSE loss
loss = F.mse_loss(u_t, v_t)               # L374
```

**要点**：
- 每个样本一次梯度更新 = 1 次 forward，**没有任何迭代**。
- 训练完全不用 KV cache（`use_cache=False`，[L356](src/openpi/models_pytorch/pi0_pytorch.py#L356)）。
- 模型学到的是"任意 t 下速度场的瞬时值"，不是"如何去噪一整条轨迹"。

### Step 3. 推理逻辑（`sample_actions` + `denoise_step`）

```python
# 1. prefix 只前向 1 次，把 K/V 存进 cache
past_key_values = paligemma(prefix_embs, use_cache=True)   # L394-400

# 2. 初始化
x_t = noise                # shape = (bsize, horizon, action_dim)
t = 1.0
dt = -1.0 / num_steps      # 默认 num_steps=10，所以 dt=-0.1

# 3. Euler 积分 10 步，每步只跑小 Expert
while t >= -dt/2:
    suffix_embs = embed_suffix(state, x_t, t)
    v_t = expert(suffix_embs, past_key_values=prefix_KV)  # 读缓存，省掉 prefix 计算
    x_t = x_t + dt * v_t                                  # L418
    t  += dt

return x_t   # 最终的 action chunk
```

**要点**：
- **prefix 只跑 1 次**；之后 10 步每一步"只跑小塔 Expert"，这就是 KV cache 省计算的地方。
- dt 是负的，t 从 1.0 递减到 0.0 —— 模型"走"的方向是 noise → action。
- 步数 `num_steps=10` 是 inference 阶段的超参，跟训练无关；你可以改成 5 或 20 观察质量-速度权衡。

### Step 4. 一个样本的数据流图（自己在本子上画一遍）

```
推理：
obs = {images, prompt, state}
  ├─ preprocess ──► [imgs, lang_tokens, state]
  ├─ embed_prefix: SigLIP(imgs) ⊕ embed(lang) ──► prefix_embs
  │                              ▼
  │                  PaliGemma.forward(use_cache=True) ──► past_key_values   ★一次
  │
  ▼  x_t = randn(bsize, horizon, action_dim);  t = 1.0
  for step in 1..10:                                                          ★十次
      suffix_embs = embed_suffix(state, x_t, t)           # state+action+time
      v_t = Expert.forward(suffix_embs, past_key_values)  # 读 prefix KV
      x_t = x_t + dt * v_t
      t  += dt  (dt=-0.1)
  ▼
  action_chunk (B, horizon, action_dim) ──► 机器人执行
```

```
训练：
batch = {obs, actions}                     # actions 是 ground truth
  ├─ sample t ~ Beta;   sample noise ~ N(0,I)
  ├─ x_t = t·noise + (1-t)·actions;   u_t = noise - actions
  ├─ embed_prefix + embed_suffix
  ├─ PaliGemma+Expert(全部拼起来, use_cache=False) ──► v_t           ★一次
  └─ loss = MSE(u_t, v_t)  ──► backward
```

### Step 5. 自检问题（答得出就算读懂了）

1. 训练时 `t` 怎么采？为什么用 Beta(1.5, 1.0) 而不是均匀分布？（提示：偏向小 t，关注靠近真实 action 的区域）
2. 为什么 KV cache 只在推理用、训练不用？（提示：prefix 和 suffix 的 attention 依赖关系 + 训练时输入每轮都变）
3. `num_steps` 调大/调小会怎样？（积分更准 vs 推理更慢）
4. Euler step `x ← x + dt·v`，`dt` 为什么是负号？
5. 训练的 loss 只用 `u_t` 和 `v_t` 的 MSE，没有任何"最终 action 的监督"—— 为什么这样就能学会？（这是 flow matching 的整个理论基础）

### Step 6. 动手验证（可选但推荐）

跑 Part 1 跑通后，在一个 Python shell 里 load 一次模型，手动调用 `sample_actions`，把 `num_steps` 改成 1、2、5、10、20，打印每次返回的 `x_t` 的 L2 距离差异：

```python
import torch
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_libero")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
policy = policy_config.create_trained_policy(config, ckpt)

# 构造一个假观测（shape 参考 libero_policy.py）
example = {...}
# policy.infer 底层会调 sample_actions；可以直接跳进去改 num_steps
```

能观察到："步数太少 → action 抖动 / 不连贯；10 步已经基本收敛"。

---

## 交作业前的 checklist

- [ ] Docker / 非 Docker 任一方案跑通 LIBERO，至少看到一个 task suite 的 success rate
- [ ] 打开 1-2 个成功/失败的 mp4，对 VLA 的行为有直观印象
- [ ] 能用自己的话回答上面 Step 5 的 5 个自检问题
- [ ] 能在白板上画出训练 / 推理两张数据流图，不看代码
- [ ] 能指出"训练 vs 推理不一样"的至少 3 个具体差异（t 采样方式、KV cache、前向次数、时间方向、等）
