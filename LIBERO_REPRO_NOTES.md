# LIBERO 复现心路历程

Ubuntu 22.04 + RTX 4070 + 澳洲网络环境下，按 `LEARN_PI0.md` Part 1 用 Docker 复现 pi05_libero 推理的完整过程和踩过的坑。

## 最终跑通的命令

```bash
sudo xhost +local:docker
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

一条命令同时拉起两个容器，跑完 500 episodes (~30-40 min)，视频存在 `examples/libero/data/libero/videos/`。

## 过程中的几个关键认知

### 1. 为什么是"两个容器"不是一个

`compose.yml` 里定义了两个 service：
- **openpi_server** (Python 3.11)：加载 pi0.5 模型权重，开 websocket 等请求
- **runtime** (Python 3.8 + PyTorch cu113)：跑 LIBERO / MuJoCo 仿真，通过 websocket 向 server 要动作

分两个容器是因为 LIBERO 依赖锁死在老 Python/老 PyTorch，和现代 JAX/Flax 的模型环境**根本装不到一起**。容器隔离正好解决这个问题。

`docker compose up` 默认启动所有 service，`depends_on` 保证 openpi_server 先起。

### 2. `SERVER_ARGS="..." docker compose up` 为什么这么写

`SERVER_ARGS` 是环境变量，通过 `compose.yml` 的 `environment:` 段注入到 openpi_server 容器，最终被 `serve_policy.Dockerfile` 里的 `CMD uv run scripts/serve_policy.py $SERVER_ARGS` 展开。流程：

shell → compose.yml → Dockerfile CMD → 实际进程参数

改 checkpoint / 改任务套件都是往这两个变量里加参数，不用改任何代码。

### 3. Checkpoint 存在哪

宿主机：`~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/`
容器里映射为：`/openpi_assets/checkpoints/pi05_libero/`

由 `compose.yml` 的 `${OPENPI_DATA_HOME:-~/.cache/openpi}:/openpi_assets` 做 bind mount。关掉容器不会丢，下次跳过下载。

### 4. 跑完的标志

lazydocker 里 runtime 显示 `exited (0)` 就是成功（退出码 0）。结尾有个 `Exception ignored in: EGLGLContext.__del__` 的 traceback 看着吓人，其实是 robosuite 退出时清理 OpenGL 上下文的老问题，和结果无关。openpi_server 不会自己退，要手动 Ctrl+C 或 `docker compose down`。

## 踩过的坑

### 坑 1: sudo 在 Claude Code 终端被禁

> `sudo: The 'no new privileges' flag is set`

Claude Code 的沙箱限制。开一个**普通系统终端**执行 sudo 命令。

### 坑 2: apt 锁被 packagekitd 占着

装 nvidia-container-toolkit 时 apt update 卡死 / 装不上。

```bash
sudo systemctl stop packagekit
sudo lsof /var/lib/apt/lists/lock   # 确认没人占
sudo apt update && sudo apt install nvidia-container-toolkit
```

### 坑 3: docker socket 权限拒绝

> `permission denied while trying to connect to the docker API`

`sudo usermod -aG docker $USER` 之后用户组只在**新登录会话**生效。解决：
- 关机重启（最彻底）
- 或注销重登
- 或当前终端临时 `newgrp docker`

### 坑 4: 容器内 apt-get 连不上 archive.ubuntu.com（本次最难）

构建时卡在：
```
Err:1 http://archive.ubuntu.com/ubuntu jammy InRelease
  Connection timed out
```

**诊断过程**：
```bash
curl --max-time 5 -I http://archive.ubuntu.com/     # timeout
curl --max-time 5 -I https://archive.ubuntu.com/    # 200 OK
```

→ **不是 DNS、不是防火墙、不是 Great Firewall**（我在澳洲），是 ISP 路径上某处把 HTTP (port 80) 挡了，HTTPS (port 443) 没事。

**解决**：在 `examples/libero/Dockerfile` 和 `scripts/docker/serve_policy.Dockerfile` 的 apt 调用之前加一行 sed，把源换成澳洲 AARnet 的 HTTPS 镜像：

```dockerfile
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirror.aarnet.edu.au/pub/ubuntu/archive|g; s|http://security.ubuntu.com/ubuntu|https://mirror.aarnet.edu.au/pub/ubuntu/archive|g' /etc/apt/sources.list
```

教训：网络问题先用 curl 按 **HTTP vs HTTPS + 不同域名**正交切分，再决定是换协议、换镜像还是走代理。

### 坑 5: Ubuntu 自带视频播放器打不开 mp4

> `H.264 (High Profile) 解码器`

Totem 缺编码器。

```bash
sudo apt install -y mpv
mpv examples/libero/data/libero/videos/rollout_*_success.mp4
```

### 坑 6: 为什么视频只有十几个而不是 500 个

`examples/libero/main.py` 里文件名是 `f"rollout_{task_segment}_{suffix}.mp4"`——**只用任务名 + success/failure，没有 episode 编号**。同任务后面的 episode 覆盖前面的。libero_spatial 有 10 个任务 × 2 种结果，上限 20 个 mp4。想看全部得改 main.py 加 episode id 到文件名。

## 概念澄清

- **Dockerfile**：一份"怎么从零搭这个环境"的脚本。每个 `RUN` 是一层，层会缓存。
- **nvidia-container-toolkit**：Docker 和 NVIDIA 驱动之间的桥。没它容器看不到 GPU。
- **MuJoCo**：物理仿真引擎（DeepMind 开源），LIBERO 用它模拟机械臂动力学、接触、渲染。不是 CAD——CAD 画车，MuJoCo 开车。
- **EGL**：GPU 的"离屏渲染"接口。`MUJOCO_GL=egl` 让 MuJoCo 不开窗口、直接把画面画到显存里再存 mp4。所以 LIBERO 全程是 headless 的。
- **uv**：Rust 写的 Python 包管理器，比 pip 快 10x+，Dockerfile 里用它装依赖。

## 文件改动汇总

本次复现修改了两个文件（都是加 AARnet 镜像那一行）：
- `examples/libero/Dockerfile`
- `scripts/docker/serve_policy.Dockerfile`

其他全部按官方文档走。
