<aside> 💡
基于 π₀.5 的工厂柔性分拣机械臂操作 2026.03 – 2026.04

OpenPI + PyTorch + JAX + PaliGemma + Flow Matching + LeRobot + 智元 G1 + A100

项目描述：面向工厂柔性分拣场景（零件分类入盒、动态抓取等典型任务），负责 π₀.5 (OpenPI) 从数据采集、训练微调、策略推理到智元 G1 真机执行的全链路落地，解决原 Diffusion Policy 方案在多品类物体随机位姿下泛化偏弱的问题。

VLA 训推链路打通： 在充分理解训推逻辑基础上，基于 OpenPI 框架完成原始数据清洗、正运动学转换（关节角 → 末端笛卡尔位姿）、LeRobot v2 格式打包、RepackTransform 字段映射、compute_norm_stats 归一化统计量生成、Dataset/DataLoader 适配、Policy 模块二次开发（action_dim / action_horizon 调整、flow matching 推理步数调参）、训练配置修改、推理服务封装全链路；在 8 卡 A100 完成全量微调，并以 WebsocketPolicyServer 形态封装策略服务、websocket_client_policy 形态封装真机/仿真评测客户端，打通训练 → 推理 → 评测闭环。
数据采集与质控： 基于智元 G1 真机 VR 遥操，围绕动态抓取与供件等典型分拣任务构建训练与验证流程，采集 100+ 条演示轨迹；严格控制轨迹平滑性，显式区分去程与回程的空间路径以避免相同观测下出现矛盾动作标注，完成原始轨迹清洗、归一化与数据质检。
Benchmark 与真机验证： 在 LIBERO 仿真任务上完成 π₀.5 训练与评测链路适配，作为代码改动的回归测试；在 G1 真机上部署动态抓取与供件任务，采用端云协同架构（真机 WebSocket 上传观测 → A100 服务端推理 → action chunk 回传执行），相比原 Diffusion Policy 方案成功率由 40%+ 提升至 70%+，验证 π₀.5 在真机复杂接触任务中的落地效果。


#### 0.项目背景与动机

工厂柔性分拣场景下，机械臂需要抓取位置、形状不固定的物件。团队此前用 Diffusion Policy 跑这个任务，成功率约 40%+——瓶颈在泛化能力：物件位置稍偏离训练分布就容易失败。项目目标是用 π₀.5（VLA 模型） 替换 Diffusion Policy，借助它在大规模机器人数据上预训练得到的视觉-语言泛化能力来提升成功率。

#### 一、真机数据采集

硬件配置：智元 G1 半人形机器人 + VR 遥操设备（头显 + 手柄）。操作员通过 VR 远程控制机械臂，按键控制夹爪开合。

采集流程：在智元平台发布任务后，通过真机背屏启动采集，操作员按"抓取 → 移动 → 放置"的轨迹执行。单任务（如 Pick and Place）采集 50–100 条轨迹，覆盖不同物件位置和形状，最终采集 100+ 条数据。

数据质量控制：

轨迹平滑性：避免手抖造成的 action 跳变。
避免同观测对应矛盾动作——这是模仿学习里最容易"学炸"的坑。当机械臂的关节状态和相机画面高度一致，但下一步标注一会儿是"向下抓"、一会儿是"向上抬"时，模型会在该状态上学到方差极大的输出，策略崩溃。实操办法是采集时让去程与回程走不同的空间路径，从源头杜绝相同状态-不同动作对的出现。

#### 二、数据处理与格式转换

数据下载与解析：从智元平台下载原始数据到 GPU 服务器。原始轨迹记录的是各关节角度（joint angles, rad）+ 相机图像 + 夹爪状态，但 π₀.5 的 action 标签设计为末端执行器在笛卡尔空间下的位姿（xyz + 旋转表示）——原因是模型最终控制的是末端，关节空间维度高、冗余、且不同本体不可迁移。这一步用公司内部的正运动学（FK）工具完成 joint → EE pose 转换。

轨迹清洗：剔除失败 episode（抓空、碰撞）、截掉首尾空闲段、检查 action 与 state 的时间戳对齐、过滤 NaN 与异常跳变。

LeRobot v2 格式打包：OpenPI 的 data_loader.py 直接基于 LeRobotDataset 加载，因此输入必须是 LeRobot v2.x 标准布局：


<repo_id>/
├── meta/
│   ├── info.json          # features schema、fps、shape（state/action 维度、相机分辨率）
│   ├── episodes.jsonl     # 每条 episode 的长度与对应 task
│   ├── tasks.jsonl        # 自然语言任务描述（π₀.5 训练要用）
│   └── episodes_stats.jsonl
├── data/chunk-000/episode_XXXXXX.parquet      # 低维数据按 episode 写 parquet
└── videos/chunk-000/observation.images.<cam>/episode_XXXXXX.mp4
                                              # 相机流默认编码成 MP4，按帧索引解码
归一化统计量生成（与 LeRobot 自带 stats 解耦）：OpenPI 的归一化是另一套——格式打包完后单独跑 scripts/compute_norm_stats.py，把 state/action 的 quantile（或 mean/std）统计量写到 assets/<config_name>/<repo_id>/norm_stats.json，训练时由 DataConfig 中的 Normalize transform 在线读取做归一化。

#### 三、Benchmark 仿真验证（LIBERO）

上真机之前，先在 LIBERO 仿真环境验证训推链路。LIBERO 是标准化的桌面操作任务 benchmark，包含多个预定义任务，社区使用同一套评估协议。

LIBERO 在本项目里不是用来刷分，而是充当"链路单元测试"：

代码回归测试：OpenPI 原生支持 LIBERO 且有官方报告的成功率。在对框架做了大量二次开发（RepackTransform、action_dim / action_horizon 调整、policy 头微改）之后，先在这个"已知答案"的任务上跑一遍，确认指标和官方接近，证明改动没引入 bug。
训推链路连通性：从 dataset 加载 → 训练 → checkpoint 保存 → WebSocket 推理 → 评估，整条链路完整跑通，确保不存在维度错配、归一化键名错位等问题。
超参试探：学习率、batch size 在 LIBERO 上几分钟一轮快速试，再迁移到真机数据；真机一轮可能要几小时。
链路无误后切换到 G1 真机数据训练与部署。

#### 四、模型训练

训练资源：8 卡 A100，π₀ 训练约 5–7 小时，π₀.5 约 9 小时。训练总步数 30k，沿用 OpenPI 官方所有微调 preset 的默认值（业界已验证的有效 horizon）。

训练配置：π₀ 用 batch_size=32，π₀.5 用 batch_size=256（与 OpenPI 官方 pi05_* preset 一致），全量微调（非 LoRA），所有参数参与梯度更新；FSDP + bf16 混精以塞进 A100 80GB 显存，必要时用梯度累积调 micro-batch。

Dataset / DataLoader 适配：因为 G1 的本体维度跟 π₀.5 预训练时不同，主要改动如下——

RepackTransform：把数据集字段重映射成模型期望的标准键（state / actions / image / prompt），这是适配新本体的真正落地点；
action_dim：模型 config 里只有这一个维度字段，state 会被右侧 zero-pad 到 action_dim（默认 32），G1 的 action 维度若超过 32 需要相应放大；
action_horizon（action chunk 长度）：π₀.5 官方默认 10 或 15，根据 G1 控制频率调整；
norm_stats.json：跑一次 compute_norm_stats.py 出新的统计量。

Policy 模块二次开发：π₀.5 的网络结构是 PaliGemma（SigLIP-So400m 视觉编码器 + Gemma-2B 语言主干）作为 VLM backbone，配一个独立的 Gemma-300M action expert 作为 flow matching 头，通过 adaRMSNorm 把 flow timestep 注入 action expert。二次开发主要集中在 action 头侧——调整 action chunk 长度、和 G1 的实际控制周期对齐；flow matching 的去噪步数 num_steps 是推理时超参（默认 10），在精度与延迟之间折中调试。

“**Policy 是 OpenPI 的推理封装层，承上接 WebSocket server，承下接模型；新本体接入时核心工作是写一份 `<robot>_policy.py` 实现 Inputs/Outputs 两个 Transform，处理字段重映射、图像格式、相机补齐、state/action 维度对齐这些适配工作**"。”

训练结果：单任务（简单 Pick and Place）在数据质量好的情况下成功率可超 90%。模型对文本指令的泛化较弱，主要依赖视觉输入，且缺乏长序列记忆能力，因此多任务混训时单任务成功率会下降。

#### 五、真机部署（端云协同）

架构是 Policy Server（GPU 服务端） + Evaluation Client（真机 / 仿真端） 的分离部署，这套接口对真机和 LIBERO 仿真复用同一个 server，只换 client 与权重：

GPU 服务端（Policy Server）：在 A100 服务器上加载训好的 checkpoint，启动 OpenPI 的 WebsocketPolicyServer（src/openpi/serving/websocket_policy_server.py）。接收客户端发来的观测 dict，跑前向推理（含 flow matching 去噪），返回 action chunk。obs / action 走 msgpack-numpy 序列化。

真机客户端（Evaluation Client）：通过电脑连接 G1，执行 source 与 ros2 相关命令进入开发者模式，启动测试脚本。客户端基于 websocket_client_policy 与服务端 WebSocket 通信——每个时间步从相机 + 编码器读取当前观测（图像 + joint state），发送给 server，收到 action chunk 后逐步执行。客户端侧的 RepackTransform 与预处理需要单独写：真机的图像源、state 排布、prompt 文本与 LIBERO 仿真都不一样，但 server 端是统一的。真机端只跑 SDK 与控制循环，不承担推理算力。

验证结果：动态抓取任务成功率从 Diffusion Policy 的 40%+ 提升到 π₀.5 的 70%+。提升主要来自 π₀.5 预训练带来的视觉泛化——物件位置、姿态变化下比 Diffusion Policy 显著更鲁棒。

面试讲述顺序
问题（40% 太低）→ 方案选择（为什么用 π₀.5：预训练 VLM 泛化能力）→ 数据采集（VR 遥操 + 去/回程路径分离的质控细节）→ 数据处理（FK 关节角→末端、LeRobot v2 打包、compute_norm_stats 单独跑归一化）→ LIBERO 链路验证（回归测试 + 调参）→ 训练（8×A100 / 30k steps / pi05 bs=256 / RepackTransform 适配 / PaliGemma+Gemma-300M action expert + flow matching）→ 部署（Policy Server + WebSocket + 真机客户端，server 复用、client 各写一份）→ 结果（70%+）。