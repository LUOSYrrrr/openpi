import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    # 训练时调用的主入口。每个 batch 调一次，算一次 Flow Matching loss。
    # ───────────────────────────────────────────────────────────
    # 架构：类 MoE 设计（论文说 "analogous to a mixture of experts with two mixture elements"）
    #   - VLM backbone（PaliGemma, 3B 预训练）：处理图像 + 语言
    #   - Action Expert（300M 从头训）：处理 state + action + time
    #   - 两套权重独立，但共享 attention；routing 是固定的（按 token 类型分配）
    # ───────────────────────────────────────────────────────────
    # 整体流程（9 个阶段，和下面注释的 === N. === 一一对应）：
    #   ① 拆 observation → 图像 / 语言 token / 机器人状态
    #   ② 采样 Flow Matching 的噪声 ε 和时间 t（t ~ Beta(1.5, 1)，偏向噪声多的难样本）
    #   ③ 构造插值点 x_t = t·ε + (1-t)·A，目标速度 u_t = ε - A
    #   ④ 两套专家分别 embed：
    #        - VLM backbone embed 图像+语言 → prefix_embs
    #        - Action Expert embed 状态+x_t+t → suffix_embs（pi0.5 还返回 adarms_cond）
    #   ⑤ bf16 精度对齐
    #   ⑥ 拼接 prefix+suffix，构造 prefix-LM 风格的 attention mask
    #        （图像/语言全连通；状态/动作可看图像语言但反向屏蔽；动作内部双向）
    #   ⑦ 联合走 PaliGemmaWithExpertModel.forward（两套专家共享 attention，权重独立）
    #   ⑧ 取最后 H 个动作 token，过 action_out_proj → 预测速度场 v_t
    #   ⑨ Flow Matching loss = MSE(u_t, v_t)，外部 .mean().backward() 完成一步训练
    # ───────────────────────────────────────────────────────────
    # 注意：
    # - 代码约定 t=0 干净、t=1 噪声；和论文 τ 方向相反（t_code = 1 - τ_paper）
    # - 真实动作 A 由调用方传入，即参数 `actions`，形状 [B, action_horizon, action_dim]
    # - noise / time 默认 None 表示内部采样；外部可传固定值用于复现或单元测试
    # - 训练不用 KV cache（`use_cache=False`），推理路径（sample_actions）才会用
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        # === 1. 准备输入：把 observation 拆成图像 / 语言 token / 机器人状态 ===
        # images: list of [B, C, H, W]；lang_tokens: [B, L]；state: [B, state_dim]
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
          observation, train=True)

        # === 2. 采样 Flow Matching 需要的噪声 ε 和时间 t ===
        if noise is None:
            # 形状和 actions 完全一样 [B, horizon, action_dim]；值从 N(0,1) 独立采样
            noise = self.sample_noise(actions.shape, actions.device)  # ε ~ N(0,1)

        if time is None:
            # 每条样本一个 t ∈ (0.001, 1]；Beta(1.5, 1) 偏向大 t（噪声多的难样本）
            time = self.sample_time(actions.shape[0], actions.device)  # t ~ Beta(1.5, 1) 偏向难时刻

        # === 3. 构造 Flow Matching 的插值点 x_t 和真实速度 u_t ===
        # 注意：代码约定和论文相反！论文是 A^τ = τ·A + (1-τ)·ε (τ=1 干净, τ=0 噪声)
        #       代码是 x_t = t·ε + (1-t)·A (t=0 干净, t=1 噪声)
        # 两者数学等价，换元 t_code = 1 - τ_paper 即可互相转换
        # 代码用这个约定是因为推理时 t 从 1 走到 0，方向和传统 diffusion denoising 一致

        # time_expanded: [B] -> [B, 1, 1]，用于广播到 [B, horizon, action_dim]
        time_expanded = time[:, None, None]
        # x_t 构造插值点
        # x_t：t=0 是干净动作 A，t=1 是纯噪声 ε，中间线性插值
        # x_1.0  shape=[B, 50, action_dim]  ← 纯噪声，50 帧都是乱的
        #   ↓ 模型预测速度 v，走一步
        # x_0.9  shape=[B, 50, action_dim]  ← 50 帧一起变清晰一点点
        #   ↓
        # x_0.8  shape=[B, 50, action_dim]
        #   ↓
        # ...
        #   ↓
        # x_0.0  shape=[B, 50, action_dim]  ← 最终干净的 50 步动作序列

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # u_t 构造速度目标
        # u_t = ε - A：从干净动作指向噪声的"真实速度场"
        # 训练目标：让模型预测的 v_t 逼近 u_t（方向固定，所以 flow matching 比 diffusion 好学）
        # 推理时 dt<0 从 t=1 走到 t=0：x_{t+dt} = x_t + dt·v，两个负号抵消正好去噪
        u_t = noise - actions

        # === 4. 两套专家分别 embed（类 MoE：VLM backbone + Action Expert）===
        # prefix（VLM backbone / PaliGemma）：图像 + 语言 token
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        # suffix（Action Expert）：状态 + 噪声动作 x_t + 时间 t
        # adarms_cond：只有 pi0.5 有，时间经 MLP 变成 adaRMS 的条件信号；pi0 时为 None
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        # === 5. 精度对齐：模型是 bf16 的话，输入 embed 也转成 bf16 ===
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # === 6. 拼接 mask，构造两套专家共享的 attention 结构 ===
        # 在 token 维（dim=1）把 prefix 和 suffix 的 mask 拼起来
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # att_masks 是 1D 的"块分隔标志"，这里展开成 [B, L, L] 的 bool 注意力矩阵
        # 可见性规则（块编号越大能看越多，只能往前看）：
        #   块0（图像/语言）：只能看自己，看不到 state/action
        #   块1（state）    ：能看图像/语言，看不到 action
        #   块2（action）   ：能看图像/语言 + state + action（action 内部双向全注意力）
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # position_ids：按 pad_masks 累加，让 padding 位置不占位置编码
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # 把 bool mask 转成 transformer 要的 4D 加性 mask（True→0，False→-inf）
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # === 7. 联合 transformer 主 forward（VLM backbone + Action Expert 共享 attention）===
        # 包一层函数是为了能套 gradient checkpointing（显存换计算）
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            # inputs_embeds 是长度 2 的 list：[VLM backbone 输入, Action Expert 输入]
            # 两套专家权重独立，但共享 attention——每层 K/V 跨专家拼接，所以动作 token 的 Q 能看到图像/语言的 K/V
            # 返回 ([VLM backbone 输出, Action Expert 输出], past_kv)；训练不需要 cache
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],  # 只给 Action Expert 注入时间条件（pi0.5）
            )
            return suffix_out

        # 开了 gradient checkpointing 就走 checkpoint 省显存，否则直接调用
        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # === 8. 取动作部分的输出，投影回动作维度 ===
        # suffix_out 前面可能还有 state token（pi0），只要最后 H 个动作 token
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # 投影前转 float32，避免 bf16 数值精度不够
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            # [B, H, expert_width] -> [B, H, action_dim]
            return self.action_out_proj(suffix_out)

        # v_t：模型预测的速度场
        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # === 9. Flow Matching loss：让模型预测的速度逼近真实速度 ===
        # reduction="none" 返回逐元素 loss [B, H, action_dim]，外部可按 mask 加权求均值
        return F.mse_loss(u_t, v_t, reduction="none")

    # @torch.no_grad()：推理不需要梯度，省显存、省计算
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            # 初始化纯噪声 x_1 ~ N(0,1)，shape=[B, action_horizon, action_dim]
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # === 第一阶段：VLM 只跑一次，把图像+语言的 KV 缓存下来 ===
        # 推理时 VLM 的输入（图像/语言）不变，没必要每步都重新算
        # use_cache=True 让 transformers 把每层的 K/V 存进 past_key_values
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # inputs_embeds=[prefix_embs, None]：只跑 VLM，Expert 输入为 None，一次就好
        # 返回的 past_key_values 是所有层的 K/V cache，后面 10 步复用
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # === 第二阶段：Action Expert 迭代去噪 num_steps 步（默认 10 步）===
        # dt=-0.1，每步让 time 从 1.0 走到 0.0，即 x_1 → x_0.9 → ... → x_0
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise                                          # 从纯噪声 x_1 出发
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:                              # 循环直到 time ≈ 0
            expanded_time = time.expand(bsize)              # 标量 → [B]，每样本同一个 t
            # denoise_step：只跑 Action Expert，复用 VLM 的 past_key_values
            # 输入当前带噪动作 x_t 和时刻 t，输出预测速度 v_t
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler 积分一步：x_{t+dt} = x_t + dt * v_t
            # dt 为负，所以实际上是往 t 减小的方向走（去噪方向）
            x_t = x_t + dt * v_t
            time += dt

        # x_t 此时是 x_0，即完全去噪的 50 步动作序列 [B, action_horizon, action_dim]
        # 直接发给机器人执行
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # === 1. 把 state + x_t + timestep 编码成 suffix token 序列 ===
        # suffix_embs: [B, L_suf, expert_width]，包含 state token + 50 个 action token
        # timestep 通过 sine-cosine 编码后融入每个 action token
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # === 2. 构造推理时的 attention mask ===
        # 推理时 prefix（图像/语言）的 KV 已经缓存在 past_key_values 里
        # 但 attention mask 仍然需要告诉模型 suffix 的每个 token 能看到 prefix 的哪些位置
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # prefix_pad_2d_masks: [B, suffix_len, prefix_len]
        # suffix 的每个 token 都能看到 prefix 的全部有效位置（padding 除外）
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # suffix 内部的可见性：state 看不到 action，action 内部双向全注意力
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # 拼成完整的 2D mask：[B, suffix_len, prefix_len + suffix_len]
        # 列方向：前半段对应 prefix（KV cache），后半段对应 suffix 自身
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # === 3. 计算 position_ids ===
        # suffix token 的位置编号要接在 prefix 之后，保证 RoPE 的相对距离正确
        # prefix_offsets：每条样本 prefix 里有效 token 的数量
        # suffix 的 position_ids 从 prefix 结束的位置继续累加
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # === 4. Action Expert forward（复用 VLM 的 KV cache）===
        # inputs_embeds=[None, suffix_embs]：只跑 Action Expert，不重跑 VLM
        # past_key_values：VLM 预先算好的图像/语言 KV，每层 attention 时自动拼到前面
        # 这样 Expert 能"看到"图像语言，但 VLM 不需要重复计算，节省 10x 推理时间
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # === 5. 取 action token 的输出，投影成速度 v_t ===
        # outputs_embeds[1] 是 Action Expert 的输出 [B, L_suf, expert_width]
        # L_suf = state(1) + action(50)，只取最后 50 个 action token
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]   # [B, 50, expert_width]
        suffix_out = suffix_out.to(dtype=torch.float32)
        # action_out_proj: [B, 50, expert_width] -> [B, 50, action_dim]
        # 返回的就是 v_t，即"当前 x_t 应该往哪个方向走"
        return self.action_out_proj(suffix_out)
