from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


# ═══════════════════════════════════════════════════════════════════════════
# pi0 / pi0.5 的"双专家共享 attention"主干
# ═══════════════════════════════════════════════════════════════════════════
# 架构关系（类 MoE，两个专家）：
#   ┌──────────────────────┐      ┌───────────────────────┐
#   │ VLM backbone         │      │ Action Expert         │
#   │ (PaliGemma, 3B)      │      │ (Gemma, 300M)         │
#   │ 处理 prefix:         │      │ 处理 suffix:          │
#   │   图像 + 语言        │      │   state + x_t + time  │
#   │ 权重预训练           │      │ 权重从头训            │
#   └──────────────────────┘      └───────────────────────┘
#             │                              │
#             └──── 共享 attention ──────────┘
#                   （每层 K/V 跨专家拼接）
#
# 关键设计：两个 nn.Module 权重完全独立，但在每一个 transformer 层里，
# attention 计算是"混合"的——Q/K/V 跨两个专家拼起来一起算 attention，
# 然后再按各自长度切回去走各自的 o_proj / MLP / LayerNorm。
# 这样动作 token 每一层都能看到 VLM 的 K/V，信息逐层融合，而不是串行两次 forward。
# ═══════════════════════════════════════════════════════════════════════════


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        # use_adarms = [vlm 是否用 adaRMS, expert 是否用 adaRMS]
        # pi0:   [False, False]（传统 RMSNorm）
        # pi0.5: [False, True]（只给 Action Expert 注入时间条件到 RMSNorm）
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # ─── 专家 1：VLM backbone（PaliGemma）的 HF 配置 ───
        # PaliGemma = SigLIP 视觉编码器 + Gemma 语言模型，3B 参数，Google 预训练权重
        # 用来把 "图像 + 语言指令" 编码成高维理解
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        # ─── 专家 2：Action Expert（纯 Gemma）的 HF 配置 ───
        # 只是一个小 Gemma 语言模型（~300M），从头训练
        # 专门处理 state / 噪声动作 / 时间，生成动作速度场 v_t
        # 注意：层数、head_dim 需要和 VLM 对齐，才能在 attention 里拼 Q/K/V
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # ─── 实例化两个完全独立的模型（权重不共享）───
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)  # VLM 专家
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)      # Action 专家
        # Action Expert 不做 token 查表（它吃的是连续动作向量，不是离散 token）
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    # ═══════════════════════════════════════════════════════════════════════
    # 主 forward：三种调用模式
    # ═══════════════════════════════════════════════════════════════════════
    # inputs_embeds 是长度 2 的 list = [prefix_embs, suffix_embs]
    #   1. inputs_embeds=[X, None]  → 只跑 VLM（推理时预缓存图像/语言的 K/V cache）
    #   2. inputs_embeds=[None, X]  → 只跑 Action Expert（推理去噪循环，复用上面的 K/V cache）
    #   3. inputs_embeds=[X, Y]     → 训练主路径，两个专家共享 attention 一起跑
    # ═══════════════════════════════════════════════════════════════════════
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        # ── 模式 1：只有 prefix，跑 VLM backbone 一次（推理时生成 K/V cache）──
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        # ── 模式 2：只有 suffix，跑 Action Expert（推理去噪循环，复用缓存的 VLM K/V）──
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        # ══════════════════════════════════════════════════════════════════
        # ── 模式 3：训练主路径 —— 两个专家共享 attention 一起跑每一层 ──
        # ══════════════════════════════════════════════════════════════════
        else:
            # 两个专家按顺序放进 list，后面 for i in [0, 1] 统一处理
            # [0] = VLM 的语言模型部分；[1] = Action Expert
            # 注意：VLM 需要层数和 Action Expert 一致，才能整齐地跨专家拼 Q/K/V
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # ══════════════════════════════════════════════════════════════
            # 核心：单层"双专家共享 attention"的完整计算
            # ══════════════════════════════════════════════════════════════
            # 每层的执行顺序：
            #   ① 各自 LayerNorm（权重独立）
            #   ② 各自 Q/K/V 投影（权重独立）
            #   ③ ⭐ 沿 token 维拼 Q/K/V（跨专家混合）→ 算一次 attention
            #   ④ 按长度切回去 → 各自 o_proj + residual
            #   ⑤ 各自 post_attention_layernorm + MLP + residual（权重独立）
            # ══════════════════════════════════════════════════════════════
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                # ─── ① + ② 各自做 LayerNorm + Q/K/V 投影（权重完全独立）───
                query_states = []
                key_states = []
                value_states = []
                gates = []  # adaRMS 的门控（pi0.5 用；pi0 为 None 则 LayerNorm 退化为普通 RMSNorm）
                for i, hidden_states in enumerate(inputs_embeds):
                    # 选专家 i 对应的这一层权重（i=0 是 VLM, i=1 是 Action Expert）
                    layer = models[i].layers[layer_idx]

                    # pre-norm：每个专家用自己的 input_layernorm
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    # Q/K/V 投影：每个专家用自己的 q_proj/k_proj/v_proj 权重
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # ─── ③ ⭐ 核心：跨专家拼 Q/K/V，共享 attention ───
                # 拼接前：query_states = [Q_vlm [B,H,L_pre,D], Q_exp [B,H,L_suf,D]]
                # 拼接后：Q = [B, H, L_pre+L_suf, D]
                # 这一步让两个专家的 token "混"在同一条序列里算 attention，
                # 是整个架构"共享注意力"的关键——从这里开始到 o_proj 之前，
                # Q/K/V 不再区分"哪个专家"，统一算一次 attention（由 2D mask 控制可见性）
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                # Rotary Position Embedding（RoPE）：用 VLM 的 rotary_emb 给拼起来的 Q/K 打位置编码
                # 两个专家共用同一套位置编码（position_ids 是全序列的），保证跨专家注意力坐标统一
                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # ⭐⭐⭐ 真正的 attention 机制就在这里 ⭐⭐⭐
                # 输入：拼接好的 Q/K/V（[B, H, L_pre+L_suf, D]）+ 2D 注意力掩码
                # 内部做：softmax(Q·K^T / √d + mask) · V
                # mask（att_2d_masks_4d）控制可见性（块编号大的能看小的，反之不行）：
                #   块0 图像/语言：只能互相看，看不到 state 和 action
                #   块1 state    ：能看图像/语言，看不到 action
                #   块2 action   ：能看图像/语言 + state，action 内部双向全注意力
                # 用的是 VLM 这一层的 self_attn 对象（但它在这里只是传参用，K/V/Q 已经拼好了）
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # ─── ④ 按长度切回去 + ⑤ 各自过自己的 o_proj / MLP / 残差 ───
                # 此处两个专家"重新分开"，各自走独立的权重：
                #   att_output[0:L_pre] → VLM 的 o_proj / MLP
                #   att_output[L_pre:]  → Action Expert 的 o_proj / MLP
                # 这种"共享 attention + 独立 FFN"的结构就是论文里说的
                # "mixture of experts with two mixture elements"（两元素的 MoE）
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    # 切出属于当前专家的那段 attention 输出，走自己的 o_proj
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # 第一次残差：h + attn(h)（pi0.5 带门控，pi0 普通残差）
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    # post-norm + MLP（又是各专家独立的权重）
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # 第二次残差：h + mlp(h)
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                # 返回 [VLM 这层输出, Expert 这层输出]，供下一层继续
                return outputs_embeds

            # ─── 循环所有层，每层都是"双专家共享 attention"的结构 ───
            # 开 gradient checkpointing 时走 torch 的 checkpoint（显存换计算）
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # ─── 所有层跑完后，各自过自己的 model.norm（最终 LayerNorm）───
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            # 分别拿到两个专家的最终输出：
            #   prefix_output：VLM 对 图像+语言 的最终理解（pi0 训练时直接丢弃不用）
            #   suffix_output：Action Expert 对 state+动作 的输出 → 进 action_out_proj 预测 v_t
            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
