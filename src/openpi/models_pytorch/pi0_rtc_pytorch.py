import torch
from torch import Tensor

from openpi.models_pytorch import pi0_pytorch as _pi0_pytorch


# 按论文算法 1 构造干净动作前缀、逐动作流时间和前缀掩码。
def _build_rtc_training_inputs(actions, noise, time, delay):
    action_horizon = actions.shape[-2]
    prefix_mask = torch.arange(action_horizon, device=actions.device)[None, :] < delay[:, None]

    # 论文令前缀 tau=1 (数据端); openpi 的流方向相反, 因此这里对应 time=0 (干净数据端)。
    token_time = torch.where(prefix_mask, 0.0, time[:, None])
    x_t = token_time[..., None] * noise + (1 - token_time[..., None]) * actions
    return x_t, token_time, prefix_mask


# 将前缀损失置零并补偿有效元素数量, 使训练脚本的 mean 只平均后缀。
def _mask_rtc_loss(loss, prefix_mask):
    postfix_mask = torch.logical_not(prefix_mask)[..., None].expand_as(loss)
    normalizer = loss.numel() / postfix_mask.sum().clamp_min(1)
    return loss * postfix_mask * normalizer


class PI0RTCPytorch(_pi0_pytorch.PI0Pytorch):
    """通过干净动作前缀实现 Training-Time RTC 的 PyTorch Pi0 子类。"""

    def __init__(self, config):
        super().__init__(config)
        # max_delay 只参与 RTC 控制流, 不注册参数或 buffer, 因而 state_dict 保持不变。
        self.max_delay = config.max_delay

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """计算逐 token 时间条件下的 RTC 后缀损失。"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        # 显式输入用于确定性诊断; 未提供时继续复用官方 noise 和 Beta 时间采样器。
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        return self._compute_loss_training_time_rtc(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )

    def _compute_loss_training_time_rtc(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
    ):
        """构造干净前缀并只保留动作后缀的训练梯度。"""
        # 论文允许选择延迟分布; 这里沿用当前实现的均匀整数分布 [0, max_delay)。
        delay = torch.randint(0, self.max_delay, (actions.shape[0],), device=actions.device)
        # 前缀保持干净, 后缀继续使用官方 Beta 时间和加噪方向。
        x_t, model_time, prefix_mask = _build_rtc_training_inputs(actions, noise, time, delay)
        loss = self._compute_loss_from_inputs(
            images, img_masks, lang_tokens, lang_masks, state, x_t, noise - actions, model_time
        )
        # 论文第 3 节要求梯度只来自动作后缀。
        return _mask_rtc_loss(loss, prefix_mask)

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        action_prefix=None,
        delay=None,
    ) -> Tensor:
        """在每个 Euler 步冻结已执行动作前缀。"""
        # 普通请求直接复用父类采样器, 保持标准 PyTorch Pi0 的数值路径。
        if action_prefix is None and delay is None:
            return super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
        if action_prefix is None or delay is None:
            raise ValueError("action_prefix and delay must be provided together")

        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # 固定协议形状使逐 token 条件与动作时域一一对应。
        expected_prefix_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        if action_prefix.shape != expected_prefix_shape:
            raise ValueError(f"Expected action_prefix shape {expected_prefix_shape}, got {action_prefix.shape}")
        if delay.shape != (bsize,):
            raise ValueError(f"Expected delay shape {(bsize,)}, got {delay.shape}")
        action_prefix = action_prefix.to(device=device, dtype=noise.dtype)
        delay = delay.to(device=device, dtype=torch.long)
        action_prefix_mask = torch.arange(self.config.action_horizon, device=device)[None, :] < delay[:, None]

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = _pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 前缀 KV cache 与标准采样器一致, RTC 只改变动作 suffix 的条件。
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            # 每个 Euler 步重新注入干净前缀, 防止已执行动作被去噪更新污染。
            x_t = torch.where(action_prefix_mask[..., None], action_prefix, x_t)
            expanded_time = torch.where(
                action_prefix_mask,
                0.0,
                time.expand(bsize, self.config.action_horizon),
            )
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt

        # 最后一次 Euler 更新仍会改变前缀, 因此返回前再次回填以严格保持冻结不变量。
        return torch.where(action_prefix_mask[..., None], action_prefix, x_t)
