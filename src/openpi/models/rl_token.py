import dataclasses
from typing import Any

import chex
from einops import einops
from flax import linen as nn
import jax
import jax.numpy as jnp

from openpi.models.utils.fsq_tokenizer import CrossAttentionLayer
from openpi.models.utils.fsq_tokenizer import GeGLU
from openpi.models.utils.fsq_tokenizer import sinusoidal_pe_init


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    num_rl_tokens: int = 1
    num_layers: int = 2
    embed_dim: int = 512  # Internal dimension of encoder-decoder (lightweight)
    input_dim: int = 2048  # VLA prefix embedding dimension (Gemma 2B hidden size)
    mlp_ratio: float = 4.0
    num_heads: int = 8
    dropout_rate: float = 0.0


class RLTokenEncoder(nn.Module):
    """Compresses VLA prefix embeddings [b, seq, input_dim] into RL tokens [b, num_rl_tokens, embed_dim]."""

    config: RLTokenConfig

    @nn.compact
    def __call__(
        self,
        prefix_embs: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        train: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config

        # Project from VLA dim to internal dim if they differ
        if cfg.input_dim != cfg.embed_dim:
            prefix_embs = nn.Dense(cfg.embed_dim, name="input_proj")(prefix_embs)

        x = self.param("q_embed", sinusoidal_pe_init, (cfg.num_rl_tokens, cfg.embed_dim))
        x = jnp.broadcast_to(x, prefix_embs.shape[:-2] + x.shape[-2:])

        if mask is not None:
            chex.assert_equal_shape([prefix_embs[..., 0], mask])
            attn_mask = einops.repeat(mask, "... kv -> ... 1 q kv", q=cfg.num_rl_tokens)
        else:
            attn_mask = jnp.ones((*prefix_embs.shape[:-2], 1, cfg.num_rl_tokens, prefix_embs.shape[-2]))

        y = prefix_embs + self.param("y_pos_enc", sinusoidal_pe_init, prefix_embs.shape[-2:])

        for _ in range(cfg.num_layers):
            x = CrossAttentionLayer(
                dropout_rate=cfg.dropout_rate,
                num_heads=cfg.num_heads,
                causal=False,
                mlp_ratio=cfg.mlp_ratio,
            )(x, y, train=train, mask_self=None, mask_cross=attn_mask)

        return x


class RLTokenDecoderLayer(nn.Module):
    """Causal self-attention block used by the autoregressive RLT decoder."""

    config: RLTokenConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        attention_mask: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config
        d_embed = x.shape[-1]

        # 论文/RLinf 的 decoder 是 causal self-attention; 原有 cross-attention 无法阻止读取未来 token.
        skip = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads or d_embed // 64,
            dropout_rate=cfg.dropout_rate,
            deterministic=not train,
        )(x, x, x, mask=attention_mask)
        x = skip + x

        skip = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(int(d_embed * cfg.mlp_ratio))(x)
        x = nn.Dropout(cfg.dropout_rate)(x, deterministic=not train)
        x = GeGLU()(x)
        x = nn.Dense(d_embed)(x)
        return skip + x


class RLTokenDecoder(nn.Module):
    """Autoregressively reconstructs prefix embeddings from RL tokens."""

    config: RLTokenConfig

    @nn.compact
    def __call__(
        self,
        rl_tokens: jnp.ndarray,
        target_embeddings: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        train: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config
        target_seq_len = target_embeddings.shape[-2]
        if target_seq_len < 1:
            raise ValueError("target_embeddings must contain at least one token")

        # mask 同时约束 teacher 输入和注意力, 避免 padding embedding 泄漏到有效位置.
        if mask is not None:
            chex.assert_equal_shape([target_embeddings[..., 0], mask])

        # RLT 论文要求 teacher forcing; 必须右移 detached target, 才能让位置 t 预测 target[t].
        shifted_targets = jax.lax.stop_gradient(target_embeddings[..., :-1, :])
        if mask is not None:
            shifted_targets = jnp.where(mask[..., :-1, None], shifted_targets, 0)
        if cfg.input_dim != cfg.embed_dim:
            shifted_targets = nn.Dense(cfg.embed_dim, name="teacher_input_proj")(shifted_targets)

        decoder_inputs = jnp.concatenate([rl_tokens, shifted_targets], axis=-2)
        decoder_input_len = decoder_inputs.shape[-2]
        # 位置编码对应 RL token 前缀加右移 target 的完整自回归序列, 替代原并行 query 编码.
        pos_enc = self.param("decoder_pos_enc", sinusoidal_pe_init, (decoder_input_len, cfg.embed_dim))
        decoder_inputs = decoder_inputs + pos_enc[:decoder_input_len]

        if mask is None:
            decoder_valid = jnp.ones(decoder_inputs.shape[:-1], dtype=jnp.bool_)
        else:
            rl_valid = jnp.ones((*mask.shape[:-1], cfg.num_rl_tokens), dtype=jnp.bool_)
            decoder_valid = jnp.concatenate([rl_valid, mask[..., :-1]], axis=-1)
        padding_mask = nn.make_attention_mask(decoder_valid, decoder_valid).astype(jnp.bool_)
        causal_mask = nn.make_causal_mask(decoder_inputs[..., 0]).astype(jnp.bool_)
        # Flax 不同版本的 attention mask dtype 可能不同, 显式转 bool 后才能安全合并两个约束.
        attention_mask = padding_mask & causal_mask

        x = decoder_inputs
        for _ in range(cfg.num_layers):
            x = RLTokenDecoderLayer(config=cfg)(x, attention_mask=attention_mask, train=train)

        # 最后一个 RL token 位置负责预测第一个 target, 保证输出仍为 [batch, target_seq_len, input_dim].
        target_start = cfg.num_rl_tokens - 1
        x = x[..., target_start : target_start + target_seq_len, :]

        # Project back to VLA dim if they differ
        if cfg.input_dim != cfg.embed_dim:
            x = nn.Dense(cfg.input_dim, name="output_proj")(x)

        return x


class RLTokenModel(nn.Module):
    """RL Token encoder-decoder: compresses VLA prefix embeddings into RL tokens via cross-attention."""

    config: RLTokenConfig

    def setup(self):
        self.encoder = RLTokenEncoder(config=self.config)
        self.decoder = RLTokenDecoder(config=self.config)

    def encode(
        self,
        prefix_embs: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        train: bool = True,
    ) -> jnp.ndarray:
        return self.encoder(prefix_embs, mask, train=train)

    def decode(
        self,
        rl_tokens: jnp.ndarray,
        target_embeddings: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        train: bool = True,
    ) -> jnp.ndarray:
        # decode 必须拿到真实 target 才能执行论文规定的 teacher forcing, 而不是仅按长度并行生成.
        return self.decoder(rl_tokens, target_embeddings, mask, train=train)

    def loss(
        self,
        prefix_embs: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        train: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        # 论文/RLinf 对 encoder 输入、teacher input 和重建目标都 stop-gradient, 避免 L_ro 反传进 VLA.
        target = jax.lax.stop_gradient(prefix_embs)
        rl_tokens = self.encode(target, mask, train=train)
        reconstructed = self.decode(rl_tokens, target, mask, train=train)

        sq_error = jnp.square(reconstructed - target)

        if mask is not None:
            mask_expanded = mask[..., None].astype(sq_error.dtype)
            masked_sq_error = sq_error * mask_expanded
            num_valid = jnp.sum(mask_expanded) * prefix_embs.shape[-1]
            mse = jnp.sum(masked_sq_error) / jnp.maximum(num_valid, 1.0)
        else:
            mse = jnp.mean(sq_error)

        return mse, {"mse": mse}

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Dummy for .init"""
        return self.loss(*args, **kwargs)
