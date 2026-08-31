# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class SpaceTimeSeparableBlock(nn.Module):
    """在空间 ViT block 后复用同一注意力参数做时间注意力。"""

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    num_timesteps: int = 1

    @nn.compact
    def __call__(self, x, deterministic=True, frame_mask=None):  # noqa: FBT002
        bk, n, d = x.shape
        k = self.num_timesteps
        if k < 1 or bk % k:
            raise ValueError(f"expected flattened (B*K, N, D) with K={k}, got {x.shape}")
        b = bk // k
        out = {}
        x = sharding.activation_sharding_constraint(x)

        # 参数名与标准 Encoder1DBlock 完全一致; temporal 分支复用同一参数。
        norm = nn.LayerNorm(name="LayerNorm_0", dtype=self.dtype_mm)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
            name="MultiHeadDotProductAttention_0",
        )
        normalized = norm(x)
        y = out["sa"] = attn(normalized, normalized)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            name="MlpBlock_0",
        )(nn.LayerNorm(name="LayerNorm_1", dtype=self.dtype_mm)(x), deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        if k == 1:
            return x, out

        # 每个 patch 沿时间轴做 causal attention, 同时屏蔽 episode 起始 padding。
        x_t = x.reshape(b, k, n, d).transpose(0, 2, 1, 3).reshape(b * n, k, d)
        x_t = x_t + self._sinusoidal_time_emb(k, d).astype(x_t.dtype)[None]
        causal = jnp.tril(jnp.ones((k, k), dtype=jnp.bool_))[None]
        if frame_mask is None:
            valid = jnp.ones((b, k), dtype=jnp.bool_)
        else:
            valid = jnp.asarray(frame_mask, dtype=jnp.bool_)
            if valid.shape != (b, k):
                raise ValueError(f"frame_mask must have shape {(b, k)}, got {valid.shape}")
        valid = jnp.repeat(valid, n, axis=0)
        valid_queries = valid[:, :, None]
        valid_keys = valid[:, None, :]
        padding_self = ~valid_queries & jnp.eye(k, dtype=jnp.bool_)[None]
        temporal_mask = (causal & ((valid_queries & valid_keys) | padding_self))[:, None]
        temporal_normalized = norm(x_t)
        y_t = out["temporal"] = attn(temporal_normalized, temporal_normalized, mask=temporal_mask)
        x_t = x_t + nn.Dropout(rate=self.dropout)(y_t, deterministic)
        x = x_t.reshape(b, n, k, d).transpose(0, 2, 1, 3).reshape(bk, n, d)
        out["+temporal"] = x
        return sharding.activation_sharding_constraint(x), out

    @staticmethod
    def _sinusoidal_time_emb(k: int, d: int):
        """生成固定时间位置编码, 并将当前帧(最后位置)平移到零。"""
        if d % 2:
            raise ValueError(f"embedding dimension must be even for temporal embedding, got {d}")
        half = d // 2
        # 输入按最旧到当前排列, 因此直接使用负的相对时间坐标 t in [-(K-1), 0]。
        positions = jnp.arange(k, dtype=jnp.float32) - (k - 1)
        freq = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / jnp.maximum(half - 1, 1))
        embedding = jnp.concatenate(
            [jnp.sin(positions[:, None] * freq[None, :]), jnp.cos(positions[:, None] * freq[None, :])], axis=-1
        )
        # cos(0)=1, 减去当前帧常量后满足论文边界条件 e(0)=0。
        return embedding - embedding[-1:]


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    num_timesteps: int = 1
    temporal_attn_every: int = 4
    drop_history_after_layer: int = -4

    @nn.compact
    def __call__(self, x, deterministic=True, frame_mask=None):  # noqa: FBT002
        out = {}
        k = self.num_timesteps
        use_video = k > 1

        if not use_video and self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            if use_video and self.temporal_attn_every < 1:
                raise ValueError("temporal_attn_every must be positive")
            drop_after = (
                self.depth + self.drop_history_after_layer
                if self.drop_history_after_layer < 0
                else self.drop_history_after_layer
            )
            if use_video and not 0 <= drop_after < self.depth:
                raise ValueError(f"drop_history_after_layer resolves to {drop_after}")
            for lyr in range(self.depth):
                if use_video and lyr % self.temporal_attn_every == 0:
                    block_cur = SpaceTimeSeparableBlock(
                        name=f"encoderblock_{lyr}",
                        dtype_mm=self.dtype_mm,
                        mlp_dim=self.mlp_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                        num_timesteps=k,
                    )
                    x, out[f"block{lyr:02d}"] = block_cur(x, deterministic, frame_mask=frame_mask)
                else:
                    block_cur = Encoder1DBlock(
                        name=f"encoderblock_{lyr}",
                        dtype_mm=self.dtype_mm,
                        mlp_dim=self.mlp_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                    )
                    x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
                if use_video and lyr == drop_after:
                    bk, n, d = x.shape
                    x = x.reshape(bk // k, k, n, d)[:, -1]
                    use_video = False
            out["pre_ln"] = x

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    num_timesteps: int = 1
    temporal_attn_every: int = 4
    drop_history_after_layer: int = -4

    @nn.compact
    def __call__(self, image, *, train=False, frame_mask=None):
        out = {}

        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        image = jnp.asarray(image, jnp.float32)

        # Patch extraction
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            num_timesteps=self.num_timesteps,
            temporal_attn_every=self.temporal_attn_every,
            drop_history_after_layer=self.drop_history_after_layer,
            name="Transformer",
        )(x, deterministic=not train, frame_mask=frame_mask)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        # 历史 token 可能已丢弃, 因此使用实际 batch 维。
        x_2d = jnp.reshape(encoded, [encoded.shape[0], h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
