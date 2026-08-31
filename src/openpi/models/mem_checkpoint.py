"""标准单帧 checkpoint 到 MEM 非 scan 视觉编码器的迁移工具。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flax import traverse_util
import numpy as np


def migrate_scan_params(
    source_params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    *,
    allow_missing_lora: bool = False,
) -> dict:
    """按层拆分 scan 参数, 并仅以 target 初始化值填充明确允许的新参数。"""
    source = traverse_util.flatten_dict(source_params, sep="/")
    target = traverse_util.flatten_dict(target_params, sep="/")
    output = {}
    consumed_source_keys = set()
    for key, target_value in target.items():
        candidate = source.get(key)
        if candidate is not None:
            consumed_source_keys.add(key)
        if candidate is None:
            # 按路径段解析, 兼容完整 Pi0 树和单独 SigLIP 参数树的顶层命名。
            path_parts = key.split("/")
            block_indices = [
                index
                for index, part in enumerate(path_parts)
                if part.startswith("encoderblock_") and part[len("encoderblock_") :].isdigit()
            ]
            if block_indices:
                block_index = block_indices[-1]
                prefix_parts = path_parts[:block_index]
                layer_text = path_parts[block_index][len("encoderblock_") :]
                suffix_parts = path_parts[block_index + 1 :]
                stacked_key = "/".join([*prefix_parts, "encoderblock", *suffix_parts])
                stacked = source.get(stacked_key)
                if stacked is not None:
                    consumed_source_keys.add(stacked_key)
                    layer = int(layer_text)
                    stacked_shape = tuple(np.shape(stacked))
                    if not stacked_shape or layer >= stacked_shape[0]:
                        raise ValueError(f"cannot extract layer {layer} from checkpoint leaf {stacked_key}")
                    # 直接切片以保留 NumPy/JAX 后端, 避免把完整视觉权重回拷到主机。
                    candidate = stacked[layer]
        if candidate is None:
            # state-history 是 MEM 相对标准 checkpoint 唯一新增的模型参数;
            # LoRA 仅允许训练 WeightLoader 按原有约定显式补齐。
            allow_initialized_value = "state_history_proj" in key or (allow_missing_lora and "lora" in key.lower())
            if not allow_initialized_value:
                raise ValueError(f"checkpoint is missing required parameter {key}")
            output[key] = target_value
            continue
        candidate_shape = tuple(np.shape(candidate))
        target_shape = tuple(target_value.shape) if hasattr(target_value, "shape") else tuple(np.shape(target_value))
        if candidate_shape != target_shape:
            raise ValueError(f"checkpoint shape mismatch for {key}: source {candidate_shape}, target {target_shape}")
        output[key] = candidate
    unused_source_keys = sorted(set(source) - consumed_source_keys)
    if unused_source_keys:
        preview = ", ".join(unused_source_keys[:5])
        suffix = " ..." if len(unused_source_keys) > 5 else ""
        raise ValueError(f"checkpoint contains unexpected parameters: {preview}{suffix}")
    return traverse_util.unflatten_dict(output, sep="/")


def migrate_checkpoint_params(source_params: Mapping[str, Any], target_params: Mapping[str, Any]) -> dict:
    """公开别名, 便于训练脚本调用。"""
    return migrate_scan_params(source_params, target_params)


def load_mem_checkpoint(config, params: Mapping[str, Any]):
    """通过配置对象加载 MEM 参数; 标准 checkpoint 的迁移由 ``config.load`` 完成。"""
    return config.load(params)
