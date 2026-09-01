from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from openpi import transforms as _transforms
from openpi.models import tokenizer as _tokenizer
from openpi.models.value_function import DistributionalValueModel
from openpi.models.value_model_config import ValueModelConfig


@struct.dataclass
class ValueTrainState:
    """可被标准 jax.jit 接收的纯 value 训练状态。"""

    step: jax.Array
    params: nnx.State
    model_def: nnx.GraphDef
    opt_state: optax.OptState


def _is_value_backbone(path: nnx.filterlib.PathParts, _value: Any) -> bool:
    return bool(path) and str(path[0]) == "PaliGemma"


def _trainable_filter(freeze_mode: str) -> nnx.filterlib.Filter:
    if freeze_mode == "none":
        return nnx.Param
    if freeze_mode == "backbones":
        return nnx.All(nnx.Param, nnx.Not(_is_value_backbone))
    raise ValueError(f"unsupported value freeze mode {freeze_mode!r}")


def _stack_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not observations:
        raise ValueError("cannot stack an empty observation list")
    keys = set(observations[0])
    if any(set(item) != keys for item in observations):
        raise ValueError("all value observations must have identical keys")
    result: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in observations]
        if isinstance(values[0], dict):
            nested_keys = set(values[0])
            if any(not isinstance(value, dict) or set(value) != nested_keys for value in values):
                raise ValueError(f"nested value observation field {key!r} has inconsistent keys")
            result[key] = {nested: np.stack([value[nested] for value in values]) for nested in nested_keys}
        else:
            if any(isinstance(value, dict) for value in values):
                raise ValueError(f"value observation field {key!r} mixes mapping and array values")
            result[key] = np.stack(values)
    return result


class ValueObservationEncoder:
    """将 episode JSON 的 canonical observation 转为 value VLM 的数值输入。"""

    def __init__(self, config: ValueModelConfig):
        self.config = config
        self._tokenizer: _tokenizer.PaligemmaTokenizer | None = None

    def _tokenize(self, prompt: Any) -> tuple[np.ndarray, np.ndarray]:
        if self._tokenizer is None:
            self._tokenizer = _tokenizer.PaligemmaTokenizer(self.config.max_token_len)
        if not isinstance(prompt, str):
            prompt = str(np.asarray(prompt).item())
        tokens, mask = self._tokenizer.tokenize(prompt)
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=np.bool_)

    def encode(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = observation.get("image", observation.get("images", {}))
        image_masks = observation.get("image_mask", observation.get("image_masks", {}))
        if "state" not in observation:
            raise ValueError("value observation must contain state")
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.ndim != 1 or not np.all(np.isfinite(state)):
            raise ValueError("each value observation state must be a finite one-dimensional array")
        if not isinstance(images, dict) or not isinstance(image_masks, dict):
            raise TypeError("value observation image and image_mask must be dictionaries")
        if not images:
            # Value VLM 必须观察视觉输入; 静默退化成纯状态/语言模型会破坏论文中的 critic 语义。
            raise ValueError("value observation must contain at least one image")
        selected_images = {}
        for key in sorted(images)[: self.config.num_cameras]:
            selected_images[key] = _transforms.canonicalize_recap_image(images[key], key=key)
        selected_masks = {}
        for key in selected_images:
            mask = np.asarray(image_masks.get(key, True), dtype=np.bool_)
            if mask.size != 1:
                raise ValueError(f"value image mask {key!r} must contain one flag, got {mask.shape}")
            # LeRobot 有时返回 shape [1]; 单帧编码统一为标量, batch stack 后才得到 [B]。
            selected_masks[key] = mask.reshape(())
        if "tokenized_prompt" in observation or "tokenized_prompt_mask" in observation:
            if "tokenized_prompt" not in observation or "tokenized_prompt_mask" not in observation:
                raise ValueError("tokenized_prompt and tokenized_prompt_mask must be supplied together")
            tokens = np.asarray(observation["tokenized_prompt"], dtype=np.int32)
            token_mask = np.asarray(observation["tokenized_prompt_mask"], dtype=np.bool_)
        else:
            if "prompt" not in observation:
                raise ValueError("value observation must contain prompt or tokenized_prompt fields")
            tokens, token_mask = self._tokenize(observation["prompt"])
        if tokens.shape != (self.config.max_token_len,) or token_mask.shape != tokens.shape:
            raise ValueError(
                f"value prompt tokens must have shape {(self.config.max_token_len,)}, got {tokens.shape}"
            )
        return {
            "image": selected_images,
            "image_mask": selected_masks,
            "state": state,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
        }

    def encode_batch(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        return _stack_observations([self.encode(observation) for observation in observations])


class ValueTrainer:
    """独立的 distributional value 训练循环, 便于在线 runner 注入或替换。"""

    def __init__(self, config: ValueModelConfig, *, seed: int = 0):
        self.config = config
        self.seed = seed
        self.encoder = ValueObservationEncoder(config)
        self.tx = optax.adam(config.learning_rate)
        model = DistributionalValueModel(config, rngs=nnx.Rngs(jax.random.key(seed)))
        params = nnx.state(model)
        opt_state = self.tx.init(params.filter(_trainable_filter(config.freeze_mode)))
        self._initialize_runtime(model, opt_state, step=0)

    def _initialize_runtime(self, model: DistributionalValueModel, opt_state: optax.OptState, *, step: int) -> None:
        self.state = ValueTrainState(
            step=jnp.asarray(step, dtype=jnp.int32),
            params=nnx.state(model),
            model_def=nnx.graphdef(model),
            opt_state=opt_state,
        )
        trainable_filter = _trainable_filter(self.config.freeze_mode)
        tx = self.tx
        clip_norm = self.config.gradient_clip_norm

        def train_fn(state: ValueTrainState, batch: dict[str, Any], targets: jax.Array):
            model = nnx.merge(state.model_def, state.params)

            def loss_fn(value_model):
                return value_model.compute_loss(batch, targets, train=True)

            diff_state = nnx.DiffState(0, trainable_filter)
            loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)
            grad_norm = optax.global_norm(grads)
            # 在 optimizer 外裁剪, 保持旧版 Adam checkpoint 的 opt-state 结构兼容。
            scale = jnp.minimum(1.0, clip_norm / (grad_norm + 1e-6))
            grads = jax.tree.map(lambda grad: grad * scale, grads)
            params = state.params.filter(trainable_filter)
            updates, opt_state = tx.update(grads, state.opt_state, params)
            nnx.update(model, optax.apply_updates(params, updates))
            return state.replace(
                step=state.step + 1,
                params=nnx.state(model),
                opt_state=opt_state,
            ), loss, grad_norm

        def eval_fn(params: nnx.State, model_def: nnx.GraphDef, batch: dict[str, Any], targets: jax.Array):
            return nnx.merge(model_def, params).compute_loss(batch, targets, train=False)

        def predict_fn(params: nnx.State, model_def: nnx.GraphDef, batch: dict[str, Any]):
            return nnx.merge(model_def, params).compute_value(batch)

        # 纯 state JIT 避免 nnx.jit 对大模型的额外可变状态开销。
        self._train_jit = jax.jit(train_fn)
        self._eval_jit = jax.jit(eval_fn)
        self._predict_jit = jax.jit(predict_fn)
        self.last_grad_norm: float | None = None

    @property
    def model(self) -> DistributionalValueModel:
        return nnx.merge(self.state.model_def, self.state.params)

    @property
    def opt_state(self) -> optax.OptState:
        return self.state.opt_state

    @property
    def step(self) -> int:
        return int(np.asarray(jax.device_get(self.state.step)))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        seed: int = 0,
        restore_optimizer: bool = True,
        learning_rate: float | None = None,
        gradient_clip_norm: float | None = None,
        freeze_mode: str | None = None,
    ) -> ValueTrainer:
        """恢复 value checkpoint; 新一轮 fine-tune 可显式重建 optimizer。"""
        path = Path(path).resolve()
        model, config, step = load_value_checkpoint(path, seed=seed)
        optimization_overrides = {
            key: value
            for key, value in {
                "learning_rate": learning_rate,
                "gradient_clip_norm": gradient_clip_norm,
                "freeze_mode": freeze_mode,
            }.items()
            if value is not None
        }
        if optimization_overrides:
            config = dataclasses.replace(config, **optimization_overrides)
            # 这些字段只影响 trainer, 但同步到模型配置以确保后续 checkpoint 自描述。
            model.config = config
        trainer = cls.__new__(cls)
        trainer.config = config
        trainer.seed = seed
        trainer.encoder = ValueObservationEncoder(config)
        trainer.tx = optax.adam(config.learning_rate)
        params = nnx.state(model)
        expected_opt_state = trainer.tx.init(params.filter(_trainable_filter(config.freeze_mode)))
        if not restore_optimizer:
            trainer._initialize_runtime(model, expected_opt_state, step=0)  # noqa: SLF001
            return trainer
        try:
            # 提供目标树让 Orbax 重建 Optax namedtuple; 无 target 时会退化成 list/dict。
            restored = ocp.PyTreeCheckpointer().restore(
                path / "params",
                item={"params": nnx.state(model, nnx.Param).to_pure_dict(), "opt_state": expected_opt_state},
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("value checkpoint does not contain a compatible optimizer state") from exc
        restored_opt_state = restored["opt_state"]
        # 让损坏或不匹配的 checkpoint 在恢复阶段失败, 而不是在下一次 update 才失败。
        def _check_shape(expected: Any, actual: Any) -> None:
            if getattr(expected, "shape", None) != getattr(actual, "shape", None):
                raise ValueError("optimizer state shape mismatch")

        try:
            jax.tree_util.tree_map(_check_shape, expected_opt_state, restored_opt_state)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("optimizer state shape mismatch; cannot resume training") from exc
        trainer._initialize_runtime(model, restored_opt_state, step=step)  # noqa: SLF001
        return trainer

    def train_step(self, observations: list[dict[str, Any]], targets: np.ndarray) -> float:
        if not observations:
            raise ValueError("observations must be non-empty")
        batch = self.encoder.encode_batch(observations)
        targets = jnp.asarray(targets, dtype=jnp.float32)
        if targets.ndim == 0:
            targets = targets[None]
        if targets.shape[0] != len(observations):
            raise ValueError("targets must contain one value per observation")

        self.state, loss, grad_norm = self._train_jit(self.state, batch, targets)
        self.last_grad_norm = float(grad_norm)
        return float(loss)

    def fit(self, observations: list[dict[str, Any]], targets: np.ndarray, *, steps: int, batch_size: int = 32) -> list[float]:
        if len(observations) != len(targets) or not observations:
            raise ValueError("observations and targets must be non-empty and have equal length")
        if steps < 1 or batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        rng = np.random.default_rng(self.seed)
        history = []
        for _ in range(steps):
            indices = rng.integers(0, len(observations), size=min(batch_size, len(observations)))
            history.append(self.train_step([observations[int(index)] for index in indices], targets[indices]))
        return history

    def predict(self, observations: list[dict[str, Any]], *, batch_size: int = 32) -> np.ndarray:
        if not observations:
            return np.empty(0, dtype=np.float32)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        predictions = []
        for start in range(0, len(observations), batch_size):
            batch = self.encoder.encode_batch(observations[start : start + batch_size])
            predictions.append(np.asarray(self._predict_jit(self.state.params, self.state.model_def, batch)))
        return np.concatenate(predictions).astype(np.float32, copy=False)

    def evaluate(self, observations: list[dict[str, Any]], targets: np.ndarray, *, batch_size: int = 32) -> float:
        """在独立 episode split 上计算 distributional cross-entropy, 不更新参数。"""
        if len(observations) != len(targets) or not observations:
            raise ValueError("observations and targets must be non-empty and have equal length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        weighted_loss = 0.0
        for start in range(0, len(observations), batch_size):
            end = min(start + batch_size, len(observations))
            batch = self.encoder.encode_batch(observations[start:end])
            loss = self._eval_jit(
                self.state.params,
                self.state.model_def,
                batch,
                jnp.asarray(targets[start:end], dtype=jnp.float32),
            )
            weighted_loss += float(loss) * (end - start)
        return weighted_loss / len(observations)


class ValuePredictor:
    """只恢复模型参数的轻量推理封装, label 阶段不依赖 optimizer state。"""

    def __init__(self, model: DistributionalValueModel, config: ValueModelConfig):
        self.encoder = ValueObservationEncoder(config)
        self._params = nnx.state(model)
        self._model_def = nnx.graphdef(model)

        def predict_fn(params: nnx.State, model_def: nnx.GraphDef, batch: dict[str, Any]):
            return nnx.merge(model_def, params).compute_value(batch)

        self._predict_jit = jax.jit(predict_fn)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, seed: int = 0) -> ValuePredictor:
        model, config, _ = load_value_checkpoint(path, seed=seed)
        return cls(model, config)

    def predict(self, observations: list[dict[str, Any]], *, batch_size: int = 32) -> np.ndarray:
        if not observations:
            return np.empty(0, dtype=np.float32)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        predictions = []
        for start in range(0, len(observations), batch_size):
            batch = self.encoder.encode_batch(observations[start : start + batch_size])
            predictions.append(np.asarray(self._predict_jit(self._params, self._model_def, batch)))
        return np.concatenate(predictions).astype(np.float32, copy=False)


def save_value_checkpoint(
    model: DistributionalValueModel,
    path: str | Path,
    config: ValueModelConfig,
    *,
    step: int = 0,
    opt_state: Any | None = None,
) -> None:
    """用 Orbax 保存纯参数和配置, label 阶段可在无训练进程时恢复。"""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint = {"params": nnx.state(model, nnx.Param).to_pure_dict()}
    if opt_state is not None:
        # 保存 optimizer state 使长 value 训练可以从同一 checkpoint 继续, 推理恢复仍只读取 params。
        checkpoint["opt_state"] = opt_state
    checkpointer.save(path / "params", checkpoint, force=True)
    (path / "config.json").write_text(
        json.dumps({"config": dataclasses.asdict(config), "step": step}, indent=2), encoding="utf-8"
    )


def load_value_checkpoint(path: str | Path, *, seed: int = 0) -> tuple[DistributionalValueModel, ValueModelConfig, int]:
    path = Path(path).resolve()
    metadata = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config = ValueModelConfig(**metadata["config"])
    model = DistributionalValueModel(config, rngs=nnx.Rngs(jax.random.key(seed)))
    restored = ocp.PyTreeCheckpointer().restore(path / "params")
    state = nnx.state(model, nnx.Param)
    # 兼容旧 checkpoint (直接保存参数树) 和新 checkpoint (附带 opt_state)。
    restored_params = restored["params"] if isinstance(restored, dict) and "params" in restored else restored
    state.replace_by_pure_dict(restored_params)
    nnx.update(model, state)
    return model, config, int(metadata.get("step", 0))
