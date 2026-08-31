"""MEM 推理阶段的低频高层策略。"""

from collections.abc import Callable
import dataclasses
import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_HIGH_LEVEL_PROMPT = """\
You are controlling a robot to complete a task.

Task goal: {task_goal}

Memory of completed steps so far:
{language_memory}

Based on the current camera image and memory, choose the next short and concrete
robot subtask and update the compressed memory of successfully completed steps.
Do not include failed attempts. Respond with valid JSON only:
{{"subtask": "<next robot action>", "updated_memory": "<completed steps>"}}
"""


@dataclasses.dataclass(frozen=True)
class HighLevelPolicyConfig:
    """高层 VLM 的触发频率和输出长度配置。"""

    subtask_trigger_steps: int = 50
    use_completion_detection: bool = True
    max_subtask_tokens: int = 64
    max_memory_tokens: int = 256

    def __post_init__(self) -> None:
        for name in ("subtask_trigger_steps", "max_subtask_tokens", "max_memory_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class HighLevelPolicy:
    """封装可注入 VLM 回调, 并维护当前子任务和压缩语言记忆。"""

    def __init__(
        self,
        vlm_inference_fn: Callable[[np.ndarray, str, int], str],
        tokenizer: Any,
        config: HighLevelPolicyConfig | None = None,
    ) -> None:
        self._vlm_inference_fn = vlm_inference_fn
        self._tokenizer = tokenizer
        self.config = config or HighLevelPolicyConfig()
        self.reset()

    def reset(self, task_goal: str = "") -> None:
        """清空单个 episode 的高层状态。"""
        self._task_goal = task_goal
        self._language_memory = ""
        self._current_subtask = ""
        self._steps_since_update = 0

    def should_update(self, *, subtask_completed: bool = False) -> bool:
        """在完成信号到达或固定低层步数后请求一次高层更新。"""
        if subtask_completed and self.config.use_completion_detection:
            return True
        self._steps_since_update += 1
        return self._steps_since_update >= self.config.subtask_trigger_steps

    def update(self, observation_image: np.ndarray, *, subtask_success: bool = True) -> tuple[str, str]:
        """调用高层 VLM; 任何回调或 JSON 错误都保留原状态。"""
        prompt = _HIGH_LEVEL_PROMPT.format(
            task_goal=self._task_goal or "(not specified)",
            language_memory=self._language_memory or "(none yet)",
        )
        max_tokens = self.config.max_subtask_tokens + self.config.max_memory_tokens
        self._steps_since_update = 0
        try:
            raw = self._vlm_inference_fn(observation_image, prompt, max_tokens)
            parsed = json.loads(raw.strip())
            if not isinstance(parsed, dict):
                raise ValueError("VLM output must be a JSON object")
            subtask = parsed.get("subtask")
            updated_memory = parsed.get("updated_memory")
            if not isinstance(subtask, str) or not isinstance(updated_memory, str):
                raise ValueError("VLM JSON must contain string fields 'subtask' and 'updated_memory'")
        except Exception:
            logger.warning("MEM high-level update failed; keeping previous state", exc_info=True)
            return self._current_subtask, self._language_memory

        self._current_subtask = subtask.strip()
        # 失败尝试不能进入长期记忆, 否则会造成论文所述的训练/推理分布偏移。
        if subtask_success:
            self._language_memory = updated_memory.strip()
        return self._current_subtask, self._language_memory

    def tokenize_memory(self, memory: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """把语言记忆转换成固定长度的 token 和有效位 mask。"""
        text = self._language_memory if memory is None else memory
        max_len = self.config.max_memory_tokens
        if not text:
            return np.zeros(max_len, dtype=np.int32), np.zeros(max_len, dtype=np.bool_)

        supplied_mask = None
        if hasattr(self._tokenizer, "tokenize"):
            encoded = self._tokenizer.tokenize(text)
            if isinstance(encoded, tuple):
                encoded, supplied_mask = encoded[0], encoded[1] if len(encoded) > 1 else None
        elif hasattr(self._tokenizer, "encode"):
            encoded = self._tokenizer.encode(text)
        elif callable(self._tokenizer):
            encoded = self._tokenizer(text)
            if isinstance(encoded, tuple):
                encoded, supplied_mask = encoded[0], encoded[1] if len(encoded) > 1 else None
        else:
            raise TypeError("tokenizer must provide tokenize(), encode(), or be callable")

        ids = np.asarray(encoded, dtype=np.int32).reshape(-1)[:max_len]
        tokens = np.zeros(max_len, dtype=np.int32)
        mask = np.zeros(max_len, dtype=np.bool_)
        tokens[: len(ids)] = ids
        if supplied_mask is None:
            mask[: len(ids)] = True
        else:
            valid = np.asarray(supplied_mask, dtype=np.bool_).reshape(-1)[: len(ids)]
            mask[: len(valid)] = valid
        return tokens, mask

    @property
    def task_goal(self) -> str:
        return self._task_goal

    @property
    def current_subtask(self) -> str:
        return self._current_subtask

    @property
    def language_memory(self) -> str:
        return self._language_memory
