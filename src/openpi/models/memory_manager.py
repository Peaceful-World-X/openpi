"""离线生成和读写 MEM 长期语言记忆标签。"""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any

_MEMORY_PROMPT = """\
Summarize the successfully completed robot subtasks below. Keep only information
needed for future steps, omit failed attempts, use first-person past tense, and
stay under {max_length} characters.

Completed subtasks:
{history}

Compressed memory summary:
"""


@dataclasses.dataclass(frozen=True)
class MemoryLabel:
    """一个时刻的 memory 输入和执行后的 memory 目标。"""

    episode_id: str
    timestep: int
    subtask_instruction: str
    subtask_success: bool
    memory_before: str
    memory_after: str


@dataclasses.dataclass(frozen=True)
class MemoryGenerationConfig:
    """离线压缩回调参数。"""

    max_memory_length: int = 512
    temperature: float = 0.3
    max_tokens: int = 256

    def __post_init__(self) -> None:
        if self.max_memory_length < 1 or self.max_tokens < 1:
            raise ValueError("memory length and token limit must be positive")


class MemoryDataGenerator:
    """根据成功子任务序列生成逐时刻 JSONL 标签。"""

    def __init__(self, llm_client: Any, config: MemoryGenerationConfig | None = None) -> None:
        self._llm_client = llm_client
        self.config = config or MemoryGenerationConfig()

    def generate_labels_for_episode(
        self,
        episode_id: str,
        subtasks: Sequence[Mapping[str, Any]],
    ) -> list[MemoryLabel]:
        """失败步骤复用旧 memory, 且不会进入后续压缩历史。"""
        labels: list[MemoryLabel] = []
        successful_instructions: list[str] = []
        current_memory = ""
        for index, subtask in enumerate(subtasks):
            instruction = str(subtask.get("instruction", subtask.get("subtask", ""))).strip()
            if not instruction:
                raise ValueError(f"subtask {index} is missing instruction")
            success = bool(subtask.get("success", True))
            memory_before = current_memory
            if success:
                successful_instructions.append(instruction)
                current_memory = self._generate_memory(successful_instructions)
            # 允许标注沿用数据集帧号, 而不是强制使用列表序号。
            timestep = int(subtask.get("frame_index", subtask.get("timestep", index)))
            labels.append(
                MemoryLabel(
                    episode_id=str(episode_id),
                    timestep=timestep,
                    subtask_instruction=instruction,
                    subtask_success=success,
                    memory_before=memory_before,
                    memory_after=current_memory,
                )
            )
        return labels

    def _generate_memory(self, successful_instructions: Sequence[str]) -> str:
        history = "\n".join(f"Step {index}: {text}" for index, text in enumerate(successful_instructions, 1))
        prompt = _MEMORY_PROMPT.format(max_length=self.config.max_memory_length, history=history)
        if hasattr(self._llm_client, "generate"):
            result = self._llm_client.generate(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif callable(self._llm_client):
            result = self._llm_client(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            raise TypeError("llm_client must provide generate() or be callable")
        return str(result).strip()[: self.config.max_memory_length]

    @staticmethod
    def save_labels(labels: Iterable[MemoryLabel], output_path: str | Path) -> None:
        save_memory_labels(labels, output_path)

    @staticmethod
    def load_labels(input_path: str | Path) -> list[MemoryLabel]:
        return load_memory_labels(input_path)


def save_memory_labels(labels: Iterable[MemoryLabel], output_path: str | Path) -> None:
    """以 UTF-8 JSONL 保存标签。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(dataclasses.asdict(label), ensure_ascii=False) + "\n")


def load_memory_labels(input_path: str | Path) -> list[MemoryLabel]:
    """从 JSONL 严格恢复 MemoryLabel。"""
    labels: list[MemoryLabel] = []
    with Path(input_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                labels.append(MemoryLabel(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid memory label at line {line_number}") from exc
    return labels
