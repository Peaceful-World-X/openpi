#!/usr/bin/env python3
"""调用 Anthropic VLM/LLM 离线生成 MEM language-memory JSONL。"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AnthropicMemoryClient:
    """仅在脚本执行时加载 Anthropic SDK, 避免污染 OpenPI 核心依赖。"""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Anthropic SDK is required: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                return str(text)
        raise RuntimeError("Anthropic response does not contain a text block")


def load_episodes(path: Path) -> list[dict[str, Any]]:
    """读取每行一个 episode 的 subtask_annotations.jsonl。"""
    episodes = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if "episode_id" not in item or not isinstance(item.get("subtasks"), list):
                raise ValueError(f"{path}:{line_number} must contain episode_id and subtasks")
            episodes.append(item)
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline MEM language-memory labels")
    parser.add_argument("--input", type=Path, required=True, help="Input subtask_annotations.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output memory_labels.jsonl")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_memory_len", type=int, default=512)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    return parser.parse_args()


def main() -> None:
    """生成标签; Anthropic 和核心生成器均延迟到 CLI 执行时导入。"""
    args = parse_args()
    if not args.api_key:
        raise ValueError("pass --api-key or set ANTHROPIC_API_KEY")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max_episodes must be positive")

    from openpi.models.memory_manager import MemoryDataGenerator
    from openpi.models.memory_manager import MemoryGenerationConfig

    episodes = load_episodes(args.input)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    generator = MemoryDataGenerator(
        AnthropicMemoryClient(args.api_key, args.model),
        MemoryGenerationConfig(max_memory_length=args.max_memory_len),
    )
    labels = []
    for index, episode in enumerate(episodes, 1):
        labels.extend(generator.generate_labels_for_episode(episode["episode_id"], episode["subtasks"]))
        logger.info("Generated episode %d/%d", index, len(episodes))
    generator.save_labels(labels, args.output)
    logger.info("Saved %d labels to %s", len(labels), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
