# 載入 Prefix checkpoint 進行推論。

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from RL_main.generator import EmotionGenerator
from RL_main.utils import select_device


def checkpoint_metadata(
    checkpoint_path: str | Path,
    fallback_emotions: list[str],
) -> tuple[list[str], int]:
    # 情緒順序與 Prefix 長度以 checkpoint 為準。
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 Prefix checkpoint：{path}")
    payload = torch.load(path, map_location="cpu")
    # 讀取單獨的 Tensor 或 checkpoint 字典，形狀為 [emotion, prefix, hidden]。
    prefix = (
        payload.get("prefix_embeddings")
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 3:
        raise ValueError("checkpoint 必須包含三維 prefix_embeddings。")

    emotion_mapping = (
        payload.get("emotion2id")
        if isinstance(payload, dict)
        else None
    )
    # 若 checkpoint 有 emotion2id，就依儲存的索引還原標籤順序。
    if isinstance(emotion_mapping, dict) and emotion_mapping:
        try:
            emotions = [
                str(label).lower()
                for label, _ in sorted(
                    emotion_mapping.items(),
                    key=lambda item: int(item[1]),
                )
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint emotion2id 格式不合法。") from exc
    else:
        emotions = [str(item).lower() for item in fallback_emotions]

    if len(emotions) != prefix.shape[0]:
        raise ValueError(
            "checkpoint 情緒數與 prefix_embeddings 第一維不一致："
            f"{len(emotions)} != {prefix.shape[0]}"
        )
    return emotions, int(prefix.shape[1])


def load_generator(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> EmotionGenerator:
    # 先取得 checkpoint 的情緒順序與長度，再建立相同結構的生成器。
    path = Path(checkpoint_path).expanduser()
    model_config = config["model"]
    emotions, prefix_length = checkpoint_metadata(
        path,
        model_config["emotions"],
    )
    resolved_device = device or select_device()
    # MPS 推論使用 float16；Prefix 載入後在串接時轉成底模 dtype。
    model_dtype = torch.float16 if resolved_device.type == "mps" else torch.float32
    generator = EmotionGenerator(
        model_path=model_config["path"],
        emotion_labels=emotions,
        device=resolved_device,
        prefix_length=prefix_length,
        model_dtype=model_dtype,
    )
    if not generator.load_adapter(path):
        raise RuntimeError(f"Prefix checkpoint 載入失敗：{path}")
    # 推論模式不保留 Prefix 生成路徑的梯度。
    generator.eval()
    return generator
