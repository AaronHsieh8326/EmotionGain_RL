# GoEmotions 的 28 類標籤對應到六類情緒。

from __future__ import annotations

from typing import Any


# 依研究分組將細分類合併，例如 admiration、gratitude 歸到 joy。
EMOTION_MAPPING = {
    "joy": "joy",
    "excitement": "joy",
    "amusement": "joy",
    "admiration": "joy",
    "approval": "joy",
    "love": "joy",
    "gratitude": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "caring": "joy",
    "desire": "joy",
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "embarrassment": "fear",
    "surprise": "surprise",
    "curiosity": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "neutral": "neutral",
}


def normalize_emotion(value: Any) -> str | None:
    # 多標籤取第一項，回傳六類情緒之一；未知標籤回傳 None。
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    label = str(value).split(",", 1)[0].strip().lower()
    return EMOTION_MAPPING.get(label)
