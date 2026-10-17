# 抽取中性原句，分配 Warm-up 的目標情緒。

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL_main.utils import load_config, load_dataset, resolve_path, save_json
from warmup_training.emotion_mapping import normalize_emotion


CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def sample_prompts(
    dataframe: pd.DataFrame,
    source_emotion: str,
    samples_per_target_emotion: dict[str, Any],
    seed: int,
) -> list[dict[str, str]]:
    # 每個原句只分配給一種目標情緒，不跨組重複。
    working = dataframe.copy()
    # 先轉成六類標籤，再只取指定來源情緒。
    working["main_emotion"] = working["emotion"].map(normalize_emotion)
    working = working.dropna(subset=["main_emotion"])
    if working.empty:
        raise ValueError("GoEmotions 映射後沒有可用資料。")

    source = str(source_emotion).strip().lower()
    source_rows = working[working["main_emotion"] == source].copy()
    source_rows["text"] = source_rows["text"].map(
        lambda value: " ".join(str(value).split()).strip()
    )
    source_rows = source_rows[source_rows["text"] != ""]
    # 忽略大小寫去重，避免同一句話分配到不同目標組。
    source_rows["text_key"] = source_rows["text"].str.casefold()
    source_rows = source_rows.drop_duplicates("text_key")
    if source_rows.empty:
        raise ValueError(f"來源情緒 {source!r} 沒有可用資料。")

    target_counts: list[tuple[str, int]] = []
    for emotion, raw_count in samples_per_target_emotion.items():
        target = str(emotion).strip().lower()
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"目標情緒 {target} 的抽樣數必須大於 0。")
        if target == source:
            raise ValueError("Warm-up 目標情緒不可等於來源情緒。")
        target_counts.append((target, count))

    # 各組所需原句數加總後，一次做不放回抽樣。
    total_count = sum(count for _, count in target_counts)
    if len(source_rows) < total_count:
        raise ValueError(
            f"來源情緒 {source} 只有 {len(source_rows)} 個唯一句子，"
            f"不足以供 {total_count} 個跨情緒 prompt 使用。"
        )
    shuffled = source_rows.sample(
        n=total_count,
        replace=False,
        random_state=seed,
    ).reset_index(drop=True)
    output: list[dict[str, str]] = []
    offset = 0
    # 將洗牌後的原句依各組需求切成連續區段。
    for target, count in target_counts:
        chunk = shuffled.iloc[offset : offset + count]
        output.extend(
            {
                "prompt": str(row["text"]),
                "source_emotion": source,
                "emotion": target,
            }
            for _, row in chunk.iterrows()
        )
        offset += count
    return output


def main() -> None:
    # 讀取設定與來源資料，輸出下一步 API 要使用的 prompt 清單。
    config = load_config(CONFIG_PATH)
    generation_config = config.get("warmup", {}).get("generate", {})
    source_path = resolve_path(
        generation_config.get("source_path", config["data"]["path"]),
        PROJECT_ROOT,
    )
    output_path = resolve_path(generation_config["input_path"], PROJECT_ROOT)
    dataframe = load_dataset(source_path, "text", "emotion")
    samples_per_target_emotion = generation_config.get(
        "samples_per_target_emotion",
        {
            "joy": 120,
            "sadness": 120,
            "anger": 120,
            "fear": 120,
            "surprise": 120,
        },
    )
    if not isinstance(samples_per_target_emotion, dict):
        raise ValueError("samples_per_target_emotion 必須是 mapping。")
    supported_emotions = {
        str(emotion).strip().lower()
        for emotion in config.get("model", {}).get("emotions", [])
    }
    target_emotions = {
        str(emotion).strip().lower()
        for emotion in samples_per_target_emotion
    }
    unsupported = target_emotions - supported_emotions
    if unsupported:
        raise ValueError(f"Warm-up 目標情緒不在 model.emotions：{unsupported}")
    output_data = sample_prompts(
        dataframe,
        str(generation_config.get("source_emotion", "neutral")),
        samples_per_target_emotion,
        int(generation_config.get("sample_seed", 42)),
    )
    save_json(output_data, output_path)
    print(f"Warm-up prompts: {len(output_data)} samples, output={output_path}")


if __name__ == "__main__":
    main()
