# 將 GoEmotions 的情緒標籤合併為六類。

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL_main.utils import load_config, resolve_path, save_json
from warmup_training.emotion_mapping import normalize_emotion


CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_records(file_path: str | Path) -> list[dict[str, Any]]:
    # 依檔案內容讀取 JSON 陣列或 JSONL。
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 GoEmotions 資料：{path}")
    content = path.read_text(encoding="utf-8").strip()
    if content.startswith("["):
        data = json.loads(content)
    else:
        data = [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]
    if not isinstance(data, list):
        raise ValueError("GoEmotions JSON 頂層必須是 list。")
    return [item for item in data if isinstance(item, dict)]


def convert_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    # 多標籤取第一項，再合併為六類情緒。
    converted: list[dict[str, Any]] = []
    statistics: Counter[str] = Counter()
    # 逐筆檢查文字與標籤，不能映射的資料另外計數。
    for item in records:
        text = str(item.get("text", "")).strip()
        source_label = item.get("emotions", item.get("emotion"))
        emotion = normalize_emotion(source_label)
        if not text:
            statistics["skipped_empty_text"] += 1
            continue
        if emotion is None:
            statistics["skipped_unknown_emotion"] += 1
            continue
        converted.append({"text": text, "emotions": [emotion]})
        statistics[emotion] += 1
    return converted, statistics


def main() -> None:
    # 依 raw_path 讀入原始資料，轉換後寫到 data.path。
    config = load_config(CONFIG_PATH)
    data_config = config["data"]
    input_path = resolve_path(data_config["raw_path"], PROJECT_ROOT)
    output_path = resolve_path(data_config["path"], PROJECT_ROOT)
    records = load_records(input_path)
    converted, statistics = convert_records(records)
    if not converted:
        raise ValueError("沒有任何資料成功映射，拒絕寫入空輸出。")
    save_json(converted, output_path)
    print(f"Converted {len(converted)}/{len(records)} records to {output_path}")
    for key in sorted(statistics):
        print(f"{key}: {statistics[key]}")


if __name__ == "__main__":
    main()
