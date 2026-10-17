# 整理 RL 資料，切分訓練與驗證樣本。

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.I)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
PLACEHOLDER_TEXTS = {"[deleted]", "[removed]", "deleted", "removed"}


@dataclass(frozen=True)
class PreparedRLData:
    train: pd.DataFrame
    validation: pd.DataFrame
    statistics: dict[str, Any]


def normalize_text_key(text: str) -> str:
    # 合併多餘空白並忽略大小寫，讓去重與排除使用同一個比對方式。
    return " ".join(str(text).split()).casefold()


def source_quality_failure(
    text: str,
    min_words: int,
    max_words: int,
) -> str | None:
    # 通過時回傳 None，否則回傳第一個失敗原因。
    stripped = " ".join(str(text).split()).strip()
    if not stripped:
        return "empty_text"
    if stripped.casefold() in PLACEHOLDER_TEXTS:
        return "placeholder_text"
    if URL_PATTERN.search(stripped):
        return "contains_url"
    word_count = len(WORD_PATTERN.findall(stripped))
    if word_count < min_words:
        return "too_short"
    if word_count > max_words:
        return "too_long"
    return None


def load_excluded_prompts(paths: Iterable[str | Path]) -> set[str]:
    # 讀取 Warm-up 用過的原句，之後從 RL 資料中排除。
    excluded: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"找不到資料排除檔：{path}")
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"資料排除檔為空：{path}")
        # 排除清單可使用 JSON 陣列或逐行 JSON。
        records = (
            json.loads(content)
            if content.startswith("[")
            else [json.loads(line) for line in content.splitlines() if line.strip()]
        )
        if not isinstance(records, list):
            raise ValueError(f"資料排除檔頂層必須是 list：{path}")
        found_in_file = 0
        for item in records:
            if not isinstance(item, dict):
                continue
            text = str(item.get("prompt", item.get("text", ""))).strip()
            if text:
                excluded.add(normalize_text_key(text))
                found_in_file += 1
        if found_in_file == 0:
            raise ValueError(f"資料排除檔沒有 prompt/text：{path}")
    return excluded


def stable_validation_member(text_key: str, seed: int, fraction: float) -> bool:
    # 用 SHA-256 固定分組；Python hash 會隨程序改變。
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction 必須位於 [0, 1)。")
    digest = hashlib.sha256(f"{seed}\0{text_key}".encode("utf-8")).digest()
    # 把雜湊前八個位元組映射到 [0, 1)，以比例門檻決定驗證集成員。
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


def prepare_rl_data(
    dataframe: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    source_emotions: list[str],
    target_emotion: str,
    supported_emotions: list[str],
    excluded_prompts: set[str] | None = None,
    min_words: int = 4,
    max_words: int = 30,
    validation_fraction: float = 0.10,
    split_seed: int = 42,
) -> PreparedRLData:
    # 統一來源與目標標籤，再清理文字、排除重複並切分資料。
    if min_words <= 0 or max_words < min_words:
        raise ValueError("資料詞數上下限不合法。")
    supported = {emotion.casefold() for emotion in supported_emotions}
    sources = {emotion.casefold() for emotion in source_emotions}
    target = target_emotion.casefold()
    if not sources <= supported:
        raise ValueError(f"未知來源情緒：{sorted(sources - supported)}")
    if target not in supported:
        raise ValueError(f"未知目標情緒：{target}")
    if target in sources:
        raise ValueError(
            "RL 必須是明確跨情緒轉換；target_emotion 不可同時是 source_emotion。"
        )

    excluded = excluded_prompts or set()
    statistics: Counter[str] = Counter()
    statistics["input_rows"] = len(dataframe)
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    # 只留下指定來源情緒，並累計各種排除原因。
    for _, row in dataframe.iterrows():
        source_emotion = str(row[label_column]).strip().casefold()
        if source_emotion not in sources:
            statistics["filtered_source_emotion"] += 1
            continue
        text = " ".join(str(row[text_column]).split()).strip()
        failure = source_quality_failure(text, min_words, max_words)
        if failure:
            statistics[f"filtered_{failure}"] += 1
            continue
        # 先排除 Warm-up 原句，再去掉這份 RL 資料本身的重複句。
        text_key = normalize_text_key(text)
        if text_key in excluded:
            statistics["filtered_warmup_overlap"] += 1
            continue
        if text_key in seen:
            statistics["filtered_duplicate"] += 1
            continue
        seen.add(text_key)
        records.append(
            {
                "text": text,
                "source_emotion": source_emotion,
                "target_emotion": target,
                "text_key": text_key,
            }
        )

    if not records:
        raise ValueError("RL 資料清理後沒有可用來源句。")
    cleaned = pd.DataFrame.from_records(records)
    # 每個句子的分組由內容與種子決定，與資料排列順序無關。
    validation_mask = cleaned["text_key"].map(
        lambda key: stable_validation_member(key, split_seed, validation_fraction)
    )
    validation = cleaned.loc[validation_mask].reset_index(drop=True)
    train = cleaned.loc[~validation_mask].reset_index(drop=True)
    if train.empty:
        raise ValueError("RL train split 為空。")
    if validation_fraction > 0.0 and validation.empty:
        raise ValueError("RL validation split 為空；請增加資料或比例。")

    if set(train["text_key"]) & set(validation["text_key"]):
        raise RuntimeError("RL train 與 validation 發生文本重疊。")
    # 附上清理後筆數與分組統計，text_key 只供內部比對，不放進訓練輸出。
    statistics["clean_unique_rows"] = len(cleaned)
    statistics["train_rows"] = len(train)
    statistics["validation_rows"] = len(validation)
    statistics["excluded_prompt_count"] = len(excluded)
    statistics["split_seed"] = split_seed
    return PreparedRLData(
        train=train.drop(columns=["text_key"]),
        validation=validation.drop(columns=["text_key"]),
        statistics=dict(statistics),
    )
