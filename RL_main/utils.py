# 設定檔、資料讀寫與日誌工具。

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    # 讀入 YAML；空檔案回傳空字典，其他頂層型別不接受。
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到設定檔：{path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"設定檔頂層必須是 mapping：{path}")
    return config


def save_json(data: Any, file_path: str | Path, *, indent: int = 2) -> None:
    # 建立輸出目錄後寫入 JSON，中文以原字元保存。
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)


def load_json(file_path: str | Path) -> Any:
    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed: int) -> None:
    # 分別設定 Python、NumPy 與 PyTorch 的亂數種子。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # CUDA 關閉自動挑選 cuDNN 演算法，使用可重現的設定。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def resolve_path(path_value: str | Path, base_dir: str | Path) -> Path:
    # 絕對路徑原樣使用，相對路徑接在指定的專案目錄後。
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else Path(base_dir) / path


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    # 同名 logger 不重複加入相同的輸出 handler。
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not any(getattr(handler, "_textgan_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler._textgan_console = True  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    # 檔案輸出依完整路徑辨識，避免同一個日誌被重複掛上。
    if log_file is not None:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        has_same_file = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == log_path
            for handler in logger.handlers
        )
        if not has_same_file:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def _first_label(value: Any) -> str | None:
    # 多標籤資料取第一個標籤。
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def load_dataset(
    file_path: str | Path,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    # 接受 JSON array 或 JSONL，並對應 text／emotion 的欄位別名。
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到資料集：{path}")

    dataframe: pd.DataFrame | None = None
    errors: list[str] = []
    # 先嘗試 JSONL，失敗後再以 JSON 陣列讀取。
    for as_lines in (True, False):
        try:
            dataframe = pd.read_json(path, orient="records", lines=as_lines)
            break
        except ValueError as exc:
            errors.append(f"lines={as_lines}: {exc}")

    if dataframe is None:
        raise ValueError(f"無法讀取 JSON/JSONL：{'; '.join(errors)}")

    # 讓 prompt／emotions 等常見欄位名對應到設定檔要求的名稱。
    aliases = {
        text_column: ("text", "prompt"),
        label_column: ("emotion", "emotions", "label", "target"),
    }
    for expected, candidates in aliases.items():
        if expected in dataframe.columns:
            continue
        source = next((name for name in candidates if name in dataframe.columns), None)
        if source is None:
            raise ValueError(f"資料集缺少欄位：{expected}")
        dataframe = dataframe.rename(columns={source: expected})

    # 只取文字與標籤，清除空值；多標籤依目前規則取第一項。
    cleaned = dataframe[[text_column, label_column]].copy()
    cleaned[text_column] = cleaned[text_column].map(
        lambda value: str(value).strip() if value is not None else None
    )
    cleaned[label_column] = cleaned[label_column].map(_first_label)
    cleaned = cleaned.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
    cleaned = cleaned.dropna(subset=[text_column, label_column]).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("清理後資料集為空，請檢查輸入欄位與內容。")
    return cleaned
