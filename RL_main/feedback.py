# 訓練回饋、EMA baseline 與權重調整。

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from RL_main.utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
logger = logging.getLogger(__name__)


class FeedbackMemory:
    def __init__(
        self,
        config_or_path: str | Path | dict[str, Any] = DEFAULT_CONFIG_PATH,
    ) -> None:
        config = (
            config_or_path
            if isinstance(config_or_path, dict)
            else load_config(config_or_path)
        )
        feedback_config = config.get("feedback", {})
        reward_config = config.get("scorer", {}).get(
            "reward",
            config.get("reward", {}),
        )

        # 這裡管理動態調整的情緒／流暢度比例；四維最終權重由評分器正規化。
        initial_emotion = float(reward_config.get("emotion_weight", 0.7))
        initial_fluency = float(reward_config.get("fluency_weight", 0.3))
        total = max(initial_emotion + initial_fluency, 1e-6)
        self.w_emotion = initial_emotion / total
        self.w_fluency = initial_fluency / total
        self.weights = {
            "emotion": self.w_emotion,
            "fluency": self.w_fluency,
        }

        self.emotion_weight_min = float(
            feedback_config.get("emotion_weight_min", 0.40)
        )
        self.emotion_weight_max = float(
            feedback_config.get("emotion_weight_max", 0.80)
        )
        self.fluency_weight_min = float(
            feedback_config.get("fluency_weight_min", 0.20)
        )
        self.fluency_weight_max = float(
            feedback_config.get("fluency_weight_max", 0.60)
        )
        # 近期平均若落後目標值，就增加對應維度的權重。
        self.targets = {
            "emo_gain": float(feedback_config.get("target_emo_gain", 0.30)),
            "fluency": float(feedback_config.get("target_fluency", 0.50)),
        }
        self.momentum = float(feedback_config.get("momentum", 0.9))
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("feedback.momentum 必須位於 [0, 1)。")

        history_path = Path(
            feedback_config.get("history_path", "logs/training_history.json")
        )
        self.history_path = (
            history_path
            if history_path.is_absolute()
            else PROJECT_ROOT / history_path
        )
        self.alpha = float(feedback_config.get("alpha", 0.1))
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("feedback.alpha 必須位於 (0, 1]。")

        # 逐句回饋保存在 history，EMA baseline 與 run 日誌狀態另外維護。
        self.history: list[dict[str, Any]] = []
        self.ema_baseline = 0.0
        self._baseline_initialized = False
        self.run_dir: Path | None = None
        self.samples_path: Path | None = None
        self.steps_path: Path | None = None
        self.snapshots_path: Path | None = None
        self.last_adjustment_stats: dict[str, Any] = {}

    def configure_run(self, run_dir: str | Path, started_at: str) -> None:
        # 建立本次訓練的逐句、逐步與事件紀錄檔。
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.run_dir / "feedback_samples.jsonl"
        self.steps_path = self.run_dir / "feedback_steps.jsonl"
        self.snapshots_path = self.run_dir / "feedback_snapshots.jsonl"
        for path in (self.samples_path, self.steps_path, self.snapshots_path):
            path.write_text("", encoding="utf-8")

        self.log_snapshot(
            {
                "event": "run_start",
                "started_at": started_at,
                "weights": self.weights.copy(),
                "targets": self.targets.copy(),
                "semantic_role": "threshold_and_continuous_reward",
            }
        )

    def _now(self) -> str:
        return datetime.now().astimezone().isoformat()

    def _jsonable(self, value: Any) -> Any:
        # 把 Tensor 與 NumPy 值轉成 JSON 可寫入的型別。
        if isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            return detached.item() if detached.numel() == 1 else detached.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        return value

    def _append_jsonl(
        self,
        path: Path | None,
        record: dict[str, Any],
    ) -> None:
        # 還沒設定 run 目錄時，不寫入日誌。
        if path is None:
            return
        # 每次附加一行 JSON，不重寫整份日誌。
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(self._jsonable(record), ensure_ascii=False) + "\n")

    def log_sample(self, record: dict[str, Any]) -> None:
        # 保存單一來源句與改寫句的評分結果。
        payload = {"timestamp": self._now(), **record}
        self._append_jsonl(self.samples_path, payload)

    def log_step(self, record: dict[str, Any]) -> None:
        # 保存一個訓練步驟的彙總指標。
        payload = {"timestamp": self._now(), **record}
        self._append_jsonl(self.steps_path, payload)

    def log_snapshot(self, record: dict[str, Any]) -> None:
        # 保存開始、權重調整及結束等訓練事件。
        payload = {"timestamp": self._now(), **record}
        self._append_jsonl(self.snapshots_path, payload)

    def store_feedback(
        self,
        prompt: str,
        response: str,
        scores: dict[str, float],
        reward: float,
    ) -> None:
        # 保存這筆評分，並用 reward 更新 EMA baseline。
        numeric_reward = float(reward)
        self.history.append(
            {
                "prompt": prompt,
                "response": response,
                "scores": dict(scores),
                "reward": numeric_reward,
            }
        )
        # 第一筆直接作為初始 baseline，之後才做 EMA 平滑。
        if not self._baseline_initialized:
            self.ema_baseline = numeric_reward
            self._baseline_initialized = True
        else:
            self.ema_baseline = (
                (1.0 - self.alpha) * self.ema_baseline
                + self.alpha * numeric_reward
            )

    def get_baseline(self) -> float:
        return float(self.ema_baseline)

    def dynamic_weight_adjustment(self, n: int = 50) -> dict[str, float]:
        # 根據近期情緒增益與流暢度調整兩者權重。
        if n <= 0:
            raise ValueError("動態權重視窗 n 必須大於 0。")
        if len(self.history) < n:
            self.last_adjustment_stats = {
                "window_size": n,
                "history_size": len(self.history),
                "skipped": True,
                "reason": "not_enough_history",
            }
            return self.weights.copy()

        # 以最近 n 筆的情緒增益與改寫流暢度估計目前表現。
        recent = self.history[-n:]
        average_emotion_gain = float(
            np.mean([item["scores"].get("emo_gain", 0.0) for item in recent])
        )
        average_fluency = float(
            np.mean(
                [
                    item["scores"].get(
                        "flu_gen",
                        item["scores"].get("fluency", 0.5),
                    )
                    for item in recent
                ]
            )
        )
        # 只計算低於目標的缺口，已達標的維度不再增加缺口。
        emotion_gap = max(0.0, self.targets["emo_gain"] - average_emotion_gain)
        fluency_gap = max(0.0, self.targets["fluency"] - average_fluency)
        total_gap = emotion_gap + fluency_gap
        before_weights = self.weights.copy()

        if total_gap <= 1e-6:
            raw_emotion_weight = self.w_emotion
        else:
            gap_ratio = emotion_gap / total_gap
            raw_emotion_weight = self.emotion_weight_min + (
                self.emotion_weight_max - self.emotion_weight_min
            ) * gap_ratio

        # 先套用兩個維度的權重上下限，再用 momentum 平滑更新。
        raw_emotion_weight = min(
            self.emotion_weight_max,
            max(self.emotion_weight_min, raw_emotion_weight),
        )
        raw_fluency_weight = min(
            self.fluency_weight_max,
            max(self.fluency_weight_min, 1.0 - raw_emotion_weight),
        )
        raw_emotion_weight = 1.0 - raw_fluency_weight

        self.w_emotion = (
            self.momentum * self.w_emotion
            + (1.0 - self.momentum) * raw_emotion_weight
        )
        self.w_fluency = 1.0 - self.w_emotion
        self.weights = {
            "emotion": self.w_emotion,
            "fluency": self.w_fluency,
        }
        self.last_adjustment_stats = {
            "window_size": n,
            "history_size": len(self.history),
            "skipped": False,
            "before_weights": before_weights,
            "after_weights": self.weights.copy(),
            "avg_emo_gain": average_emotion_gain,
            "avg_flu_gen": average_fluency,
            "target_emo_gain": self.targets["emo_gain"],
            "target_fluency": self.targets["fluency"],
            "gap_emo": emotion_gap,
            "gap_flu": fluency_gap,
            "raw_w_emotion": raw_emotion_weight,
            "raw_w_fluency": raw_fluency_weight,
        }
        logger.info(
            "Updated reward weights: emotion=%.3f, fluency=%.3f",
            self.w_emotion,
            self.w_fluency,
        )
        return self.weights.copy()

    def save_history(self, output_path: str | Path | None = None) -> Path:
        # 輸出位置依序使用指定路徑、run 目錄、設定檔的預設位置。
        if output_path is not None:
            path = Path(output_path)
        elif self.run_dir is not None:
            path = self.run_dir / "feedback_memory.json"
        else:
            path = self.history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self._jsonable(self.history),
                file,
                indent=2,
                ensure_ascii=False,
            )
        return path

    def get_current_weights_tensor(self, device: torch.device) -> torch.Tensor:
        # 舊介面的順序為 [semantic, emotion, fluency]；semantic 權重固定為 0。
        return torch.tensor(
            [0.0, self.w_emotion, self.w_fluency],
            device=device,
            dtype=torch.float32,
        )
