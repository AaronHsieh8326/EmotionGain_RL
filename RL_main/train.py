# 情緒 Prefix 的 REINFORCE 訓練。

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL_main import utils
from RL_main.Scorer_NLI import EmotionScorer
from RL_main.data_pipeline import load_excluded_prompts, prepare_rl_data
from RL_main.feedback import FeedbackMemory
from RL_main.generator import EmotionGenerator


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SCORE_KEYS = [
    "emo_orig",
    "emo_gen",
    "emo_gain",
    "flu_orig",
    "flu_gen",
    "semantic",
    "sem_cosine",
    "sem_nli",
    "parrot",
    "diversity_ratio",
]


class SourceBatchSampler:
    # shuffled_epoch 用完一輪再洗牌；with_replacement 每次獨立抽樣。

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        mode: str = "shuffled_epoch",
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size 必須大於 0。")
        if batch_size <= 0:
            raise ValueError("batch_size 必須大於 0。")
        if mode not in {"shuffled_epoch", "with_replacement"}:
            raise ValueError(
                "sampling_mode 必須是 shuffled_epoch 或 with_replacement。"
            )
        if mode == "shuffled_epoch" and batch_size > dataset_size:
            raise ValueError(
                "shuffled_epoch 模式的 batch_size 不可大於資料集大小。"
            )

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.mode = mode
        # 抽樣器有自己的亂數來源，不依賴其他程式用了多少次亂數。
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(dataset_size)
        self.cursor = 0
        self.completed_epochs = 0
        self.seen_indices: set[int] = set()

    def next_batch(self) -> list[int]:
        # 跨過資料尾端時，接著取下一輪的索引。
        if self.mode == "with_replacement":
            indices = self.rng.integers(
                0,
                self.dataset_size,
                size=self.batch_size,
            ).tolist()
            self.seen_indices.update(indices)
            return indices

        indices: list[int] = []
        while len(indices) < self.batch_size:
            remaining = self.dataset_size - self.cursor
            take_count = min(self.batch_size - len(indices), remaining)
            indices.extend(
                self.order[self.cursor : self.cursor + take_count].tolist()
            )
            self.cursor += take_count
            # 資料用完就重新洗牌，因此一個 batch 可能跨越兩輪。
            if self.cursor == self.dataset_size:
                self.completed_epochs += 1
                self.order = self.rng.permutation(self.dataset_size)
                self.cursor = 0

        self.seen_indices.update(indices)
        return indices

    @property
    def coverage(self) -> float:
        return len(self.seen_indices) / self.dataset_size


def robust_clean(text: str) -> tuple[str, bool]:
    # 在送進評分器前，先去掉提示標記與明顯無效的生成文字。
    if not isinstance(text, str) or not text.strip():
        return "", False

    leakage_markers = [
        "Output:",
        "Response:",
        "Input:",
        "Emotion:",
        "Example",
        "In Figure",
        "The system",
        "Note:",
        "Task:",
        "Rewrite:",
        "Now you:",
        "\n\n",
    ]
    cleaned = text
    positions = [
        cleaned.find(marker)
        for marker in leakage_markers
        if marker in cleaned
    ]
    if positions:
        cleaned = cleaned[: min(positions)]
    cleaned = re.sub(r"<[^>]+>", "", cleaned).strip().strip('"').strip("'")

    # 這裡只做基本過濾，完整格式檢查留給評分器。
    words = cleaned.split()
    if len(words) < 2:
        return cleaned, False
    if len(set(words)) < len(words) * 0.4:
        return cleaned, False
    return cleaned, True


def select_source_emotions(
    model_emotions: list[str],
    train_config: dict[str, Any],
) -> list[str]:
    # 來源情緒先看 source_emotions，再看 include_emotions，最後套用排除名單。
    excluded = {
        str(item).lower()
        for item in train_config.get("exclude_emotions", [])
    }
    source = train_config.get("source_emotions")
    included = train_config.get("include_emotions")

    if source:
        selected = [str(item).lower() for item in source]
    elif included:
        include_set = {str(item).lower() for item in included}
        selected = [
            emotion
            for emotion in model_emotions
            if emotion.lower() in include_set
        ]
    else:
        selected = list(model_emotions)

    selected = [
        emotion
        for emotion in selected
        if emotion.lower() not in excluded
    ]
    unsupported = [
        emotion
        for emotion in selected
        if emotion not in model_emotions
    ]
    if unsupported:
        raise ValueError(f"source_emotions 含未知標籤：{unsupported}")
    if not selected:
        raise ValueError("情緒篩選後沒有可訓練資料。")
    return selected


def normalize_advantages(
    advantages: torch.Tensor,
    mode: str = "none",
    clip_value: float = 2.0,
    rolling_rewards: deque[float] | None = None,
) -> tuple[torch.Tensor, float]:
    # 依設定做 batch 或 rolling 縮放，再裁切 advantage。
    normalized = advantages
    scale = 1.0

    if mode == "batch" and advantages.numel() > 1:
        standard_deviation = advantages.std()
        if standard_deviation > 1e-6:
            normalized = (
                advantages - advantages.mean()
            ) / (standard_deviation + 1e-8)
            scale = float(standard_deviation.item())
    elif mode == "rolling" and rolling_rewards and len(rolling_rewards) > 1:
        standard_deviation = float(np.std(list(rolling_rewards)))
        if standard_deviation > 1e-6:
            normalized = advantages / standard_deviation
            scale = standard_deviation

    return torch.clamp(normalized, -clip_value, clip_value), scale


def mean_score_dict(
    score_records: list[dict[str, float]],
    keys: list[str],
) -> dict[str, float]:
    # 將同一個 batch 的評分彙整成各欄平均值。
    if not score_records:
        return {key: 0.0 for key in keys}
    return {
        key: float(np.mean([record.get(key, 0.0) for record in score_records]))
        for key in keys
    }


def invalid_score_record() -> dict[str, float | bool]:
    # 無效生成也使用相同的評分欄位，方便後面統一寫入紀錄。
    return {
        "emo_orig": 0.0,
        "emo_gen": 0.0,
        "semantic": 0.0,
        "emotion": 0.0,
        "fluency": 0.0,
        "parrot": 1.0,
        "emo_gain": 0.0,
        "flu_orig": 0.0,
        "flu_gen": 0.0,
        "sem_cosine": 0.0,
        "sem_nli": 0.0,
        "diversity_ratio": 0.0,
        "gate_passed": False,
    }


def flatten_score_result(result: dict[str, Any]) -> dict[str, float | bool]:
    # 將評分器的巢狀結果整理成訓練紀錄使用的平面欄位。
    raw = result["raw"]
    return {
        "emo_orig": raw["emo_orig"],
        "emo_gen": raw["emo_gen"],
        "semantic": raw["sem_combined"],
        "emotion": raw["emo_gen"],
        "fluency": raw["flu_gen"],
        "parrot": raw["parrot"],
        "emo_gain": raw["emo_gain"],
        "flu_orig": raw["flu_orig"],
        "flu_gen": raw["flu_gen"],
        "sem_cosine": raw["sem_cosine"],
        "sem_nli": raw["sem_nli"],
        "diversity_ratio": raw["diversity_ratio"],
        "gate_passed": result["gate_passed"],
    }


def generate_batch(
    generator: EmotionGenerator,
    prompts: list[str],
    source_emotions: list[str],
    emotions: list[str],
    model_config: dict[str, Any],
    logger: Any,
    reference_prefix: torch.Tensor,
) -> list[dict[str, Any]]:
    # 生成失敗的樣本仍記錄在 batch 中，標為無效。
    generated: list[dict[str, Any]] = []
    if not (len(prompts) == len(source_emotions) == len(emotions)):
        raise ValueError("prompts、source_emotions 與 emotions 長度必須相同。")
    for prompt, source_emotion, emotion in zip(
        prompts,
        source_emotions,
        emotions,
    ):
        try:
            # 生成設定來自 model 區塊，包含 top_k、top_p 與停止條件。
            output = generator.generate_with_logprobs(
                emotion_label=emotion,
                prompt_text=prompt,
                max_new_tokens=int(model_config.get("max_new_tokens", 40)),
                temperature=float(model_config.get("temperature", 0.9)),
                top_k=int(model_config.get("top_k", 50)),
                top_p=float(model_config.get("top_p", 0.9)),
                repetition_penalty=float(
                    model_config.get("repetition_penalty", 1.1)
                ),
                stop_after_first_sentence=bool(
                    model_config.get("stop_after_first_sentence", True)
                ),
                reference_prefix=reference_prefix,
            )
            # 文字交給評分器，logprob 與 KL 則保留給後面的 loss。
            clean_text, is_valid = robust_clean(str(output["text"]))
            generated.append(
                {
                    "prompt": prompt,
                    "source_emotion": source_emotion,
                    "emotion": emotion,
                    "raw_text": output["text"],
                    "clean_text": clean_text,
                    "logprob_sum": output["logprob_sum"],
                    "kl_mean": output["kl_mean"],
                    "kl_token_count": output["kl_token_count"],
                    "token_count": output["token_count"],
                    "stop_reason": output["stop_reason"],
                    "is_valid": is_valid,
                }
            )
        except Exception as exc:
            logger.warning("生成失敗：%s", exc)
            generated.append(
                {
                    "prompt": prompt,
                    "source_emotion": source_emotion,
                    "emotion": emotion,
                    "raw_text": "",
                    "clean_text": "",
                    "logprob_sum": None,
                    "kl_mean": None,
                    "kl_token_count": 0,
                    "token_count": 0,
                    "stop_reason": "generation_error",
                    "is_valid": False,
                }
            )
    return generated


def score_batch_and_record(
    generated: list[dict[str, Any]],
    scorer: EmotionScorer,
    memory: FeedbackMemory,
    step: int,
    invalid_penalty: float,
) -> tuple[torch.Tensor, list[dict[str, Any]], int, Counter[str]]:
    rewards: list[float] = []
    score_records: list[dict[str, Any]] = []
    passed_count = 0
    fail_counts: Counter[str] = Counter()

    for sample_id, item in enumerate(generated):
        # 生成失敗或清理後無效的文字，直接給 invalid_penalty。
        if not item["is_valid"]:
            reward = -invalid_penalty
            fail_counts["invalid"] += 1
            score_record = invalid_score_record()
            threshold_info = {
                "passed": False,
                "fail_reasons": ["invalid"],
                "penalty": invalid_penalty,
            }
            breakdown = {
                "reward_gen": 0.0,
                "reward_orig": 0.0,
                "reward_base": 0.0,
                "diversity_bonus": 0.0,
                "gate_penalty": invalid_penalty,
            }
        else:
            # 其餘句子交給完整評分器，取得 Threshold 與 reward。
            result = scorer.score_single(
                item["prompt"],
                item["clean_text"],
                item["emotion"],
            )
            reward = float(result["reward"])
            score_record = flatten_score_result(result)
            if result["gate_passed"]:
                passed_count += 1
            for reason in result.get("gate_fail_reasons", []):
                fail_counts[reason] += 1
            breakdown = result.get("breakdown", {})
            threshold_info = {
                "passed": result["gate_passed"],
                "fail_reasons": result.get("gate_fail_reasons", []),
                "penalty": breakdown.get("gate_penalty", 0.0),
            }

        rewards.append(reward)
        score_records.append(score_record)
        # 先保存逐句回饋並更新 EMA，main 會使用進入這批前的 baseline 計算 advantage。
        memory.store_feedback(
            item["prompt"],
            item["clean_text"],
            score_record,
            reward,
        )
        # 同時記錄原句、實際輸出及各維度分數，之後可回查失敗案例。
        memory.log_sample(
            {
                "step": step,
                "sample_id": sample_id,
                "prompt": item["prompt"],
                "response": item["clean_text"],
                "raw_response": item["raw_text"],
                "target_emotion": item["emotion"],
                "source_emotion": item["source_emotion"],
                "is_valid": item["is_valid"],
                "stop_reason": item["stop_reason"],
                "reward": reward,
                "scores": {
                    "emo_orig": score_record["emo_orig"],
                    "emo_gen": score_record["emo_gen"],
                    "emo_gain": score_record["emo_gain"],
                    "flu_orig": score_record["flu_orig"],
                    "flu_gen": score_record["flu_gen"],
                    "sem_combined": score_record["semantic"],
                    "sem_cosine": score_record["sem_cosine"],
                    "sem_nli": score_record["sem_nli"],
                    "parrot": score_record["parrot"],
                    "diversity_ratio": score_record["diversity_ratio"],
                },
                "gate": threshold_info,
                "breakdown": breakdown,
                "weights": {
                    **scorer.reward_weights(),
                },
                "ema_baseline_after": memory.get_baseline(),
            }
        )

    return (
        torch.tensor(rewards, dtype=torch.float32, device=scorer.device),
        score_records,
        passed_count,
        fail_counts,
    )


def calculate_policy_loss(
    generated: list[dict[str, Any]],
    advantages: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    # 文字品質不合格但已生成的軌跡仍計入 loss，更新方向由 advantage 決定。
    sample_losses: list[torch.Tensor] = []
    for index, item in enumerate(generated):
        log_probability = item["logprob_sum"]
        token_count = int(item["token_count"])
        if token_count <= 0 or log_probability is None:
            continue
        # 用 token 數正規化 log probability，再乘上該句 advantage。
        normalized_log_probability = log_probability / token_count
        sample_loss = -normalized_log_probability * advantages[index]
        sample_losses.append(torch.clamp(sample_loss, -5.0, 5.0))

    if not sample_losses:
        return torch.zeros((), device=device), 0
    return torch.stack(sample_losses).mean(), len(sample_losses)


def calculate_token_kl_loss(
    generated: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    # 只平均有 reference KL 的生成軌跡。
    values = [
        item["kl_mean"]
        for item in generated
        if item.get("kl_mean") is not None
        and int(item.get("kl_token_count", 0)) > 0
    ]
    if not values:
        return torch.zeros((), device=device), 0
    return torch.stack(values).mean(), len(values)


def apply_optimizer_step(
    generator: EmotionGenerator,
    optimizer: AdamW,
    grad_clip: float,
    logger: Any,
    step: int,
) -> tuple[float, float, list[str]]:
    # 梯度非有限值時略過這次更新，並清除梯度。
    active_emotions = generator.gradient_emotions()
    # 保留裁切前後的梯度大小，方便觀察訓練是否穩定。
    raw_norm_tensor = torch.nn.utils.clip_grad_norm_(
        generator.trainable_prefix_parameters(),
        max_norm=grad_clip,
    )
    raw_grad_norm = float(raw_norm_tensor.item())
    clipped_grad_norm = generator.get_grad_norm()
    if not math.isfinite(raw_grad_norm) or not math.isfinite(clipped_grad_norm):
        logger.warning("Step %s 梯度不是有限值，略過更新。", step)
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not math.isfinite(raw_grad_norm) or not math.isfinite(clipped_grad_norm):
        return 0.0, 0.0, active_emotions
    return raw_grad_norm, clipped_grad_norm, active_emotions


def save_checkpoint(
    path: Path,
    generator: EmotionGenerator,
    step: int,
    baseline: float,
    weights: dict[str, float],
) -> None:
    # 只存 Prefix 與訓練狀態，不存底模。
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prefix_embeddings": generator.prefix_embeddings.detach().cpu(),
            "emotion2id": generator.emotion2id,
            "step": step,
            "baseline": baseline,
            "reward_weights": dict(weights),
            "prefix_parameterization": "independent_per_emotion",
            "regularization": "token_distribution_forward_kl",
        },
        path,
    )


def smooth_curve(values: list[float], window: int = 10) -> np.ndarray:
    # 以固定視窗做移動平均，前後無法填滿視窗的部分不輸出。
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def safe_float(value: Any) -> float:
    # 缺少或無法轉換的紀錄用 NaN 表示，繪圖時再略過。
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def moving_average_full(values: list[float], window: int = 15) -> list[float]:
    # 使用較短的開頭視窗，讓結果長度與原序列相同。
    output: list[float] = []
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1) : index + 1]
        clean = [item for item in chunk if not math.isnan(item)]
        output.append(sum(clean) / len(clean) if clean else float("nan"))
    return output


def mean_non_nan(values: list[float]) -> float:
    # 摘要統計忽略缺值，整段都缺值時保留 NaN。
    clean = [value for value in values if not math.isnan(value)]
    return statistics.mean(clean) if clean else float("nan")


def linear_slope(x_values: list[int], y_values: list[float]) -> float:
    # 以有效資料點計算線性趨勢斜率，用來比較指標隨步數的變化。
    points = [
        (x_value, y_value)
        for x_value, y_value in zip(x_values, y_values)
        if not math.isnan(y_value)
    ]
    if len(points) < 2:
        return 0.0
    x_mean = sum(point[0] for point in points) / len(points)
    y_mean = sum(point[1] for point in points) / len(points)
    covariance = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in points
    )
    variance = sum((x_value - x_mean) ** 2 for x_value, _ in points)
    return covariance / variance if variance else 0.0


def plot_focus_metrics(
    step_log_path: Path,
    plot_dir: Path,
    window: int = 15,
) -> Path | None:
    # 從逐步日誌繪製主要指標，另存前後期平均與趨勢摘要。
    if not step_log_path.is_file():
        return None
    rows = [
        json.loads(line)
        for line in step_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None

    # 將日誌拆成情緒、照抄、通過率與 reward 等時間序列。
    steps = [int(row.get("step", index + 1)) for index, row in enumerate(rows)]
    emotion_generated = [
        safe_float(row.get("scores_mean", {}).get("emo_gen"))
        for row in rows
    ]
    emotion_gain = [
        safe_float(row.get("scores_mean", {}).get("emo_gain"))
        for row in rows
    ]
    parrot = [
        safe_float(row.get("scores_mean", {}).get("parrot"))
        for row in rows
    ]
    threshold_pass = [
        safe_float(row.get("gate_pass_rate"))
        for row in rows
    ]
    reward = [safe_float(row.get("reward_mean")) for row in rows]

    # 各取最多 50 步比較開頭與結尾，資料不足時使用實際筆數。
    first_count = min(50, len(rows))
    last_count = min(50, len(rows))
    summary = {
        "count": len(rows),
        "step_start": steps[0],
        "step_end": steps[-1],
        "emo_gen_first50_mean": mean_non_nan(
            emotion_generated[:first_count]
        ),
        "emo_gen_last50_mean": mean_non_nan(
            emotion_generated[-last_count:]
        ),
        "emo_gain_first50_mean": mean_non_nan(emotion_gain[:first_count]),
        "emo_gain_last50_mean": mean_non_nan(emotion_gain[-last_count:]),
        "parrot_first50_mean": mean_non_nan(parrot[:first_count]),
        "parrot_last50_mean": mean_non_nan(parrot[-last_count:]),
        "threshold_first50_mean": mean_non_nan(
            threshold_pass[:first_count]
        ),
        "threshold_last50_mean": mean_non_nan(
            threshold_pass[-last_count:]
        ),
        "reward_first50_mean": mean_non_nan(reward[:first_count]),
        "reward_last50_mean": mean_non_nan(reward[-last_count:]),
        "emo_gen_slope_per100": (
            linear_slope(steps, emotion_generated) * 100
        ),
        "emo_gain_slope_per100": linear_slope(steps, emotion_gain) * 100,
        "parrot_slope_per100": linear_slope(steps, parrot) * 100,
    }

    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    series = [
        (
            axes[0],
            [
                (emotion_generated, "tab:red", "emo_gen"),
                (emotion_gain, "tab:purple", "emo_gain"),
            ],
            "Emotion metrics",
        ),
        (
            axes[1],
            [
                (parrot, "tab:blue", "parrot"),
                (threshold_pass, "tab:green", "threshold pass"),
            ],
            "Parrot / Threshold",
        ),
        (
            axes[2],
            [(reward, "tab:orange", "reward")],
            "Reward",
        ),
    ]
    # 淡線保留原始波動，粗線顯示移動平均。
    for axis, metrics, title in series:
        for values, color, label in metrics:
            axis.plot(steps, values, color=color, alpha=0.22, label=label)
            axis.plot(
                steps,
                moving_average_full(values, window),
                color=color,
                linewidth=2.2,
                label=f"{label} MA({window})",
            )
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_xlabel("Step")

    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / "emotion_parrot_focus.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    utils.save_json(summary, plot_dir / "emotion_parrot_focus_summary.json")
    return output_path


def plot_training_overview(
    history: dict[str, Any],
    plot_dir: Path,
) -> Path:
    # 總覽圖分別顯示 loss、reward／baseline、評分及梯度大小。
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["loss"], alpha=0.35, label="total")
    axes[0, 0].plot(history["policy_loss"], alpha=0.45, label="policy")
    axes[0, 0].plot(history["kl_loss"], alpha=0.45, label="token KL")
    if len(history["loss"]) >= 10:
        axes[0, 0].plot(smooth_curve(history["loss"]), label="smoothed")
    axes[0, 0].set_title("Policy / Token KL Loss")

    axes[0, 1].plot(history["reward"], alpha=0.35, label="reward")
    axes[0, 1].plot(history["baseline"], linestyle="--", label="baseline")
    axes[0, 1].set_title("Reward vs Baseline")

    for key, values in history["scores"].items():
        axes[1, 0].plot(values, label=key)
    axes[1, 0].set_title("Scores")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(
        history["grad_norm_raw"],
        color="tab:red",
        alpha=0.45,
        label="raw",
    )
    axes[1, 1].plot(
        history["grad_norm"],
        color="tab:purple",
        label="clipped",
    )
    axes[1, 1].set_title("Gradient Norm Before / After Clipping")

    for axis in axes.flat:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    output_path = plot_dir / "training_overview.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def training_parameters(train_config: dict[str, Any]) -> dict[str, Any]:
    # 集中讀取 train 區塊，將數值轉成訓練迴圈使用的型別。
    parameters = {
        "max_steps": int(train_config.get("max_steps", 100)),
        "batch_size": int(train_config.get("batch_size", 2)),
        "accumulation_steps": int(
            train_config.get("accumulation_steps", 4)
        ),
        "learning_rate": float(train_config.get("learning_rate", 2e-4)),
        "weight_decay": float(train_config.get("weight_decay", 0.0)),
        "grad_clip": float(train_config.get("grad_clip", 1.0)),
        "invalid_penalty": float(
            train_config.get("invalid_penalty", 0.5)
        ),
        "parrot_start": float(
            train_config.get("curriculum_parrot_start", 0.85)
        ),
        "parrot_end": float(
            train_config.get("curriculum_parrot_end", 0.85)
        ),
        "parrot_moderate_offset": float(
            train_config.get("parrot_moderate_offset", 0.10)
        ),
        "kl_weight": float(train_config.get("kl_weight", 0.05)),
        "logging_steps": int(train_config.get("logging_steps", 10)),
        "save_steps": int(train_config.get("save_steps", 50)),
        "dynamic_weight_interval": int(
            train_config.get("dynamic_weight_interval", 50)
        ),
        "dynamic_weight_window": int(
            train_config.get("dynamic_weight_window", 50)
        ),
        "advantage_mode": str(
            train_config.get("advantage_normalization", "none")
        ).lower(),
        "advantage_clip": float(
            train_config.get("advantage_clip", 2.0)
        ),
        "reward_norm_window": int(
            train_config.get("reward_norm_window", 100)
        ),
        "sampling_mode": str(
            train_config.get("sampling_mode", "shuffled_epoch")
        ).lower(),
        "sampling_seed": int(
            train_config.get(
                "sampling_seed",
                train_config.get("seed", 42),
            )
        ),
    }
    for key in (
        "max_steps",
        "batch_size",
        "accumulation_steps",
        "logging_steps",
        "save_steps",
        "dynamic_weight_interval",
        "dynamic_weight_window",
        "reward_norm_window",
    ):
        if parameters[key] <= 0:
            raise ValueError(f"train.{key} 必須大於 0。")
    if parameters["advantage_mode"] not in {"none", "batch", "rolling"}:
        raise ValueError(
            "train.advantage_normalization 必須是 none、batch 或 rolling。"
        )
    if parameters["sampling_mode"] not in {
        "shuffled_epoch",
        "with_replacement",
    }:
        raise ValueError(
            "train.sampling_mode 必須是 shuffled_epoch 或 with_replacement。"
        )
    if parameters["learning_rate"] <= 0.0:
        raise ValueError("train.learning_rate 必須大於 0。")
    if parameters["weight_decay"] < 0.0 or parameters["kl_weight"] < 0.0:
        raise ValueError("train.weight_decay 與 train.kl_weight 不得小於 0。")
    if parameters["grad_clip"] <= 0.0:
        raise ValueError("train.grad_clip 必須大於 0。")
    return parameters


def main() -> None:
    # 先建立資料切分，再載入模型、準備 optimizer，最後進入 RL 迴圈。
    config = utils.load_config(CONFIG_PATH)
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]
    parameters = training_parameters(train_config)

    output_dir = utils.resolve_path(train_config["output_dir"], PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = utils.get_logger("Trainer", output_dir / "train.log")
    utils.set_seed(int(train_config.get("seed", 42)))
    device = utils.select_device()

    model_emotions = [
        str(emotion).lower()
        for emotion in model_config.get(
            "emotions",
            ["joy", "sadness", "anger", "fear", "surprise", "neutral"],
        )
    ]
    # 明確區分來源情緒與改寫目標，預設任務為 neutral → joy。
    target_override = train_config.get("target_emotion_override")
    if not target_override:
        raise ValueError(
            "train.target_emotion_override 必須明確指定，禁止把來源標籤直接當 RL 目標。"
        )
    target_override = str(target_override).lower()
    if target_override not in model_emotions:
        raise ValueError(
            f"target_emotion_override 不在 model.emotions：{target_override}"
        )
    source_emotions = select_source_emotions(model_emotions, train_config)

    raw_dataframe = utils.load_dataset(
        utils.resolve_path(data_config["path"], PROJECT_ROOT),
        data_config["text_column"],
        data_config["label_column"],
    )
    exclusion_paths = [
        utils.resolve_path(path, PROJECT_ROOT)
        for path in data_config.get("exclude_prompt_paths", [])
    ]
    # 排除 Warm-up 用過的來源句，再做固定的 train／validation 切分。
    excluded_prompts = load_excluded_prompts(exclusion_paths)
    prepared_data = prepare_rl_data(
        raw_dataframe,
        text_column=data_config["text_column"],
        label_column=data_config["label_column"],
        source_emotions=source_emotions,
        target_emotion=target_override,
        supported_emotions=model_emotions,
        excluded_prompts=excluded_prompts,
        min_words=int(data_config.get("min_words", 4)),
        max_words=int(data_config.get("max_words", 30)),
        validation_fraction=float(
            data_config.get("validation_fraction", 0.10)
        ),
        split_seed=int(data_config.get("split_seed", 42)),
    )
    dataframe = prepared_data.train
    prompts = dataframe["text"].tolist()
    labels = dataframe["target_emotion"].tolist()
    source_labels = dataframe["source_emotion"].tolist()

    # 每次訓練建立獨立日誌目錄，保存資料筆數與驗證句清單。
    run_started = datetime.now().astimezone()
    run_id = run_started.strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "RL_main" / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    utils.save_json(
        prepared_data.statistics,
        run_dir / "data_manifest.json",
    )
    utils.save_json(
        prepared_data.validation.to_dict(orient="records"),
        run_dir / "validation_sources.json",
    )

    logger.info("Training device: %s", device)
    logger.info("Dataset size: %s", len(dataframe))
    logger.info("Source emotions: %s", source_emotions)
    logger.info("Target override: %s", target_override or "none")
    logger.info("Data preparation: %s", prepared_data.statistics)
    source_sampler = SourceBatchSampler(
        dataset_size=len(prompts),
        batch_size=parameters["batch_size"],
        seed=parameters["sampling_seed"],
        mode=parameters["sampling_mode"],
    )
    logger.info(
        "Source sampling: mode=%s seed=%s planned_draws=%s",
        parameters["sampling_mode"],
        parameters["sampling_seed"],
        parameters["max_steps"] * parameters["batch_size"],
    )

    # 載入底模與 Prefix 結構，再接續 Warm-up 權重。
    generator = EmotionGenerator(
        model_path=model_config["path"],
        emotion_labels=model_emotions,
        device=device,
        prefix_length=int(model_config.get("prefix_length", 20)),
    ).to(device)
    warmup_value = train_config.get("warmup_checkpoint")
    if warmup_value:
        warmup_path = utils.resolve_path(warmup_value, PROJECT_ROOT)
        if not generator.load_adapter(warmup_path):
            raise RuntimeError(
                "Warm-up checkpoint 不存在、格式錯誤或 Prefix shape 與 "
                f"model.prefix_length 不一致：{warmup_path}。"
                "請先以目前 config 重新執行 warmup_training/warmup.py。"
            )
    else:
        logger.warning("train.warmup_checkpoint 未設定，將從初始化 Prefix 訓練。")

    scorer = EmotionScorer(config)
    memory = FeedbackMemory(config)
    memory.configure_run(run_dir, run_started.isoformat())

    optimizer = AdamW(
        generator.trainable_prefix_parameters(),
        lr=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"],
    )
    # 複製並凍結起始 Prefix，整段 RL 都以這份權重計算 KL。
    reference_prefix = generator.prefix_embeddings.detach().clone()
    reward_config = config.get("scorer", {}).get("reward", {})
    dynamic_weights = bool(reward_config.get("dynamic_weights", True))

    # history 保留每步的摘要；逐句詳細資料另寫入 JSONL。
    history: dict[str, Any] = {
        "loss": [],
        "reward": [],
        "baseline": [],
        "grad_norm": [],
        "grad_norm_raw": [],
        "policy_loss": [],
        "kl_loss": [],
        "active_prefixes": [],
        "scores": {
            "semantic": [],
            "emotion": [],
            "fluency": [],
            "parrot": [],
        },
        "weights": [],
        "gate_pass_rate": [],
    }
    rolling_rewards: deque[float] = deque(
        maxlen=parameters["reward_norm_window"]
    )
    previous_reward_mean: float | None = None
    pending_gradient_steps = 0

    logger.info(
        "RL start | steps=%s batch=%s accumulation=%s lr=%s token_kl=%s",
        parameters["max_steps"],
        parameters["batch_size"],
        parameters["accumulation_steps"],
        parameters["learning_rate"],
        parameters["kl_weight"],
    )
    generator.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(range(parameters["max_steps"]), desc="RL Training")

    # 每一步依序進行：抽樣、生成、評分、計算 loss、累積梯度、寫入紀錄。
    for step_index in progress:
        step = step_index + 1
        ratio = step_index / max(parameters["max_steps"] - 1, 1)
        # 依訓練進度調整照抄門檻；起訖相同時門檻維持固定。
        severe_threshold = (
            parameters["parrot_start"]
            - (
                parameters["parrot_start"]
                - parameters["parrot_end"]
            )
            * ratio
        )
        scorer.parrot_threshold_severe = severe_threshold
        scorer.parrot_threshold_moderate = max(
            0.0,
            severe_threshold - parameters["parrot_moderate_offset"],
        )

        # 學習率線性降到起始值的一半。
        current_learning_rate = parameters["learning_rate"] * (
            1.0 - 0.5 * ratio
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate

        # 取出這批來源句及其情緒標籤，再交給生成器。
        indices = source_sampler.next_batch()
        batch_prompts = [prompts[index] for index in indices]
        batch_source_emotions = [source_labels[index] for index in indices]
        batch_emotions = [labels[index] for index in indices]
        generated = generate_batch(
            generator,
            batch_prompts,
            batch_source_emotions,
            batch_emotions,
            model_config,
            logger,
            reference_prefix,
        )

        # 先讀取舊 baseline；評分這批資料時，memory 會更新下一批要用的 baseline。
        baseline_before_batch = memory.get_baseline()
        rewards, score_records, passed_count, fail_counts = (
            score_batch_and_record(
                generated,
                scorer,
                memory,
                step,
                parameters["invalid_penalty"],
            )
        )
        rewards = rewards.to(device)
        pass_rate = passed_count / parameters["batch_size"]
        # 高於 baseline 的句子得到正 advantage，低於它的句子得到負值。
        advantages = rewards - baseline_before_batch
        raw_advantage_mean = float(advantages.mean().item())
        advantages, normalization_scale = normalize_advantages(
            advantages,
            mode=parameters["advantage_mode"],
            clip_value=parameters["advantage_clip"],
            rolling_rewards=rolling_rewards,
        )
        used_advantage_mean = float(advantages.mean().item())
        rolling_rewards.extend(rewards.detach().cpu().tolist())

        # policy loss 調整生成機率，KL loss 約束目前策略與 Warm-up 的差距。
        policy_loss, trajectory_count = calculate_policy_loss(
            generated,
            advantages,
            device,
        )
        kl_loss, kl_trajectory_count = calculate_token_kl_loss(
            generated,
            device,
        )
        total_loss = torch.zeros((), device=device)
        if trajectory_count or kl_trajectory_count:
            # loss 先除以累積步數，累積多批梯度後才更新一次。
            total_loss = torch.clamp(
                policy_loss + parameters["kl_weight"] * kl_loss,
                -10.0,
                10.0,
            )
            (
                total_loss / parameters["accumulation_steps"]
            ).backward()
            pending_gradient_steps += 1

        raw_grad_norm = 0.0
        grad_norm = 0.0
        active_prefixes: list[str] = []
        # 達到累積批數後裁切梯度、更新 Prefix，並清空梯度。
        if pending_gradient_steps >= parameters["accumulation_steps"]:
            raw_grad_norm, grad_norm, active_prefixes = apply_optimizer_step(
                generator,
                optimizer,
                parameters["grad_clip"],
                logger,
                step,
            )
            pending_gradient_steps = 0

        if (
            dynamic_weights
            and step % parameters["dynamic_weight_interval"] == 0
        ):
            # 若啟用動態權重，定期依近期情緒與流暢度的差距調整。
            before_weights = memory.weights.copy()
            new_weights = memory.dynamic_weight_adjustment(
                n=parameters["dynamic_weight_window"]
            )
            scorer.W_EMOTION = new_weights["emotion"]
            scorer.W_FLUENCY = new_weights["fluency"]
            memory.log_snapshot(
                {
                    "event": "dynamic_weight_adjustment",
                    "step": step,
                    "before_weights": before_weights,
                    "after_weights": new_weights,
                    "window_stats": memory.last_adjustment_stats,
                }
            )

        average_reward = float(rewards.mean().item())
        reward_minimum = float(rewards.min().item())
        reward_maximum = float(rewards.max().item())
        current_baseline = memory.get_baseline()
        scores_mean = mean_score_dict(score_records, SCORE_KEYS)
        scores_mean["sem_combined"] = scores_mean.pop("semantic")

        # 將目前 batch 的 loss、reward、權重與梯度加入時間序列。
        history["loss"].append(float(total_loss.detach().item()))
        history["reward"].append(average_reward)
        history["baseline"].append(current_baseline)
        history["grad_norm"].append(grad_norm)
        history["grad_norm_raw"].append(raw_grad_norm)
        history["policy_loss"].append(float(policy_loss.detach().item()))
        history["kl_loss"].append(float(kl_loss.detach().item()))
        history["active_prefixes"].append(active_prefixes)
        history["gate_pass_rate"].append(pass_rate)
        history["weights"].append(
            scorer.reward_weights()
        )
        # 評分曲線排除基本清理就失敗的輸出；Threshold 失敗但可評分者仍保留。
        valid_scores = [
            record
            for record, item in zip(score_records, generated)
            if item["is_valid"]
        ]
        for key in history["scores"]:
            history["scores"][key].append(
                float(
                    np.mean([record[key] for record in valid_scores])
                )
                if valid_scores
                else 0.0
            )

        reward_delta_previous = (
            0.0
            if previous_reward_mean is None
            else average_reward - previous_reward_mean
        )
        previous_reward_mean = average_reward
        # 寫入本步摘要，包含分數、失敗原因、更新狀態與資料覆蓋率。
        memory.log_step(
            {
                "step": step,
                "reward_mean": average_reward,
                "reward_max": reward_maximum,
                "reward_min": reward_minimum,
                "ema_baseline_before": baseline_before_batch,
                "ema_baseline": current_baseline,
                "advantage_mean": raw_advantage_mean,
                "advantage_mean_used": used_advantage_mean,
                "advantage_normalization": parameters["advantage_mode"],
                "advantage_norm_scale": normalization_scale,
                "advantage_clip": parameters["advantage_clip"],
                "reward_delta_vs_baseline": (
                    average_reward - baseline_before_batch
                ),
                "reward_delta_vs_prev": reward_delta_previous,
                "gate_pass_rate": pass_rate,
                "gate_fail_counts": dict(fail_counts),
                "parrot_fail_count": (
                    fail_counts.get("parrot_severe", 0)
                    + fail_counts.get("parrot_moderate", 0)
                ),
                "semantic_fail_count": fail_counts.get(
                    "semantic_drift",
                    0,
                ),
                "scores_mean": scores_mean,
                "weights": {
                    **scorer.reward_weights(),
                },
                "semantic_role": "threshold_and_continuous_reward",
                "parrot_threshold_severe": (
                    scorer.parrot_threshold_severe
                ),
                "parrot_threshold_moderate": (
                    scorer.parrot_threshold_moderate
                ),
                "learning_rate": current_learning_rate,
                "reward_mode": scorer.reward_mode,
                "dynamic_weights": dynamic_weights,
                "policy_loss": history["policy_loss"][-1],
                "token_kl_loss": history["kl_loss"][-1],
                "kl_weight": parameters["kl_weight"],
                "policy_trajectory_count": trajectory_count,
                "kl_trajectory_count": kl_trajectory_count,
                "active_prefixes": active_prefixes,
                "grad_norm_raw": raw_grad_norm,
                "grad_norm": grad_norm,
                "loss": history["loss"][-1],
                "unique_source_samples": len(source_sampler.seen_indices),
                "source_coverage": source_sampler.coverage,
                "completed_data_epochs": source_sampler.completed_epochs,
            }
        )

        # 進度列只顯示主要指標，完整數字留在日誌。
        progress.set_postfix(
            {
                "R": f"{average_reward:.2f}",
                "B": f"{current_baseline:.2f}",
                "G": f"{grad_norm:.2f}",
                "Thr%": f"{pass_rate * 100:.0f}",
                "PThr": f"{severe_threshold:.2f}",
                "Data%": f"{source_sampler.coverage * 100:.1f}",
            }
        )
        if step % parameters["logging_steps"] == 0:
            logger.info(
                "Step %s | reward=%.3f baseline=%.3f emotion=%.3f "
                "parrot=%.3f threshold_pass=%.1f%% data_coverage=%.1f%% "
                "kl=%.4f active=%s lr=%.6f",
                step,
                average_reward,
                current_baseline,
                history["scores"]["emotion"][-1],
                history["scores"]["parrot"][-1],
                pass_rate * 100,
                source_sampler.coverage * 100,
                history["kl_loss"][-1],
                active_prefixes or "accumulating",
                current_learning_rate,
            )
        # 定期保存 Prefix，供後續比較不同訓練階段。
        if step % parameters["save_steps"] == 0:
            checkpoint_path = output_dir / f"rl_prefix_step{step}.pt"
            save_checkpoint(
                checkpoint_path,
                generator,
                step,
                current_baseline,
                scorer.reward_weights(),
            )
            logger.info("Checkpoint: %s", checkpoint_path)

    # 最後不足一個累積週期的梯度，也在結束前更新一次。
    if pending_gradient_steps:
        final_raw_grad_norm, final_grad_norm, final_active_prefixes = (
            apply_optimizer_step(
                generator,
                optimizer,
                parameters["grad_clip"],
                logger,
                parameters["max_steps"],
            )
        )
        if history["grad_norm"]:
            history["grad_norm"][-1] = final_grad_norm
            history["grad_norm_raw"][-1] = final_raw_grad_norm
            history["active_prefixes"][-1] = final_active_prefixes

    # 保存最終權重、歷史資料與兩組訓練圖表。
    final_checkpoint = output_dir / "rl_prefix_final.pt"
    save_checkpoint(
        final_checkpoint,
        generator,
        parameters["max_steps"],
        memory.get_baseline(),
        scorer.reward_weights(),
    )
    history_path = memory.save_history(run_dir / "training_history.json")
    plot_dir = run_dir / "plots"
    overview_path = plot_training_overview(history, plot_dir)
    focus_path = plot_focus_metrics(
        run_dir / "feedback_steps.jsonl",
        plot_dir,
    )

    run_ended = datetime.now().astimezone()
    final_reward = (
        float(np.mean(history["reward"][-50:]))
        if history["reward"]
        else 0.0
    )
    # 結束摘要包含耗時、最後 50 步 reward、輸出位置及抽樣覆蓋率。
    memory.log_snapshot(
        {
            "event": "run_end",
            "started_at": run_started.isoformat(),
            "ended_at": run_ended.isoformat(),
            "duration_seconds": (run_ended - run_started).total_seconds(),
            "final_baseline": memory.get_baseline(),
            "final_reward_last_50": final_reward,
            "final_weights": scorer.reward_weights(),
            "checkpoint": final_checkpoint,
            "history": history_path,
            "training_overview_plot": overview_path,
            "emotion_parrot_focus_plot": focus_path,
            "sampling_mode": parameters["sampling_mode"],
            "unique_source_samples": len(source_sampler.seen_indices),
            "source_coverage": source_sampler.coverage,
            "completed_data_epochs": source_sampler.completed_epochs,
            "data_manifest": run_dir / "data_manifest.json",
            "validation_sources": run_dir / "validation_sources.json",
            "kl_type": "token_distribution_forward_kl",
            "kl_weight": parameters["kl_weight"],
            "final_token_kl": history["kl_loss"][-1] if history["kl_loss"] else 0.0,
        }
    )
    logger.info("Training complete: %s", final_checkpoint)
    logger.info(
        "Final baseline=%.4f | reward(last 50)=%.4f",
        memory.get_baseline(),
        final_reward,
    )


if __name__ == "__main__":
    main()
