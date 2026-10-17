# 依研究彙總數字繪製 README 圖表。

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


def save_human_method_figure() -> None:
    # 數字為逐題平均後的研究彙總，不是由原始問卷重新計算。
    methods = ["Zero-shot", "Few-shot", "Prefix Step750"]
    dimensions = ["Emotion", "Semantic", "Fluency"]
    # 每列是一種方法，每欄依序為情緒、語意與流暢度。
    scores = np.array(
        [
            [3.48, 3.01, 3.52],
            [3.67, 2.62, 3.37],
            [4.02, 3.48, 3.73],
        ]
    )
    positions = np.arange(len(dimensions))
    width = 0.24
    colors = ["#b0b0b0", "#707070", "#202020"]

    figure, axis = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    # 以不同水平偏移將三種方法並排在同一評分維度下。
    for index, method in enumerate(methods):
        axis.bar(
            positions + (index - 1) * width,
            scores[index],
            width,
            label=method,
            color=colors[index],
            edgecolor="black",
            linewidth=0.6,
        )
    axis.set_title("Human evaluation: question-level mean ratings")
    axis.set_ylabel("Mean rating (1, 3, 5 scale)")
    axis.set_xticks(positions, dimensions)
    axis.set_ylim(0, 5)
    axis.set_yticks(range(0, 6))
    axis.grid(axis="y", color="#d0d0d0", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncols=3, loc="upper left")
    figure.savefig(ASSET_DIR / "human_evaluation_summary.png", dpi=180)
    plt.close(figure)


def save_scorer_alignment_figure() -> None:
    # 繪製三個評分維度與真人評分的 Spearman 相關係數。
    labels = ["Emotion gain", "Semantic", "Fluency"]
    values = [0.32, 0.67, 0.05]
    figure, axis = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    bars = axis.bar(
        labels,
        values,
        color="#4c4c4c",
        edgecolor="black",
        linewidth=0.7,
    )
    axis.set_title("Scorer-human rank agreement")
    axis.set_ylabel("Spearman rho")
    axis.set_ylim(0, 0.8)
    axis.grid(axis="y", color="#d0d0d0", linewidth=0.6)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.2f}",
            ha="center",
        )
    figure.savefig(ASSET_DIR / "scorer_human_correlation.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    save_human_method_figure()
    save_scorer_alignment_figure()
