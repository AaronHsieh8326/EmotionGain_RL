# 以目標改寫句預訓練情緒 Prefix。

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL_main.generator import EmotionGenerator
from RL_main.Scorer_NLI import analyze_response_quality
from RL_main.utils import (
    get_logger,
    load_config,
    resolve_path,
    select_device,
    set_seed,
)


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
logger = get_logger("Warmup")


class WarmupDataset(Dataset):
    # 每筆資料含 prompt、source_emotion、emotion、target_text。

    def __init__(
        self,
        file_path: str | Path,
        *,
        allowed_emotions: list[str],
        expected_source_emotion: str,
        max_invalid_fraction: float = 0.05,
    ) -> None:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 Warm-up 資料：{path}")

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Warm-up 資料為空：{path}")
        # 資料可放在一個 JSON 陣列中，也可逐行存成 JSONL。
        if content.startswith("["):
            raw_data = json.loads(content)
        else:
            raw_data = [
                json.loads(line)
                for line in content.splitlines()
                if line.strip()
            ]
        if not isinstance(raw_data, list):
            raise ValueError("Warm-up JSON 頂層必須是 list。")

        if not 0.0 <= max_invalid_fraction < 1.0:
            raise ValueError("max_invalid_fraction 必須位於 [0, 1)。")
        allowed = {emotion.strip().lower() for emotion in allowed_emotions}
        expected_source = expected_source_emotion.strip().lower()
        self.data: list[dict[str, str]] = []
        rejected: Counter[str] = Counter()
        # 逐筆整理必要欄位，累計無效原因，再保留可訓練的配對。
        for item in raw_data:
            if not isinstance(item, dict):
                rejected["not_mapping"] += 1
                continue
            prompt = str(item.get("prompt", item.get("text", ""))).strip()
            target = clean_target_text(str(
                item.get("target_text", item.get("response", ""))
            ).strip())
            emotion_value = item.get("emotion", "")
            if isinstance(emotion_value, list):
                emotion_value = emotion_value[0] if emotion_value else ""
            emotion = str(emotion_value).strip().lower()
            source_emotion = str(
                item.get("source_emotion", "")
            ).strip().lower()
            if not prompt or not target or not emotion or not source_emotion:
                rejected["missing_required_field"] += 1
                continue
            if emotion not in allowed:
                rejected["unsupported_target_emotion"] += 1
                continue
            if source_emotion != expected_source:
                rejected["unexpected_source_emotion"] += 1
                continue
            if source_emotion == emotion:
                rejected["not_cross_emotion"] += 1
                continue
            # 目標句沿用 RL 的文字格式檢查，但此處不載入神經評分模型。
            quality = analyze_response_quality(prompt, target)
            if quality["fail_reasons"]:
                for reason in quality["fail_reasons"]:
                    rejected[f"target_{reason}"] += 1
                continue
            self.data.append(
                {
                    "prompt": prompt,
                    "source_emotion": source_emotion,
                    "emotion": emotion,
                    "target_text": target,
                }
            )
        if not self.data:
            raise ValueError("Warm-up 資料沒有有效樣本。")
        # 資料清理完後檢查無效比例，並統計各目標情緒的有效筆數。
        invalid_fraction = (len(raw_data) - len(self.data)) / len(raw_data)
        if invalid_fraction > max_invalid_fraction:
            raise ValueError(
                f"Warm-up 無效資料比例 {invalid_fraction:.1%} 超過上限 "
                f"{max_invalid_fraction:.1%}：{dict(rejected)}"
            )
        self.rejected = dict(rejected)
        self.emotion_counts = Counter(item["emotion"] for item in self.data)
        logger.info("Loaded %s Warm-up samples from %s", len(self.data), path)
        logger.info("Warm-up targets=%s rejected=%s", self.emotion_counts, rejected)

    def validate_coverage(
        self,
        required_emotions: list[str],
        minimum_per_emotion: int,
    ) -> None:
        # 每個預定目標都需達到最低筆數，不只檢查資料總量。
        if minimum_per_emotion <= 0:
            raise ValueError("minimum_per_emotion 必須大於 0。")
        insufficient = {
            emotion: self.emotion_counts.get(emotion, 0)
            for emotion in required_emotions
            if self.emotion_counts.get(emotion, 0) < minimum_per_emotion
        }
        if insufficient:
            raise ValueError(
                f"Warm-up 目標情緒樣本不足，最低 {minimum_per_emotion}："
                f"{insufficient}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.data[index]


def collate_records(batch: list[dict[str, str]]) -> list[dict[str, str]]:
    # 先保留原始字典，每筆送入模型時才建立 embedding。
    return batch


def clean_target_text(text: str) -> str:
    # 移除 API 可能帶回的 Response 等標籤及外層引號。
    cleaned = text.strip()
    for marker in ("Output:", "Rewritten:", "Response:", "Example:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()
    return cleaned.strip().strip('"').strip("'")


def calculate_sample_loss(
    generator: EmotionGenerator,
    prompt: str,
    emotion: str,
    target_text: str,
) -> torch.Tensor:
    # 只對 target 與 EOS 計算交叉熵，Prefix 和 prompt 的 labels 設為 -100。
    generator._emotion_id(emotion)
    target = clean_target_text(target_text)
    if not target:
        raise ValueError("清理後 target_text 為空。")

    formatted_prompt = generator.format_prompt(emotion, prompt)
    prompt_ids = generator.tokenizer.encode(
        formatted_prompt,
        add_special_tokens=False,
    )
    target_ids = generator.tokenizer.encode(
        target,
        add_special_tokens=False,
    )
    bos_id = generator.tokenizer.bos_token_id
    eos_id = generator.tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer 必須提供 BOS 與 EOS token。")

    # 將提示與正確目標串在同一序列，讓模型逐 token 預測目標句。
    full_ids = [bos_id, *prompt_ids, *target_ids, eos_id]
    full_ids_tensor = torch.tensor(
        [full_ids],
        device=generator.device,
        dtype=torch.long,
    )
    with torch.no_grad():
        text_embeddings = generator.model.get_input_embeddings()(
            full_ids_tensor
        )

    # 取出這筆目標情緒的 Prefix，接在文字 embedding 前。
    prefix_embeddings = generator.get_prefix_parameter(emotion).unsqueeze(0)
    prefix_embeddings = prefix_embeddings.to(text_embeddings.dtype)
    input_embeddings = torch.cat(
        [prefix_embeddings, text_embeddings],
        dim=1,
    )
    # 標籤與完整輸入對齊；BOS、提示與 Prefix 的位置不計入 loss。
    ignored_prompt_labels = [-100] * (1 + len(prompt_ids))
    text_labels = torch.tensor(
        [[*ignored_prompt_labels, *target_ids, eos_id]],
        device=generator.device,
        dtype=torch.long,
    )
    prefix_labels = torch.full(
        (1, generator.prefix_length),
        -100,
        device=generator.device,
        dtype=torch.long,
    )
    labels = torch.cat([prefix_labels, text_labels], dim=1)
    attention_mask = torch.ones(
        input_embeddings.shape[:2],
        device=generator.device,
        dtype=torch.long,
    )
    # 模型內部會位移 labels，計算下一個 token 的交叉熵。
    output = generator.model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        labels=labels,
    )
    if output.loss is None or not torch.isfinite(output.loss):
        raise ValueError("模型回傳無效 Warm-up loss。")
    return output.loss


def train_one_epoch(
    generator: EmotionGenerator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    epoch: int,
    epochs: int,
) -> float:
    # 逐批計算目標句 loss，只更新 Prefix。
    total_loss = 0.0
    valid_batches = 0
    progress = tqdm(dataloader, desc=f"Warm-up {epoch}/{epochs}")

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        sample_losses: list[torch.Tensor] = []
        # 每筆序列長度不同，分開前向計算後再平均成 batch loss。
        for item in batch:
            try:
                sample_losses.append(
                    calculate_sample_loss(
                        generator,
                        item["prompt"],
                        item["emotion"],
                        item["target_text"],
                    )
                )
            except ValueError as exc:
                logger.warning("略過 Warm-up 樣本：%s", exc)

        if not sample_losses:
            continue
        batch_loss = torch.stack(sample_losses).mean()
        # 對平均 loss 反向傳播、裁切梯度，然後做一次 optimizer 更新。
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            generator.trainable_prefix_parameters(),
            max_norm=grad_clip,
        )
        optimizer.step()
        total_loss += float(batch_loss.detach().item())
        valid_batches += 1
        progress.set_postfix(loss=f"{batch_loss.item():.4f}")

    if valid_batches == 0:
        raise RuntimeError(f"Epoch {epoch} 沒有有效 Warm-up batch。")
    return total_loss / valid_batches


def plot_loss_curve(loss_history: list[float], output_dir: Path) -> Path:
    # 繪製每個 epoch 的平均交叉熵。
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(loss_history) + 1)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(epochs, loss_history, "b-o", label="Training Loss")
    axis.set_title("Warm-up Cross-Entropy Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-Entropy Loss")
    axis.set_xticks(list(epochs))
    axis.grid(True, linestyle="--", alpha=0.7)
    axis.legend()
    figure.tight_layout()
    output_path = plot_dir / "warmup_loss_curve.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main() -> None:
    # 依序準備資料、模型與 optimizer，再開始各 epoch 的 Warm-up。
    config = load_config(CONFIG_PATH)
    warmup_config = config.get("warmup", {})
    model_config = config["model"]
    train_config = config["train"]
    set_seed(int(warmup_config.get("seed", train_config.get("seed", 42))))

    output_dir = resolve_path(train_config["output_dir"], PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_path(warmup_config["dataset_path"], PROJECT_ROOT)
    generation_config = warmup_config.get("generate", {})
    required_targets = list(
        generation_config.get("samples_per_target_emotion", {}).keys()
    )
    dataset = WarmupDataset(
        dataset_path,
        allowed_emotions=model_config["emotions"],
        expected_source_emotion=str(
            generation_config.get("source_emotion", "neutral")
        ),
        max_invalid_fraction=float(
            warmup_config.get("max_invalid_fraction", 0.05)
        ),
    )
    # 載入底模前先確認每個目標情緒的有效資料量。
    dataset.validate_coverage(
        required_targets,
        int(warmup_config.get("minimum_samples_per_target", 500)),
    )
    device = select_device()
    generator = EmotionGenerator(
        model_path=model_config["path"],
        emotion_labels=model_config["emotions"],
        device=device,
        prefix_length=int(model_config.get("prefix_length", 20)),
    ).to(device)
    # DataLoader 負責洗牌與分批，保留字典而不先做 padding。
    dataloader = DataLoader(
        dataset,
        batch_size=int(warmup_config.get("batch_size", 2)),
        shuffle=True,
        collate_fn=collate_records,
    )

    learning_rate = float(warmup_config.get("learning_rate", 1e-4))
    epochs = int(warmup_config.get("epochs", 10))
    grad_clip = float(warmup_config.get("grad_clip", 1.0))
    if epochs <= 0:
        raise ValueError("warmup.epochs 必須大於 0。")
    # 只把 Prefix Parameter 交給 optimizer。
    optimizer = torch.optim.AdamW(
        generator.trainable_prefix_parameters(),
        lr=learning_rate,
        weight_decay=float(warmup_config.get("weight_decay", 0.01)),
    )

    logger.info(
        "Warm-up start | device=%s samples=%s epochs=%s trainable=%s",
        device,
        len(dataset),
        epochs,
        sum(
            parameter.numel()
            for parameter in generator.trainable_prefix_parameters()
        ),
    )
    generator.train()
    loss_history: list[float] = []
    for epoch in range(1, epochs + 1):
        epoch_loss = train_one_epoch(
            generator,
            dataloader,
            optimizer,
            grad_clip,
            epoch,
            epochs,
        )
        loss_history.append(epoch_loss)
        logger.info("Epoch %s average loss: %.4f", epoch, epoch_loss)

    # 存下 Prefix、標籤順序、資料統計與 loss 歷史，供 RL 接續使用。
    checkpoint_path = output_dir / "warmup_prefix.pt"
    torch.save(
        {
            "prefix_embeddings": generator.prefix_embeddings.detach().cpu(),
            "emotion2id": generator.emotion2id,
            "epochs": epochs,
            "loss_history": loss_history,
            "final_loss": loss_history[-1],
            "source_emotion": str(
                generation_config.get("source_emotion", "neutral")
            ),
            "target_counts": dict(dataset.emotion_counts),
            "rejected_counts": dataset.rejected,
            "dataset_path": str(dataset_path),
            "prefix_parameterization": "independent_per_emotion",
        },
        checkpoint_path,
    )
    plot_path = plot_loss_curve(loss_history, output_dir)
    logger.info("Warm-up checkpoint: %s", checkpoint_path)
    logger.info("Warm-up loss plot: %s", plot_path)


if __name__ == "__main__":
    main()
