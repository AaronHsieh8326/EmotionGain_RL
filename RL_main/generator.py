# 凍結 LLM，以情緒 Prefix 控制改寫。

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)

STOP_MARKERS = (
    "Context:",
    "Emotion:",
    "Reply:",
    "Output:",
    "Response:",
    "Example:",
    "The sentence",
    "The system",
    "The response",
    "In Figure",
    "Input :",
    "[INPUT]",
    "###",
    "\n",
)

TERMINAL_SENTENCE_PATTERN = re.compile(r"[.!?]+(?:[\"')\]]+)?$")
INCOMPLETE_TITLE_PATTERN = re.compile(
    r"\b(?:mr|mrs|ms|dr|prof|sr|jr)\.$",
    re.I,
)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: Sequence[int],
    penalty: float,
) -> torch.Tensor:
    # 正 logits 除以 penalty，負 logits 乘以 penalty。
    if penalty <= 0:
        raise ValueError("repetition_penalty 必須大於 0。")
    if penalty == 1.0 or not generated_ids:
        return logits

    adjusted = logits.clone()
    for token_id in set(generated_ids):
        token_logits = adjusted[:, token_id]
        adjusted[:, token_id] = torch.where(
            token_logits < 0,
            token_logits * penalty,
            token_logits / penalty,
        )
    return adjusted


def _filter_logits(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    # 未保留的 token 設為 -inf，sampling 與 logprob 都用這個分布。
    filtered = logits
    vocabulary_size = filtered.size(-1)

    # 先保留分數最高的 k 個 token；k=0 代表停用這一步。
    if top_k > 0 and top_k < vocabulary_size:
        cutoff = torch.topk(filtered, top_k, dim=-1).values[:, -1:]
        filtered = filtered.masked_fill(filtered < cutoff, float("-inf"))

    # 再依累積機率做 nucleus 過濾，保留跨過 top_p 的第一個 token。
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_remove = cumulative > top_p
        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
        sorted_remove[:, 0] = False
        remove_mask = torch.zeros_like(sorted_remove).scatter(
            1, sorted_indices, sorted_remove
        )
        filtered = filtered.masked_fill(remove_mask, float("-inf"))

    return filtered


def token_distribution_kl(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    # KL(policy || reference)；reference 不參與梯度計算。
    if policy_logits.ndim != 2 or policy_logits.shape != reference_logits.shape:
        raise ValueError("policy 與 reference logits 必須是相同的二維形狀。")
    # 只保留兩組 logits 都是有限值的位置，再用 log_softmax 計算 KL。
    finite_mask = torch.isfinite(policy_logits) & torch.isfinite(reference_logits)
    if not finite_mask.any(dim=-1).all():
        raise ValueError("policy 或 reference 沒有可用的有限 logits。")
    policy_safe = policy_logits.float().masked_fill(~finite_mask, -1e9)
    reference_safe = reference_logits.detach().float().masked_fill(
        ~finite_mask,
        -1e9,
    )
    policy_log_probs = F.log_softmax(policy_safe, dim=-1)
    reference_log_probs = F.log_softmax(
        reference_safe,
        dim=-1,
    )
    policy_probs = policy_log_probs.exp()
    per_sample = torch.sum(
        policy_probs * (policy_log_probs - reference_log_probs),
        dim=-1,
    )
    return per_sample.mean().clamp_min(0.0)


def has_complete_sentence_end(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or not TERMINAL_SENTENCE_PATTERN.search(stripped):
        return False
    if INCOMPLETE_TITLE_PATTERN.search(stripped):
        return False

    # 等外層引號或括號閉合後再停止。
    if stripped.count('"') % 2 != 0:
        return False
    if stripped.count("“") > stripped.count("”"):
        return False
    if stripped.count("(") > stripped.count(")"):
        return False
    if stripped.count("[") > stripped.count("]"):
        return False
    return True


def _clean_generated_text(text: str) -> str:
    cleaned = text.strip()
    positions = [cleaned.find(marker) for marker in STOP_MARKERS if marker in cleaned]
    if positions:
        cleaned = cleaned[: min(positions)].strip()

    if cleaned.startswith('"'):
        closing_quote = cleaned.find('"', 1)
        if closing_quote >= 0:
            cleaned = cleaned[1:closing_quote].strip()
    return cleaned.strip().strip('"').strip("'").strip()


class EmotionGenerator(nn.Module):
    # Warm-up 與 RL 共用這個模板；更動後需重訓 Prefix。
    PROMPT_TEMPLATE = """Transform the Input into {emotion} style.
        Constraint: Output ONLY the rewritten sentence. No explanations. No options.

        Input: "I just won a prize."
        Emotion: excited
        Response: I am absolutely ecstatic that I won this prize!

        Input: "My car broke down."
        Emotion: frustration
        Response: This stupid car died on me again, what a nightmare.

        Input: "The weather is gray."
        Emotion: sadness
        Response: The depressing, gloomy sky makes everything feel hopeless.

        Input: "{prompt}"
        Emotion: {emotion}
        Response:
            """

    def __init__(
        self,
        model_path: str,
        emotion_labels: Sequence[str],
        device: torch.device | None = None,
        prefix_length: int = 10,
        model_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        normalized_labels = [str(label).strip().lower() for label in emotion_labels]
        if not normalized_labels:
            raise ValueError("emotion_labels 不可為空。")
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ValueError("emotion_labels 不可包含重複標籤。")
        if prefix_length <= 0:
            raise ValueError("prefix_length 必須大於 0。")

        self.device = device or _select_device()
        self.prefix_length = int(prefix_length)
        # 讀取 tokenizer；缺少 pad 或 BOS 時沿用 EOS。
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id

        load_kwargs: dict[str, object] = {
            "torch_dtype": model_dtype,
            "low_cpu_mem_usage": True,
        }
        # MPS 的 float16 推論直接載入裝置，減少搬移模型時的記憶體峰值。
        direct_mps_load = (
            self.device.type == "mps" and model_dtype == torch.float16
        )
        if direct_mps_load:
            load_kwargs["device_map"] = {"": "mps"}
        else:
            load_kwargs["device_map"] = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )
        if not direct_mps_load:
            self.model = self.model.to(self.device)
        # 底模不更新權重，也關閉 dropout；梯度仍可經底模傳回 Prefix。
        self.model.eval()
        self.model.requires_grad_(False)

        self.hidden_size = int(self.model.config.hidden_size)
        # 固定情緒與 Prefix 列的對應，存取 checkpoint 時使用相同順序。
        self.emotion2id = {
            emotion: index for index, emotion in enumerate(normalized_labels)
        }
        self.id2emotion = {
            index: emotion for emotion, index in self.emotion2id.items()
        }
        self._initialize_prefix_embeddings(normalized_labels)

        logger.info("EmotionGenerator initialized on %s", self.device)
        logger.info("Emotions: %s", normalized_labels)
        logger.info("Prefix shape: %s", self.prefix_shape)

    def _initialize_prefix_embeddings(self, emotion_labels: Sequence[str]) -> None:
        # 以 "feeling {emotion}" 的 embedding 初始化，不足處補 pad。
        embeddings: list[torch.Tensor] = []
        word_embeddings = self.model.get_input_embeddings()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            for emotion in emotion_labels:
                token_ids = self.tokenizer.encode(
                    f"feeling {emotion} ",
                    add_special_tokens=False,
                )
                token_ids = (token_ids + [pad_id] * self.prefix_length)[
                    : self.prefix_length
                ]
                ids_tensor = torch.tensor(token_ids, device=self.device)
                embeddings.append(word_embeddings(ids_tensor))

        # 每種情緒各存成獨立 Parameter，不把所有情緒當成一個可更新矩陣。
        initial_weights = torch.stack(embeddings).to(dtype=torch.float32)
        self.prefix_parameters = nn.ParameterList(
            [nn.Parameter(row.clone()) for row in initial_weights]
        )

    @property
    def prefix_embeddings(self) -> torch.Tensor:
        # checkpoint 使用 [emotion, prefix, hidden] 的形狀。
        return torch.stack(list(self.prefix_parameters), dim=0)

    @property
    def prefix_shape(self) -> tuple[int, int, int]:
        return (len(self.prefix_parameters), self.prefix_length, self.hidden_size)

    def trainable_prefix_parameters(self) -> list[nn.Parameter]:
        # optimizer 只取得這些 Prefix，沒有底模權重。
        return list(self.prefix_parameters)

    def get_prefix_parameter(self, emotion_label: str) -> nn.Parameter:
        # 每個情緒用獨立 Parameter，未使用的情緒維持 grad=None。
        return self.prefix_parameters[self._emotion_id(emotion_label)]

    def train(self, mode: bool = True) -> "EmotionGenerator":
        # 只切換 Prefix 的訓練狀態，底模維持 eval。
        super().train(mode)
        self.model.eval()
        return self

    def format_prompt(self, emotion: str, prompt: str) -> str:
        return self.PROMPT_TEMPLATE.format(emotion=emotion, prompt=prompt)

    def _emotion_id(self, emotion_label: str) -> int:
        normalized = str(emotion_label).strip().lower()
        if normalized not in self.emotion2id:
            supported = ", ".join(self.emotion2id)
            raise ValueError(f"未知情緒 {emotion_label!r}；可用標籤：{supported}")
        return self.emotion2id[normalized]

    def load_adapter(self, path: str | Path) -> bool:
        # 用 copy_ 載入權重，保留 optimizer 持有的 Parameter。
        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.is_file():
            logger.warning("Adapter not found: %s", checkpoint_path)
            return False

        try:
            # 接受單獨的 Prefix Tensor，或含 prefix_embeddings 的 checkpoint 字典。
            payload = torch.load(checkpoint_path, map_location="cpu")
            loaded = (
                payload.get("prefix_embeddings")
                if isinstance(payload, dict)
                else payload
            )
            if not isinstance(loaded, torch.Tensor):
                raise TypeError("checkpoint 未包含 prefix_embeddings tensor")
            # 情緒數、Prefix 長度與隱藏維度需和目前模型一致。
            if tuple(loaded.shape) != self.prefix_shape:
                logger.error(
                    "Adapter shape mismatch: loaded=%s expected=%s",
                    tuple(loaded.shape),
                    self.prefix_shape,
                )
                return False

            with torch.no_grad():
                loaded = loaded.to(device=self.device, dtype=torch.float32)
                for emotion_id, parameter in enumerate(self.prefix_parameters):
                    parameter.copy_(loaded[emotion_id])
            logger.info("Adapter loaded from %s", checkpoint_path)
            return True
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Failed to load adapter %s: %s", checkpoint_path, exc)
            return False

    def get_prefix_and_input_embeds(
        self,
        emotion_label: str,
        prompt_text: str,
        formatted_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # 輸入順序為 Prefix、BOS、prompt；回傳 embedding、mask 和文字 token 數。
        formatted = formatted_prompt or self.format_prompt(emotion_label, prompt_text)
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(
            formatted,
            add_special_tokens=False,
        )
        input_tensor = torch.tensor([input_ids], device=self.device)
        # 文字 embedding 不需梯度，只有接在前面的 Prefix 需要更新。
        with torch.no_grad():
            input_embeds = self.model.get_input_embeddings()(input_tensor)

        emotion_id = self._emotion_id(emotion_label)
        prefix_embeds = self.prefix_parameters[emotion_id].unsqueeze(0)
        prefix_embeds = prefix_embeds.to(input_embeds.dtype)
        # 沿序列維度串接，mask 長度也包含 Prefix。
        inputs_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            device=self.device,
            dtype=torch.long,
        )
        return inputs_embeds, attention_mask, len(input_ids)

    def generate_with_logprobs(
        self,
        emotion_label: str,
        prompt_text: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        min_new_tokens: int = 3,
        repetition_penalty: float = 1.2,
        stop_after_first_sentence: bool = True,
        *,
        use_prefix: bool = True,
        reference_prefix: torch.Tensor | None = None,
        formatted_prompt: str | None = None,
    ) -> dict[str, str | int | torch.Tensor]:
        # reference_prefix 用於逐 token KL；純提示基準不加入 Prefix。
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必須大於 0。")
        if temperature <= 0:
            raise ValueError("temperature 必須大於 0。")
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens 不可為負數。")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p 必須位於 (0, 1]。")
        if reference_prefix is not None and not use_prefix:
            raise ValueError("純提示模式不可同時提供 reference_prefix。")

        normalized_emotion = str(emotion_label).strip().lower()
        self._emotion_id(normalized_emotion)
        formatted = formatted_prompt or self.format_prompt(
            normalized_emotion,
            prompt_text,
        )

        # Prefix 模式以 embedding 輸入；純提示基準則直接輸入 token ID。
        if use_prefix:
            context_embeds, attention_mask, _ = self.get_prefix_and_input_embeds(
                normalized_emotion,
                prompt_text,
                formatted,
            )
            first_step_inputs: dict[str, torch.Tensor] = {
                "inputs_embeds": context_embeds
            }
        else:
            input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                formatted,
                add_special_tokens=False,
            )
            input_tensor = torch.tensor([input_ids], device=self.device)
            attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
            first_step_inputs = {"input_ids": input_tensor}

        reference_first_step_inputs: dict[str, torch.Tensor] | None = None
        reference_attention_mask: torch.Tensor | None = None
        # 固定參考 Prefix 與目前 Prefix 共用同一段提示文字。
        if reference_prefix is not None:
            if reference_prefix.ndim == 3:
                if tuple(reference_prefix.shape) != self.prefix_shape:
                    raise ValueError(
                        "reference_prefix shape 不符："
                        f"{tuple(reference_prefix.shape)} != {self.prefix_shape}"
                    )
                reference_active = reference_prefix[
                    self._emotion_id(normalized_emotion)
                ]
            elif reference_prefix.ndim == 2:
                expected = (self.prefix_length, self.hidden_size)
                if tuple(reference_prefix.shape) != expected:
                    raise ValueError(
                        "reference_prefix shape 不符："
                        f"{tuple(reference_prefix.shape)} != {expected}"
                    )
                reference_active = reference_prefix
            else:
                raise ValueError("reference_prefix 必須是二維或三維 tensor。")
            static_text_embeds = context_embeds[:, self.prefix_length :, :].detach()
            reference_context = torch.cat(
                [
                    reference_active.to(
                        device=self.device,
                        dtype=static_text_embeds.dtype,
                    ).detach().unsqueeze(0),
                    static_text_embeds,
                ],
                dim=1,
            )
            reference_first_step_inputs = {"inputs_embeds": reference_context}
            reference_attention_mask = attention_mask.detach().clone()

        # 目前策略與參考策略各保留自己的 KV cache。
        past_key_values = None
        reference_past_key_values = None
        next_token: torch.Tensor | None = None
        generated_ids: list[int] = []
        logprob_terms: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        # 只有 Prefix 訓練保留生成路徑的梯度，推論與純提示基準不保留。
        requires_policy_grad = bool(self.training and use_prefix)
        stop_reason = "max_new_tokens"

        # 每次產生一個 token；第一步讀完整提示，後續沿用 KV cache。
        for step in range(max_new_tokens):
            with torch.set_grad_enabled(requires_policy_grad):
                if step == 0:
                    outputs = self.model(
                        **first_step_inputs,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                else:
                    outputs = self.model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                # 取最後位置預測下一個 token，套用溫度與重複懲罰。
                policy_logits = outputs.logits[:, -1, :] / temperature
                policy_logits = _apply_repetition_penalty(
                    policy_logits,
                    generated_ids,
                    repetition_penalty,
                )
                # 未達最少 token 數前，先遮住 EOS。
                if (
                    len(generated_ids) < min_new_tokens
                    and self.tokenizer.eos_token_id is not None
                ):
                    policy_logits = policy_logits.clone()
                    policy_logits[:, self.tokenizer.eos_token_id] = float("-inf")

                # 參考策略沿用同一個已生成前綴，不另抽另一條句子。
                if reference_first_step_inputs is not None:
                    with torch.no_grad():
                        if step == 0:
                            reference_outputs = self.model(
                                **reference_first_step_inputs,
                                attention_mask=reference_attention_mask,
                                use_cache=True,
                            )
                        else:
                            reference_outputs = self.model(
                                input_ids=next_token,
                                attention_mask=reference_attention_mask,
                                past_key_values=reference_past_key_values,
                                use_cache=True,
                            )
                        reference_logits = (
                            reference_outputs.logits[:, -1, :] / temperature
                        )
                        reference_logits = _apply_repetition_penalty(
                            reference_logits,
                            generated_ids,
                            repetition_penalty,
                        )
                        if (
                            len(generated_ids) < min_new_tokens
                            and self.tokenizer.eos_token_id is not None
                        ):
                            reference_logits = reference_logits.clone()
                            reference_logits[:, self.tokenizer.eos_token_id] = float(
                                "-inf"
                            )
                    kl_terms.append(
                        token_distribution_kl(policy_logits, reference_logits)
                    )
                    reference_past_key_values = reference_outputs.past_key_values

                # sampling 與所記錄的 log probability 都使用相同的 top-k／top-p 分布。
                filtered_logits = _filter_logits(policy_logits, top_k, top_p)
                log_probs = torch.log_softmax(filtered_logits, dim=-1)

            # 抽樣本身不反向傳播；選中 token 的 log probability 仍保留梯度。
            probabilities = torch.softmax(filtered_logits.detach(), dim=-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Sampling probabilities 包含 NaN 或 infinity。")
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            logprob_terms.append(log_probs.gather(1, next_token).squeeze())
            past_key_values = outputs.past_key_values
            # 增加一個 token 後，同步延長目前策略與參考策略的 attention mask。
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        device=self.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
            if reference_attention_mask is not None:
                reference_attention_mask = torch.cat(
                    [
                        reference_attention_mask,
                        torch.ones(
                            (1, 1),
                            device=self.device,
                            dtype=reference_attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )

            if (
                token_id == self.tokenizer.eos_token_id
                and len(generated_ids) >= min_new_tokens
            ):
                stop_reason = "eos"
                break

            # 依設定在第一個完整句末停止，不一定要等到 EOS。
            if stop_after_first_sentence:
                partial_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                if has_complete_sentence_end(partial_text):
                    stop_reason = "sentence_end"
                    break

        # 解碼成文字供評分；log probability 則對整條生成軌跡加總。
        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        text = _clean_generated_text(decoded)
        if logprob_terms:
            logprob_sum = torch.stack(logprob_terms).sum()
        else:
            logprob_sum = torch.zeros(
                (),
                device=self.device,
                requires_grad=requires_policy_grad,
            )
        # KL 對已比較的 token 取平均，回傳時一併保留計算筆數。
        kl_mean = (
            torch.stack(kl_terms).mean()
            if kl_terms
            else torch.zeros((), device=self.device)
        )

        return {
            "text": text,
            "logprob_sum": logprob_sum,
            "kl_mean": kl_mean,
            "kl_token_count": len(kl_terms),
            "token_count": len(generated_ids),
            "use_prefix": use_prefix,
            "stop_reason": stop_reason,
        }

    def generate_without_prefix(
        self,
        emotion_label: str,
        prompt_text: str,
        **generation_kwargs: object,
    ) -> dict[str, str | int | torch.Tensor]:
        # 基準方法沿用相同解碼流程，只關閉 Prefix。
        return self.generate_with_logprobs(
            emotion_label,
            prompt_text,
            use_prefix=False,
            **generation_kwargs,
        )

    def get_grad_norm(self) -> float:
        # 將各情緒已存在的梯度平方加總，得到整體 L2 norm。
        squared_norm = sum(
            float(parameter.grad.detach().float().pow(2).sum().item())
            for parameter in self.prefix_parameters
            if parameter.grad is not None
        )
        return math.sqrt(squared_norm) if squared_norm > 0.0 else 0.0

    def gradient_emotions(self) -> list[str]:
        # 列出這次梯度累積涉及的情緒，供訓練紀錄使用。
        return [
            self.id2emotion[index]
            for index, parameter in enumerate(self.prefix_parameters)
            if parameter.grad is not None
        ]
