# Threshold 檢查與多維 reward 評分。

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from RL_main.utils import get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 兩個分類器的標籤不同，先各自合併到同一組六類情緒。
EMOTION_CLUSTERS = {
    "joy": {
        "roberta": [
            "joy",
            "amusement",
            "approval",
            "excitement",
            "gratitude",
            "love",
            "optimism",
            "pride",
            "relief",
            "admiration",
        ],
        "distilbert": ["joy", "love"],
    },
    "sadness": {
        "roberta": [
            "sadness",
            "disappointment",
            "embarrassment",
            "grief",
            "remorse",
        ],
        "distilbert": ["sadness"],
    },
    "anger": {
        "roberta": ["anger", "annoyance", "disapproval", "disgust"],
        "distilbert": ["anger"],
    },
    "fear": {
        "roberta": ["fear", "nervousness"],
        "distilbert": ["fear"],
    },
    "surprise": {
        "roberta": ["surprise", "realization", "confusion", "curiosity"],
        "distilbert": ["surprise"],
    },
    "neutral": {
        "roberta": ["neutral"],
        "distilbert": ["neutral"],
    },
}

# 攔下「以下是改寫」或評論句子本身的文字，不把它們當成改寫結果。
META_COMMENTARY_PATTERNS = (
    re.compile(r"\b(?:the|this) (?:sentence|statement|phrase|text)\b", re.I),
    re.compile(r"\bthe original (?:sentence|statement|text|input)\b", re.I),
    re.compile(r"^(?:here(?:'s| is)|rewritten (?:sentence|version)|rewrite:)\b", re.I),
    re.compile(r"\b(?:this|it) (?:expresses|conveys|demonstrates)\b", re.I),
    re.compile(r"\btarget emotion\b", re.I),
)

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
SENTENCE_END_PATTERN = re.compile(r"[.!?]+(?:[\"')\]]+)?(?=\s|$)")
TERMINAL_PUNCTUATION_PATTERN = re.compile(r"[.!?]+(?:[\"')\]]+)?$")
DANGLING_END_PATTERN = re.compile(
    r"(?:[,;:\-]|\b(?:and|or|but|because|although|which|that|to|with|of|for))\s*$",
    re.I,
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.:/]\d+)*\b")
NUMBER_WORD_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}

REPETITION_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "she",
    "so",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}

PROTECTED_ABBREVIATION_PATTERN = re.compile(
    r"\b(?:a\.m|p\.m|e\.g|i\.e|mr|mrs|ms|dr|prof|sr|jr)\.",
    re.I,
)


def tokenize_words(text: str) -> list[str]:
    return [token.lower() for token in WORD_PATTERN.findall(text or "")]


def extract_numeric_details(text: str) -> set[str]:
    # 數字詞與數字用同一表示法，例如 nine → 9。
    details = set(NUMBER_PATTERN.findall(text or ""))
    details.update(
        NUMBER_WORD_MAP[word]
        for word in tokenize_words(text)
        if word in NUMBER_WORD_MAP
    )
    return details


def count_sentences(text: str) -> int:
    # a.m. 等縮寫裡的句點不算句子結尾。
    protected = PROTECTED_ABBREVIATION_PATTERN.sub(
        lambda match: match.group(0).replace(".", "<DOT>"),
        text or "",
    )
    return len(SENTENCE_END_PATTERN.findall(protected))


def analyze_response_quality(
    prompt: str,
    response: str,
    *,
    max_length_ratio: float = 1.7,
    max_extra_words: int = 8,
    min_length_ratio: float = 0.5,
    max_sentences: int = 1,
    require_terminal_punctuation: bool = True,
) -> dict[str, Any]:
    # 這裡只查格式、長度與重複等文字規則，不判斷語意或情緒。
    prompt_words = tokenize_words(prompt)
    response_words = tokenize_words(response)
    stripped = (response or "").strip()
    source_count = len(prompt_words)
    response_count = len(response_words)
    # 長度上限取「原句多幾詞」與「原句長度倍率」中較寬的一個。
    maximum_words = max(
        source_count + max_extra_words,
        math.ceil(source_count * max_length_ratio),
    )
    minimum_words = max(2, math.floor(source_count * min_length_ratio))
    sentence_count = count_sentences(stripped)
    has_terminal_punctuation = bool(
        TERMINAL_PUNCTUATION_PATTERN.search(stripped)
    )
    meta_matches = [
        pattern.pattern
        for pattern in META_COMMENTARY_PATTERNS
        if pattern.search(stripped)
    ]

    # 扣掉常見功能詞後，檢查單詞反覆出現及相鄰片語重複。
    content_words = [
        word for word in response_words if word not in REPETITION_STOP_WORDS
    ]
    content_counts = Counter(content_words)
    repeated_content_count = max(content_counts.values(), default=0)
    repeated_bigram = any(
        response_words[index : index + 2]
        == response_words[index + 2 : index + 4]
        for index in range(max(0, len(response_words) - 3))
    )
    repetition_ratio = 1.0 - (
        len(set(content_words)) / max(1, len(content_words))
    )

    # 比較原句與改寫句的數字，找出改寫時新增的數字細節。
    source_numbers = extract_numeric_details(prompt)
    response_numbers = extract_numeric_details(response)
    unsupported_numbers = sorted(response_numbers - source_numbers)

    # 把不合格原因與較輕微的警告分開，供後面的 Threshold 使用。
    fail_reasons: list[str] = []
    warnings: list[str] = []
    if not stripped or not response_words:
        fail_reasons.append("empty_response")
    if require_terminal_punctuation and not has_terminal_punctuation:
        fail_reasons.append("missing_terminal_punctuation")
    if sentence_count > max_sentences:
        fail_reasons.append("multiple_sentences")
    if response_count > maximum_words:
        fail_reasons.append("response_too_long")
    if response_words and response_count < minimum_words:
        fail_reasons.append("response_too_short")
    if meta_matches:
        fail_reasons.append("meta_commentary")
    without_terminal = TERMINAL_PUNCTUATION_PATTERN.sub("", stripped).strip()
    if without_terminal and DANGLING_END_PATTERN.search(without_terminal):
        fail_reasons.append("incomplete_response")
    if repeated_content_count >= 3 or repeated_bigram:
        fail_reasons.append("excessive_repetition")
    elif repetition_ratio > 0.35:
        warnings.append("lexical_repetition")
    if unsupported_numbers:
        fail_reasons.append("unsupported_numeric_detail")

    # 比原句最多多三詞時精簡度為 1，之後隨額外詞數遞減。
    if response_count <= source_count + 3:
        concision_score = 1.0
    else:
        available_extra = max(1, maximum_words - source_count - 3)
        excess = response_count - source_count - 3
        concision_score = max(0.0, 1.0 - excess / available_extra)
    if response_count < minimum_words:
        concision_score = 0.0

    return {
        "fail_reasons": fail_reasons,
        "warnings": warnings,
        "source_word_count": source_count,
        "response_word_count": response_count,
        "minimum_words": minimum_words,
        "maximum_words": maximum_words,
        "length_ratio": response_count / max(1, source_count),
        "sentence_count": sentence_count,
        "has_terminal_punctuation": has_terminal_punctuation,
        "meta_commentary_matches": meta_matches,
        "repetition_ratio": repetition_ratio,
        "unsupported_numbers": unsupported_numbers,
        "concision_score": concision_score,
    }


class EmotionScorer:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        load_models: bool = True,
    ) -> None:
        # load_models=False 可跳過模型載入。
        self.config = config or {}
        self.logger = get_logger(
            "EmotionScorer",
            log_file=PROJECT_ROOT / "logs" / "scorer.log",
        )
        self.device, self.pipeline_device = self._select_device()

        scorer_config = self.config.get("scorer", {})
        threshold_config = scorer_config.get(
            "threshold",
            scorer_config.get("gate", {}),
        )
        reward_config = scorer_config.get("reward", {})
        model_config = scorer_config.get("models", {})

        # 語意與照抄門檻，以及允許的句長和句數。
        self.sem_threshold = float(
            threshold_config.get("semantic_min", 0.60)
        )
        self.cosine_threshold = float(
            threshold_config.get("cosine_min", 0.58)
        )
        self.contradiction_hard_fail = bool(
            threshold_config.get("contradiction_hard_fail", True)
        )
        self.parrot_threshold_moderate = float(
            threshold_config.get("parrot_soft_start", 0.75)
        )
        self.parrot_threshold_severe = float(
            threshold_config.get("parrot_severe", 0.85)
        )
        self.max_length_ratio = float(
            threshold_config.get("max_length_ratio", 1.7)
        )
        self.max_extra_words = int(
            threshold_config.get("max_extra_words", 8)
        )
        self.min_length_ratio = float(
            threshold_config.get("min_length_ratio", 0.5)
        )
        self.max_sentences = int(threshold_config.get("max_sentences", 1))
        self.require_terminal_punctuation = bool(
            threshold_config.get("require_terminal_punctuation", True)
        )
        self.require_all_models = bool(
            scorer_config.get("require_all_models", True)
        )
        self.strict_runtime = bool(scorer_config.get("strict_runtime", True))

        # reward 的權重、尺度與分數上限。
        self.reward_mode = str(
            reward_config.get("mode", "bounded_multidimensional")
        )
        self.W_EMOTION = float(reward_config.get("emotion_weight", 0.35))
        self.semantic_weight = float(
            reward_config.get("semantic_weight", 0.30)
        )
        self.W_FLUENCY = float(reward_config.get("fluency_weight", 0.20))
        self.concision_weight = float(
            reward_config.get("concision_weight", 0.15)
        )
        self.emotion_gain_cap = float(
            reward_config.get("emotion_gain_cap", 0.50)
        )
        self.semantic_reward_floor = float(
            reward_config.get("semantic_reward_floor", self.sem_threshold)
        )
        self.semantic_reward_ceiling = float(
            reward_config.get("semantic_reward_ceiling", 0.85)
        )
        self.fluency_delta_scale = float(
            reward_config.get("fluency_delta_scale", 0.25)
        )
        self.parrot_soft_penalty_max = float(
            reward_config.get("parrot_soft_penalty_max", 0.25)
        )
        self.nli_asymmetry_penalty = float(
            reward_config.get("nli_asymmetry_penalty", 0.05)
        )
        self.minimum_positive_gain = float(
            reward_config.get("minimum_positive_gain", 0.05)
        )
        self.non_positive_gain_reward_cap = float(
            reward_config.get("non_positive_gain_reward_cap", 0.0)
        )
        self.small_gain_reward_cap = float(
            reward_config.get("small_gain_reward_cap", 0.15)
        )
        self.threshold_fail_reward = float(
            reward_config.get("threshold_fail_reward", -0.50)
        )
        self.severe_fail_reward = float(
            reward_config.get("severe_fail_reward", -0.80)
        )
        self.reward_min = float(reward_config.get("reward_min", -1.0))
        self.reward_max = float(reward_config.get("reward_max", 1.0))

        # 各評分維度使用的模型，可在設定檔中替換。
        self.model_names = {
            "semantic": model_config.get(
                "semantic",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            "nli": model_config.get(
                "nli",
                "cross-encoder/nli-deberta-v3-small",
            ),
            "emotion_roberta": model_config.get(
                "emotion_roberta",
                "SamLowe/roberta-base-go_emotions",
            ),
            "emotion_distilbert": model_config.get(
                "emotion_distilbert",
                "bhadresh-savani/distilbert-base-uncased-emotion",
            ),
            "fluency": model_config.get("fluency", "gpt2"),
        }
        # NLI 類別先換成分數，再依模型輸出的機率加權。
        self.nli_score_map = {
            "entailment": 1.0,
            "neutral": 0.6,
            "contradiction": 0.2,
        }

        self._validate_config()
        self._reset_models()
        if load_models:
            self._load_models()
        self.logger.info(
            "EmotionScorer ready | device=%s | mode=%s | models=%s",
            self.device,
            self.reward_mode,
            "loaded" if load_models else "skipped",
        )

    def _validate_config(self) -> None:
        # 檢查分數尺度與門檻順序；這些值會用在後面的比例換算。
        if self.reward_mode != "bounded_multidimensional":
            raise ValueError(
                "Scorer v11 僅支援 bounded_multidimensional reward mode。"
            )
        unit_values = {
            "semantic_min": self.sem_threshold,
            "cosine_min": self.cosine_threshold,
            "parrot_soft_start": self.parrot_threshold_moderate,
            "parrot_severe": self.parrot_threshold_severe,
            "semantic_reward_floor": self.semantic_reward_floor,
            "semantic_reward_ceiling": self.semantic_reward_ceiling,
        }
        invalid = [
            name for name, value in unit_values.items() if not 0.0 <= value <= 1.0
        ]
        if invalid:
            raise ValueError(f"Scorer 設定必須介於 0 到 1：{invalid}")
        if self.parrot_threshold_moderate >= self.parrot_threshold_severe:
            raise ValueError("parrot_soft_start 必須小於 parrot_severe。")
        if self.semantic_reward_floor >= self.semantic_reward_ceiling:
            raise ValueError(
                "semantic_reward_floor 必須小於 semantic_reward_ceiling。"
            )
        if self.max_length_ratio < 1.0 or self.max_extra_words < 0:
            raise ValueError("長度上限設定不合法。")
        if self.emotion_gain_cap <= 0.0 or self.fluency_delta_scale <= 0.0:
            raise ValueError("reward 正規化尺度必須大於 0。")
        if self.minimum_positive_gain < 0.0:
            raise ValueError("minimum_positive_gain 不得小於 0。")
        weights = self._normalized_reward_weights()
        if not weights:
            raise ValueError("至少需要一個正的 reward 權重。")

    @staticmethod
    def _select_device() -> tuple[torch.device, int]:
        if torch.cuda.is_available():
            return torch.device("cuda"), 0
        if torch.backends.mps.is_available():
            return torch.device("mps"), -1
        return torch.device("cpu"), -1

    def _reset_models(self) -> None:
        # 初始化子模型欄位，載入成功後再逐一填入。
        self.sem_model = None
        self.nli_model = None
        self.classifier_roberta = None
        self.classifier_distilbert = None
        self.fluency_tokenizer = None
        self.fluency_model = None

    def _load_models(self) -> None:
        # 分別載入語意、NLI、情緒及流暢度模型。
        errors: list[str] = []
        try:
            self.sem_model = SentenceTransformer(self.model_names["semantic"])
        except Exception as exc:
            errors.append(f"semantic: {exc}")

        try:
            # top_k=None 讓 pipeline 回傳所有類別分數，而不只最高分標籤。
            self.nli_model = pipeline(
                "text-classification",
                model=self.model_names["nli"],
                device=self.pipeline_device,
                top_k=None,
            )
        except Exception as exc:
            errors.append(f"nli: {exc}")

        try:
            # 情緒由 RoBERTa 與 DistilBERT 兩個分類器共同評分。
            self.classifier_roberta = pipeline(
                "text-classification",
                model=self.model_names["emotion_roberta"],
                top_k=None,
                device=self.pipeline_device,
            )
        except Exception as exc:
            errors.append(f"emotion_roberta: {exc}")

        try:
            self.classifier_distilbert = pipeline(
                "text-classification",
                model=self.model_names["emotion_distilbert"],
                top_k=None,
                device=self.pipeline_device,
            )
        except Exception as exc:
            errors.append(f"emotion_distilbert: {exc}")

        try:
            # GPT-2 用自身文字的交叉熵估計流暢度，不參與訓練更新。
            model_name = self.model_names["fluency"]
            self.fluency_tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.fluency_tokenizer.pad_token is None:
                self.fluency_tokenizer.pad_token = self.fluency_tokenizer.eos_token
            self.fluency_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.fluency_model.to(self.device)
            self.fluency_model.eval()
        except Exception as exc:
            self.fluency_tokenizer = None
            self.fluency_model = None
            errors.append(f"fluency: {exc}")

        # 彙整載入錯誤，依 require_all_models 決定是否停止。
        if errors:
            message = "評分模型載入不完整：" + " | ".join(errors)
            if self.require_all_models:
                raise RuntimeError(message)
            self.logger.error(message)

    def _metric_error(self, metric: str, exc: Exception, fallback: float) -> float:
        # 推論錯誤依 strict_runtime 決定拋出例外或回傳預設分數。
        if self.strict_runtime:
            raise RuntimeError(f"{metric} 評分失敗：{exc}") from exc
        self.logger.error("%s 評分失敗，使用保守分數 %.3f：%s", metric, fallback, exc)
        return fallback

    def _get_emotion_score(self, text: str, target_emotion: str) -> float:
        # RoBERTa 與 DistilBERT 的分數按 0.7／0.3 加權。
        if not text or len(text.strip()) < 2:
            return 0.0
        target = target_emotion.lower()
        # 例如 joy 會合併 excitement、gratitude 等相關子情緒。
        cluster = EMOTION_CLUSTERS.get(
            target,
            {"roberta": [target], "distilbert": [target]},
        )
        roberta_score = self._classifier_cluster_score(
            self.classifier_roberta,
            text,
            cluster["roberta"],
            "RoBERTa",
        )
        distilbert_score = self._classifier_cluster_score(
            self.classifier_distilbert,
            text,
            cluster.get("distilbert", []),
            "DistilBERT",
        )
        if self.classifier_roberta is None:
            return distilbert_score
        if self.classifier_distilbert is None:
            return roberta_score
        return float(0.7 * roberta_score + 0.3 * distilbert_score)

    def _classifier_cluster_score(
        self,
        classifier: Any,
        text: str,
        labels: list[str],
        classifier_name: str,
    ) -> float:
        if classifier is None or not labels:
            if self.require_all_models:
                raise RuntimeError(f"{classifier_name} 情緒分類器未載入。")
            return 0.0
        try:
            result = classifier(text, truncation=True, max_length=512)
            if result and isinstance(result[0], list):
                result = result[0]
            if not result or not isinstance(result[0], dict):
                raise ValueError("分類器回傳格式不符預期。")
            # 只加總這個情緒群組的標籤機率，總分最高為 1。
            wanted = {label.lower() for label in labels}
            return float(
                min(
                    1.0,
                    sum(
                        float(item["score"])
                        for item in result
                        if str(item["label"]).lower() in wanted
                    ),
                )
            )
        except Exception as exc:
            return self._metric_error(classifier_name, exc, 0.0)

    def _calculate_fluency(self, text: str) -> float:
        # 將 GPT-2 的交叉熵映射到 0–1。
        if not text or len(text.strip()) < 3:
            return 0.0
        if self.fluency_model is None or self.fluency_tokenizer is None:
            if self.require_all_models:
                raise RuntimeError("GPT-2 流暢度模型未載入。")
            return 0.0
        try:
            inputs = self.fluency_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self.device)
            with torch.no_grad():
                output = self.fluency_model(
                    inputs["input_ids"],
                    labels=inputs["input_ids"],
                )
            # 交叉熵越低分數越高，最後裁切到 0–1。
            loss = float(output.loss.item())
            if not math.isfinite(loss):
                raise ValueError(f"GPT-2 loss 非有限值：{loss}")
            return max(0.0, min(1.0, 1.0 - loss / 8.0))
        except Exception as exc:
            return self._metric_error("fluency", exc, 0.0)

    def _calculate_cosine_similarity(self, prompt: str, response: str) -> float:
        # 將餘弦相似度從 [-1, 1] 映射到 [0, 1]。
        if self.sem_model is None:
            if self.require_all_models:
                raise RuntimeError("SentenceTransformer 未載入。")
            return 0.0
        try:
            embeddings = self.sem_model.encode(
                [prompt, response],
                convert_to_tensor=True,
            )
            cosine = torch.nn.functional.cosine_similarity(
                embeddings[0].unsqueeze(0),
                embeddings[1].unsqueeze(0),
            ).item()
            return float(max(0.0, min(1.0, (cosine + 1.0) / 2.0)))
        except Exception as exc:
            return self._metric_error("semantic cosine", exc, 0.0)

    @staticmethod
    def _normalize_nli_label(label: str) -> str:
        # 將模型的文字標籤或 LABEL_n 轉成統一的三類名稱。
        normalized = label.lower()
        if "contradict" in normalized or normalized == "label_0":
            return "contradiction"
        if "entail" in normalized or normalized == "label_1":
            return "entailment"
        if "neutral" in normalized or normalized == "label_2":
            return "neutral"
        raise ValueError(f"無法辨識 NLI 標籤：{label}")

    def _parse_nli_result(self, result: Any) -> dict[str, float | str]:
        # 整理 NLI 的標籤、信心值與機率加權分數。
        if isinstance(result, dict):
            candidates = [result]
        elif isinstance(result, list) and result:
            candidates = result[0] if isinstance(result[0], list) else result
        else:
            raise ValueError("NLI pipeline 回傳空值或未知格式。")
        if not candidates or not all(isinstance(item, dict) for item in candidates):
            raise ValueError("NLI pipeline 缺少標籤字典。")

        # 先合併同義標籤並正規化機率，再計算整體 NLI 分數。
        probabilities: dict[str, float] = {}
        for item in candidates:
            label = self._normalize_nli_label(str(item.get("label", "")))
            probabilities[label] = probabilities.get(label, 0.0) + float(
                item.get("score", 0.0)
            )
        total = sum(probabilities.values())
        if total <= 0.0:
            raise ValueError("NLI 標籤機率總和為零。")
        probabilities = {
            label: score / total for label, score in probabilities.items()
        }
        best_label = max(probabilities, key=probabilities.get)
        expected_score = sum(
            probabilities.get(label, 0.0) * mapped_score
            for label, mapped_score in self.nli_score_map.items()
        )
        return {
            "label": best_label,
            "confidence": probabilities[best_label],
            "score": expected_score,
        }

    def _calculate_nli_direction(
        self,
        premise: str,
        hypothesis: str,
        direction: str,
    ) -> dict[str, float | str]:
        # premise 是前提，hypothesis 是待判斷能否由前提支持的句子。
        if self.nli_model is None:
            if self.require_all_models:
                raise RuntimeError("NLI 模型未載入。")
            return {
                "direction": direction,
                "label": "contradiction",
                "confidence": 1.0,
                "score": 0.0,
            }
        try:
            raw = self.nli_model(
                {"text": premise, "text_pair": hypothesis},
                truncation=True,
            )
            parsed = self._parse_nli_result(raw)
            return {"direction": direction, **parsed}
        except Exception as exc:
            fallback = self._metric_error(f"NLI {direction}", exc, 0.0)
            return {
                "direction": direction,
                "label": "contradiction",
                "confidence": 1.0,
                "score": fallback,
            }

    def _calculate_semantic_scores(
        self,
        prompt: str,
        response: str,
    ) -> dict[str, Any]:
        # 雙向 NLI 取較低分，減少單向判斷漏掉新增內容的情況。
        cosine = self._calculate_cosine_similarity(prompt, response)
        # 原句推改寫、改寫推原句各做一次，檢查是否新增或遺漏內容。
        forward = self._calculate_nli_direction(
            prompt,
            response,
            "source_to_response",
        )
        reverse = self._calculate_nli_direction(
            response,
            prompt,
            "response_to_source",
        )
        conservative_nli = min(float(forward["score"]), float(reverse["score"]))
        # 句向量與雙向 NLI 各占一半，另保留方向差距作為軟懲罰依據。
        combined = 0.5 * cosine + 0.5 * conservative_nli
        return {
            "combined": combined,
            "cosine": cosine,
            "nli": conservative_nli,
            "forward": forward,
            "reverse": reverse,
            "asymmetry": abs(float(forward["score"]) - float(reverse["score"])),
        }

    def _apply_thresholds(
        self,
        parrot_score: float,
        semantic: dict[str, Any],
        quality: dict[str, Any],
    ) -> tuple[bool, list[str], list[str], float]:
        # 回傳通過與否、失敗原因、警告和軟懲罰。
        fail_reasons = list(quality["fail_reasons"])
        warnings = list(quality["warnings"])

        # 嚴重照抄直接失敗，較輕微的照抄只列警告並扣分。
        if parrot_score > self.parrot_threshold_severe:
            fail_reasons.append("parrot_severe")
        elif parrot_score > self.parrot_threshold_moderate:
            warnings.append("parrot_soft")

        # 組合語意分數與單獨的句向量分數都要達標。
        if float(semantic["combined"]) < self.sem_threshold:
            fail_reasons.append("semantic_drift")
        if float(semantic["cosine"]) < self.cosine_threshold:
            fail_reasons.append("low_cosine_similarity")

        forward_label = str(semantic["forward"]["label"])
        reverse_label = str(semantic["reverse"]["label"])
        # 任一方向判為矛盾時，依設定列為硬性失敗。
        if self.contradiction_hard_fail:
            if forward_label == "contradiction":
                fail_reasons.append("nli_contradiction_source_to_response")
            if reverse_label == "contradiction":
                fail_reasons.append("nli_contradiction_response_to_source")

        if float(semantic["asymmetry"]) >= 0.25:
            warnings.append("nli_direction_asymmetry")

        fail_reasons = list(dict.fromkeys(fail_reasons))
        warnings = list(dict.fromkeys(warnings))
        # 軟懲罰由照抄程度與 NLI 方向差距組成。
        soft_penalty = self._parrot_soft_penalty(parrot_score)
        if "nli_direction_asymmetry" in warnings:
            soft_penalty += self.nli_asymmetry_penalty
        return not fail_reasons, fail_reasons, warnings, soft_penalty

    def _parrot_soft_penalty(self, parrot_score: float) -> float:
        # 只在 soft 到 severe 的照抄區間內增加懲罰。
        if parrot_score <= self.parrot_threshold_moderate:
            return 0.0
        width = self.parrot_threshold_severe - self.parrot_threshold_moderate
        ratio = (parrot_score - self.parrot_threshold_moderate) / width
        return self.parrot_soft_penalty_max * min(1.0, max(0.0, ratio))

    def _normalized_reward_weights(self) -> dict[str, float]:
        # 負權重視為零，再將四個權重正規化成總和 1。
        raw = {
            "emotion": max(0.0, self.W_EMOTION),
            "semantic": max(0.0, self.semantic_weight),
            "fluency": max(0.0, self.W_FLUENCY),
            "concision": max(0.0, self.concision_weight),
        }
        total = sum(raw.values())
        if total <= 0.0:
            return {}
        return {name: value / total for name, value in raw.items()}

    def reward_weights(self) -> dict[str, float]:
        return self._normalized_reward_weights()

    def _calculate_reward_components(
        self,
        emotion_original: float,
        emotion_generated: float,
        fluency_original: float,
        fluency_generated: float,
        semantic_score: float,
        concision_score: float,
        soft_penalty: float,
    ) -> dict[str, float | str | dict[str, float]]:
        # 分別計算四個維度的分數、權重與加權貢獻。
        # 用改寫前後的情緒差值，而非只看改寫句的情緒分數。
        emotion_gain = emotion_generated - emotion_original
        emotion_component = max(
            -1.0,
            min(1.0, emotion_gain / self.emotion_gain_cap),
        )
        # 語意從設定的 floor 到 ceiling 線性換算；超過範圍就裁切。
        semantic_width = (
            self.semantic_reward_ceiling - self.semantic_reward_floor
        )
        semantic_component = max(
            0.0,
            min(
                1.0,
                (semantic_score - self.semantic_reward_floor) / semantic_width,
            ),
        )
        # 流暢度同時考慮改寫句本身，以及相對原句改善或退步的程度。
        fluency_delta = fluency_generated - fluency_original
        fluency_relative = max(
            -1.0,
            min(1.0, fluency_delta / self.fluency_delta_scale),
        )
        fluency_component = 0.6 * fluency_generated + 0.4 * fluency_relative
        fluency_component = max(-1.0, min(1.0, fluency_component))

        # 加總四維貢獻後扣除軟懲罰，這時還沒套用 Threshold 的失敗分數。
        weights = self._normalized_reward_weights()
        contributions = {
            "emotion": weights["emotion"] * emotion_component,
            "semantic": weights["semantic"] * semantic_component,
            "fluency": weights["fluency"] * fluency_component,
            "concision": weights["concision"] * concision_score,
        }
        reward_base = sum(contributions.values()) - soft_penalty
        return {
            "reward_gen": reward_base,
            "reward_orig": 0.0,
            "reward_base": reward_base,
            "emotion_component": emotion_component,
            "semantic_component": semantic_component,
            "fluency_component": fluency_component,
            "concision_component": concision_score,
            "emotion_level_reward": contributions["emotion"],
            "semantic_reward": contributions["semantic"],
            "fluency_reward": contributions["fluency"],
            "concision_reward": contributions["concision"],
            "improvement_bonus": contributions["emotion"],
            "weakening_penalty": max(0.0, -contributions["emotion"]),
            "parrot_penalty": soft_penalty,
            "soft_penalty": soft_penalty,
            "weights": weights,
            "reward_formula": "bounded_multidimensional_v1",
        }

    def score_single(
        self,
        prompt: str,
        response: str,
        target_emotion: str,
    ) -> dict[str, Any]:
        # 回傳各維度分數、Threshold 判斷和最終 reward。
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必須是非空字串。")
        if not isinstance(response, str):
            raise ValueError("response 必須是字串。")
        if not isinstance(target_emotion, str) or not target_emotion.strip():
            raise ValueError("target_emotion 必須是非空字串。")

        # 先查文字格式，再分別計算情緒、流暢度、照抄和語意。
        quality = analyze_response_quality(
            prompt,
            response,
            max_length_ratio=self.max_length_ratio,
            max_extra_words=self.max_extra_words,
            min_length_ratio=self.min_length_ratio,
            max_sentences=self.max_sentences,
            require_terminal_punctuation=self.require_terminal_punctuation,
        )
        emotion_original = self._get_emotion_score(prompt, target_emotion)
        emotion_generated = self._get_emotion_score(response, target_emotion)
        emotion_gain = emotion_generated - emotion_original
        fluency_original = self._calculate_fluency(prompt)
        fluency_generated = self._calculate_fluency(response)
        # 照抄程度使用字元序列相似度，與後面的語意相似度不同。
        parrot_score = SequenceMatcher(
            None,
            prompt.lower().strip(),
            response.lower().strip(),
        ).ratio()
        semantic = self._calculate_semantic_scores(prompt, response)

        # 新詞比例只作為紀錄，不直接當成越高越好的 reward。
        prompt_words = set(tokenize_words(prompt))
        response_words = set(tokenize_words(response))
        diversity_ratio = len(response_words - prompt_words) / max(
            1,
            len(response_words),
        )
        passed, fail_reasons, warnings, soft_penalty = self._apply_thresholds(
            parrot_score,
            semantic,
            quality,
        )

        components = self._calculate_reward_components(
            emotion_original,
            emotion_generated,
            fluency_original,
            fluency_generated,
            float(semantic["combined"]),
            float(quality["concision_score"]),
            soft_penalty,
        )
        # 依失敗原因區分一般負分與較重負分。
        severe_reasons = {
            "empty_response",
            "meta_commentary",
            "incomplete_response",
            "multiple_sentences",
            "response_too_long",
            "excessive_repetition",
            "unsupported_numeric_detail",
            "parrot_severe",
            "nli_contradiction_source_to_response",
            "nli_contradiction_response_to_source",
        }
        reward_cap_reason = "none"
        # 通過門檻仍需限制情緒增益太小的情況，不能只靠其他維度拿高分。
        if passed:
            reward_before_clip = float(components["reward_base"])
            if emotion_gain <= 0.0:
                reward_before_clip = min(
                    reward_before_clip,
                    self.non_positive_gain_reward_cap,
                )
                reward_cap_reason = "non_positive_emotion_gain"
            elif emotion_gain < self.minimum_positive_gain:
                reward_before_clip = min(
                    reward_before_clip,
                    self.small_gain_reward_cap,
                )
                reward_cap_reason = "small_emotion_gain"
        elif severe_reasons.intersection(fail_reasons):
            reward_before_clip = self.severe_fail_reward
            reward_cap_reason = "severe_threshold_failure"
        else:
            reward_before_clip = self.threshold_fail_reward
            reward_cap_reason = "threshold_failure"
        reward = float(
            max(self.reward_min, min(self.reward_max, reward_before_clip))
        )

        # 拆開保存原始量測、獎勵構成及失敗原因，供訓練紀錄查閱。
        breakdown = {
            **components,
            "diversity_bonus": 0.0,
            "threshold_penalty": soft_penalty,
            "gate_penalty": soft_penalty,
            "semantic_penalty": 0.0,
            "reward_before_cap": reward_before_clip,
            "reward_before_clip": reward_before_clip,
            "reward_cap_reason": reward_cap_reason,
        }
        raw = {
            "emo_orig": emotion_original,
            "emo_gen": emotion_generated,
            "emo_gain": emotion_gain,
            "flu_orig": fluency_original,
            "flu_gen": fluency_generated,
            "parrot": parrot_score,
            "sem_cosine": float(semantic["cosine"]),
            "sem_nli": float(semantic["nli"]),
            "sem_nli_forward": float(semantic["forward"]["score"]),
            "sem_nli_reverse": float(semantic["reverse"]["score"]),
            "sem_combined": float(semantic["combined"]),
            "nli_asymmetry": float(semantic["asymmetry"]),
            "diversity_ratio": diversity_ratio,
            "concision": float(quality["concision_score"]),
        }
        return {
            "raw": raw,
            "quality": quality,
            "semantic": semantic,
            "threshold_passed": passed,
            "threshold_fail_reasons": fail_reasons,
            "threshold_warnings": warnings,
            "gate_passed": passed,
            "gate_fail_reasons": fail_reasons,
            "reward": reward,
            "breakdown": breakdown,
        }

    def score_batch(
        self,
        prompts: list[str],
        responses: list[str],
        emotions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # details 欄位依序為：語意、情緒、流暢度、照抄程度。
        if not (len(prompts) == len(responses) == len(emotions)):
            raise ValueError("prompts、responses 與 emotions 長度必須相同。")

        details: list[list[float]] = []
        rewards: list[float] = []
        for prompt, response, emotion in zip(prompts, responses, emotions):
            result = self.score_single(prompt, response, emotion)
            raw = result["raw"]
            details.append(
                [
                    raw["sem_combined"],
                    raw["emo_gen"],
                    raw["flu_gen"],
                    raw["parrot"],
                ]
            )
            rewards.append(result["reward"])
        # 轉成放在評分器裝置上的 Tensor，供訓練迴圈使用。
        return (
            torch.tensor(details, dtype=torch.float32, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
        )
