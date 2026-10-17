# 呼叫 API 生成 Warm-up 目標句。

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
# API 提示詞限制改寫內容，JSON 回覆格式供後面的解析器使用。
SYSTEM_PROMPT = (
    "You create concise, natural rewrites for an emotion style-transfer "
    "research dataset. Preserve the source event and logic while making the "
    "requested emotion clear. Return only valid JSON."
)
USER_PROMPT_TEMPLATE = """Rewrite the input in {count} different ways.

Requirements:
1. Preserve the core event and logic of the input.
2. Strongly express the target emotion.
3. Keep each rewrite within 30 words.
4. Vary wording and sentence structure.
5. Do not add explanations, emojis, or facts absent from the input.

Input: "{prompt}"
Target emotion: {emotion}

Return exactly {count} objects as a JSON array:
[
  {{"style": "short style label", "response": "rewritten sentence"}}
]"""


def load_input_prompts(file_path: str | Path) -> list[dict[str, str]]:
    # 每筆需有 prompt、source_emotion 和不同於來源的 emotion。
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Warm-up input file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Warm-up prompt JSON must be an array.")
    # 清除空白並統一標籤大小寫，來源與目標情緒需不同。
    normalized: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        emotion = str(item.get("emotion") or "").strip().lower()
        if not prompt or not emotion:
            continue
        source = str(item.get("source_emotion") or "").strip().lower()
        if not source:
            raise ValueError("Every Warm-up prompt needs source_emotion.")
        if source == emotion:
            raise ValueError("Warm-up source and target emotions must differ.")
        normalized.append({"prompt": prompt, "source_emotion": source,
                           "emotion": emotion})
    if not normalized:
        raise ValueError("No valid Warm-up prompts were found.")
    return normalized


def validate_generation_plan(
    input_data: list[dict[str, str]],
    generation_config: dict[str, Any],
    minimum_samples_per_target: int,
) -> dict[str, int]:
    # 先用原句數 × 每句改寫數，檢查是否可能達到最低筆數。
    responses = int(generation_config.get("responses_per_prompt", 5))
    retries = int(generation_config.get("retry_limit", 3))
    if responses <= 0 or retries <= 0 or minimum_samples_per_target <= 0:
        raise ValueError("Response, retry, and minimum counts must be positive.")
    expected_source = str(generation_config.get("source_emotion", "neutral"))
    expected_targets = list(generation_config["samples_per_target_emotion"])
    if not expected_targets:
        raise ValueError("At least one target emotion is required.")
    for item in input_data:
        if item["source_emotion"] != expected_source:
            raise ValueError("Input source_emotion does not match the configuration.")
        if item["emotion"] not in expected_targets:
            raise ValueError("An input target is absent from the generation plan.")
    # 這是理論最高筆數；API 失敗和品質篩選還會減少結果。
    counts = Counter(item["emotion"] for item in input_data)
    maximum_pairs = {target: counts[target] * responses for target in expected_targets}
    insufficient = {target: count for target, count in maximum_pairs.items()
                    if count < minimum_samples_per_target}
    if insufficient:
        raise ValueError(
            f"Generation cannot reach {minimum_samples_per_target} pairs per target; "
            f"maximum pairs: {insufficient}. Increase source prompts or responses "
            "before making API requests."
        )
    return maximum_pairs


def parse_response_array(content: str) -> list[dict[str, str]]:
    # 接受 JSON array，也接受包在 Markdown code fence 裡的回覆。
    cleaned = content.strip()
    # 先拆掉 Markdown code fence，再取 JSON 陣列所在的範圍。
    code_fence = chr(96) * 3
    if code_fence in cleaned:
        parts = cleaned.split(code_fence)
        cleaned = parts[1] if len(parts) >= 3 else cleaned
        cleaned = cleaned.removeprefix("json").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end < start:
        raise ValueError("API response has no JSON array.")
    parsed = json.loads(cleaned[start:end + 1])
    if not isinstance(parsed, list):
        raise ValueError("API JSON must be an array.")
    # 只保留有實際改寫文字的項目，缺少 style 時使用預設值。
    responses: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        response = str(item.get("response") or "").strip()
        if response:
            responses.append({"style": str(item.get("style", "unspecified")).strip(),
                              "response": response})
    if not responses:
        raise ValueError("API response has no valid rewrites.")
    return responses


def call_model(
    client: OpenAI, prompt: str, emotion: str, *, model: str,
    response_count: int, retry_limit: int, retry_delay: float, temperature: float,
) -> list[dict[str, str]]:
    # 每次請求一個原句的多種改寫；回覆格式錯誤也會進入重試。
    user_prompt = USER_PROMPT_TEMPLATE.format(count=response_count, prompt=prompt,
                                            emotion=emotion)
    for attempt in range(1, retry_limit + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            content = completion.choices[0].message.content or ""
            # 最多取設定數量，避免 API 回傳額外項目改變資料規模。
            return parse_response_array(content)[:response_count]
        except Exception as exc:
            # 只印錯誤類型，API 錯誤內容可能包含請求資料。
            print(f"Generation attempt {attempt}/{retry_limit} failed "
                  f"({type(exc).__name__}).")
            if attempt < retry_limit:
                time.sleep(retry_delay)
    return []


def generate_dataset(
    client: OpenAI, input_data: list[dict[str, str]],
    generation_config: dict[str, Any],
) -> tuple[list[dict[str, str]], int, int]:
    # 沿用 Warm-up 的文字品質檢查，保留通過的改寫句。
    from tqdm import tqdm
    from RL_main.Scorer_NLI import analyze_response_quality

    model = str(generation_config.get("model", "gpt-4o-mini"))
    response_count = int(generation_config.get("responses_per_prompt", 5))
    retry_limit = int(generation_config.get("retry_limit", 3))
    retry_delay = float(generation_config.get("retry_delay", 1.0))
    call_delay = float(generation_config.get("call_delay", 1.0))
    temperature = float(generation_config.get("temperature", 1.0))
    if response_count <= 0 or retry_limit <= 0:
        raise ValueError("responses_per_prompt and retry_limit must be positive.")
    if retry_delay < 0 or call_delay < 0:
        raise ValueError("API delays must not be negative.")
    results: list[dict[str, str]] = []
    failed_count = invalid_response_count = 0
    # 逐個原句呼叫 API，並分開統計請求失敗與不合格改寫。
    for index, item in enumerate(tqdm(input_data, desc="Warm-up data")):
        responses = call_model(
            client, item["prompt"], item["emotion"], model=model,
            response_count=response_count, retry_limit=retry_limit,
            retry_delay=retry_delay, temperature=temperature,
        )
        if not responses:
            failed_count += 1
        for response in responses:
            quality = analyze_response_quality(item["prompt"], response["response"])
            if quality["fail_reasons"]:
                invalid_response_count += 1
                continue
            # 保存原句、來源情緒、目標情緒和 target_text，供 Warm-up 直接讀取。
            results.append({"prompt": item["prompt"],
                            "source_emotion": item["source_emotion"],
                            "emotion": item["emotion"],
                            "target_text": response["response"],
                            "style": response["style"], "source": model})
        if index + 1 < len(input_data) and call_delay > 0:
            time.sleep(call_delay)
    return results, failed_count, invalid_response_count


def save_new_dataset(results: list[dict[str, str]], output_path: Path) -> None:
    # 使用 x 模式建立新檔，不覆寫已存在的資料。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def main() -> None:
    # 先檢查路徑與筆數規劃，再建立 API client 生成資料。
    from RL_main.utils import load_config, resolve_path

    config = load_config(CONFIG_PATH)
    warmup = config["warmup"]
    generation = warmup["generate"]
    input_path = resolve_path(generation["input_path"], PROJECT_ROOT)
    output_path = resolve_path(warmup["dataset_path"], PROJECT_ROOT)
    # 輸出檔已存在就停止，另選路徑後再生成。
    if output_path.exists():
        raise FileExistsError(
            "Warm-up output already exists; keep it and choose a fresh "
            "warmup.dataset_path instead of overwriting it."
        )
    inputs = load_input_prompts(input_path)
    minimum = int(warmup.get("minimum_samples_per_target", 500))
    maximum_pairs = validate_generation_plan(inputs, generation, minimum)
    print(f"Planned maximum pairs per target: {maximum_pairs}; required: {minimum}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment before generation.")
    from openai import OpenAI

    results, failed_count, invalid_count = generate_dataset(
        OpenAI(api_key=api_key), inputs, generation,
    )
    if not results:
        raise RuntimeError("No accepted Warm-up pairs; no empty dataset was written.")
    # 先保存非空成果，再檢查每組筆數是否足夠開始訓練。
    accepted = Counter(item["emotion"] for item in results)
    save_new_dataset(results, output_path)
    print(f"Warm-up data: {len(results)} pairs, {failed_count} failed prompts, "
          f"{invalid_count} rejected rewrites; output={output_path}")
    print(f"Accepted pairs per target: {dict(accepted)}")
    insufficient = {target: accepted[target] for target in maximum_pairs
                    if accepted[target] < minimum}
    if insufficient:
        raise RuntimeError(
            f"Partial dataset saved for inspection, but targets below {minimum}: "
            f"{insufficient}. Do not start Warm-up; review or supplement the data."
        )


if __name__ == "__main__":
    main()
