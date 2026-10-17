# 訓練與重現流程

流程為：**轉換來源資料 → 抽取原句 → 生成目標改寫 → 檢查資料 → Warm-up → RL**。Llama 權重與 GoEmotions 需另行取得，並依原始條款使用。正式評估結果的適用範圍見 [研究限制](limitations.md)。

## 一、環境與設定

使用 Python 3.12，從專案根目錄安裝套件並建立本機設定檔：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml
```

將 `model.path` 指向已取得使用授權的本機 Llama-3.2-3B。範例設定中的相對路徑都以專案根目錄為基準。每次訓練使用新的 `train.output_dir`，並讓 `train.warmup_checkpoint` 指向該目錄產生的 `warmup_prefix.pt`。

## 二、轉換來源資料

從 [GoEmotions 官方資料夾](https://github.com/google-research/google-research/tree/master/goemotions)取得資料。`convert_dataset.py` 接受 JSON 陣列或 JSONL，每筆需有 `text`，以及以情緒名稱表示的 `emotion` 或 `emotions`：

```json
{"text": "The report was completed this morning.", "emotions": ["neutral"]}
```

程式不直接讀取官方 TSV／CSV、數字標籤或 one-hot 欄位，需先轉成上述格式，再設定 `data.raw_path`。多標籤資料取第一個標籤，因此標籤順序會影響結果。

```bash
python warmup_training/convert_dataset.py
```

輸出寫入 `data.path`，將原始情緒名稱合併為六類，例如 `{"text": "...", "emotions": ["neutral"]}`。共用讀取器接受 `emotions` 作為 `emotion` 的欄位別名；空白文字及未知標籤會略過。

## 三、準備來源句

```bash
python warmup_training/prepare_inputprompt.py
```

程式讀取 `warmup.generate.source_path`，整理標籤、去除重複文字，再依 `sample_seed` 抽取中性句。預設五個目標情緒各分配 120 個不同原句，合計需要 600 個不重複的中性句。每個原句只分配給一組，資料不足時會停止。

輸出位於 `warmup.generate.input_path`：

```json
{
  "prompt": "The report was completed this morning.",
  "source_emotion": "neutral",
  "emotion": "joy"
}
```

這時只有原句與目標情緒，下一步加入 `target_text` 後才是 Warm-up 訓練資料。

## 四、生成並檢查目標句

`generate_warmup_data.py` 透過 Chat Completions 介面呼叫 `warmup.generate.model`，預設為 `gpt-4o-mini`。每句要求 `responses_per_prompt` 種改寫，解析回覆的 `style` 與 `response`，再用 `Scorer_NLI.analyze_response_quality` 篩選。

將 `OPENAI_API_KEY` 設為環境變數；程式不會自動載入 `.env`。金鑰不要加入版本控制。

```bash
python warmup_training/generate_warmup_data.py
```

這個步驟會將來源句送往外部 API，並產生使用費用，請使用有權傳送的資料。`retry_limit` 控制程式的重試次數，SDK 也可能自行重試。間隔與溫度等參數設於 `warmup.generate`。

通過篩選的資料會寫入 `warmup.dataset_path`：

```json
{
  "prompt": "The report was completed this morning.",
  "source_emotion": "neutral",
  "emotion": "joy",
  "target_text": "The report was completed this morning, marking a welcome achievement.",
  "style": "concise",
  "source": "gpt-4o-mini"
}
```

篩選涵蓋句末標點、句數、長度、重複、解說性文字、未完成的句尾及新增數字。語意、情緒、否定、事實與照抄情形仍需人工檢查。編輯資料時需保留 `prompt`、`source_emotion`、`emotion`、`target_text` 四個訓練欄位。

### 筆數與生成失敗

- `samples_per_target_emotion: 120` 指原句數，不是最後的訓練筆數。
- `responses_per_prompt: 5` 表示每句最多取得五種改寫。
- 每組最多 600 筆，五組合計最多 3,000 筆。
- 扣除請求失敗及品質篩選後，每組需達 `minimum_samples_per_target: 500`。

程式在送出請求前先檢查規劃筆數，結束後印出各組有效筆數。輸出採新建檔案，不覆寫既有資料。有部分成果但筆數不足時，仍會存下資料並回報錯誤；完全失敗則不寫入空資料集。目前沒有自動續跑或補足功能，可檢查已存結果後補入人工確認的資料，或改用新的 `warmup.dataset_path` 再生成。

也可自行準備符合格式的資料，不使用 API 生成器。`WarmupDataset` 會檢查標籤、必填欄位、文字品質與各組筆數；無效比例超過 `max_invalid_fraction`（預設 5%）時，會拒絕該資料集。

## 五、先 Warm-up，再 RL

確認目標資料的品質與筆數後執行：

```bash
python warmup_training/warmup.py
python RL_main/train.py
```

Warm-up 使用 teacher forcing，只對目標改寫句與 EOS 計算交叉熵，更新 Prefix，不更新底模。輸出為 `train.output_dir` 下的 `warmup_prefix.pt`。

RL 載入這個 checkpoint，再使用另一份來源資料訓練。`data.exclude_prompt_paths` 需包含 `warmup.dataset_path`，資料流程會先排除 Warm-up 用過的原句，再切分訓練與驗證資料。

`data/example/` 的五個句子僅示範格式，無法滿足預設 600 個原句的抽樣規劃。本機資料、生成目標句、權重、checkpoint 與訓練輸出都已列入 Git 忽略規則。
