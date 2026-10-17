# EmotionGain_RL

EmotionGain_RL 是情緒強化改寫專題的研究程式。研究以凍結的 Llama-3.2-3B 為底模，嘗試將中性的英文任務通知改寫得更有喜悅感，同時保留原事件、減少照抄，並維持可讀性。

訓練只更新各情緒的 Prefix 向量。六種情緒、Prefix 長度 20、隱藏維度 3072，共有 368,640 個可訓練數值，約為底模參數量的 0.012%。

## 資料夾內容

本資料夾包含 Warm-up、REINFORCE 訓練、多維評分器、API 目標句生成、範例資料及研究結果彙總圖。模型權重、Prefix checkpoint、外部資料集、生成的訓練資料與原始問卷資料不包含在內。

保留一切權利，詳見 [著作權說明](COPYRIGHT.md)。

## 方法概述

1. 凍結底模，只更新目標情緒對應的 Prefix。
2. 先以 Warm-up 的 teacher forcing 學習改寫格式。
3. 在 RL 階段生成候選句，檢查格式、照抄、語意保留、NLI 矛盾、情緒增益、流暢度與精簡度。
4. 未通過 Threshold 的句子給予負 reward；通過者依多維評分計算 reward，再以 REINFORCE 更新 Prefix。
5. 加入逐 token 的 KL 正則化，約束目前策略與固定 Warm-up 策略之間的差距。

通過 Threshold 後，基礎獎勵的預設公式為：

`R = 0.35e + 0.30s + 0.20f + 0.15c - P`

各分數的意義與後續限制見 [方法說明](docs/method.md)。

## 真人評估

正式比較使用 60 個來源句、三種方法（Zero-shot、Few-shot、Prefix Step750），共有 38 位一般讀者完成評估。每個來源句有 1–6 份評分，因此先計算逐題平均，再進行配對分析。

![各方法的逐題平均真人評分](assets/human_evaluation_summary.png)

經 Holm 多重比較校正後，Prefix 的情緒評分顯著高於 Zero-shot，語意保留評分顯著高於 Few-shot，整體最佳選擇率也顯著高於兩種基準方法。其他維度比較未達校正後的顯著門檻，因此不能解讀成所有面向都優於基準。

## 評分器與真人評分的關係

評分器提供訓練訊號，不等同真人判斷。主要 Spearman 等級相關係數為：情緒增益 0.32、語意保留 0.67、流暢度 0.05。

![評分器與真人評分的等級相關](assets/scorer_human_correlation.png)

語意保留是目前與真人評分最一致的部分；情緒的對齊程度有限，GPT-2 流暢度特徵也不能代替真人對自然程度的判斷。詳見 [研究限制](docs/limitations.md)。

## 環境與執行

在專案根目錄建立環境、安裝套件，並複製設定檔：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml
```

在 `config.yaml` 填入已取得使用授權的 Llama 模型路徑與資料路徑。原始輸入需為使用情緒名稱標籤的 JSON／JSONL；官方 TSV／CSV 及數字標籤需先轉換。接著準備來源句：

```bash
python warmup_training/convert_dataset.py
python warmup_training/prepare_inputprompt.py
```

透過 OpenAI API 生成目標改寫句。這個步驟會將來源句傳給 API，並產生使用費用。先將 `OPENAI_API_KEY` 設為環境變數，金鑰不要加入 Git：

```bash
python warmup_training/generate_warmup_data.py
```

預設每個目標情緒使用 120 個原句，每句要求五種改寫，篩選後每組至少需保留 500 筆。先檢查生成資料，再開始訓練；自行準備的目標句也可使用相同格式。

```bash
python warmup_training/warmup.py
python RL_main/train.py
```

資料格式與生成失敗的處理方式見 [訓練流程](docs/training.md)。`data/example/` 的五個句子只用來示範格式。

離線測試使用模擬 API 回覆，不載入模型權重，也不產生 API 費用。公開檢查腳本需先初始化 Git，再執行：

```bash
python -m unittest discover -s tests -v
python scripts/audit_public_release.py
```

GoEmotions 請從 [Google Research 官方資料夾](https://github.com/google-research/google-research/tree/master/goemotions)取得，並依其條款使用。本機資料、模型權重、checkpoint 與訓練輸出已列入 Git 忽略規則。

## 目錄結構

```text
RL_main/             Prefix 生成器、評分器、資料切分、回饋與 RL 訓練
warmup_training/     情緒映射、來源句抽樣、API 目標句生成與 Warm-up
data/example/        手寫的非敏感範例資料
docs/                方法、訓練流程與研究限制
assets/              兩張去識別化的研究彙總圖
scripts/             公開檔案檢查與彙總圖繪製
tests/               設定、資料生成與採樣參數的離線測試
```

## 公開前檢查

新複本先初始化 Git 並加入檔案，再從專案根目錄執行下列檢查：

```bash
git init
git add .
git status --short
git ls-files
git grep -nE 'sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}'
git grep -nE '/Users/|Google Forms|班級|座號|answer_key|rater_assignment' || true
python -m compileall RL_main warmup_training
```

先在私人 GitHub 儲存庫確認檔案清單與 README 顯示結果，再改為公開。啟用敏感資訊掃描與推送保護；若不希望提交紀錄顯示本機電子郵件，可使用 GitHub 的 noreply 地址。
