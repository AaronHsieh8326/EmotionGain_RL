# 檢查範例設定與文件中的執行指令。

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PublicConfigurationTest(unittest.TestCase):
    def test_example_config_has_required_sections(self) -> None:
        config = yaml.safe_load((ROOT / "config.example.yaml").read_text())
        self.assertTrue({"model", "data", "warmup", "train", "scorer", "feedback"} <= set(config))
        self.assertFalse(Path(config["model"]["path"]).is_absolute())
        self.assertFalse(Path(config["data"]["path"]).is_absolute())
        self.assertFalse(Path(config["train"]["output_dir"]).is_absolute())

    def test_public_example_data_is_non_sensitive(self) -> None:
        sample = (ROOT / "data" / "example" / "source_sentences.json").read_text()
        self.assertNotIn("/Users/", sample)
        self.assertNotIn("answer_key", sample)

    def test_generation_plan_can_reach_training_minimum(self) -> None:
        # 檢查每組原句數乘改寫數，是否足以達到訓練門檻。
        config = yaml.safe_load((ROOT / "config.example.yaml").read_text())
        warmup = config["warmup"]
        generation = warmup["generate"]
        responses = generation["responses_per_prompt"]
        self.assertGreater(responses, 0)
        self.assertTrue(generation["model"])
        self.assertGreater(generation["retry_limit"], 0)
        for count in generation["samples_per_target_emotion"].values():
            self.assertGreaterEqual(count * responses, warmup["minimum_samples_per_target"])
        self.assertIn(warmup["dataset_path"], config["data"]["exclude_prompt_paths"])

    def test_documented_generation_command_exists(self) -> None:
        command = "python warmup_training/generate_warmup_data.py"
        self.assertTrue((ROOT / "warmup_training/generate_warmup_data.py").is_file())
        for document in ("README.md", "docs/training.md"):
            self.assertIn(command, (ROOT / document).read_text())


if __name__ == "__main__":
    unittest.main()
