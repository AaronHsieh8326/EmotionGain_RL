# 用模擬 API 回覆測試 Warm-up 資料生成。

from __future__ import annotations

import ast
import io
import json
import math
import re
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from difflib import SequenceMatcher
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from warmup_training import generate_warmup_data as generation


ROOT = Path(__file__).resolve().parents[1]
PROMPT = "The report was completed this morning."
REWRITE = "The report was completed this morning, marking a welcome achievement."


def completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_client(*responses: object) -> SimpleNamespace:
    # 依序回傳準備好的回覆或例外，不送出網路請求。
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=Mock(side_effect=list(responses)),
    )))


def surface_module() -> ModuleType:
    # 只載入文字規則，測試不需要評分模型。
    path = ROOT / "RL_main/Scorer_NLI.py"
    tree = ast.parse(path.read_text())
    nodes: list[ast.stmt] = [ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    )]
    # 取出第一個類別前的資料與純文字函式，略過模型類別。
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            break
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef)):
            nodes.append(node)
    module = ModuleType("RL_main.Scorer_NLI")
    module.__dict__.update({"__file__": str(path), "Path": Path, "math": math,
                            "re": re, "Counter": Counter, "SequenceMatcher": SequenceMatcher})
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), "exec"), module.__dict__)
    return module


class WarmupGenerationTest(unittest.TestCase):
    def test_parse_fenced_response_array(self) -> None:
        fence = chr(96) * 3
        content = fence + "json\n" + json.dumps([{"style": "concise", "response": REWRITE}]) + "\n" + fence
        self.assertEqual(generation.parse_response_array(content)[0]["response"], REWRITE)

    def test_empty_responses_are_rejected(self) -> None:
        for content in ("not JSON", "[]", '[{"response": ""}]'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                generation.parse_response_array(content)

    def test_input_requires_source_and_cross_emotion(self) -> None:
        for row in ({"prompt": PROMPT, "emotion": "joy"},
                    {"prompt": PROMPT, "emotion": "joy", "source_emotion": "joy"}):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "inputs.json"
                path.write_text(json.dumps([row]))
                with self.assertRaises(ValueError):
                    generation.load_input_prompts(path)

    def test_input_preserves_source_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.json"
            path.write_text(json.dumps([{"prompt": PROMPT, "emotion": "JOY", "source_emotion": "neutral"}]))
            rows = generation.load_input_prompts(path)
        self.assertEqual(rows, [{"prompt": PROMPT, "emotion": "joy", "source_emotion": "neutral"}])

    def test_120_sources_can_yield_600_pairs(self) -> None:
        rows = [{"prompt": f"Source {i}.", "source_emotion": "neutral", "emotion": "joy"} for i in range(120)]
        config = {"samples_per_target_emotion": {"joy": 120}, "responses_per_prompt": 5}
        self.assertEqual(generation.validate_generation_plan(rows, config, 500), {"joy": 600})

    def test_impossible_or_missing_target_plan_is_rejected(self) -> None:
        rows = [{"prompt": PROMPT, "source_emotion": "neutral", "emotion": "joy"}]
        cases = [({"joy": 1}, 5, 500), ({"joy": 1, "anger": 1}, 5, 1), ({"joy": 1}, 0, 1)]
        for targets, responses, minimum in cases:
            with self.subTest(targets=targets, responses=responses), self.assertRaises(ValueError):
                generation.validate_generation_plan(rows, {
                    "samples_per_target_emotion": targets, "responses_per_prompt": responses,
                }, minimum)

    def test_api_response_is_bounded_and_uses_working_model(self) -> None:
        client = fake_client(completion(json.dumps([{"response": REWRITE}] * 6)))
        rows = generation.call_model(client, PROMPT, "joy", model="gpt-4o-mini",
                                     response_count=5, retry_limit=3, retry_delay=0, temperature=1)
        self.assertEqual(len(rows), 5)
        arguments = client.chat.completions.create.call_args.kwargs
        self.assertEqual(arguments["model"], "gpt-4o-mini")
        self.assertEqual(arguments["max_tokens"], 500)
        self.assertIn("5 different ways", arguments["messages"][1]["content"])

    def test_parse_failure_is_retried_offline(self) -> None:
        client = fake_client(completion("bad JSON"), completion(json.dumps([{"response": REWRITE}])))
        with redirect_stdout(io.StringIO()):
            rows = generation.call_model(client, PROMPT, "joy", model="gpt-4o-mini",
                                         response_count=5, retry_limit=3, retry_delay=0, temperature=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_failed_api_error_body_is_not_logged(self) -> None:
        client = fake_client(RuntimeError("sensitive provider detail"))
        output = io.StringIO()
        with redirect_stdout(output):
            rows = generation.call_model(client, PROMPT, "joy", model="gpt-4o-mini",
                                         response_count=5, retry_limit=1, retry_delay=0, temperature=1)
        self.assertEqual(rows, [])
        self.assertNotIn("sensitive provider detail", output.getvalue())

    def test_generated_schema_and_actual_surface_filter(self) -> None:
        client = fake_client(completion(json.dumps([
            {"response": REWRITE, "style": "concise"},
            {"response": REWRITE.rstrip(".")},
        ])))
        progress = ModuleType("tqdm")
        progress.tqdm = lambda iterable, **kwargs: iterable
        with patch.dict("sys.modules", {"RL_main.Scorer_NLI": surface_module(), "tqdm": progress}):
            rows, failures, invalid = generation.generate_dataset(client, [
                {"prompt": PROMPT, "source_emotion": "neutral", "emotion": "joy"},
            ], {"call_delay": 0, "retry_delay": 0})
        self.assertEqual((len(rows), failures, invalid), (1, 0, 1))
        self.assertEqual(rows[0], {"prompt": PROMPT, "source_emotion": "neutral", "emotion": "joy",
                                   "target_text": REWRITE, "style": "concise", "source": "gpt-4o-mini"})

    def run_main_fixture(self, temporary: str, results: list[dict], *, existing: bool = False,
                         key: bool = True) -> tuple[Path, Mock]:
        # 以暫存資料、模擬 client 與設定，測試 main 的存檔及失敗處理。
        folder = Path(temporary)
        inputs = folder / "inputs.json"
        output = folder / "targets.json"
        inputs.write_text(json.dumps([{"prompt": PROMPT, "source_emotion": "neutral", "emotion": "joy"}]))
        if existing:
            output.write_text("existing dataset")
        config = {"warmup": {"dataset_path": "targets.json", "minimum_samples_per_target": 2,
                             "generate": {"input_path": "inputs.json", "responses_per_prompt": 5,
                                          "samples_per_target_emotion": {"joy": 1}}}}
        utilities = ModuleType("RL_main.utils")
        utilities.load_config = lambda path: config
        utilities.resolve_path = lambda path, base: folder / path
        sdk = ModuleType("openai")
        sdk.OpenAI = Mock(return_value=object())
        environment = {"OPENAI_API_KEY": "offline-test-placeholder"} if key else {}
        with patch.dict("sys.modules", {"RL_main.utils": utilities, "openai": sdk}), \
                patch.dict("os.environ", environment, clear=True), \
                patch.object(generation, "generate_dataset", return_value=(results, 0, 0)), \
                redirect_stdout(io.StringIO()):
            generation.main()
        return output, sdk.OpenAI

    def test_main_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                self.run_main_fixture(temporary, [], existing=True)
            self.assertEqual((Path(temporary) / "targets.json").read_text(), "existing dataset")

    def test_exclusive_save_preserves_file_created_after_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.json"
            path.write_text("another run's dataset")
            with self.assertRaises(FileExistsError):
                generation.save_new_dataset([], path)
            self.assertEqual(path.read_text(), "another run's dataset")

    def test_main_missing_key_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                self.run_main_fixture(temporary, [], key=False)
            self.assertFalse((Path(temporary) / "targets.json").exists())

    def test_main_does_not_write_empty_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "no empty dataset"):
                self.run_main_fixture(temporary, [])
            self.assertFalse((Path(temporary) / "targets.json").exists())

    def test_main_retains_partial_pairs_but_rejects_training_readiness(self) -> None:
        row = {"prompt": PROMPT, "source_emotion": "neutral", "emotion": "joy", "target_text": REWRITE}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "Partial dataset"):
                self.run_main_fixture(temporary, [row])
            self.assertEqual(json.loads((Path(temporary) / "targets.json").read_text()), [row])

    def test_main_accepts_sufficient_coverage(self) -> None:
        row = {"prompt": PROMPT, "source_emotion": "neutral", "emotion": "joy", "target_text": REWRITE}
        with tempfile.TemporaryDirectory() as temporary:
            path, constructor = self.run_main_fixture(temporary, [row, row])
            self.assertEqual(len(json.loads(path.read_text())), 2)
            constructor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
