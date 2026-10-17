# 檢查 RL 訓練是否傳入採樣設定。

import unittest
from unittest.mock import Mock, patch, sentinel

# 採樣測試不需要載入評分模型。
with patch.dict("sys.modules", {"RL_main.Scorer_NLI": Mock()}):
    from RL_main.train import generate_batch


class TrainSamplingTest(unittest.TestCase):
    def generation_arguments(self, model_config: dict) -> dict:
        generator = Mock()
        generator.generate_with_logprobs.return_value = {
            "text": "The report was completed this morning.",
            "logprob_sum": sentinel.logprob,
            "kl_mean": sentinel.kl,
            "kl_token_count": 7,
            "token_count": 7,
            "stop_reason": "sentence_end",
        }
        logger = Mock()

        # 呼叫真正的訓練函式，只以 Mock 替代生成器。
        batch = generate_batch(
            generator,
            prompts=["The report was completed this morning."],
            source_emotions=["neutral"],
            emotions=["joy"],
            model_config=model_config,
            logger=logger,
            reference_prefix=sentinel.reference_prefix,
        )

        logger.warning.assert_not_called()
        generator.generate_with_logprobs.assert_called_once()
        self.assertEqual(len(batch), 1)
        self.assertTrue(batch[0]["is_valid"])
        return generator.generate_with_logprobs.call_args.kwargs

    def test_configured_top_k_is_forwarded(self) -> None:
        arguments = self.generation_arguments({"top_k": 7})
        self.assertEqual(arguments["top_k"], 7)

    def test_zero_disables_top_k_filtering(self) -> None:
        arguments = self.generation_arguments({"top_k": 0})
        self.assertEqual(arguments["top_k"], 0)

    def test_missing_top_k_keeps_default(self) -> None:
        arguments = self.generation_arguments({})
        self.assertEqual(arguments["top_k"], 50)

    def test_top_k_is_converted_to_integer(self) -> None:
        arguments = self.generation_arguments({"top_k": "7"})
        self.assertEqual(arguments["top_k"], 7)
        self.assertIsInstance(arguments["top_k"], int)


if __name__ == "__main__":
    unittest.main()
