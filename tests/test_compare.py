from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from balatro_ai.eval.compare import compare_paired_results, load_run_results
from balatro_ai.eval.metrics import RunResult


def result(seed: int, *, bot: str, won: bool, ante: int, score: int) -> RunResult:
    return RunResult(
        bot_version=bot,
        seed=seed,
        stake="white",
        won=won,
        ante_reached=ante,
        final_score=score,
        final_money=0,
        runtime_seconds=1.0,
    )


class CompareTests(unittest.TestCase):
    def test_compare_paired_results_reports_seed_aligned_deltas(self) -> None:
        bot_a = (
            result(1, bot="bot_a", won=False, ante=1, score=100),
            result(2, bot="bot_a", won=False, ante=2, score=200),
            result(3, bot="bot_a", won=True, ante=8, score=1000),
            result(4, bot="bot_a", won=True, ante=8, score=900),
        )
        bot_b = (
            result(1, bot="bot_b", won=True, ante=8, score=1200),
            result(2, bot="bot_b", won=True, ante=8, score=1000),
            result(3, bot="bot_b", won=True, ante=8, score=1100),
            result(4, bot="bot_b", won=False, ante=4, score=350),
        )

        comparison = compare_paired_results(bot_a, bot_b, bootstrap_samples=100, rng_seed=1)

        self.assertEqual(comparison.seed_count, 4)
        self.assertEqual(comparison.wins_flipped, 2)
        self.assertEqual(comparison.wins_lost, 1)
        self.assertAlmostEqual(comparison.bot_a_win_rate, 0.5)
        self.assertAlmostEqual(comparison.bot_b_win_rate, 0.75)
        self.assertAlmostEqual(comparison.win_rate_delta, 0.25)
        self.assertAlmostEqual(comparison.average_ante_delta, 2.25)
        self.assertAlmostEqual(comparison.average_score_delta, 362.5)
        self.assertGreaterEqual(comparison.mcnemar_p_value, 0.0)
        self.assertLessEqual(comparison.mcnemar_p_value, 1.0)
        self.assertGreaterEqual(comparison.wilcoxon_ante_p_value, 0.0)
        self.assertLessEqual(comparison.wilcoxon_ante_p_value, 1.0)
        self.assertIn("Wins flipped", comparison.to_text())
        self.assertEqual(comparison.to_json_dict()["win_rate"]["wins_flipped"], 2)

    def test_compare_uses_only_common_seeds(self) -> None:
        bot_a = (
            result(1, bot="bot_a", won=False, ante=1, score=100),
            result(2, bot="bot_a", won=False, ante=2, score=200),
        )
        bot_b = (
            result(2, bot="bot_b", won=True, ante=8, score=1000),
            result(3, bot="bot_b", won=True, ante=8, score=1200),
        )

        comparison = compare_paired_results(bot_a, bot_b, bootstrap_samples=10)

        self.assertEqual(comparison.seed_count, 1)
        self.assertEqual(comparison.wins_flipped, 1)
        self.assertAlmostEqual(comparison.average_ante_delta, 6.0)

    def test_load_run_results_from_replay_summary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_dir = Path(directory)
            (replay_dir / "bot_white_1.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"state": "ignored"},
                        {
                            "record_type": "run_summary",
                            "bot_version": "summary_bot",
                            "seed": 1,
                            "stake": "white",
                            "won": True,
                            "ante": 8,
                            "final_score": 12345,
                            "final_money": 67,
                            "runtime_seconds": 12.5,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            results = load_run_results((replay_dir,), default_bot="fallback", default_stake="white")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].bot_version, "summary_bot")
        self.assertEqual(results[0].seed, 1)
        self.assertTrue(results[0].won)
        self.assertEqual(results[0].final_score, 12345)

    def test_load_run_results_from_progress_log_keeps_retry_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.out.txt"
            log_path.write_text(
                "\n".join(
                    (
                        "Bot: basic_strategy_bot",
                        "Stake: white",
                        "[1/2] seed=1 won=False ante=4 score=5000 money=12 runtime=10.00s",
                        "[2/2] seed=2 won=False ante=0 score=0 money=0 runtime=2.00s death=error:ConnectionError: bad",
                        "[retry 1] seed=2 won=True ante=8 score=200000 money=99 runtime=20.00s",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            results = load_run_results((log_path,), default_bot="fallback", default_stake="white")

        by_seed = {result.seed: result for result in results}
        self.assertEqual(len(results), 2)
        self.assertEqual(by_seed[1].bot_version, "basic_strategy_bot")
        self.assertFalse(by_seed[1].won)
        self.assertTrue(by_seed[2].won)
        self.assertIsNone(by_seed[2].death_reason)


if __name__ == "__main__":
    unittest.main()
