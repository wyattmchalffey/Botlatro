from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.eval.metrics import RunResult, summarize_runs


class MetricsTests(unittest.TestCase):
    def test_summarize_runs(self) -> None:
        results = [
            RunResult("bot_v1", 1, "white", True, 8, 10000, 20, 10.0),
            RunResult("bot_v1", 2, "white", False, 5, 4000, 8, 20.0),
        ]

        summary = summarize_runs(results)

        self.assertEqual(summary.run_count, 2)
        self.assertEqual(summary.win_rate, 0.5)
        self.assertEqual(summary.average_ante, 6.5)
        self.assertEqual(summary.average_runtime_seconds, 15.0)
        self.assertIn("Profile: unknown", summary.to_text())

    def test_empty_summary_fails(self) -> None:
        with self.assertRaises(ValueError):
            summarize_runs([])


if __name__ == "__main__":
    unittest.main()
