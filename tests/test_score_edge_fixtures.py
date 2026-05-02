from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import context  # noqa: F401
from balatro_ai.eval.score_edge_fixtures import main, run_fixture_paths


class ScoreEdgeFixtureTests(unittest.TestCase):
    def test_score_edge_fixtures_pass(self) -> None:
        results = run_fixture_paths((Path("tests/fixtures/score_edges"),))

        self.assertGreaterEqual(len(results), 10)
        self.assertTrue(all(result.passed for result in results), "\n".join(result.to_text() for result in results))

    def test_score_edge_fixture_cli_prints_results(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["tests/fixtures/score_edges/scoring_visible_counters.json"])

        self.assertEqual(result, 0)
        text = output.getvalue()
        self.assertIn("[PASS] scoring_visible_counters.json::square_joker_scores_current_four_card_gain", text)
        self.assertIn("[PASS] scoring_visible_counters.json::loyalty_card_active_text_scores_this_hand", text)


if __name__ == "__main__":
    unittest.main()
