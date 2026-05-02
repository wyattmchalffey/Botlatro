from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import context  # noqa: F401
from balatro_ai.eval.score_dataset import check_score_dataset, main


class ScoreDatasetTests(unittest.TestCase):
    def test_score_dataset_cli_passes_fixture_suite(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--fixtures", "tests/fixtures/score_edges"])

        self.assertEqual(result, 0)
        text = output.getvalue()
        self.assertIn("Score dataset check", text)
        self.assertIn("Fixture failures: 0", text)
        self.assertIn("Result: PASS", text)

    def test_score_dataset_checks_recomputed_replay_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            replay_path.write_text(
                json.dumps(
                    {
                        "seed": 123,
                        "chosen_action": {"type": "play_hand", "card_indices": [0]},
                        "extra": {
                            "score_audit": {
                                "cards": ["AS"],
                                "card_details": [{"rank": "A", "suit": "S", "debuffed": False}],
                                "hand_before_details": [{"rank": "A", "suit": "S", "debuffed": False}],
                                "joker_details": [],
                                "hand_type": "High Card",
                                "predicted_score": 16,
                                "actual_score_delta": 16,
                                "ante": 1,
                                "blind": "Small Blind",
                                "deck_size": 40,
                                "hand_levels": {"High Card": 1},
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            check = check_score_dataset(fixture_paths=(), replay_paths=(Path(directory),))

        self.assertTrue(check.passed)
        self.assertEqual(len(check.replay_rows_checked), 1)
        self.assertEqual(len(check.replay_misses), 0)

    def test_score_dataset_fails_on_recomputed_replay_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            replay_path.write_text(
                json.dumps(
                    {
                        "seed": 123,
                        "chosen_action": {"type": "play_hand", "card_indices": [0]},
                        "extra": {
                            "score_audit": {
                                "cards": ["AS"],
                                "card_details": [{"rank": "A", "suit": "S", "debuffed": False}],
                                "hand_before_details": [{"rank": "A", "suit": "S", "debuffed": False}],
                                "joker_details": [],
                                "hand_type": "High Card",
                                "predicted_score": 16,
                                "actual_score_delta": 15,
                                "ante": 1,
                                "blind": "Small Blind",
                                "deck_size": 40,
                                "hand_levels": {"High Card": 1},
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            check = check_score_dataset(fixture_paths=(), replay_paths=(Path(directory),))

        self.assertFalse(check.passed)
        self.assertEqual(len(check.replay_misses), 1)
        self.assertIn("Replay misses: 1", check.to_text())


if __name__ == "__main__":
    unittest.main()
