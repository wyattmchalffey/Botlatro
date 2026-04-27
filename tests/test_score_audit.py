from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.eval.score_audit import audit_replays


class ScoreAuditTests(unittest.TestCase):
    def test_audit_replays_summarizes_score_audit_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            replay_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "seed": 123,
                                "extra": {
                                    "score_audit": {
                                        "cards": ["AS", "AD"],
                                        "hand_type": "Pair",
                                        "predicted_score": 141,
                                        "actual_score_delta": 141,
                                        "ante": 1,
                                        "blind": "Small Blind",
                                        "jokers": [],
                                    }
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "seed": 123,
                                "extra": {
                                    "score_audit": {
                                        "cards": ["KS"],
                                        "hand_type": "High Card",
                                        "predicted_score": 15,
                                        "actual_score_delta": 19,
                                        "ante": 1,
                                        "blind": "Small Blind",
                                        "jokers": ["Joker"],
                                    }
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = audit_replays((Path(directory),))

        self.assertEqual(summary.files_scanned, 1)
        self.assertEqual(summary.played_hands, 2)
        self.assertEqual(summary.exact_matches, 1)
        self.assertEqual(summary.mean_absolute_error, 2)
        self.assertIn("Worst supported 2", summary.to_text())

    def test_audit_marks_known_uncertain_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            replay_path.write_text(
                json.dumps(
                    {
                        "seed": 123,
                        "extra": {
                            "score_audit": {
                                "cards": ["KS"],
                                "hand_type": "High Card",
                                "predicted_score": 15,
                                "actual_score_delta": 33,
                                "ante": 1,
                                "blind": "Small Blind",
                                "jokers": ["Misprint"],
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = audit_replays((Path(directory),))

        self.assertEqual(len(summary.supported_records), 0)
        self.assertEqual(len(summary.uncertain_records), 1)
        self.assertIn("Misprint has random mult", summary.to_text())


if __name__ == "__main__":
    unittest.main()
