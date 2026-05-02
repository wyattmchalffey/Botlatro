from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from balatro_ai.eval.balatrobench_score_audit import audit_balatrobench_runs


class BalatroBenchScoreAuditTests(unittest.TestCase):
    def test_audit_infers_played_cards_from_consecutive_gamestates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            rows = [
                {
                    "state": "SELECTING_HAND",
                    "ante_num": 1,
                    "round_num": 1,
                    "round": {"chips": 0, "hands_left": 4, "discards_left": 3},
                    "blinds": {"current": {"name": "Big Blind", "score": 450}},
                    "hand": {
                        "cards": [
                            {"id": 1, "value": {"rank": "A", "suit": "S", "effect": "+11 chips"}, "label": "Base Card"},
                            {"id": 2, "value": {"rank": "A", "suit": "D", "effect": "+11 chips"}, "label": "Base Card"},
                            {"id": 3, "value": {"rank": "K", "suit": "H", "effect": "+10 chips"}, "label": "Base Card"},
                        ]
                    },
                    "hands": {"Pair": {"level": 1, "played": 0, "played_this_round": 0}},
                },
                {
                    "state": "SELECTING_HAND",
                    "ante_num": 1,
                    "round_num": 1,
                    "round": {"chips": 64, "hands_left": 3, "discards_left": 3},
                    "blinds": {"current": {"name": "Big Blind", "score": 450}},
                    "hand": {
                        "cards": [
                            {"id": 3, "value": {"rank": "K", "suit": "H", "effect": "+10 chips"}, "label": "Base Card"},
                            {"id": 4, "value": {"rank": "2", "suit": "C", "effect": "+2 chips"}, "label": "Base Card"},
                            {"id": 5, "value": {"rank": "3", "suit": "C", "effect": "+3 chips"}, "label": "Base Card"},
                        ]
                    },
                    "hands": {"Pair": {"level": 1, "played": 1, "played_this_round": 1}},
                },
            ]
            (run_dir / "gamestates.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            audit = audit_balatrobench_runs((Path(directory),))

        self.assertEqual(audit.runs_scanned, 1)
        self.assertEqual(len(audit.records), 1)
        self.assertEqual(audit.records[0].cards, ("AS", "AD"))
        self.assertEqual(audit.records[0].predicted_score, 64)
        self.assertEqual(audit.records[0].actual_score_delta, 64)
        self.assertEqual(audit.supported_misses, ())

    def test_audit_marks_played_lucky_cards_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            rows = [
                {
                    "state": "SELECTING_HAND",
                    "ante_num": 1,
                    "round_num": 1,
                    "round": {"chips": 0, "hands_left": 4, "discards_left": 3},
                    "blinds": {"current": {"name": "Big Blind", "score": 450}},
                    "hand": {
                        "cards": [
                            {
                                "id": 1,
                                "value": {"rank": "A", "suit": "S", "effect": "+11 chips 1 in 5 chance for +20 Mult"},
                                "label": "Lucky Card",
                            },
                            {"id": 2, "value": {"rank": "A", "suit": "D", "effect": "+11 chips"}, "label": "Base Card"},
                            {"id": 3, "value": {"rank": "K", "suit": "H", "effect": "+10 chips"}, "label": "Base Card"},
                        ]
                    },
                    "hands": {"Pair": {"level": 1, "played": 0, "played_this_round": 0}},
                },
                {
                    "state": "SELECTING_HAND",
                    "ante_num": 1,
                    "round_num": 1,
                    "round": {"chips": 64, "hands_left": 3, "discards_left": 3},
                    "blinds": {"current": {"name": "Big Blind", "score": 450}},
                    "hand": {
                        "cards": [
                            {"id": 3, "value": {"rank": "K", "suit": "H", "effect": "+10 chips"}, "label": "Base Card"},
                            {"id": 4, "value": {"rank": "2", "suit": "C", "effect": "+2 chips"}, "label": "Base Card"},
                        ]
                    },
                    "hands": {"Pair": {"level": 1, "played": 1, "played_this_round": 1}},
                },
            ]
            (run_dir / "gamestates.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            audit = audit_balatrobench_runs((Path(directory),))

        self.assertEqual(len(audit.records), 1)
        self.assertIn("Lucky Card", audit.records[0].uncertainty_reason or "")
        self.assertEqual(audit.supported_records, ())


if __name__ == "__main__":
    unittest.main()
