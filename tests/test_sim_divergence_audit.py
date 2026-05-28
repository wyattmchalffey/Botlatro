from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from balatro_ai.eval.sim_divergence_audit import audit_runs


def _hand_card(card_id: int, rank: str, suit: str, chips: int) -> dict:
    return {
        "id": card_id,
        "value": {"rank": rank, "suit": suit, "effect": f"+{chips} chips"},
        "label": "Base Card",
    }


class SimDivergenceAuditTests(unittest.TestCase):
    def test_clean_pair_play_records_no_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            rows = [
                {
                    "state": "SELECTING_HAND",
                    "ante_num": 1,
                    "round_num": 1,
                    "money": 4,
                    "round": {"chips": 0, "hands_left": 4, "discards_left": 3},
                    "blinds": {"current": {"name": "Big Blind", "score": 450}},
                    "hand": {
                        "cards": [
                            _hand_card(1, "A", "S", 11),
                            _hand_card(2, "A", "D", 11),
                            _hand_card(3, "K", "H", 10),
                            _hand_card(4, "2", "C", 2),
                        ]
                    },
                    "hands": {"Pair": {"level": 1, "played": 0, "played_this_round": 0}},
                },
                {
                    "state": "SELECTING_HAND",
                    "ante_num": 1,
                    "round_num": 1,
                    "money": 4,
                    "round": {"chips": 64, "hands_left": 3, "discards_left": 3},
                    "blinds": {"current": {"name": "Big Blind", "score": 450}},
                    "hand": {
                        "cards": [
                            _hand_card(3, "K", "H", 10),
                            _hand_card(4, "2", "C", 2),
                        ]
                    },
                    "hands": {"Pair": {"level": 1, "played": 1, "played_this_round": 1}},
                },
            ]
            (run_dir / "gamestates.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )

            audit = audit_runs((Path(directory),))

        self.assertEqual(audit.runs_scanned, 1)
        self.assertEqual(len(audit.records), 1)
        self.assertEqual(audit.exact, 1)
        self.assertEqual(audit.diverged, ())

    def test_money_mismatch_is_reported_as_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            # Same setup as the clean test, but the post-state claims the
            # player gained $7 that the sim has no reason to produce.
            rows = [
                {
                    "state": "SELECTING_HAND",
                    "ante_num": 1,
                    "round_num": 1,
                    "money": 4,
                    "round": {"chips": 0, "hands_left": 4, "discards_left": 3},
                    "blinds": {"current": {"name": "Big Blind", "score": 450}},
                    "hand": {
                        "cards": [
                            _hand_card(1, "A", "S", 11),
                            _hand_card(2, "A", "D", 11),
                            _hand_card(3, "K", "H", 10),
                            _hand_card(4, "2", "C", 2),
                        ]
                    },
                    "hands": {"Pair": {"level": 1, "played": 0, "played_this_round": 0}},
                },
                {
                    "state": "SELECTING_HAND",
                    "ante_num": 1,
                    "round_num": 1,
                    "money": 11,
                    "round": {"chips": 64, "hands_left": 3, "discards_left": 3},
                    "blinds": {"current": {"name": "Big Blind", "score": 450}},
                    "hand": {
                        "cards": [
                            _hand_card(3, "K", "H", 10),
                            _hand_card(4, "2", "C", 2),
                        ]
                    },
                    "hands": {"Pair": {"level": 1, "played": 1, "played_this_round": 1}},
                },
            ]
            (run_dir / "gamestates.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )

            audit = audit_runs((Path(directory),))

        self.assertEqual(len(audit.diverged), 1)
        diverged = audit.diverged[0]
        money_diffs = [d for d in diverged.divergences if d.field == "money"]
        self.assertEqual(len(money_diffs), 1)
        self.assertEqual(money_diffs[0].predicted, 4)
        self.assertEqual(money_diffs[0].actual, 11)


if __name__ == "__main__":
    unittest.main()
