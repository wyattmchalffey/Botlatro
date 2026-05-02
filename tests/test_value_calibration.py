from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import context  # noqa: F401
from balatro_ai.search.value_calibration import calibrate_clear_probability


def state_detail(
    *,
    phase: str = "selecting_hand",
    ante: int = 1,
    blind: str = "Small Blind",
    current_score: int = 0,
    required_score: int = 300,
    hands_remaining: int = 4,
    hand: list[str] | None = None,
) -> dict[str, object]:
    return {
        "phase": phase,
        "ante": ante,
        "blind": blind,
        "required_score": required_score,
        "current_score": current_score,
        "hands_remaining": hands_remaining,
        "discards_remaining": 4,
        "money": 4,
        "deck_size": 52,
        "hand": [{"rank": card[:-1], "suit": card[-1], "name": card} for card in (hand or [])],
        "hand_levels": {},
        "hands": {},
        "jokers": [],
        "known_deck": [],
    }


class ValueCalibrationTests(unittest.TestCase):
    def test_calibrate_clear_probability_bins_replay_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(hand=["AS", "KH"], required_score=10),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="round_eval",
                        current_score=16,
                        required_score=10,
                        hands_remaining=3,
                        hand=[],
                    ),
                    "chosen_action": {"type": "cash_out"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(phase="shop", current_score=0, required_score=10, hand=[]),
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        ante=1,
                        blind="Big Blind",
                        required_score=9999,
                        hand=["2C", "3D"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="run_over",
                        ante=1,
                        blind="Big Blind",
                        current_score=10,
                        required_score=9999,
                        hands_remaining=0,
                        hand=[],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            with patch("balatro_ai.search.value_calibration.clear_probability", side_effect=(0.85, 0.15)):
                summary = calibrate_clear_probability((replay_path,), samples=1)

        self.assertEqual(summary.files_scanned, 1)
        self.assertEqual(summary.blind_count, 2)
        self.assertTrue(summary.examples[0].actual_clear)
        self.assertFalse(summary.examples[1].actual_clear)
        populated = [item for item in summary.bins if item.count]
        self.assertEqual([(item.label, item.count) for item in populated], [("10-20%", 1), ("80-90%", 1)])
        self.assertIn("State-value calibration", summary.to_text())


if __name__ == "__main__":
    unittest.main()
