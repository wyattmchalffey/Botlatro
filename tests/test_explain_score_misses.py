from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.eval.explain_score_misses import explain_replays


class ExplainScoreMissesTests(unittest.TestCase):
    def test_explain_replays_prints_context_for_largest_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            replay_path.write_text(
                json.dumps(
                    {
                        "seed": 123,
                        "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                        "extra": {
                            "score_audit": {
                                "cards": ["AS", "AD"],
                                "card_details": [
                                    {"rank": "A", "suit": "S", "debuffed": False},
                                    {"rank": "A", "suit": "D", "debuffed": False},
                                ],
                                "hand_before_details": [
                                    {"rank": "A", "suit": "S", "debuffed": False},
                                    {"rank": "A", "suit": "D", "debuffed": False},
                                    {"rank": "Q", "suit": "C", "debuffed": False},
                                ],
                                "held_card_details": [
                                    {"rank": "Q", "suit": "C", "debuffed": False},
                                ],
                                "joker_details": [
                                    {
                                        "name": "Green Joker",
                                        "sell_value": 2,
                                        "metadata": {
                                            "value": {
                                                "effect": "+1 Mult per hand played -1 Mult per discard (Currently +1 Mult)"
                                            }
                                        },
                                    }
                                ],
                                "hand_type": "Pair",
                                "predicted_score": 141,
                                "actual_score_delta": 94,
                                "ante": 1,
                                "blind": "Small Blind",
                                "discards_remaining": 2,
                                "deck_size": 40,
                                "hand_levels": {"Pair": 1},
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            text = explain_replays((Path(directory),), worst_count=1)

        self.assertIn("Score miss explanations", text)
        self.assertIn("cards=AS AD indexes=0,1", text)
        self.assertIn("held=QC", text)
        self.assertIn("Green Joker current counter timing", text)


if __name__ == "__main__":
    unittest.main()
