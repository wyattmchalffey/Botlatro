from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.eval.replay_analyzer import analyze_replays


class ReplayAnalyzerTests(unittest.TestCase):
    def test_analyze_replays_summarizes_shop_and_hand_efficiency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            replay_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "seed": 123,
                            "state": "phase=shop ante=1 blind=Small Blind score=0/300 money=8 hands=4 discards=4",
                            "chosen_action": {
                                "type": "buy",
                                "metadata": {
                                    "reason": "shop_value value=44.0 pressure=1.15 target=450 capacity=390",
                                    "shop_audit": {
                                        "decision": "take",
                                        "build_profile": {
                                            "preferred_hand": "Pair",
                                            "missing_roles": ["xmult", "scaling"],
                                        },
                                        "threshold": 24.0,
                                        "chosen_value": 44.0,
                                        "pressure": {"ratio": 1.15},
                                        "chosen_item": {"name": "Cavendish"},
                                        "options": [
                                            {"type": "buy", "value": 44.0, "item": {"name": "Cavendish"}},
                                            {"type": "reroll", "value": 19.0},
                                        ],
                                    },
                                },
                            },
                            "chosen_item": {"name": "Cavendish"},
                        },
                        {
                            "seed": 123,
                            "state": "phase=selecting_hand ante=1 blind=Big Blind score=0/450 money=4 hands=4 discards=4 hand=[AS AH 7C 2D] jokers=[Cavendish]",
                            "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                        },
                        {
                            "seed": 123,
                            "state": "phase=selecting_hand ante=1 blind=Big Blind score=100/450 money=4 hands=3 discards=4 hand=[KS KH 7C 7D] jokers=[Cavendish]",
                            "chosen_action": {"type": "play_hand", "card_indices": [0, 1, 2, 3]},
                        },
                        {
                            "seed": 123,
                            "state": "phase=round_eval ante=1 blind=Big Blind score=460/450 money=4 hands=2 discards=4",
                            "chosen_action": {"type": "cash_out"},
                        },
                        {
                            "seed": 123,
                            "state": "phase=shop ante=2 blind=Small Blind score=0/800 money=9 hands=4 discards=4",
                            "chosen_action": {
                                "type": "sell",
                                "metadata": {"reason": "replace Joker value=10.0 with Cavendish upgrade=50.0 pressure=0.80"},
                            },
                        },
                        {
                            "record_type": "run_summary",
                            "seed": 123,
                            "won": True,
                            "outcome": "win",
                            "ante": 9,
                            "final_score": 215800,
                            "final_money": 211,
                            "final_state_detail": {
                                "phase": "run_over",
                                "ante": 9,
                                "blind": "Cerulean Bell",
                                "current_score": 215800,
                                "required_score": 120000,
                                "money": 211,
                                "hands_remaining": 2,
                                "discards_remaining": 0,
                                "jokers": [{"name": "Cavendish"}],
                            },
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            analysis = analyze_replays((Path(directory),))

        self.assertEqual(analysis.files_scanned, 1)
        self.assertEqual(analysis.average_ante, 9)
        self.assertEqual(analysis.observed_wins, 1)
        self.assertEqual(analysis.outcome_distribution["win"], 1)
        self.assertEqual(analysis.runs[0].final_jokers, ("Cavendish",))
        self.assertEqual(analysis.action_counts["buy"], 1)
        self.assertEqual(analysis.action_counts["sell"], 1)
        self.assertEqual(analysis.shop_reason_counts["shop_value"], 1)
        self.assertEqual(analysis.shop_reason_counts["replace"], 1)
        self.assertEqual(analysis.shop_audit_count, 1)
        self.assertEqual(analysis.chosen_shop_items["buy:Cavendish"], 1)
        self.assertEqual(analysis.played_hand_types["Pair"], 1)
        self.assertEqual(analysis.played_hand_types["Two Pair"], 1)
        self.assertEqual(analysis.dominant_played_hands["Pair"], 1)
        self.assertEqual(analysis.preferred_hand_counts["Pair"], 1)
        self.assertEqual(analysis.final_preferred_hands["Pair"], 1)
        self.assertEqual(analysis.missing_role_counts["xmult"], 1)
        self.assertEqual(analysis.missing_role_counts["scaling"], 1)
        self.assertEqual(analysis.pressure_values, (1.15, 0.8))
        self.assertEqual(analysis.hands_per_blind, (2,))
        self.assertEqual(len(analysis.deep_losses), 0)
        self.assertEqual(len(analysis.early_failures), 0)
        self.assertIn("Average played hands per blind: 2.00", analysis.to_text())
        self.assertIn("Archetypes:", analysis.to_text())
        self.assertEqual(analysis.to_json_dict()["reach_counts"], {"2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "8": 1})
        self.assertEqual(analysis.to_json_dict()["outcome_distribution"], {"win": 1})
        self.assertEqual(analysis.to_json_dict()["chosen_shop_items"], {"buy:Cavendish": 1})
        self.assertEqual(
            analysis.to_json_dict()["archetypes"]["played_hand_types"],
            {"Pair": 1, "Two Pair": 1},
        )
        self.assertEqual(analysis.to_json_dict()["pressure"]["at_least_1"], 1)

    def test_analyze_replays_reports_early_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "early.jsonl"
            replay_path.write_text(
                '{"seed": 7, "state": "phase=selecting_hand ante=1 blind=The Wall score=200/1200 money=3 hands=1 discards=0 hand=[AS] jokers=[Joker]", "chosen_action": {"type": "play_hand"}}\n',
                encoding="utf-8",
            )

            analysis = analyze_replays((replay_path,))

        self.assertEqual(len(analysis.early_failures), 1)
        self.assertEqual(analysis.early_failures[0].final_jokers, ("Joker",))
        self.assertIn("Early failures:", analysis.to_text())
        self.assertEqual(analysis.to_json_dict()["early_failures"]["count"], 1)

    def test_analyze_replays_counts_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "bad.jsonl"
            replay_path.write_text(
                '{"seed": 1, "state": "phase=shop ante=1", "chosen_action": {"type": "end_shop"}}\n'
                "{not-json}\n",
                encoding="utf-8",
            )

            analysis = analyze_replays((replay_path,))

        self.assertEqual(analysis.files_scanned, 1)
        self.assertEqual(analysis.malformed_rows, 1)
        self.assertEqual(analysis.action_counts["end_shop"], 1)


if __name__ == "__main__":
    unittest.main()
