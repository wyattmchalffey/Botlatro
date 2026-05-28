from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from balatro_ai.eval.loss_audit import audit_late_losses
from balatro_ai.eval.compare import load_run_results
from balatro_ai.eval.decision_flip_audit import load_decision_traces


class LossAuditTests(unittest.TestCase):
    def test_late_loss_audit_flags_seen_unbought_boss_reroll_voucher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "results.jsonl"
            traces_path = root / "trace.jsonl"
            results_path.write_text(
                json.dumps(
                    {
                        "record_type": "run_summary",
                        "bot_version": "basic_strategy_bot",
                        "seed": 7,
                        "stake": "white",
                        "won": False,
                        "ante": 8,
                        "ante_reached": 8,
                        "final_score": 240000,
                        "final_money": 31,
                        "runtime_seconds": 1.0,
                        "death_reason": "Violet Vessel",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = [
                {
                    "record_type": "local_decision_trace",
                    "seed": 7,
                    "step": 3,
                    "ante": 7,
                    "blind": "Small Blind",
                    "phase": "shop",
                    "money": 40,
                    "action": {"type": "end_shop", "metadata": {}},
                    "action_stable_key": "end_shop|||",
                    "chosen_item": None,
                    "voucher_cards": [{"name": "Director's Cut", "set": "VOUCHER"}],
                    "shop_audit": {
                        "chosen_value": 0.0,
                        "pressure": {"ratio": 1.2, "build_capacity": 120000},
                        "options": [{"stable_key": "end_shop|||", "value": 0.0}],
                    },
                },
                {
                    "record_type": "local_decision_trace",
                    "seed": 7,
                    "step": 9,
                    "ante": 8,
                    "blind": "Violet Vessel",
                    "phase": "selecting_hand",
                    "money": 31,
                    "current_score": 200000,
                    "post_current_score": 240000,
                    "post_required_score": 300000,
                    "post_run_over": True,
                    "post_won": False,
                    "action": {"type": "play_hand", "metadata": {}},
                    "action_stable_key": "play_hand|1,2,3,4,5||",
                    "voucher_cards": [],
                },
            ]
            traces_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            audit = audit_late_losses(
                load_run_results([results_path], default_bot="basic_strategy_bot", default_stake="white"),
                load_decision_traces([traces_path]),
                min_ante=8,
            )

        self.assertEqual(len(audit.losses), 1)
        loss = audit.losses[0]
        self.assertEqual(loss.score_gap, 60000)
        self.assertIn("boss_reroll_voucher_seen_unbought", loss.signals)
        self.assertIn("late_bank_unspent", loss.signals)
        self.assertIn("Violet Vessel", audit.to_text())


if __name__ == "__main__":
    unittest.main()
