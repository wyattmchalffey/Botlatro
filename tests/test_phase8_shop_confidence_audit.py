from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from scripts import phase8_shop_confidence_audit as script


def _record(*, terminal_won: bool, heuristic_values: list[float], candidate_values: list[float]) -> dict:
    return {
        "terminal_won": terminal_won,
        "selection_reason": "win" if terminal_won else "reached_ante_8",
        "candidates": [
            {
                "action_key": "end_shop",
                "action": {"type": "end_shop"},
                "mean_value": sum(heuristic_values) / len(heuristic_values),
                "rollout_values": heuristic_values,
                "is_heuristic_action": True,
            },
            {
                "action_key": "open_pack",
                "action": {"type": "open_pack"},
                "mean_value": sum(candidate_values) / len(candidate_values),
                "rollout_values": candidate_values,
                "is_heuristic_action": False,
            },
        ],
    }


class Phase8ShopConfidenceAuditTests(unittest.TestCase):
    def test_block_quality_marks_signal_blocks(self) -> None:
        records = [
            _record(terminal_won=index % 2 == 0, heuristic_values=[1.0] * 4, candidate_values=[1.4] * 4)
            for index in range(8)
        ]
        confidence = {
            "best_practical_high_conf_beats_heuristic_rate": 1.0,
            "any_high_conf_practical_override_candidate_rate": 1.0,
            "mean_best_lcb_advantage_vs_heuristic": 0.4,
        }

        summary = script.block_quality_summary(records, confidence)

        self.assertEqual(summary["decision"], "strong_signal")
        self.assertEqual(summary["recommendation"], "select_and_confirm_candidates")
        self.assertEqual(summary["records_by_terminal_won"], {"False": 4, "True": 4})

    def test_block_quality_marks_flat_blocks_as_calibration(self) -> None:
        records = [
            _record(terminal_won=False, heuristic_values=[1.0] * 4, candidate_values=[1.03] * 4)
            for _ in range(8)
        ]
        confidence = {
            "best_practical_high_conf_beats_heuristic_rate": 0.0,
            "any_high_conf_practical_override_candidate_rate": 0.0,
            "mean_best_lcb_advantage_vs_heuristic": -0.02,
        }

        summary = script.block_quality_summary(records, confidence)

        self.assertEqual(summary["decision"], "calibration_only")
        self.assertEqual(summary["recommendation"], "keep_for_calibration_or_holdout")
        self.assertEqual(summary["heuristic_within_practical_margin_rate"], 1.0)

    def test_main_writes_block_quality_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "labels.jsonl"
            output_path = Path(tmpdir) / "metrics.json"
            input_path.write_text(
                "\n".join(
                    json.dumps(
                        _record(
                            terminal_won=index == 0,
                            heuristic_values=[1.0] * 4,
                            candidate_values=[1.4] * 4,
                        )
                    )
                    for index in range(8)
                )
                + "\n",
                encoding="utf-8",
            )

            rc = script.main(["--input", str(input_path), "--out", str(output_path)])
            metrics = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(metrics["records"], 8)
        self.assertEqual(metrics["block_quality"]["decision"], "strong_signal")


if __name__ == "__main__":
    unittest.main()
