from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from balatro_ai.eval.decision_flip_audit import audit_decision_flips, first_action_divergence, load_decision_traces
from balatro_ai.eval.metrics import RunResult


def result(seed: int, *, bot: str, won: bool, ante: int) -> RunResult:
    return RunResult(
        bot_version=bot,
        seed=seed,
        stake="white",
        won=won,
        ante_reached=ante,
        final_score=0,
        final_money=0,
        runtime_seconds=1.0,
    )


def trace_row(seed: int, step: int, action_key: str, action_type: str, *, shop_audit: dict[str, object] | None = None):
    metadata = {"shop_audit": shop_audit} if shop_audit else {}
    return {
        "record_type": "local_decision_trace",
        "seed": seed,
        "step": step,
        "ante": 5,
        "blind": "Small Blind",
        "phase": "shop",
        "money": 20,
        "action_stable_key": action_key,
        "action_reason": f"reason for {action_key}",
        "action": {"type": action_type, "metadata": metadata},
    }


class DecisionFlipAuditTests(unittest.TestCase):
    def test_load_traces_groups_and_sorts_by_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        trace_row(2, 2, "buy||card|0", "buy"),
                        {"record_type": "run_summary", "seed": 2},
                        trace_row(2, 1, "end_shop|||", "end_shop"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            traces = load_decision_traces((path,))

        self.assertEqual(tuple(traces), (2,))
        self.assertEqual([row["step"] for row in traces[2]], [1, 2])

    def test_first_action_divergence_returns_first_changed_action(self) -> None:
        rows_a = (
            trace_row(1, 0, "select_blind|||", "select_blind"),
            trace_row(1, 1, "end_shop|||", "end_shop"),
        )
        rows_b = (
            trace_row(1, 0, "select_blind|||", "select_blind"),
            trace_row(1, 1, "buy||card|0", "buy"),
        )

        index, row_a, row_b = first_action_divergence(rows_a, rows_b)

        self.assertEqual(index, 1)
        assert row_a is not None
        assert row_b is not None
        self.assertEqual(row_a["action_stable_key"], "end_shop|||")
        self.assertEqual(row_b["action_stable_key"], "buy||card|0")

    def test_audit_decision_flips_summarizes_shop_options(self) -> None:
        shop_audit = {
            "decision": "take",
            "threshold": 10.0,
            "options": (
                {
                    "stable_key": "buy||card|0",
                    "value": 12.5,
                    "item": {"name": "Cavendish"},
                    "planner_terms": {"enabled": True, "legacy_value": 9.0, "late_conversion_value": 3.5},
                },
            ),
        }
        traces_a = {
            1: (
                trace_row(1, 0, "select_blind|||", "select_blind"),
                trace_row(1, 1, "end_shop|||", "end_shop"),
            )
        }
        traces_b = {
            1: (
                trace_row(1, 0, "select_blind|||", "select_blind"),
                trace_row(1, 1, "buy||card|0", "buy", shop_audit=shop_audit),
            )
        }

        audit = audit_decision_flips(
            (result(1, bot="legacy", won=False, ante=5),),
            (result(1, bot="candidate", won=True, ante=8),),
            traces_a,
            traces_b,
        )

        self.assertEqual(audit.flip_counts["candidate_win"], 1)
        self.assertEqual(audit.category_counts["end_shop->buy"], 1)
        text = audit.to_text()
        self.assertIn("seed=1 candidate_win", text)
        self.assertIn("Cavendish", text)
        self.assertIn("late_conversion_value", text)


if __name__ == "__main__":
    unittest.main()
