from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from scripts import phase8_deepening_confirmation_audit as script


class Phase8DeepeningConfirmationAuditTests(unittest.TestCase):
    def test_join_confirmations_classifies_solver_label(self) -> None:
        deepening = [
            {
                "seed": "s",
                "state_index": 1,
                "deepening_candidate_action_key": "candidate",
                "deepening_candidate_action_type": "buy",
                "deepening_heuristic_action_type": "end_shop",
                "deepening_rollouts": 4,
                "deepening_mean_advantage": 1.0,
                "deepening_sem": 0.25,
                "deepening_lcb": 0.75,
                "deepening_positive_rate": 1.0,
            }
        ]
        solver = [
            {
                "seed": "s",
                "state_index": 1,
                "candidates": [
                    {
                        "action_key": "heuristic",
                        "action": {"type": "end_shop"},
                        "rollout_values": [2.0, 2.0, 2.0, 2.0],
                        "is_heuristic_action": True,
                    },
                    {
                        "action_key": "candidate",
                        "action": {"type": "buy"},
                        "rollout_values": [4.0, 4.0, 4.0, 4.0],
                        "is_heuristic_action": False,
                    },
                ],
            }
        ]

        rows = script._join_confirmations(deepening_records=deepening, solver_records=solver, margin=0.10)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].solver_label, "positive")
        self.assertEqual(rows[0].candidate_action_type, "buy")
        self.assertEqual(rows[0].solver_lcb, 2.0)

    def test_main_writes_filter_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deepening_path = Path(tmpdir) / "deepening.jsonl"
            solver_path = Path(tmpdir) / "solver.jsonl"
            metrics_path = Path(tmpdir) / "metrics.json"
            deepening_path.write_text(
                json.dumps(
                    {
                        "seed": "s",
                        "state_index": 1,
                        "deepening_candidate_action_key": "candidate",
                        "deepening_candidate_action_type": "buy",
                        "deepening_heuristic_action_type": "end_shop",
                        "deepening_rollouts": 4,
                        "deepening_mean_advantage": 1.0,
                        "deepening_sem": 0.25,
                        "deepening_lcb": 0.75,
                        "deepening_positive_rate": 1.0,
                        "ranker_margin": 0.42,
                        "ranker_baseline_margin": 0.35,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            solver_path.write_text(
                json.dumps(
                    {
                        "seed": "s",
                        "state_index": 1,
                        "candidates": [
                            {
                                "action_key": "heuristic",
                                "action": {"type": "end_shop"},
                                "rollout_values": [2.0, 2.0],
                                "is_heuristic_action": True,
                            },
                            {
                                "action_key": "candidate",
                                "action": {"type": "buy"},
                                "rollout_values": [4.0, 4.0],
                                "is_heuristic_action": False,
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rc = script.main(
                [
                    "--deepening",
                    str(deepening_path),
                    "--solver",
                    str(solver_path),
                    "--metrics",
                    str(metrics_path),
                    "--max-sem",
                    "0.45",
                ]
            )

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(metrics["joined_records"], 1)
        self.assertEqual(metrics["filter_summaries"]["max_sem_0.450"]["positive"], 1)
        self.assertEqual(metrics["label_stats"]["positive"]["mean_ranker_margin"], 0.42)
        self.assertEqual(metrics["label_stats"]["positive"]["mean_ranker_baseline_margin"], 0.35)


if __name__ == "__main__":
    unittest.main()
