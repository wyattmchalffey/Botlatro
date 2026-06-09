from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from scripts import phase8_select_deepening_states as script


class Phase8SelectDeepeningStatesTests(unittest.TestCase):
    def test_best_opportunities_selects_candidate_beating_heuristic(self) -> None:
        record = _record(
            seed="a",
            heuristic_values=[2.0, 2.0],
            candidate_values=[4.0, 4.0],
            weak_values=[2.5, 2.5],
        )

        opportunities = script._best_opportunities(
            [record],
            z=1.0,
            min_mean_advantage=0.0,
            min_lcb=0.1,
            min_positive_rate=0.5,
            min_rollouts=2,
            max_sem=None,
            min_lcb_sem_ratio=None,
            candidate_action_types=(),
            exclude_candidate_action_types=(),
            excluded_candidate_keys=set(),
            preferred_candidate_types=("open_pack", "buy"),
        )

        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0].candidate_action_type, "open_pack")
        self.assertEqual(opportunities[0].heuristic_action_type, "end_shop")
        self.assertEqual(opportunities[0].lcb, 2.0)

    def test_best_opportunities_can_exclude_candidate_action_types(self) -> None:
        record = _record(
            seed="a",
            heuristic_values=[2.0, 2.0],
            candidate_values=[4.0, 4.0],
            weak_values=[3.0, 3.0],
        )
        record["candidates"].append(
            {
                "action_key": "skip",
                "action": {"type": "end_shop"},
                "rollout_values": [5.0, 5.0],
                "is_heuristic_action": False,
            }
        )

        opportunities = script._best_opportunities(
            [record],
            z=1.0,
            min_mean_advantage=0.0,
            min_lcb=0.1,
            min_positive_rate=0.5,
            min_rollouts=2,
            max_sem=None,
            min_lcb_sem_ratio=None,
            candidate_action_types=("buy", "open_pack"),
            exclude_candidate_action_types=("end_shop",),
            excluded_candidate_keys=set(),
            preferred_candidate_types=("buy", "open_pack"),
        )

        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0].candidate_action_type, "open_pack")
        self.assertNotEqual(opportunities[0].candidate_action_key, "skip")

    def test_best_opportunities_can_filter_noisy_candidates_by_sem(self) -> None:
        record = {
            "seed": "a",
            "state_index": 4,
            "source_bot": "bot",
            "ante": 2,
            "money": 10,
            "state_snapshot": {"phase": "shop"},
            "candidates": [
                {
                    "action_key": "h",
                    "action": {"type": "end_shop"},
                    "rollout_values": [2.0, 2.0, 2.0, 2.0],
                    "is_heuristic_action": True,
                },
                {
                    "action_key": "risky",
                    "action": {"type": "open_pack"},
                    "rollout_values": [20.0, 0.0, 20.0, 0.0],
                    "is_heuristic_action": False,
                },
                {
                    "action_key": "steady",
                    "action": {"type": "buy"},
                    "rollout_values": [5.0, 5.0, 5.0, 5.0],
                    "is_heuristic_action": False,
                },
            ],
        }

        opportunities = script._best_opportunities(
            [record],
            z=1.0,
            min_mean_advantage=0.0,
            min_lcb=0.0,
            min_positive_rate=0.5,
            min_rollouts=2,
            max_sem=0.5,
            min_lcb_sem_ratio=None,
            candidate_action_types=(),
            exclude_candidate_action_types=(),
            excluded_candidate_keys=set(),
            preferred_candidate_types=("open_pack", "buy"),
        )

        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0].candidate_action_key, "steady")
        self.assertEqual(opportunities[0].candidate_action_type, "buy")

    def test_best_opportunities_skips_excluded_candidate_keys(self) -> None:
        record = _record(
            seed="a",
            heuristic_values=[2.0, 2.0],
            candidate_values=[4.0, 4.0],
            weak_values=[3.0, 3.0],
        )

        opportunities = script._best_opportunities(
            [record],
            z=1.0,
            min_mean_advantage=0.0,
            min_lcb=0.1,
            min_positive_rate=0.5,
            min_rollouts=2,
            max_sem=None,
            min_lcb_sem_ratio=None,
            candidate_action_types=(),
            exclude_candidate_action_types=(),
            excluded_candidate_keys={("a", 4, "strong")},
            preferred_candidate_types=("open_pack", "buy"),
        )

        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0].candidate_action_key, "weak")

    def test_candidate_exclusion_keys_reads_deepening_and_solver_records(self) -> None:
        keys = script._candidate_exclusion_keys(
            [
                {
                    "seed": "a",
                    "state_index": 1,
                    "deepening_candidate_action_key": "from-deepening",
                },
                {
                    "seed": "b",
                    "state_index": 2,
                    "candidates": [
                        {"action_key": "heuristic", "is_heuristic_action": True},
                        {"action_key": "from-solver", "is_heuristic_action": False},
                    ],
                },
            ]
        )

        self.assertEqual(keys, {("a", 1, "from-deepening"), ("b", 2, "from-solver")})

    def test_select_balanced_round_robins_heuristic_groups(self) -> None:
        opportunities = [
            _opportunity("sell", 2, 1.0, "a"),
            _opportunity("sell", 2, 5.0, "b"),
            _opportunity("end_shop", 2, 2.0, "c"),
            _opportunity("end_shop", 2, 6.0, "d"),
        ]

        selected = script._select_balanced(
            opportunities,
            limit=2,
            seed=0,
            balance_fields=("heuristic_action_type",),
        )

        self.assertEqual([item.record["seed"] for item in selected], ["d", "b"])

    def test_select_balanced_can_round_robin_terminal_outcomes(self) -> None:
        opportunities = [
            _opportunity("end_shop", 8, 1.0, "loss-a", terminal_won=False),
            _opportunity("end_shop", 8, 5.0, "loss-b", terminal_won=False),
            _opportunity("end_shop", 8, 2.0, "win-a", terminal_won=True),
            _opportunity("end_shop", 8, 6.0, "win-b", terminal_won=True),
        ]

        selected = script._select_balanced(
            opportunities,
            limit=2,
            seed=0,
            balance_fields=("terminal_won",),
        )

        self.assertEqual([item.record["seed"] for item in selected], ["loss-b", "win-b"])

    def test_atomic_write_jsonl_preserves_relabel_snapshot(self) -> None:
        opportunity = _opportunity("end_shop", 3, 2.0, "seed-a")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selected.jsonl"
            script._atomic_write_jsonl(path, [opportunity])
            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(row["seed"], "seed-a")
        self.assertEqual(row["state_snapshot"], {"phase": "shop"})
        self.assertEqual(row["deepening_candidate_action_type"], "open_pack")
        self.assertEqual(row["deepening_rollouts"], 2)
        self.assertEqual(row["deepening_lcb"], 2.0)
        self.assertEqual(row["terminal_won"], False)
        self.assertEqual(row["selection_reason"], "reached_ante_8")


def _record(
    *,
    seed: str,
    heuristic_values: list[float],
    candidate_values: list[float],
    weak_values: list[float],
) -> dict:
    return {
        "seed": seed,
        "state_index": 4,
        "source_bot": "bot",
        "ante": 2,
        "money": 10,
        "state_snapshot": {"phase": "shop"},
        "candidates": [
            {
                "action_key": "h",
                "action": {"type": "end_shop"},
                "rollout_values": heuristic_values,
                "is_heuristic_action": True,
            },
            {
                "action_key": "strong",
                "action": {"type": "open_pack"},
                "rollout_values": candidate_values,
                "is_heuristic_action": False,
            },
            {
                "action_key": "weak",
                "action": {"type": "buy"},
                "rollout_values": weak_values,
                "is_heuristic_action": False,
            },
        ],
    }


def _opportunity(
    heuristic_type: str,
    ante: int,
    score: float,
    seed: str,
    terminal_won: bool = False,
) -> script.DeepeningOpportunity:
    return script.DeepeningOpportunity(
        record={"seed": seed, "state_index": 1, "state_snapshot": {"phase": "shop"}},
        source_bot="bot",
        ante=ante,
        money=10,
        terminal_won=terminal_won,
        selection_reason="win" if terminal_won else "reached_ante_8",
        heuristic_action_type=heuristic_type,
        candidate_action_type="open_pack",
        candidate_action_key="candidate",
        heuristic_action_key="heuristic",
        n=2,
        mean_advantage=score,
        sem=0.0,
        lcb=score,
        positive_rate=1.0,
        score=score,
    )


if __name__ == "__main__":
    unittest.main()
