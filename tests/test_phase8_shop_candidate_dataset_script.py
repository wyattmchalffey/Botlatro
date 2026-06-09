from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import context  # noqa: F401
from balatro_ai.api.actions import ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.ml.shop_candidate_dataset import state_snapshot
from scripts import phase8_shop_candidate_dataset as script


class Phase8ShopCandidateDatasetScriptTests(unittest.TestCase):
    def test_select_states_uses_deterministic_shuffle_before_limit(self) -> None:
        states = [
            ("bot_a", "0000001", 1, object()),
            ("bot_a", "0000002", 2, object()),
            ("bot_b", "0000003", 3, object()),
            ("bot_b", "0000004", 4, object()),
        ]

        selected = script._select_states(states, limit=2, selection_seed=2)

        self.assertEqual([item[1] for item in selected], ["0000002", "0000003"])

    def test_select_states_can_balance_source_bots(self) -> None:
        states = [
            ("bot_a", "a1", 1, object()),
            ("bot_a", "a2", 2, object()),
            ("bot_a", "a3", 3, object()),
            ("bot_b", "b1", 4, object()),
            ("bot_b", "b2", 5, object()),
            ("bot_b", "b3", 6, object()),
        ]

        selected = script._select_states(
            states,
            limit=4,
            selection_seed=0,
            balance_sources=True,
        )

        self.assertEqual([item[0] for item in selected], ["bot_a", "bot_b", "bot_a", "bot_b"])

    def test_select_states_can_balance_sources_and_antes(self) -> None:
        states = [
            ("bot_a", "a2-1", 1, SimpleNamespace(ante=2)),
            ("bot_a", "a2-2", 2, SimpleNamespace(ante=2)),
            ("bot_a", "a3-1", 3, SimpleNamespace(ante=3)),
            ("bot_a", "a3-2", 4, SimpleNamespace(ante=3)),
            ("bot_b", "b2-1", 5, SimpleNamespace(ante=2)),
            ("bot_b", "b2-2", 6, SimpleNamespace(ante=2)),
            ("bot_b", "b3-1", 7, SimpleNamespace(ante=3)),
            ("bot_b", "b3-2", 8, SimpleNamespace(ante=3)),
        ]

        selected = script._select_states(
            states,
            limit=4,
            selection_seed=0,
            balance_sources=True,
            balance_antes=True,
        )

        selected_groups = {(item[0], item[3].ante) for item in selected}
        self.assertEqual(selected_groups, {("bot_a", 2), ("bot_a", 3), ("bot_b", 2), ("bot_b", 3)})

    def test_parse_action_types_csv_accepts_values_and_names(self) -> None:
        parsed = script._parse_action_types_csv("buy, OPEN_PACK, buy")

        self.assertEqual([action.value for action in parsed], ["buy", "open_pack"])

    def test_chunks_split_seeds_without_dropping_items(self) -> None:
        chunks = script._chunks(["a", "b", "c", "d", "e"], 2)

        self.assertEqual(chunks, [["a", "b", "c"], ["d", "e"]])

    def test_default_partial_path_appends_partial_suffix(self) -> None:
        self.assertEqual(
            script._default_partial_path(Path(".data/out.jsonl")),
            Path(".data/out.jsonl.partial"),
        )

    def test_write_label_progress_writes_ordered_partial_records_and_metrics(self) -> None:
        args = SimpleNamespace(
            jobs=4,
            capture_bot=["bot_a", "bot_b"],
            rollout_bot="rollout",
            heuristic_bot="heuristic",
            seed_offset=1,
            seed_count=2,
            captured_states=3,
            deduped_states=2,
            selected_states=2,
            collect_jobs=3,
            min_capture_ante=2,
            max_capture_ante=5,
            selection_seed=7,
            balance_source_bots=True,
            balance_antes=False,
            candidate_action_types=[ActionType.BUY],
            rust_bestplay=True,
            rollouts=4,
            max_antes=2,
            partial_every=1,
        )
        record_a = {
            "seed": "a",
            "source_bot": "bot_a",
            "ante": 2,
            "heuristic_action_key": "a",
            "best_action_key": "a",
            "candidates": [
                {
                    "action_key": "a",
                    "mean_value": 3.0,
                    "first_half_mean": 3.0,
                    "second_half_mean": 3.0,
                    "is_heuristic_action": True,
                },
                {
                    "action_key": "b",
                    "mean_value": 2.0,
                    "first_half_mean": 2.0,
                    "second_half_mean": 2.0,
                    "is_heuristic_action": False,
                },
            ],
        }
        record_b = {
            "seed": "b",
            "source_bot": "bot_b",
            "ante": 2,
            "heuristic_action_key": "b",
            "best_action_key": "c",
            "candidates": [
                {
                    "action_key": "c",
                    "mean_value": 4.0,
                    "first_half_mean": 4.0,
                    "second_half_mean": 4.0,
                    "is_heuristic_action": False,
                },
                {
                    "action_key": "b",
                    "mean_value": 3.5,
                    "first_half_mean": 3.5,
                    "second_half_mean": 3.5,
                    "is_heuristic_action": True,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            partial_out = Path(tmpdir) / "records.jsonl.partial"
            partial_metrics = Path(tmpdir) / "metrics.json.partial"

            script._write_label_progress(
                [(1, record_b), (0, record_a)],
                args=args,
                started=time.perf_counter(),
                expected_label_jobs=3,
                completed_label_jobs=2,
                partial_out=partial_out,
                partial_metrics=partial_metrics,
                complete=False,
            )

            rows = [json.loads(line) for line in partial_out.read_text(encoding="utf-8").splitlines()]
            metrics = json.loads(partial_metrics.read_text(encoding="utf-8"))

        self.assertEqual([row["seed"] for row in rows], ["a", "b"])
        self.assertFalse(metrics["complete"])
        self.assertEqual(metrics["expected_label_jobs"], 3)
        self.assertEqual(metrics["completed_label_jobs"], 2)
        self.assertEqual(metrics["records"], 2)
        self.assertEqual(metrics["partial_out"], str(partial_out))

    def test_load_partial_records_for_jobs_matches_current_jobs(self) -> None:
        jobs = [
            ("bot_a", "0000001", 4, object()),
            ("bot_b", "0000002", 9, object()),
        ]
        record_a = {
            "seed": "0000001",
            "state_index": 4,
            "source_bot": "bot_a",
            "candidates": [],
        }
        record_b = {
            "seed": "0000002",
            "state_index": 9,
            "source_bot": "bot_b",
            "candidates": [],
        }
        unknown_record = {
            "seed": "9999999",
            "state_index": 1,
            "source_bot": "bot_a",
            "candidates": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            partial = Path(tmpdir) / "records.jsonl.partial"
            partial.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (record_b, record_a, unknown_record, record_b)
                ),
                encoding="utf-8",
            )

            loaded = script._load_partial_records_for_jobs(partial, jobs)

        self.assertEqual([(index, record["seed"]) for index, record in loaded], [(1, "0000002"), (0, "0000001")])

    def test_load_states_from_records_uses_state_snapshot(self) -> None:
        state = GameState(phase=GamePhase.SHOP, ante=3, money=17)
        record = {
            "source_bot": "bot_a",
            "seed": "0000001",
            "state_index": 4,
            "terminal_won": True,
            "selection_reason": "win",
            "state_snapshot": state_snapshot(state),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.jsonl"
            path.write_text(json.dumps(record), encoding="utf-8")

            loaded = script._load_states_from_records([path])

        self.assertEqual(len(loaded), 1)
        source_bot, seed, state_index, loaded_state, metadata = loaded[0]
        self.assertEqual((source_bot, seed, state_index), ("bot_a", "0000001", 4))
        self.assertEqual(loaded_state.phase, GamePhase.SHOP)
        self.assertEqual(loaded_state.ante, 3)
        self.assertEqual(loaded_state.money, 17)
        self.assertEqual(metadata["terminal_won"], True)
        self.assertEqual(metadata["selection_reason"], "win")

    def test_load_states_from_records_rejects_old_records_without_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.jsonl"
            path.write_text(json.dumps({"seed": "0000001"}), encoding="utf-8")

            with self.assertRaises(ValueError):
                script._load_states_from_records([path])

    def test_state_pool_records_are_reloadable(self) -> None:
        state = GameState(phase=GamePhase.SHOP, ante=2, money=11)

        records = script._state_pool_records(
            [("bot_a", "0000001", 9, state, {"terminal_won": False, "selection_reason": "reached_ante_8"})]
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["source_bot"], "bot_a")
        self.assertEqual(record["seed"], "0000001")
        self.assertEqual(record["state_index"], 9)
        self.assertEqual(record["ante"], 2)
        self.assertEqual(record["terminal_won"], False)
        self.assertEqual(record["selection_reason"], "reached_ante_8")
        loaded_state = GameState.from_mapping(record["state_snapshot"])
        self.assertEqual(loaded_state.phase, GamePhase.SHOP)
        self.assertEqual(loaded_state.money, 11)

    def test_label_worker_preserves_source_metadata(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            money=20,
            modifiers={
                "shop_cards": (
                    {"key": "j_joker", "name": "Joker", "set": "JOKER", "cost": {"buy": 4}},
                ),
                "booster_packs": (
                    {"key": "p_buffoon_normal_1", "name": "Buffoon Pack", "set": "BOOSTER", "cost": {"buy": 4}},
                ),
            },
        )

        record = script._label_worker(
            (
                "bot_a",
                "0000001",
                2,
                state,
                {"terminal_won": True, "selection_reason": "win"},
                (1, 2),
                "basic_strategy_bot",
                "basic_strategy_bot",
                1,
                20,
                4,
                None,
                "legal",
                True,
            )
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["terminal_won"], True)
        self.assertEqual(record["selection_reason"], "win")

    def test_summarize_state_pool_reports_capture_only_metrics(self) -> None:
        args = SimpleNamespace(
            jobs=2,
            capture_bot=["bot_a", "bot_b"],
            input_records=[],
            seed_offset=1,
            seed_count=2,
            captured_states=4,
            deduped_states=3,
            selected_states=2,
            collect_jobs=2,
            min_capture_ante=2,
            max_capture_ante=3,
            selection_seed=7,
            balance_source_bots=True,
            balance_antes=True,
            rust_bestplay=True,
        )
        records = [
            {"source_bot": "bot_a", "ante": 2},
            {"source_bot": "bot_b", "ante": 3},
        ]

        metrics = script._summarize_state_pool(records, args=args, wall_s=2.0)

        self.assertTrue(metrics["capture_only"])
        self.assertEqual(metrics["records"], 2)
        self.assertEqual(metrics["records_by_source_bot"], {"bot_a": 1, "bot_b": 1})
        self.assertEqual(metrics["records_by_ante"], {"2": 1, "3": 1})
        self.assertEqual(metrics["wall_s_per_record"], 1.0)

    def test_summarize_reports_source_and_ante_distribution(self) -> None:
        args = SimpleNamespace(
            jobs=4,
            capture_bot=["bot_a", "bot_b"],
            rollout_bot="rollout",
            heuristic_bot="heuristic",
            seed_offset=1,
            seed_count=2,
            captured_states=3,
            deduped_states=2,
            selected_states=2,
            collect_jobs=3,
            min_capture_ante=2,
            max_capture_ante=5,
            selection_seed=7,
            balance_source_bots=True,
            balance_antes=False,
            candidate_action_types=[ActionType.BUY],
            rust_bestplay=True,
            rollouts=4,
            max_antes=2,
        )
        records = [
            {
                "source_bot": "bot_a",
                "ante": 2,
                "heuristic_action_key": "a",
                "best_action_key": "a",
                "candidates": [
                    {
                        "action_key": "a",
                        "mean_value": 3.0,
                        "first_half_mean": 3.1,
                        "second_half_mean": 2.9,
                        "is_heuristic_action": True,
                        "action": {"type": "buy"},
                    },
                    {
                        "action_key": "b",
                        "mean_value": 2.5,
                        "first_half_mean": 2.4,
                        "second_half_mean": 2.6,
                        "is_heuristic_action": False,
                    },
                ],
            },
            {
                "source_bot": "bot_b",
                "ante": 3,
                "heuristic_action_key": "b",
                "best_action_key": "c",
                "candidates": [
                    {
                        "action_key": "c",
                        "mean_value": 4.0,
                        "first_half_mean": 3.0,
                        "second_half_mean": 4.5,
                        "is_heuristic_action": False,
                    },
                    {
                        "action_key": "b",
                        "mean_value": 3.0,
                        "first_half_mean": 3.5,
                        "second_half_mean": 2.5,
                        "is_heuristic_action": True,
                        "action": {"type": "reroll"},
                    },
                ],
            },
        ]

        metrics = script._summarize(records, args=args, wall_s=1.25)

        self.assertEqual(metrics["estimated_candidate_continuations"], 16)
        self.assertEqual(metrics["candidate_continuations_per_wall_s"], 12.8)
        self.assertEqual(metrics["wall_s_per_record"], 0.625)
        self.assertEqual(metrics["records_by_source_bot"], {"bot_a": 1, "bot_b": 1})
        self.assertEqual(metrics["label_value_version"], 3)
        self.assertEqual(metrics["candidate_action_types"], ["buy"])
        self.assertEqual(metrics["heuristic_action_types"], {"buy": 1, "reroll": 1})
        self.assertEqual(metrics["heuristic_outside_candidate_action_types_rate"], 0.5)
        self.assertEqual(metrics["heuristic_outside_candidate_action_types_count"], 1)
        self.assertEqual(metrics["records_by_label_value_version"], {"1": 2})
        self.assertEqual(metrics["records_by_ante"], {"2": 1, "3": 1})
        self.assertEqual(metrics["selected_states"], 2)
        self.assertEqual(metrics["collect_jobs"], 3)
        self.assertEqual(metrics["min_capture_ante"], 2)
        self.assertTrue(metrics["balance_source_bots"])
        self.assertFalse(metrics["balance_antes"])
        self.assertTrue(metrics["rust_bestplay"])
        self.assertEqual(metrics["nonzero_best_margin_rate"], 1.0)
        self.assertEqual(metrics["mean_top_tie_count"], 1.0)
        self.assertEqual(metrics["mean_actions_within_0_05"], 1.0)
        self.assertEqual(metrics["mean_actions_within_0_10"], 1.0)
        self.assertEqual(metrics["heuristic_within_0_05_rate"], 0.5)
        self.assertEqual(metrics["heuristic_within_0_10_rate"], 0.5)
        self.assertEqual(metrics["split_half_best_agreement_rate"], 0.5)
        self.assertEqual(metrics["mean_best_first_half_agreement_rate"], 0.5)
        self.assertEqual(metrics["mean_best_second_half_agreement_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
