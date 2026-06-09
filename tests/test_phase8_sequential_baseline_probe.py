from __future__ import annotations

from unittest.mock import patch
import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.ml.shop_candidate_dataset import action_key, candidate_shop_actions, state_snapshot
from scripts import phase8_sequential_baseline_probe as script


def _shop_state() -> GameState:
    return GameState(
        phase=GamePhase.SHOP,
        ante=2,
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


class Phase8SequentialBaselineProbeTests(unittest.TestCase):
    def test_half_mean_handles_even_and_odd_rollout_counts(self) -> None:
        self.assertEqual(script._half_mean((1.0, 3.0, 5.0, 7.0), first=True), 2.0)
        self.assertEqual(script._half_mean((1.0, 3.0, 5.0, 7.0), first=False), 6.0)
        self.assertEqual(script._half_mean((1.0, 3.0, 5.0), first=True), 1.0)
        self.assertEqual(script._half_mean((1.0, 3.0, 5.0), first=False), 4.0)

    def test_probe_record_stops_clear_positive_and_negative_candidates(self) -> None:
        heuristic = Action(ActionType.END_SHOP)
        record = {
            "seed": "0000001",
            "state_index": 4,
            "source_bot": "test",
            "state_snapshot": state_snapshot(_shop_state()),
        }

        def value_for_action(_state, action, *, seed, rollout_bot, max_antes, max_steps):
            del _state, seed, rollout_bot, max_antes, max_steps
            if action.action_type == ActionType.OPEN_PACK:
                return 3.0
            if action.action_type == ActionType.BUY:
                return 0.0
            return 1.0

        with (
            patch("balatro_ai.ml.shop_candidate_dataset._heuristic_action", return_value=heuristic),
            patch("balatro_ai.ml.shop_candidate_dataset.rollout_value_after_action", side_effect=value_for_action),
        ):
            probed = script._probe_record(
                record,
                rollout_bot="solver_shop_basic_play_bot",
                heuristic_bot="solver_shop_basic_play_bot",
                max_antes=4,
                max_steps=100,
                max_actions=4,
                candidate_action_types=(ActionType.BUY, ActionType.OPEN_PACK, ActionType.END_SHOP),
                candidate_priority="deep_advantage",
                include_heuristic_action=True,
                min_rollouts=2,
                max_rollouts=4,
                z=1.0,
                positive_margin=0.10,
                negative_margin=0.10,
                max_wall_s_per_state=0.0,
                focus_action_key="",
                focus_deepening_candidate=False,
            )

        self.assertIsNotNone(probed)
        assert probed is not None
        self.assertTrue(probed["sequential_probe"])
        self.assertFalse(probed["sequential_timed_out"])
        stop_reasons = {
            candidate["action"]["type"]: candidate["sequential_stop_reason"]
            for candidate in probed["candidates"]
            if not candidate["is_heuristic_action"]
        }
        self.assertEqual(stop_reasons["open_pack"], "positive_lcb")
        self.assertEqual(stop_reasons["buy"], "negative_ucb")
        self.assertTrue(all(candidate["sequential_rollouts"] == 2 for candidate in probed["candidates"]))

    def test_probe_record_can_focus_deepening_candidate(self) -> None:
        state = _shop_state()
        actions = {
            action.action_type: action
            for action in candidate_shop_actions(
                state,
                max_actions=4,
                action_types={ActionType.BUY, ActionType.OPEN_PACK, ActionType.END_SHOP},
                priority="deep_advantage",
            )
        }
        heuristic = Action(ActionType.END_SHOP)
        focus = actions[ActionType.OPEN_PACK]
        record = {
            "seed": "0000001",
            "state_index": 4,
            "source_bot": "test",
            "state_snapshot": state_snapshot(state),
            "deepening_candidate_action_key": action_key(focus),
        }

        def value_for_action(_state, action, *, seed, rollout_bot, max_antes, max_steps):
            del _state, seed, rollout_bot, max_antes, max_steps
            return 3.0 if action.action_type == ActionType.OPEN_PACK else 1.0

        with (
            patch("balatro_ai.ml.shop_candidate_dataset._heuristic_action", return_value=heuristic),
            patch("balatro_ai.ml.shop_candidate_dataset.rollout_value_after_action", side_effect=value_for_action),
        ):
            probed = script._probe_record(
                record,
                rollout_bot="solver_shop_basic_play_bot",
                heuristic_bot="solver_shop_basic_play_bot",
                max_antes=4,
                max_steps=100,
                max_actions=4,
                candidate_action_types=(ActionType.BUY, ActionType.OPEN_PACK, ActionType.END_SHOP),
                candidate_priority="deep_advantage",
                include_heuristic_action=True,
                min_rollouts=2,
                max_rollouts=4,
                z=1.0,
                positive_margin=0.10,
                negative_margin=0.10,
                max_wall_s_per_state=0.0,
                focus_action_key="",
                focus_deepening_candidate=True,
            )

        self.assertIsNotNone(probed)
        assert probed is not None
        self.assertEqual(len(probed["candidates"]), 2)
        self.assertEqual(
            {candidate["action"]["type"] for candidate in probed["candidates"]},
            {"end_shop", "open_pack"},
        )
        self.assertEqual(probed["sequential_focus_action_key"], record["deepening_candidate_action_key"])

    def test_probe_record_can_focus_captured_action_outside_regenerated_budget(self) -> None:
        state = _shop_state()
        heuristic = Action(ActionType.END_SHOP)
        focus = Action(ActionType.REROLL, metadata={"kind": "", "index": 0})
        record = {
            "seed": "0000001",
            "state_index": 4,
            "source_bot": "ranker_capture",
            "state_snapshot": state_snapshot(state),
            "deepening_candidate_action_key": action_key(focus),
            "deepening_candidate_action": focus.to_json(),
        }

        def value_for_action(_state, action, *, seed, rollout_bot, max_antes, max_steps):
            del _state, seed, rollout_bot, max_antes, max_steps
            return 2.0 if action.action_type == ActionType.REROLL else 1.0

        with (
            patch("balatro_ai.ml.shop_candidate_dataset._heuristic_action", return_value=heuristic),
            patch("balatro_ai.ml.shop_candidate_dataset.rollout_value_after_action", side_effect=value_for_action),
        ):
            probed = script._probe_record(
                record,
                rollout_bot="solver_shop_basic_play_bot",
                heuristic_bot="solver_shop_basic_play_bot",
                max_antes=4,
                max_steps=100,
                max_actions=1,
                candidate_action_types=(ActionType.END_SHOP,),
                candidate_priority="legal",
                include_heuristic_action=True,
                min_rollouts=2,
                max_rollouts=2,
                z=1.0,
                positive_margin=0.10,
                negative_margin=0.10,
                max_wall_s_per_state=0.0,
                focus_action_key="",
                focus_deepening_candidate=True,
            )

        self.assertIsNotNone(probed)
        assert probed is not None
        self.assertEqual(
            {candidate["action"]["type"] for candidate in probed["candidates"]},
            {"end_shop", "reroll"},
        )

    def test_summarize_reports_stop_reasons_and_confidence(self) -> None:
        record = {
            "source_bot": "test",
            "ante": 2,
            "candidates": [
                {
                    "action_key": "pack",
                    "action": {"type": "open_pack"},
                    "rollout_values": [3.0, 3.0],
                    "mean_value": 3.0,
                    "is_heuristic_action": False,
                    "sequential_rollouts": 2,
                    "sequential_stop_reason": "positive_lcb",
                },
                {
                    "action_key": "end",
                    "action": {"type": "end_shop"},
                    "rollout_values": [1.0, 1.0],
                    "mean_value": 1.0,
                    "is_heuristic_action": True,
                    "sequential_rollouts": 2,
                    "sequential_stop_reason": "baseline",
                },
            ],
        }
        args = _Args()

        metrics = script._summarize([record], args=args, wall_s=2.0)

        self.assertEqual(metrics["records"], 1)
        self.assertEqual(metrics["input_record_count"], 1)
        self.assertEqual(metrics["skipped_record_count"], 0)
        self.assertEqual(metrics["candidate_stop_reasons"], {"positive_lcb": 1})
        self.assertEqual(metrics["estimated_candidate_continuations"], 4)
        self.assertEqual(metrics["best_high_conf_beats_heuristic_rate"], 1.0)


class _Args:
    jobs = 1
    input_records = []
    rollout_bot = "rollout"
    heuristic_bot = "heuristic"
    candidate_action_types = (ActionType.BUY, ActionType.OPEN_PACK)
    candidate_priority = "deep_advantage"
    include_heuristic_action = True
    min_rollouts = 2
    max_rollouts = 4
    max_antes = 4
    max_steps = 100
    z = 1.0
    positive_margin = 0.10
    negative_margin = 0.10
    max_wall_s_per_state = 0.0
    focus_deepening_candidate = False
    focus_action_key = ""
    rust_bestplay = True


if __name__ == "__main__":
    unittest.main()
