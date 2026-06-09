from __future__ import annotations

from unittest.mock import patch
import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.ml import shop_candidate_dataset as ds


def _shop_state() -> GameState:
    return GameState(
        phase=GamePhase.SHOP,
        ante=1,
        money=20,
        modifiers={
            "shop_cards": (
                {"key": "j_joker", "name": "Joker", "set": "JOKER", "cost": {"buy": 4}},
                {"key": "c_mercury", "name": "Mercury", "set": "PLANET", "cost": {"buy": 3}},
            ),
            "booster_packs": (
                {"key": "p_buffoon_normal_1", "name": "Buffoon Pack", "set": "BOOSTER", "cost": {"buy": 4}},
            ),
        },
    )


class ShopCandidateDatasetTests(unittest.TestCase):
    def test_candidate_shop_actions_include_buys_packs_and_end_shop(self) -> None:
        actions = ds.candidate_shop_actions(_shop_state())
        types = {action.action_type for action in actions}

        self.assertIn(ActionType.BUY, types)
        self.assertIn(ActionType.OPEN_PACK, types)
        self.assertIn(ActionType.END_SHOP, types)

    def test_candidate_shop_actions_can_filter_action_types(self) -> None:
        actions = ds.candidate_shop_actions(
            _shop_state(),
            action_types={ActionType.BUY, ActionType.END_SHOP},
        )
        types = {action.action_type for action in actions}

        self.assertIn(ActionType.BUY, types)
        self.assertIn(ActionType.END_SHOP, types)
        self.assertNotIn(ActionType.OPEN_PACK, types)

    def test_candidate_shop_actions_deep_priority_keeps_pack_and_end_shop_under_budget(self) -> None:
        actions = ds.candidate_shop_actions(
            _shop_state(),
            max_actions=2,
            action_types={ActionType.BUY, ActionType.OPEN_PACK, ActionType.END_SHOP},
            priority="deep_advantage",
        )
        types = [action.action_type for action in actions]

        self.assertEqual(types, [ActionType.END_SHOP, ActionType.OPEN_PACK])

    def test_action_key_distinguishes_shop_indices(self) -> None:
        first = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        second = Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1})

        self.assertNotEqual(ds.action_key(first), ds.action_key(second))

    def test_label_shop_state_builds_ranked_record(self) -> None:
        state = _shop_state()
        heuristic = Action(ActionType.END_SHOP)

        def value_for_action(_state, action, *, seed, rollout_bot, max_antes, max_steps):
            del _state, rollout_bot, max_antes, max_steps
            base = 3.0 if action.action_type == ActionType.OPEN_PACK else 1.0
            return base + seed * 0.01

        with (
            patch("balatro_ai.ml.shop_candidate_dataset.rollout_value_after_action", side_effect=value_for_action),
            patch("balatro_ai.ml.shop_candidate_dataset._heuristic_action_key", return_value=ds.action_key(heuristic)),
        ):
            record = ds.label_shop_state(
                state,
                seed="0000001",
                state_index=4,
                source_bot="basic_strategy_bot",
                crn_seeds=(1, 2),
                rollout_bot="basic_strategy_bot",
                max_antes=2,
            )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.seed, "0000001")
        self.assertEqual(record.state_index, 4)
        self.assertEqual(record.source_bot, "basic_strategy_bot")
        self.assertEqual(record.label_value_version, ds.LABEL_VALUE_VERSION)
        self.assertEqual(record.best_action_key, record.candidates[0].action_key)
        self.assertEqual(record.candidates[0].rank, 1)
        self.assertEqual(record.candidates[0].action["type"], "open_pack")
        self.assertEqual(record.candidates[0].shop_token_index, 2)
        self.assertEqual(record.encoding_version, record.encoded_state["version"])
        reloaded = GameState.from_mapping(record.state_snapshot)
        self.assertEqual(reloaded.phase, state.phase)
        self.assertEqual(reloaded.ante, state.ante)
        self.assertEqual(reloaded.money, state.money)
        self.assertEqual(len(reloaded.modifiers.get("shop_cards", ())), 2)
        self.assertEqual(len(reloaded.modifiers.get("booster_packs", ())), 1)

    def test_paired_rollout_advantage_uses_crn_differences(self) -> None:
        candidate = {"rollout_values": [2.0, 4.0, 6.0, 8.0]}
        baseline = {"rollout_values": [1.0, 2.0, 3.0, 4.0]}

        stats = ds.paired_rollout_advantage(candidate, baseline, z=1.0)

        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.n, 4)
        self.assertEqual(stats.mean, 2.5)
        self.assertAlmostEqual(stats.sem, 0.6454972244)
        self.assertAlmostEqual(stats.lower_bound, 1.8545027756)
        self.assertEqual(stats.positive_rate, 1.0)

    def test_rollout_confidence_summary_separates_clear_and_ambiguous_preferences(self) -> None:
        clear = {
            "candidates": [
                {
                    "action_key": "buy_a",
                    "mean_value": 3.0,
                    "rollout_values": [3.0, 3.0, 3.0, 3.0],
                    "is_heuristic_action": True,
                },
                {
                    "action_key": "buy_b",
                    "mean_value": 4.0,
                    "rollout_values": [4.0, 4.0, 4.0, 4.0],
                    "is_heuristic_action": False,
                },
                {
                    "action_key": "end_shop",
                    "mean_value": 3.2,
                    "rollout_values": [3.2, 3.2, 3.2, 3.2],
                    "is_heuristic_action": False,
                },
            ],
        }
        ambiguous = {
            "candidates": [
                {
                    "action_key": "buy_a",
                    "mean_value": 3.0,
                    "rollout_values": [2.0, 4.0, 2.0, 4.0],
                    "is_heuristic_action": True,
                },
                {
                    "action_key": "buy_b",
                    "mean_value": 3.1,
                    "rollout_values": [4.0, 2.0, 4.0, 2.4],
                    "is_heuristic_action": False,
                },
                {
                    "action_key": "end_shop",
                    "mean_value": 3.0,
                    "rollout_values": [3.0, 3.0, 3.0, 3.0],
                    "is_heuristic_action": False,
                },
            ],
        }

        summary = ds.rollout_confidence_summary([clear, ambiguous], z=1.0, practical_margin=0.10)

        self.assertEqual(summary["best_runnerup_covered_n"], 2)
        self.assertEqual(summary["heuristic_confidence_covered_n"], 2)
        self.assertEqual(summary["oracle_positive_vs_heuristic_rate"], 1.0)
        self.assertEqual(summary["best_high_conf_beats_heuristic_rate"], 0.5)
        self.assertEqual(summary["best_ambiguous_vs_heuristic_rate"], 0.5)
        self.assertEqual(summary["any_high_conf_practical_override_candidate_rate"], 0.5)

    def test_label_shop_state_respects_candidate_action_filter(self) -> None:
        state = _shop_state()

        def value_for_action(_state, action, *, seed, rollout_bot, max_antes, max_steps):
            del _state, rollout_bot, max_antes, max_steps
            return float(seed) + (1.0 if action.action_type == ActionType.BUY else 0.0)

        with (
            patch("balatro_ai.ml.shop_candidate_dataset.rollout_value_after_action", side_effect=value_for_action),
            patch("balatro_ai.ml.shop_candidate_dataset._heuristic_action_key", return_value=None),
        ):
            record = ds.label_shop_state(
                state,
                seed="0000001",
                state_index=4,
                crn_seeds=(1, 2),
                rollout_bot="basic_strategy_bot",
                max_antes=2,
                candidate_action_types={ActionType.BUY, ActionType.END_SHOP},
            )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(
            {candidate.action["type"] for candidate in record.candidates},
            {"buy", "end_shop"},
        )

    def test_label_shop_state_can_include_filtered_heuristic_action(self) -> None:
        state = _shop_state()
        heuristic = Action(ActionType.REROLL)

        def value_for_action(_state, action, *, seed, rollout_bot, max_antes, max_steps):
            del _state, rollout_bot, max_antes, max_steps
            return float(seed) + (0.5 if action.action_type == ActionType.REROLL else 0.0)

        with (
            patch("balatro_ai.ml.shop_candidate_dataset.rollout_value_after_action", side_effect=value_for_action),
            patch("balatro_ai.ml.shop_candidate_dataset._heuristic_action", return_value=heuristic),
        ):
            record = ds.label_shop_state(
                state,
                seed="0000001",
                state_index=4,
                crn_seeds=(1, 2),
                rollout_bot="basic_strategy_bot",
                max_antes=2,
                candidate_action_types={ActionType.BUY, ActionType.END_SHOP},
                include_heuristic_action=True,
            )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertIn("reroll", {candidate.action["type"] for candidate in record.candidates})
        self.assertEqual(record.heuristic_action_key, ds.action_key(heuristic))
        self.assertTrue(any(candidate.is_heuristic_action for candidate in record.candidates))

    def test_terminal_value_uses_horizon_quality_for_same_ante_shop_survivors(self) -> None:
        root = _shop_state()
        terminal = GameState(phase=GamePhase.SHOP, ante=3, money=12)

        with patch("balatro_ai.search.shop_search.shop_leaf_value", return_value=120.0):
            value = ds._rollout_terminal_value(terminal, root_state=root)

        self.assertAlmostEqual(value, 3.535)

    def test_terminal_quality_bonus_is_capped_below_next_ante(self) -> None:
        terminal = GameState(phase=GamePhase.SHOP, ante=3, money=12)

        with patch("balatro_ai.search.shop_search.shop_leaf_value", return_value=10000.0):
            value = ds._rollout_terminal_value(terminal)

        self.assertLess(value, 4.0)
        self.assertAlmostEqual(value, 3.95)

    def test_terminal_value_prefers_economy_at_same_survival_horizon(self) -> None:
        poor = GameState(phase=GamePhase.SHOP, ante=3, money=2)
        banked = GameState(phase=GamePhase.SHOP, ante=3, money=22)

        self.assertGreater(
            ds._rollout_terminal_value(banked, root_state=_shop_state()),
            ds._rollout_terminal_value(poor, root_state=_shop_state()),
        )

    def test_terminal_value_preserves_economy_when_strategic_value_ties(self) -> None:
        poor = GameState(phase=GamePhase.SHOP, ante=3, money=2)
        banked = GameState(phase=GamePhase.SHOP, ante=3, money=22)

        with patch("balatro_ai.search.shop_search.shop_leaf_value", return_value=100.0):
            poor_value = ds._rollout_terminal_value(poor, root_state=_shop_state())
            banked_value = ds._rollout_terminal_value(banked, root_state=_shop_state())

        self.assertGreater(banked_value, poor_value)

    def test_run_over_terminal_value_uses_score_fraction(self) -> None:
        terminal = GameState(
            phase=GamePhase.RUN_OVER,
            ante=2,
            current_score=50,
            required_score=100,
            run_over=True,
        )

        with patch("balatro_ai.search.shop_search.shop_leaf_value", return_value=10000.0):
            value = ds._rollout_terminal_value(terminal)

        self.assertAlmostEqual(value, 2.5)

    def test_collect_shop_states_can_quota_per_ante(self) -> None:
        states = [
            GameState(phase=GamePhase.SHOP, ante=2),
            GameState(phase=GamePhase.SHOP, ante=2),
            GameState(phase=GamePhase.SHOP, ante=3),
            GameState(phase=GamePhase.SHOP, ante=3),
            GameState(phase=GamePhase.RUN_OVER, ante=3, run_over=True),
        ]
        build_actions = (
            Action(ActionType.BUY, target_id="card", amount=0),
            Action(ActionType.END_SHOP),
        )

        class FakeBot:
            def choose_action(self, state):
                del state
                return Action(ActionType.END_SHOP)

        class FakeSeedGame:
            def __init__(self, seed, *, stake):
                del seed, stake

            def initial_state(self):
                return states[0]

        class FakeSimulator:
            def __init__(self, *, seed, stake):
                del seed, stake
                self.index = 0
                self.state = states[0]

            def step(self, action):
                del action
                self.index += 1
                self.state = states[self.index]

        with (
            patch("balatro_ai.ml.shop_candidate_dataset.candidate_shop_actions", return_value=build_actions),
            patch("balatro_ai.bots.registry.create_bot", return_value=FakeBot()),
            patch("balatro_ai.sim.local_runner.LocalBalatroSimulator", FakeSimulator),
            patch("balatro_ai.solver.seed_game.SeedGame", FakeSeedGame),
        ):
            collected = ds.collect_shop_states(
                ["0000001"],
                bot_name="fake_bot",
                cap=10,
                per_seed=1,
                max_steps=10,
                min_ante=2,
                max_ante=3,
                balance_antes=True,
            )

        self.assertEqual([state.ante for _, _, state in collected], [2, 3])


if __name__ == "__main__":
    unittest.main()
