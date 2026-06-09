"""Tests for `balatro_ai.solver.policy` (Milestone M5).

Verifies the composed solver policy:
- Delegates non-search phases (BLIND_SELECT, CASH_OUT) to the fallback.
- Routes SHOP to shop beam search, SELECTING_HAND to play beam search.
- Recovers gracefully when either sub-search returns None / raises.
- Drives a full trajectory through `generate_trajectory` without
  crashing -- the M5 acceptance bar.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState, Joker
from balatro_ai.search.shop_search import ShopSearchConfig
from balatro_ai.solver.play_search import PlaySearchPolicy
from balatro_ai.solver.policy import SolverPolicy
from balatro_ai.solver.search_v2.play import SearchV2PlayPolicy
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import generate_trajectory


class _FixedFallback:
    def __init__(self, action: Action) -> None:
        self.action = action
        self.calls = 0

    def choose_action(self, _state: GameState) -> Action:
        self.calls += 1
        return self.action


class _RecordingFallback(_FixedFallback):
    def __init__(self) -> None:
        super().__init__(Action(ActionType.END_SHOP))
        self.synced_states: list[GameState] = []
        self.recorded_actions: list[Action] = []

    def _sync_shop_memory(self, state: GameState) -> None:
        self.synced_states.append(state)

    def _record_shop_action(self, _state: GameState, action: Action) -> None:
        self.recorded_actions.append(action)


class _InfoFirstFallback(_RecordingFallback):
    def __init__(self, action: Action) -> None:
        super().__init__()
        self.action = action

    def choose_action(self, _state: GameState) -> Action:
        self.calls += 1
        return self.action


class SolverPolicyDelegationTests(unittest.TestCase):
    def test_blind_select_delegates_to_fallback(self) -> None:
        fallback = _FixedFallback(Action(action_type=ActionType.SELECT_BLIND))
        policy = SolverPolicy(fallback=fallback)
        state = SeedGame("AAAAAAA").initial_state()
        self.assertEqual(state.phase, GamePhase.BLIND_SELECT)
        action = policy.choose_action(state)
        self.assertEqual(action.action_type, ActionType.SELECT_BLIND)
        self.assertEqual(fallback.calls, 1)

    def test_callable_alias_works(self) -> None:
        fallback = _FixedFallback(Action(action_type=ActionType.SELECT_BLIND))
        policy = SolverPolicy(fallback=fallback)
        state = SeedGame("AAAAAAA").initial_state()
        action_direct = policy(state)
        self.assertEqual(action_direct.action_type, ActionType.SELECT_BLIND)

    def test_default_play_backend_is_legacy(self) -> None:
        # Default reverted from "v2" to "legacy" on 2026-05-26 after a
        # 4-seed measurement showed v2 averages ante 2.0 vs legacy 4.5.
        # See SOLVER_OPTIMIZATION_PLAN.md "Clean 4-seed measurement".
        # v2 stays available via play_backend="v2" for deep-search work.
        policy = SolverPolicy()
        self.assertIsInstance(policy.play_policy, PlaySearchPolicy)

    def test_v2_play_backend_opt_in(self) -> None:
        policy = SolverPolicy(play_backend="v2")
        self.assertIsInstance(policy.play_policy, SearchV2PlayPolicy)

    def test_legacy_play_backend_still_available(self) -> None:
        policy = SolverPolicy(play_backend="legacy")
        self.assertIsInstance(policy.play_policy, PlaySearchPolicy)

    def test_default_shop_config_can_be_overridden_by_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BALATRO_SOLVER_SHOP_BEAM_WIDTH": "3",
                "BALATRO_SOLVER_SHOP_DEPTH": "1",
                "BALATRO_SOLVER_SHOP_MIN_SEARCH_VALUE": "0",
                "BALATRO_SOLVER_SHOP_REROLL_SAMPLES": "5",
            },
        ):
            policy = SolverPolicy()

        self.assertEqual(policy.shop_config.beam_width, 3)
        self.assertEqual(policy.shop_config.depth, 1)
        self.assertEqual(policy.shop_config.min_search_value, 0.0)
        self.assertEqual(policy.shop_config.reroll_samples, 5)

    def test_opening_buffoon_can_override_first_non_joker_buy(self) -> None:
        buy_planet = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        open_pack = Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            money=10,
            modifiers={
                "shop_cards": ({"label": "Saturn", "set": "PLANET", "cost": {"buy": 3}},),
                "booster_packs": ({"label": "Buffoon Pack", "set": "Booster", "cost": {"buy": 4}},),
            },
            legal_actions=(buy_planet, open_pack),
        )
        fallback = _FixedFallback(Action(ActionType.END_SHOP))
        policy = SolverPolicy(
            fallback=fallback,
            play_policy=fallback,
            fallback_opening_buffoon_over_non_joker_buy=True,
        )

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=buy_planet):
            action = policy.choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)

    def test_negative_end_shop_can_fallback_to_basic_action(self) -> None:
        fallback_action = Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})
        solver_end = Action(ActionType.END_SHOP, metadata={"search_value": -12.0})
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            money=8,
            legal_actions=(fallback_action, solver_end),
        )
        fallback = _FixedFallback(fallback_action)
        policy = SolverPolicy(
            fallback=fallback,
            play_policy=fallback,
            fallback_negative_end_shop=True,
        )

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=solver_end):
            action = policy.choose_action(state)

        self.assertIs(action, fallback_action)

    def test_ante1_single_joker_planet_buy_can_fallback_to_joker(self) -> None:
        fallback_action = Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1})
        solver_planet = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            money=7,
            jokers=(Joker("Trading Card"),),
            modifiers={
                "shop_cards": (
                    {"label": "Earth", "set": "PLANET", "cost": {"buy": 3}},
                    {"label": "Hanging Chad", "set": "JOKER", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(solver_planet, fallback_action),
        )
        fallback = _FixedFallback(fallback_action)
        policy = SolverPolicy(
            fallback=fallback,
            play_policy=fallback,
            fallback_ante1_single_joker_planet_buy=True,
        )

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=solver_planet):
            action = policy.choose_action(state)

        self.assertIs(action, fallback_action)


class SolverPolicyShopMemoryTests(unittest.TestCase):
    def test_shop_memory_is_passed_to_shop_search_and_recorded(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=2,
            blind="Small Blind",
            legal_actions=(Action(ActionType.REROLL), Action(ActionType.END_SHOP)),
        )
        fallback = _FixedFallback(Action(ActionType.END_SHOP))
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
        )
        policy._shop_key = (state.seed, state.ante, state.blind)
        policy._rerolls_in_shop = 2
        policy._packs_opened_in_shop = 1

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=Action(ActionType.REROLL)) as mocked:
            action = policy.choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["shop_context"].rerolls_in_shop, 2)
        self.assertEqual(kwargs["shop_context"].packs_opened_in_shop, 1)
        self.assertEqual(policy._rerolls_in_shop, 3)

    def test_solver_shop_action_is_mirrored_to_fallback_shop_memory(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=2,
            blind="Small Blind",
            legal_actions=(Action(ActionType.OPEN_PACK), Action(ActionType.END_SHOP)),
        )
        fallback = _RecordingFallback()
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
        )
        chosen = Action(ActionType.OPEN_PACK)

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=chosen):
            action = policy.choose_action(state)

        self.assertIs(action, chosen)
        self.assertEqual(fallback.synced_states, [state])
        self.assertEqual(fallback.recorded_actions, [chosen])

    def test_shop_sell_can_fall_back_to_basic_action(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=2,
            blind="Small Blind",
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        fallback_action = Action(ActionType.END_SHOP)
        fallback = _FixedFallback(fallback_action)
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
            allow_shop_sells=False,
        )

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=state.legal_actions[0]):
            action = policy.choose_action(state)

        self.assertIs(action, fallback_action)
        self.assertEqual(fallback.calls, 1)

    def test_negative_shop_sell_can_fall_back_to_basic_action(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=2,
            blind="Small Blind",
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        fallback_action = Action(ActionType.END_SHOP)
        fallback = _FixedFallback(fallback_action)
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
            fallback_negative_shop_sells=True,
        )
        sell = Action(
            ActionType.SELL,
            target_id="joker",
            amount=0,
            metadata={"kind": "joker", "index": 0, "search_value": -0.1},
        )

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=sell):
            action = policy.choose_action(state)

        self.assertIs(action, fallback_action)
        self.assertEqual(fallback.calls, 1)

    def test_negative_shop_sell_can_fund_planned_buy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=1,
            blind="Small Blind",
            money=3,
            jokers=(Joker("Weak Joker", sell_value=2),),
            modifiers={
                "shop_cards": (
                    {"key": "j_devious", "name": "Devious Joker", "set": "JOKER", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        fallback = _FixedFallback(Action(ActionType.END_SHOP))
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
            fallback_negative_shop_sells=True,
        )
        sell = Action(
            ActionType.SELL,
            target_id="joker",
            amount=0,
            metadata={
                "kind": "joker",
                "index": 0,
                "search_value": -0.1,
                "search_path": (
                    {"type": "sell", "kind": "joker", "index": 0},
                    {"type": "buy", "kind": "card", "index": 0},
                ),
            },
        )

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=sell):
            action = policy.choose_action(state)

        self.assertIs(action, sell)
        self.assertEqual(fallback.calls, 0)

    def test_positive_shop_sell_does_not_fall_back_to_basic_action(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=2,
            blind="Small Blind",
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        fallback = _FixedFallback(Action(ActionType.END_SHOP))
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
            fallback_negative_shop_sells=True,
        )
        sell = Action(
            ActionType.SELL,
            target_id="joker",
            amount=0,
            metadata={"kind": "joker", "index": 0, "search_value": 0.1},
        )

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=sell):
            action = policy.choose_action(state)

        self.assertIs(action, sell)
        self.assertEqual(fallback.calls, 0)

    def test_unfunded_open_slot_shop_sell_can_fall_back_to_basic_action(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=2,
            blind="Small Blind",
            money=10,
            jokers=(Joker("Weak Joker", sell_value=2),),
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        fallback_action = Action(ActionType.END_SHOP)
        fallback = _FixedFallback(fallback_action)
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
            fallback_unfunded_open_slot_sells=True,
        )
        sell = Action(
            ActionType.SELL,
            target_id="joker",
            amount=0,
            metadata={
                "kind": "joker",
                "index": 0,
                "search_value": 10.0,
                "search_path": (
                    {"type": "sell", "kind": "joker", "index": 0},
                    {"type": "reroll", "kind": "", "index": None},
                ),
            },
        )

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=sell):
            action = policy.choose_action(state)

        self.assertIs(action, fallback_action)
        self.assertEqual(fallback.calls, 1)

    def test_negative_shop_action_can_fall_back_to_active_basic_action(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=2,
            blind="Small Blind",
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        fallback_action = state.legal_actions[0]
        fallback = _FixedFallback(fallback_action)
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
            fallback_negative_shop_actions=True,
        )
        end_shop = Action(ActionType.END_SHOP, metadata={"search_value": -1.0})

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=end_shop):
            action = policy.choose_action(state)

        self.assertIs(action, fallback_action)
        self.assertEqual(fallback.calls, 1)

    def test_negative_shop_action_keeps_solver_action_when_basic_would_end_shop(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=2,
            blind="Small Blind",
            legal_actions=(Action(ActionType.END_SHOP),),
        )
        fallback = _FixedFallback(Action(ActionType.END_SHOP))
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
            fallback_negative_shop_actions=True,
        )
        end_shop = Action(ActionType.END_SHOP, metadata={"search_value": -1.0})

        with patch("balatro_ai.solver.policy.best_shop_action", return_value=end_shop):
            action = policy.choose_action(state)

        self.assertIs(action, end_shop)
        self.assertEqual(fallback.calls, 1)

    def test_can_prefer_fallback_info_first_pack_before_shop_search(self) -> None:
        buy_action = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        info_action = Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=1,
            blind="Small Blind",
            legal_actions=(buy_action, info_action, Action(ActionType.END_SHOP)),
        )
        fallback = _RecordingFallback()
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
            prefer_fallback_info_first_shop=True,
        )

        pressure = SimpleNamespace(ratio=0.42)

        def value_for_action(_state, action, *_args, **_kwargs):
            if action is buy_action:
                return 50.0
            if action is info_action:
                return 40.0
            return 0.0

        with (
            patch("balatro_ai.bots.basic_strategy.shop_pressure._shop_pressure", return_value=pressure),
            patch("balatro_ai.bots.basic_strategy.build_profile._build_profile", return_value=object()),
            patch("balatro_ai.bots.basic_strategy.run_plan._run_plan", return_value=object()),
            patch("balatro_ai.bots.basic_strategy.shop_planner._calibrated_shop_action_value", side_effect=value_for_action),
            patch("balatro_ai.bots.basic_strategy.shop_values._shop_buy_threshold", return_value=10.0),
            patch(
                "balatro_ai.bots.basic_strategy.shop_flow._shop_information_first_action",
                return_value=(info_action, 40.0, {"name": "Joker"}),
            ),
            patch("balatro_ai.solver.policy.best_shop_action") as mocked,
        ):
            action = policy.choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertIn("shop_sequence_info_first", str(action.metadata.get("reason", "")))
        self.assertEqual(policy._packs_opened_in_shop, 1)
        self.assertEqual(fallback.calls, 0)
        self.assertEqual([recorded.action_type for recorded in fallback.recorded_actions], [ActionType.OPEN_PACK])
        mocked.assert_not_called()

    def test_can_prefer_opening_buffoon_pack_before_shop_search(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=1,
            blind="Small Blind",
            money=10,
            modifiers={
                "shop_cards": (
                    {"key": "c_saturn", "name": "Saturn", "set": "PLANET", "cost": {"buy": 3}},
                ),
                "booster_packs": (
                    {"key": "p_buffoon_normal_1", "name": "Buffoon Pack", "set": "BOOSTER", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        fallback = _RecordingFallback()
        policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            shop_config=ShopSearchConfig(depth=1, beam_width=1),
            prefer_opening_buffoon_pack=True,
        )

        with patch("balatro_ai.solver.policy.best_shop_action") as mocked:
            action = policy.choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertIn("shop_opening_buffoon_first", str(action.metadata.get("reason", "")))
        self.assertEqual(policy._packs_opened_in_shop, 1)
        self.assertEqual(fallback.calls, 0)
        self.assertEqual([recorded.action_type for recorded in fallback.recorded_actions], [ActionType.OPEN_PACK])
        mocked.assert_not_called()

    def test_shop_memory_tracks_bought_joker_and_resets_outside_shop(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            seed=123,
            ante=2,
            blind="Small Blind",
            jokers=(
                Joker("A"),
                Joker("B"),
                Joker("C"),
                Joker("D"),
            ),
            modifiers={
                "shop_cards": (
                    {"key": "j_jolly", "name": "Jolly Joker", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
        )
        fallback = _FixedFallback(Action(ActionType.END_SHOP))
        policy = SolverPolicy(play_policy=fallback, fallback=fallback)
        policy._sync_shop_memory(state)
        policy._record_shop_action(
            state,
            Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
        )

        self.assertEqual(policy._protected_shop_jokers, ("Jolly Joker",))
        self.assertTrue(policy._filled_last_joker_slot_in_shop)

        policy._record_shop_action(state, Action(ActionType.OPEN_PACK))
        policy._sync_shop_memory(GameState(phase=GamePhase.BOOSTER_OPENED))
        self.assertEqual(policy._packs_opened_in_shop, 1)

        policy._sync_shop_memory(GameState(phase=GamePhase.SELECTING_HAND))
        self.assertEqual(policy._protected_shop_jokers, ())
        self.assertEqual(policy._packs_opened_in_shop, 0)
        self.assertFalse(policy._filled_last_joker_slot_in_shop)


class SolverPolicyTrajectoryTests(unittest.TestCase):
    """M5 acceptance: full-run trajectory via the composed policy."""

    def test_full_trajectory_runs_without_crashing(self) -> None:
        # Default config; same throughput envelope as M4's trajectory test,
        # plus shop search at default ShopSearchConfig.
        policy = SolverPolicy()
        traj = generate_trajectory(
            "AAAAAAA", policy.choose_action, max_steps=2000
        )
        self.assertIn(
            traj.terminated_reason,
            ("RUN_OVER", "STEP_LIMIT"),
            f"Unexpected termination: {traj.terminated_reason}",
        )
        self.assertGreater(traj.n_steps, 10)
        self.assertGreaterEqual(traj.final_ante, 1)


if __name__ == "__main__":
    unittest.main()
