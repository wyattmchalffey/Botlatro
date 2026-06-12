from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, with_derived_legal_actions
from balatro_ai.bots import neural_shop
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.bots.registry import (
    PortfolioBot,
    RankerGuidedShopBot,
    SolverPolicyBot,
    SolverShopBasicPlayBot,
    SolverShopBasicPlayNoSellBot,
    ValueGuidedBot,
    create_bot,
)
from balatro_ai.bots.search_bot import SearchBot
from balatro_ai.bots.search_bot_v2 import SearchBotV2
from balatro_ai.ml.shop_candidate_dataset import action_key as shop_action_key
from balatro_ai.search.consumable_search import ConsumableSearchConfig, best_consumable_action, consumable_action_value
from balatro_ai.search.discard_search import DiscardSearchConfig
from balatro_ai.search.pack_search import PackSearchConfig
from balatro_ai.sim.local_runner import LocalBalatroSimulator


class SearchBotTests(unittest.TestCase):
    def test_registry_creates_search_bot_v0(self) -> None:
        bot = create_bot("search_bot_v0", seed=3)

        self.assertIsInstance(bot, SearchBot)
        self.assertEqual(bot.name, "search_bot_v0")

    def test_registry_creates_search_bot_v1_with_shop_search(self) -> None:
        bot = create_bot("search_bot_v1", seed=3)

        self.assertIsInstance(bot, SearchBot)
        self.assertEqual(bot.name, "search_bot_v1")
        self.assertTrue(bot.enable_shop_search)

    def test_registry_creates_search_bot_v2_experiment_lane(self) -> None:
        bot = create_bot("search_bot_v2", seed=3)

        self.assertIsInstance(bot, SearchBotV2)
        self.assertNotIsInstance(bot, SearchBot)
        self.assertEqual(bot.name, "search_bot_v2")
        self.assertTrue(bot.enable_shop_search)
        self.assertEqual(bot.hand_config.beam_depth, 3)
        self.assertEqual(bot.hand_config.beam_width, 2)
        self.assertEqual(bot.hand_config.draw_samples, 2)

    def test_registry_creates_solver_policy_bot(self) -> None:
        bot = create_bot("solver_policy_bot", seed=3)

        self.assertIsInstance(bot, SolverPolicyBot)
        self.assertEqual(bot.name, "solver_policy_bot")

    def test_registry_creates_solver_shop_basic_play_bot(self) -> None:
        bot = create_bot("solver_shop_basic_play_bot", seed=3)

        self.assertIsInstance(bot, SolverShopBasicPlayBot)
        self.assertEqual(bot.name, "solver_shop_basic_play_bot")

    def test_registry_creates_value_guided_aliases(self) -> None:
        basic = create_bot("basic_strategy_value_guided_bot", seed=3)
        solver = create_bot("solver_shop_basic_play_value_guided_bot", seed=3)

        self.assertIsInstance(basic, ValueGuidedBot)
        self.assertEqual(basic.name, "basic_strategy_value_guided_bot")
        self.assertIsInstance(basic.wrapped, BasicStrategyBot)
        self.assertIsInstance(solver, ValueGuidedBot)
        self.assertEqual(solver.name, "solver_shop_basic_play_value_guided_bot")
        self.assertIsInstance(solver.wrapped, SolverShopBasicPlayBot)

    def test_registry_creates_shop_ranker_aliases(self) -> None:
        basic = create_bot("basic_strategy_shop_ranker_bot", seed=3)
        solver = create_bot("solver_shop_basic_play_shop_ranker_bot", seed=3)

        self.assertIsInstance(basic, RankerGuidedShopBot)
        self.assertEqual(basic.name, "basic_strategy_shop_ranker_bot")
        self.assertIsInstance(basic.wrapped, BasicStrategyBot)
        self.assertIsInstance(solver, RankerGuidedShopBot)
        self.assertEqual(solver.name, "solver_shop_basic_play_shop_ranker_bot")
        self.assertIsInstance(solver.wrapped, SolverShopBasicPlayBot)

    def test_value_guided_bot_scopes_checkpoint_env(self) -> None:
        action = Action(ActionType.NO_OP)
        wrapped = _EnvAssertingChoose(action, expected_ckpt="model.pt")
        bot = ValueGuidedBot(wrapped=wrapped, name="guided")
        state = GameState(phase=GamePhase.SHOP)

        with patch.dict(
            os.environ,
            {
                "BALATRO_VALUE_GUIDED_CKPT": "model.pt",
                "BALATRO_VALUE_GUIDED_HEAD": "win",
                "BALATRO_VALUE_GUIDED_SCALE": "17",
            },
            clear=True,
        ):
            chosen = bot.choose_action(state)
            self.assertNotIn("BALATRO_VALUE_MODEL_CKPT", os.environ)
            self.assertNotIn("BALATRO_VALUE_MODEL_HEAD", os.environ)
            self.assertNotIn("BALATRO_VALUE_SCALE", os.environ)

        self.assertIs(chosen, action)

    def test_shop_ranker_bot_falls_back_without_checkpoint(self) -> None:
        fallback = Action(ActionType.END_SHOP)
        bot = RankerGuidedShopBot(wrapped=_FixedChoose(fallback))
        state = GameState(phase=GamePhase.SHOP)

        with patch.dict(os.environ, {}, clear=True):
            chosen = bot.choose_action(state)

        self.assertIs(chosen, fallback)

    def test_shop_ranker_bot_uses_ranker_when_checkpoint_is_set(self) -> None:
        ranked = Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})
        bot = RankerGuidedShopBot(wrapped=_RaisingChoose())
        state = GameState(phase=GamePhase.SHOP)

        with (
            patch.dict(os.environ, {"BALATRO_SHOP_RANKER_CKPT": "ranker.pt"}, clear=True),
            patch("balatro_ai.bots.neural_shop._ranker_shop_action", return_value=ranked) as ranker,
        ):
            chosen = bot.choose_action(state)

        self.assertIs(chosen, ranked)
        ranker.assert_called_once()

    def test_shop_ranker_bot_falls_back_when_ranker_declines(self) -> None:
        fallback = Action(ActionType.END_SHOP)
        bot = RankerGuidedShopBot(wrapped=_FixedChoose(fallback))
        state = GameState(phase=GamePhase.SHOP)

        with (
            patch.dict(os.environ, {"BALATRO_SHOP_RANKER_CKPT": "ranker.pt"}, clear=True),
            patch("balatro_ai.bots.neural_shop._ranker_shop_action", return_value=None),
        ):
            chosen = bot.choose_action(state)

        self.assertIs(chosen, fallback)

    def test_shop_ranker_bot_respects_ante_gate(self) -> None:
        fallback = Action(ActionType.END_SHOP)
        bot = RankerGuidedShopBot(wrapped=_FixedChoose(fallback))
        state = GameState(phase=GamePhase.SHOP, ante=3)

        with (
            patch.dict(
                os.environ,
                {
                    "BALATRO_SHOP_RANKER_CKPT": "ranker.pt",
                    "BALATRO_SHOP_RANKER_MIN_ANTE": "2",
                    "BALATRO_SHOP_RANKER_MAX_ANTE": "2",
                },
                clear=True,
            ),
            patch("balatro_ai.bots.neural_shop._ranker_shop_action", return_value=Action(ActionType.REROLL)) as ranker,
        ):
            chosen = bot.choose_action(state)

        self.assertIs(chosen, fallback)
        ranker.assert_not_called()

    def test_shop_ranker_action_type_env_parser(self) -> None:
        with patch.dict(os.environ, {"BALATRO_SHOP_RANKER_ACTION_TYPES": "buy,OPEN_PACK,bad"}, clear=True):
            parsed = neural_shop._env_action_types("BALATRO_SHOP_RANKER_ACTION_TYPES")

        self.assertEqual(parsed, frozenset({ActionType.BUY, ActionType.OPEN_PACK}))

    def test_shop_ranker_bot_can_cap_ranker_actions_per_shop(self) -> None:
        fallback = Action(ActionType.END_SHOP)
        ranked = Action(ActionType.OPEN_PACK, target_id="pack", amount=0)
        bot = RankerGuidedShopBot(wrapped=_FixedChoose(fallback))
        state = GameState(phase=GamePhase.SHOP, ante=2)

        with (
            patch.dict(
                os.environ,
                {
                    "BALATRO_SHOP_RANKER_CKPT": "ranker.pt",
                    "BALATRO_SHOP_RANKER_MAX_ACTIONS_PER_SHOP": "1",
                },
                clear=True,
            ),
            patch("balatro_ai.bots.neural_shop._ranker_shop_action", return_value=ranked) as ranker,
        ):
            first = bot.choose_action(state)
            second = bot.choose_action(state)

        self.assertIs(first, ranked)
        self.assertIs(second, fallback)
        ranker.assert_called_once()

    def test_shop_ranker_action_cap_survives_booster_opened_phase(self) -> None:
        fallback = Action(ActionType.END_SHOP)
        ranked = Action(ActionType.OPEN_PACK, target_id="pack", amount=0)
        bot = RankerGuidedShopBot(wrapped=_FixedChoose(fallback))
        shop_state = GameState(phase=GamePhase.SHOP, ante=2)
        booster_state = GameState(phase=GamePhase.BOOSTER_OPENED, ante=2)

        with (
            patch.dict(
                os.environ,
                {
                    "BALATRO_SHOP_RANKER_CKPT": "ranker.pt",
                    "BALATRO_SHOP_RANKER_MAX_ACTIONS_PER_SHOP": "1",
                },
                clear=True,
            ),
            patch("balatro_ai.bots.neural_shop._ranker_shop_action", return_value=ranked) as ranker,
        ):
            first = bot.choose_action(shop_state)
            booster = bot.choose_action(booster_state)
            second_shop = bot.choose_action(shop_state)

        self.assertIs(first, ranked)
        self.assertIs(booster, fallback)
        self.assertIs(second_shop, fallback)
        ranker.assert_called_once()

    def test_shop_ranker_bot_can_cap_ranker_actions_per_run(self) -> None:
        fallback = Action(ActionType.END_SHOP)
        ranked = Action(ActionType.OPEN_PACK, target_id="pack", amount=0)
        bot = RankerGuidedShopBot(wrapped=_FixedChoose(fallback))
        shop_state = GameState(phase=GamePhase.SHOP, ante=2)
        blind_state = GameState(phase=GamePhase.BLIND_SELECT, ante=2)

        with (
            patch.dict(
                os.environ,
                {
                    "BALATRO_SHOP_RANKER_CKPT": "ranker.pt",
                    "BALATRO_SHOP_RANKER_MAX_ACTIONS_PER_RUN": "1",
                },
                clear=True,
            ),
            patch("balatro_ai.bots.neural_shop._ranker_shop_action", return_value=ranked) as ranker,
        ):
            first = bot.choose_action(shop_state)
            between_shops = bot.choose_action(blind_state)
            second_shop = bot.choose_action(shop_state)

        self.assertIs(first, ranked)
        self.assertIs(between_shops, fallback)
        self.assertIs(second_shop, fallback)
        ranker.assert_called_once()

    def test_shop_ranker_baseline_compare_probes_without_mutating_wrapped_bot(self) -> None:
        baseline = Action(ActionType.END_SHOP)
        ranked = Action(ActionType.BUY, target_id="card", amount=0)
        wrapped = _CountingChoose(baseline)
        bot = RankerGuidedShopBot(wrapped=wrapped)
        state = GameState(phase=GamePhase.SHOP, ante=2)

        def ranker_action(_state, *, ckpt, baseline_action_getter=None):
            del ckpt
            self.assertIsNotNone(baseline_action_getter)
            assert baseline_action_getter is not None
            self.assertEqual(baseline_action_getter().stable_key, baseline.stable_key)
            return ranked

        with (
            patch.dict(
                os.environ,
                {
                    "BALATRO_SHOP_RANKER_CKPT": "ranker.pt",
                    "BALATRO_SHOP_RANKER_COMPARE_BASELINE": "1",
                },
                clear=True,
            ),
            patch("balatro_ai.bots.neural_shop._ranker_shop_action", side_effect=ranker_action) as ranker,
        ):
            chosen = bot.choose_action(state)

        self.assertIs(chosen, ranked)
        self.assertEqual(wrapped.calls, 0)
        ranker.assert_called_once()

    def test_shop_ranker_can_score_filtered_baseline_without_returning_it(self) -> None:
        buy = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        end_shop = Action(ActionType.END_SHOP)
        baseline = Action(ActionType.REROLL)
        state = GameState(phase=GamePhase.SHOP, ante=2)

        with (
            patch.dict(
                os.environ,
                {
                    "BALATRO_SHOP_RANKER_ACTION_TYPES": "buy,end_shop",
                    "BALATRO_SHOP_RANKER_MIN_BASELINE_MARGIN": "0.5",
                },
                clear=True,
            ),
            patch("balatro_ai.ml.shop_candidate_dataset.candidate_shop_actions", return_value=(buy, end_shop)),
            patch("balatro_ai.bots.neural_shop._load_ranker_model", return_value=_FixedLogitModel(1.0, 0.0, 0.2)),
        ):
            chosen = neural_shop._ranker_shop_action(
                state,
                ckpt=Path("ranker.pt"),
                baseline_action_getter=lambda: baseline,
            )

        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen.action_type, ActionType.BUY)
        self.assertEqual(chosen.metadata["shop_ranker"]["baseline_key"], shop_action_key(baseline))
        self.assertAlmostEqual(chosen.metadata["shop_ranker"]["baseline_margin"], 0.8)

    def test_shop_ranker_can_filter_override_action_kinds(self) -> None:
        voucher = Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0})
        buy = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        pack = Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})
        end_shop = Action(ActionType.END_SHOP)
        state = GameState(phase=GamePhase.SHOP, ante=2)

        with (
            patch.dict(
                os.environ,
                {
                    "BALATRO_SHOP_RANKER_ACTION_TYPES": "buy,open_pack,end_shop",
                    "BALATRO_SHOP_RANKER_ACTION_KINDS": "card,pack",
                },
                clear=True,
            ),
            patch(
                "balatro_ai.ml.shop_candidate_dataset.candidate_shop_actions",
                return_value=(voucher, buy, pack, end_shop),
            ),
            patch("balatro_ai.bots.neural_shop._load_ranker_model", return_value=_FixedLogitModel(3.0, 2.0, 1.0)),
        ):
            chosen = neural_shop._ranker_shop_action(state, ckpt=Path("ranker.pt"))

        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen.action_type, ActionType.BUY)
        self.assertEqual(chosen.metadata["kind"], "card")

    def test_registry_creates_portfolio_basic_solver_bot(self) -> None:
        bot = create_bot("portfolio_basic_solver_bot", seed=3)

        self.assertIsInstance(bot, PortfolioBot)
        self.assertEqual(bot.name, "portfolio_basic_solver_bot")

    def test_portfolio_bot_selects_best_seed_rollout(self) -> None:
        action = Action(ActionType.NO_OP)
        bot = PortfolioBot(candidate_names=("weak_bot", "strong_bot"))
        state = GameState(phase=GamePhase.BLIND_SELECT)

        def result(name: str, _seed: str) -> dict:
            if name == "strong_bot":
                return {"won": True, "ante": 8, "loss_frac": None, "score": 0}
            return {"won": False, "ante": 7, "loss_frac": 0.99, "score": 10}

        with (
            patch.dict(os.environ, {"BALATRO_RUN_SEED": "0000001"}, clear=True),
            patch("balatro_ai.bots.registry._portfolio_candidate_result", side_effect=result),
            patch("balatro_ai.bots.registry.create_bot", return_value=_FixedChoose(action)) as factory,
        ):
            chosen = bot.choose_action(state)

        self.assertIs(chosen, action)
        factory.assert_called_once_with("strong_bot", seed=None)

    def test_portfolio_bot_reuses_matching_action_plan(self) -> None:
        action = Action(ActionType.NO_OP)
        bot = PortfolioBot(candidate_names=("strong_bot",))
        state = GameState(phase=GamePhase.BLIND_SELECT, legal_actions=(action,))

        with (
            patch.dict(os.environ, {"BALATRO_RUN_SEED": "0000001"}, clear=True),
            patch(
                "balatro_ai.bots.registry._portfolio_candidate_result",
                return_value={"won": True, "ante": 8, "loss_frac": None, "score": 0, "action_plan": (action.stable_key,)},
            ),
            patch("balatro_ai.bots.registry.create_bot") as factory,
        ):
            chosen = bot.choose_action(state)

        self.assertIs(chosen, action)
        factory.assert_not_called()

    def test_portfolio_bot_falls_back_when_action_plan_diverges(self) -> None:
        fallback_action = Action(ActionType.NO_OP)
        bot = PortfolioBot(candidate_names=("strong_bot",))
        state = GameState(phase=GamePhase.BLIND_SELECT, legal_actions=(Action(ActionType.SELECT_BLIND),))

        with (
            patch.dict(os.environ, {"BALATRO_RUN_SEED": "0000001"}, clear=True),
            patch(
                "balatro_ai.bots.registry._portfolio_candidate_result",
                return_value={"won": True, "ante": 8, "loss_frac": None, "score": 0, "action_plan": (fallback_action.stable_key,)},
            ),
            patch("balatro_ai.bots.registry.create_bot", return_value=_FixedChoose(fallback_action)) as factory,
        ):
            chosen = bot.choose_action(state)

        self.assertIs(chosen, fallback_action)
        factory.assert_called_once_with("strong_bot", seed=None)

    def test_registry_creates_solver_shop_basic_play_ablation_bots(self) -> None:
        info_bot = create_bot("solver_shop_basic_play_info_bot", seed=3)
        neg_sell_bot = create_bot("solver_shop_basic_play_neg_sell_bot", seed=3)

        self.assertIsInstance(info_bot, SolverShopBasicPlayBot)
        self.assertEqual(info_bot.name, "solver_shop_basic_play_info_bot")
        self.assertTrue(info_bot.prefer_fallback_info_first_shop)
        self.assertFalse(info_bot.fallback_negative_shop_sells)
        self.assertIsInstance(neg_sell_bot, SolverShopBasicPlayBot)
        self.assertEqual(neg_sell_bot.name, "solver_shop_basic_play_neg_sell_bot")
        self.assertFalse(neg_sell_bot.prefer_fallback_info_first_shop)
        self.assertTrue(neg_sell_bot.fallback_negative_shop_sells)

    def test_registry_creates_solver_shop_basic_play_buffoon_ablation_bot(self) -> None:
        bot = create_bot("solver_shop_basic_play_buffoon_bot", seed=3)

        self.assertIsInstance(bot, SolverShopBasicPlayBot)
        self.assertEqual(bot.name, "solver_shop_basic_play_buffoon_bot")
        self.assertTrue(bot.prefer_opening_buffoon_pack)

    def test_registry_creates_solver_shop_basic_play_buffoon_nonjoker_ablation_bot(self) -> None:
        bot = create_bot("solver_shop_basic_play_buffoon_nonjoker_bot", seed=3)

        self.assertIsInstance(bot, SolverShopBasicPlayBot)
        self.assertEqual(bot.name, "solver_shop_basic_play_buffoon_nonjoker_bot")
        self.assertTrue(bot.fallback_opening_buffoon_over_non_joker_buy)

    def test_registry_creates_solver_shop_basic_play_sell_guard_ablation_bot(self) -> None:
        bot = create_bot("solver_shop_basic_play_sell_guard_bot", seed=3)

        self.assertIsInstance(bot, SolverShopBasicPlayBot)
        self.assertEqual(bot.name, "solver_shop_basic_play_sell_guard_bot")
        self.assertTrue(bot.fallback_unfunded_open_slot_sells)

    def test_registry_creates_solver_shop_basic_play_staged_ablation_bots(self) -> None:
        ante1_bot = create_bot("solver_shop_basic_play_basic_ante1_bot", seed=3)
        ante2_bot = create_bot("solver_shop_basic_play_basic_ante2_bot", seed=3)

        self.assertIsInstance(ante1_bot, SolverShopBasicPlayBot)
        self.assertEqual(ante1_bot.name, "solver_shop_basic_play_basic_ante1_bot")
        self.assertEqual(ante1_bot.fallback_shop_through_ante, 1)
        self.assertIsInstance(ante2_bot, SolverShopBasicPlayBot)
        self.assertEqual(ante2_bot.name, "solver_shop_basic_play_basic_ante2_bot")
        self.assertEqual(ante2_bot.fallback_shop_through_ante, 2)

    def test_registry_creates_solver_shop_basic_play_negative_action_ablation_bot(self) -> None:
        bot = create_bot("solver_shop_basic_play_neg_action_bot", seed=3)

        self.assertIsInstance(bot, SolverShopBasicPlayBot)
        self.assertEqual(bot.name, "solver_shop_basic_play_neg_action_bot")
        self.assertTrue(bot.fallback_negative_shop_actions)

    def test_registry_creates_solver_shop_basic_play_negative_end_ablation_bot(self) -> None:
        bot = create_bot("solver_shop_basic_play_neg_end_bot", seed=3)

        self.assertIsInstance(bot, SolverShopBasicPlayBot)
        self.assertEqual(bot.name, "solver_shop_basic_play_neg_end_bot")
        self.assertTrue(bot.fallback_negative_end_shop)

    def test_registry_creates_solver_shop_basic_play_ante1_planet_ablation_bot(self) -> None:
        bot = create_bot("solver_shop_basic_play_ante1_planet_bot", seed=3)

        self.assertIsInstance(bot, SolverShopBasicPlayBot)
        self.assertEqual(bot.name, "solver_shop_basic_play_ante1_planet_bot")
        self.assertTrue(bot.fallback_ante1_single_joker_planet_buy)

    def test_staged_solver_shop_basic_play_uses_fallback_for_early_shop(self) -> None:
        bot = create_bot("solver_shop_basic_play_basic_ante1_bot", seed=3)
        assert isinstance(bot, SolverShopBasicPlayBot)
        fallback_action = Action(ActionType.END_SHOP)
        solver_action = Action(ActionType.REROLL)
        bot._fallback = _FixedChoose(fallback_action)
        bot._policy = _FixedChoose(solver_action)
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            legal_actions=(fallback_action, solver_action),
        )

        action = bot.choose_action(state)

        self.assertIs(action, fallback_action)

    def test_staged_solver_shop_basic_play_uses_solver_after_early_shop(self) -> None:
        bot = create_bot("solver_shop_basic_play_basic_ante1_bot", seed=3)
        assert isinstance(bot, SolverShopBasicPlayBot)
        fallback_action = Action(ActionType.END_SHOP)
        solver_action = Action(ActionType.REROLL)
        bot._fallback = _FixedChoose(fallback_action)
        bot._policy = _FixedChoose(solver_action)
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            legal_actions=(fallback_action, solver_action),
        )

        action = bot.choose_action(state)

        self.assertIs(action, solver_action)

    def test_registry_creates_solver_shop_basic_play_no_sell_bot(self) -> None:
        bot = create_bot("solver_shop_basic_play_no_sell_bot", seed=3)

        self.assertIsInstance(bot, SolverShopBasicPlayNoSellBot)
        self.assertEqual(bot.name, "solver_shop_basic_play_no_sell_bot")

    def test_registry_creates_basic_strategy_legacy_without_calibrated_planner(self) -> None:
        bot = create_bot("basic_strategy_legacy", seed=3)

        self.assertIsInstance(bot, BasicStrategyBot)
        self.assertEqual(bot.name, "basic_strategy_legacy")
        self.assertIsNotNone(bot.config)
        assert bot.config is not None
        self.assertFalse(bot.config.calibrated_shop_planner_enabled)

    def test_registry_does_not_create_trace_variant_for_search_bot_v1(self) -> None:
        with self.assertRaises(ValueError):
            create_bot("search_bot_v1_trace", seed=3)

    def test_search_bot_uses_discard_search_when_discards_are_legal(self) -> None:
        bot = SearchBot(seed=1, discard_config=DiscardSearchConfig(draw_samples=2, leaf_samples=1, seed=1))
        discard_ace = Action(ActionType.DISCARD, card_indices=(0,))
        discard_king = Action(ActionType.DISCARD, card_indices=(1,))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            deck_size=2,
            hand=(Card("A", "S"), Card("K", "H"), Card("2", "D")),
            known_deck=(Card("A", "H"), Card("3", "C")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0,)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)),
                discard_ace,
                discard_king,
            ),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (1,))
        self.assertEqual(action.metadata["search"], "hand_expectimax")

    def test_search_bot_plays_when_current_hand_clears_in_one(self) -> None:
        bot = SearchBot(seed=1, discard_config=DiscardSearchConfig(draw_samples=2, leaf_samples=1, seed=1))
        play_pair = Action(ActionType.PLAY_HAND, card_indices=(0, 1))
        discard = Action(ActionType.DISCARD, card_indices=(2,))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=1,
            blind="Small Blind",
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=10,
            hand=(Card("A", "S"), Card("A", "H"), Card("2", "D")),
            known_deck=(Card("K", "C"),),
            legal_actions=(play_pair, discard),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))

    def test_search_bot_uses_integrated_hand_search_for_first_blind_hunt(self) -> None:
        bot = SearchBot(seed=1, discard_config=DiscardSearchConfig(draw_samples=2, leaf_samples=1, seed=1))
        state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.SELECTING_HAND,
                ante=1,
                blind="Small Blind",
                required_score=300,
                current_score=0,
                hands_remaining=4,
                discards_remaining=3,
                deck_size=45,
                hand=(
                    Card("9", "H"),
                    Card("7", "D"),
                    Card("6", "H"),
                    Card("6", "D"),
                    Card("5", "C"),
                    Card("4", "D"),
                    Card("3", "H"),
                    Card("3", "C"),
                ),
            )
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.metadata["search"], "hand_expectimax")

    def test_search_bot_does_not_repeat_opening_hunt_after_first_discard(self) -> None:
        bot = create_bot("search_bot_v1", seed=1758025119)
        simulator = LocalBalatroSimulator(seed=1758025119)
        state = simulator.reset()
        state = simulator.step(bot.choose_action(state))

        first_action = bot.choose_action(state)
        self.assertEqual(first_action.action_type, ActionType.DISCARD)
        # Indices updated when _sort_hand_cards switched to Balatro's
        # get_nominal ordering: rank "10" cards now sort between J and 9
        # (previously they fell to the end of the hand), shifting positions.
        self.assertEqual(first_action.card_indices, (0, 1, 5))
        state = simulator.step(first_action)
        second_action = bot.choose_action(state)

        self.assertEqual(second_action.action_type, ActionType.PLAY_HAND)

    def test_search_bot_preserves_safe_joker_setup_before_discard_search(self) -> None:
        bot = SearchBot(seed=1, discard_config=DiscardSearchConfig(draw_samples=2, leaf_samples=1, seed=1))
        play_six = Action(ActionType.PLAY_HAND, card_indices=(0,))
        play_pair = Action(ActionType.PLAY_HAND, card_indices=(1, 2))
        discard = Action(ActionType.DISCARD, card_indices=(3,))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=1,
            blind="Small Blind",
            required_score=100,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=10,
            hand=(Card("6", "S"), Card("A", "S"), Card("A", "H"), Card("2", "D")),
            jokers=(Joker("Sixth Sense"),),
            legal_actions=(play_six, play_pair, discard),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0,))
        self.assertIn("joker_setup", action.metadata["reason"])

    def test_search_bot_uses_pack_search_for_deterministic_pack_choices(self) -> None:
        bot = SearchBot(seed=1, pack_config=PackSearchConfig(leaf_samples=1, seed=1))
        choose_saturn = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=1,
            hand=(Card("2", "S"), Card("3", "H"), Card("4", "D"), Card("5", "C"), Card("6", "S")),
            hand_levels={"Straight": 1},
            pack=("Saturn",),
            modifiers={"pack_cards": ({"label": "Saturn", "set": "PLANET"},)},
            legal_actions=(skip, choose_saturn),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")
        self.assertEqual(action.metadata["search"], "pack_value")

    def test_search_bot_uses_hermit_before_shop_spending(self) -> None:
        bot = SearchBot(seed=1, consumable_config=ConsumableSearchConfig(leaf_samples=1, seed=1))
        use_hermit = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            money=10,
            consumables=("The Hermit",),
            legal_actions=(use_hermit, Action(ActionType.END_SHOP)),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.USE_CONSUMABLE)
        self.assertEqual(action.amount, 0)
        self.assertEqual(action.metadata["search"], "consumable_use")

    def test_hermit_has_premium_over_equal_temperance_payout(self) -> None:
        use_consumable = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        hermit_state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            money=10,
            consumables=("The Hermit",),
        )
        temperance_state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            money=10,
            jokers=(Joker("Sell Value", sell_value=10),),
            consumables=("Temperance",),
        )
        config = ConsumableSearchConfig(leaf_samples=1, seed=1, min_shop_delta=0.0)

        self.assertGreater(
            consumable_action_value(hermit_state, use_consumable, config=config, value_fn=lambda state: float(state.money)),
            consumable_action_value(temperance_state, use_consumable, config=config, value_fn=lambda state: float(state.money)),
        )

    def test_search_bot_v1_routes_shop_consumables_through_shop_beam(self) -> None:
        bot = SearchBot(seed=1, enable_shop_search=True, name="search_bot_v1")
        use_temperance = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            money=10,
            consumables=("Temperance",),
            legal_actions=(use_temperance, Action(ActionType.END_SHOP)),
        )
        shop_action = Action(ActionType.END_SHOP, metadata={"search": "shop_beam"})

        with patch("balatro_ai.bots.search_bot.best_consumable_action") as consumable_search:
            with patch("balatro_ai.bots.search_bot.best_shop_action", return_value=shop_action) as shop_search:
                action = bot.choose_action(state)

        consumable_search.assert_not_called()
        shop_search.assert_called_once()
        self.assertEqual(action.metadata["search"], "shop_beam")

    def test_search_bot_uses_planet_when_it_turns_current_hand_into_clear(self) -> None:
        bot = SearchBot(seed=1, consumable_config=ConsumableSearchConfig(leaf_samples=1, seed=1))
        use_saturn = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=1,
            discards_remaining=0,
            deck_size=0,
            hand=(Card("2", "S"), Card("3", "H"), Card("4", "D"), Card("5", "C"), Card("6", "S")),
            hand_levels={"Straight": 1},
            consumables=("Saturn",),
            legal_actions=(use_saturn, Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4))),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.USE_CONSUMABLE)
        self.assertEqual(action.metadata["search"], "consumable_use")

    def test_consumable_search_can_value_stochastic_wheel_outcomes(self) -> None:
        use_wheel = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            money=10,
            jokers=(Joker("Joker", sell_value=1),),
            consumables=("The Wheel of Fortune",),
            legal_actions=(use_wheel, Action(ActionType.END_SHOP)),
        )

        def edition_value(candidate: GameState) -> float:
            return sum(100.0 for joker in candidate.jokers if joker.edition)

        value = consumable_action_value(
            state,
            use_wheel,
            config=ConsumableSearchConfig(leaf_samples=1, seed=1, stochastic_samples=32, min_shop_delta=0.0),
            value_fn=edition_value,
        )
        action = best_consumable_action(
            state,
            config=ConsumableSearchConfig(leaf_samples=1, seed=1, stochastic_samples=32, min_shop_delta=0.0),
            value_fn=edition_value,
        )

        self.assertGreater(value, 0.0)
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.USE_CONSUMABLE)

    def test_consumable_search_prefers_max_tarot_targets(self) -> None:
        one_target = Action(
            ActionType.USE_CONSUMABLE,
            card_indices=(0,),
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        two_targets = Action(
            ActionType.USE_CONSUMABLE,
            card_indices=(0, 1),
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=800,
            hand=(Card("2", "S"), Card("K", "H")),
            consumables=("Strength",),
            legal_actions=(one_target, two_targets),
        )

        with patch(
            "balatro_ai.search.consumable_search.consumable_action_value",
            side_effect=lambda _state, action, **_kwargs: 10.0 if len(action.card_indices) == 1 else 1.0,
        ) as value_mock:
            action = best_consumable_action(
                state,
                config=ConsumableSearchConfig(min_blind_delta=-999.0),
                value_fn=lambda _state: 0.0,
            )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.USE_CONSUMABLE)
        self.assertEqual(action.card_indices, (0, 1))
        self.assertEqual(value_mock.call_count, 1)

    def test_high_priestess_value_drops_when_consumable_slots_are_full(self) -> None:
        use_high_priestess = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        open_state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            money=10,
            consumables=("The High Priestess",),
            modifiers={"consumable_slots": 2},
            legal_actions=(use_high_priestess, Action(ActionType.END_SHOP)),
        )
        full_state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            money=10,
            consumables=("The High Priestess", "Death"),
            modifiers={"consumable_slots": 2},
            legal_actions=(use_high_priestess, Action(ActionType.END_SHOP)),
        )

        def storage_value(candidate: GameState) -> float:
            return float(len(candidate.consumables) * 10)

        config = ConsumableSearchConfig(leaf_samples=1, seed=1, stochastic_samples=8, min_shop_delta=0.0)

        self.assertGreater(
            consumable_action_value(open_state, use_high_priestess, config=config, value_fn=storage_value),
            consumable_action_value(full_state, use_high_priestess, config=config, value_fn=storage_value),
        )

    def test_justice_values_high_scoring_card_targets(self) -> None:
        high_card = Action(
            ActionType.USE_CONSUMABLE,
            card_indices=(0,),
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        low_card = Action(
            ActionType.USE_CONSUMABLE,
            card_indices=(1,),
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        state = GameState(
            phase=GamePhase.PLAYING_BLIND,
            required_score=1000,
            hands_remaining=2,
            hand=(Card("A", "S"), Card("2", "C")),
            consumables=("Justice",),
        )
        config = ConsumableSearchConfig(leaf_samples=1, seed=1, min_blind_delta=0.0)

        self.assertGreater(
            consumable_action_value(state, high_card, config=config, value_fn=lambda _state: 0.0),
            consumable_action_value(state, low_card, config=config, value_fn=lambda _state: 0.0),
        )

    def test_search_bot_can_sell_then_take_full_slot_pack_upgrade(self) -> None:
        bot = SearchBot(seed=1, pack_config=PackSearchConfig(leaf_samples=1, seed=1))
        sell = Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0})
        choose = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=1,
            jokers=(Joker("Weak Joker", sell_value=1),),
            modifiers={
                "joker_slots": 1,
                "pack_cards": ({"label": "Photograph", "set": "JOKER", "cost": {"buy": 5}},),
            },
            legal_actions=(sell, choose, skip),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.metadata["search"], "pack_value")
        self.assertEqual(action.metadata["search_sequence"][1]["type"], "choose_pack_card")

    def test_search_bot_delegates_when_no_discard_is_legal(self) -> None:
        bot = SearchBot(seed=1)
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            legal_actions=(Action(ActionType.SELECT_BLIND),),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.SELECT_BLIND)

    def test_search_bot_handles_blind_select_without_fallback_choose(self) -> None:
        bot = SearchBot(seed=1, enable_shop_search=True, name="search_bot_v1")
        bot._fallback = _FallbackChooseRaises(bot._fallback)
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            legal_actions=(Action(ActionType.SELECT_BLIND),),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.SELECT_BLIND)

    def test_search_bot_handles_cash_out_without_fallback_choose(self) -> None:
        bot = SearchBot(seed=1, enable_shop_search=True, name="search_bot_v1")
        bot._fallback = _FallbackChooseRaises(bot._fallback)
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            legal_actions=(Action(ActionType.CASH_OUT),),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.CASH_OUT)

    def test_search_bot_plays_obvious_hand_without_full_fallback_choose(self) -> None:
        bot = SearchBot(seed=1, enable_shop_search=True, name="search_bot_v1")
        bot._fallback = _FallbackChooseRaises(bot._fallback)
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=20,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(Card("A", "S"), Card("K", "H"), Card("2", "D")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0,)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(2,)),
            ),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)

    def test_search_bot_v1_protects_joker_bought_this_shop(self) -> None:
        bot = SearchBot(seed=1, enable_shop_search=True, name="search_bot_v1")
        buy_fresh = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        first_state = GameState(
            phase=GamePhase.SHOP,
            seed=1,
            ante=1,
            blind="Small Blind",
            money=10,
            modifiers={"shop_cards": ({"key": "j_fresh", "name": "Fresh Joker", "set": "JOKER", "cost": {"buy": 1}},)},
            legal_actions=(buy_fresh,),
        )

        first_action = bot.choose_action(first_state)

        second_state = GameState(
            phase=GamePhase.SHOP,
            seed=1,
            ante=1,
            blind="Small Blind",
            money=0,
            jokers=(Joker("Fresh Joker", sell_value=1),),
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        second_action = bot.choose_action(second_state)

        self.assertEqual(first_action.action_type, ActionType.BUY)
        self.assertEqual(second_action.action_type, ActionType.END_SHOP)

    def test_search_bot_v2_allows_beam_to_override_basic_play_discard_guard(self) -> None:
        bot = SearchBotV2(seed=1)
        beam_discard = Action(ActionType.DISCARD, card_indices=(2,), metadata={"search": "hand_beam"})
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=60,
            current_score=0,
            hands_remaining=1,
            discards_remaining=1,
            deck_size=1,
            hand=(Card("A", "S"), Card("K", "H"), Card("2", "D")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0,)),
                Action(ActionType.DISCARD, card_indices=(2,)),
            ),
        )

        with patch("balatro_ai.bots.search_bot_v2.best_hand_action", return_value=beam_discard):
            action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.metadata["search"], "hand_beam")


class _FallbackChooseRaises:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.name = wrapped.name

    def choose_action(self, state: GameState) -> Action:
        raise AssertionError(f"fallback choose_action should not be called for {state.phase.value}")

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


class _FixedChoose:
    def __init__(self, action: Action) -> None:
        self.action = action

    def choose_action(self, _state: GameState) -> Action:
        return self.action


class _RaisingChoose:
    name = "raising"

    def choose_action(self, _state: GameState) -> Action:
        raise AssertionError("fallback choose_action should not be called")


class _CountingChoose:
    def __init__(self, action: Action) -> None:
        self.action = action
        self.calls = 0

    def choose_action(self, _state: GameState) -> Action:
        self.calls += 1
        return self.action


class _EnvAssertingChoose:
    def __init__(self, action: Action, *, expected_ckpt: str) -> None:
        self.action = action
        self.expected_ckpt = expected_ckpt

    def choose_action(self, _state: GameState) -> Action:
        assert os.environ["BALATRO_VALUE_MODEL_CKPT"] == self.expected_ckpt
        assert os.environ["BALATRO_VALUE_MODEL_HEAD"] == "win"
        assert os.environ["BALATRO_VALUE_SCALE"] == "17"
        return self.action


class _FixedLogitModel:
    def __init__(self, *logits: float) -> None:
        self.logits = logits

    def eval(self) -> "_FixedLogitModel":
        return self

    def __call__(self, batch):
        import torch

        values = torch.tensor(self.logits, dtype=torch.float32)
        return values.unsqueeze(0).expand(batch.action_type.shape[0], -1)


if __name__ == "__main__":
    unittest.main()
