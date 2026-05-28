from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.state import GameState
from balatro_ai.bots import basic_strategy_bot as strategy
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.bots.config import (
    DEFAULT_CONFIG,
    BotConfig,
    active_config,
    bot_config_scope,
)


class BotConfigDefaultsTests(unittest.TestCase):
    def test_default_config_matches_legacy_module_constants(self) -> None:
        # Backward compatibility: module-level aliases must equal the
        # default config so any external code reading them sees the same
        # numbers as the bot does internally.
        self.assertEqual(strategy.SHOP_VALUE_TOLERANCE, DEFAULT_CONFIG.shop_value_tolerance)
        self.assertEqual(strategy.SHOP_TARGET_SAFETY_BASE, DEFAULT_CONFIG.shop_target_safety_base)
        self.assertEqual(strategy.HAND_PACE_SAFETY_BASE, DEFAULT_CONFIG.hand_pace_safety_base)

    def test_active_config_returns_default_outside_scope(self) -> None:
        self.assertIs(active_config(), DEFAULT_CONFIG)

    def test_default_config_field_values(self) -> None:
        # Pin the historical numbers so an accidental default change
        # surfaces as a test failure rather than a silent winrate shift.
        cfg = BotConfig()
        self.assertEqual(cfg.shop_value_tolerance, 0.25)
        self.assertTrue(cfg.calibrated_shop_planner_enabled)
        self.assertEqual(cfg.calibrated_shop_legacy_weight, 1.0)
        self.assertEqual(cfg.calibrated_shop_leaf_delta_weight, 0.35)
        self.assertEqual(cfg.calibrated_shop_pressure_delta_weight, 0.65)
        self.assertEqual(cfg.calibrated_shop_role_fill_weight, 1.0)
        self.assertEqual(cfg.calibrated_shop_late_conversion_weight, 1.0)
        self.assertEqual(cfg.calibrated_shop_reserve_risk_weight, 0.0)
        self.assertEqual(cfg.calibrated_shop_slot_risk_weight, 1.0)
        self.assertEqual(cfg.calibrated_shop_boss_risk_weight, 1.0)
        self.assertEqual(cfg.shop_target_safety_base, 1.15)
        self.assertEqual(cfg.hand_pace_safety_base, 1.05)
        self.assertEqual(cfg.shop_safety_ante3_bonus, 0.10)
        self.assertEqual(cfg.shop_safety_ante4_bonus, 0.10)
        self.assertEqual(cfg.shop_safety_ante5_bonus, 0.05)
        self.assertEqual(cfg.shop_safety_full_slots_bonus, 0.05)
        self.assertEqual(cfg.shop_safety_cap, 1.45)
        self.assertEqual(cfg.capacity_no_xmult_penalty, 0.10)
        self.assertEqual(cfg.capacity_no_planet_penalty, 0.08)
        self.assertEqual(cfg.capacity_full_slots_penalty, 0.06)
        self.assertEqual(cfg.capacity_rare_hand_penalty, 0.05)
        self.assertEqual(cfg.capacity_ante5_penalty, 0.04)
        self.assertEqual(cfg.capacity_floor, 0.72)
        self.assertEqual(cfg.joker_sample_coefficient, 0.08)
        self.assertEqual(cfg.panic_discard_base_ratio, 0.45)
        self.assertEqual(cfg.panic_discard_ante4_bonus, 0.10)
        self.assertEqual(cfg.panic_discard_boss_bonus, 0.05)
        self.assertEqual(cfg.panic_discard_low_hands_bonus, 0.10)
        self.assertEqual(cfg.panic_discard_desperate_penalty, 0.25)
        self.assertEqual(cfg.panic_discard_floor, 0.20)
        self.assertEqual(cfg.panic_discard_cap, 0.70)


class BotConfigScopeTests(unittest.TestCase):
    def test_scope_swaps_active_config_and_restores_on_exit(self) -> None:
        custom = BotConfig(shop_value_tolerance=0.99)
        self.assertIs(active_config(), DEFAULT_CONFIG)
        with bot_config_scope(custom):
            self.assertIs(active_config(), custom)
        self.assertIs(active_config(), DEFAULT_CONFIG)

    def test_nested_scopes_restore_outer_config(self) -> None:
        outer = BotConfig(joker_sample_coefficient=0.20)
        inner = BotConfig(joker_sample_coefficient=0.40)
        with bot_config_scope(outer):
            self.assertEqual(active_config().joker_sample_coefficient, 0.20)
            with bot_config_scope(inner):
                self.assertEqual(active_config().joker_sample_coefficient, 0.40)
            self.assertEqual(active_config().joker_sample_coefficient, 0.20)
        self.assertIs(active_config(), DEFAULT_CONFIG)

    def test_scope_with_none_is_passthrough(self) -> None:
        # Passing None must not clobber an outer scope and must leave
        # the default in place when no scope is active.
        self.assertIs(active_config(), DEFAULT_CONFIG)
        with bot_config_scope(None):
            self.assertIs(active_config(), DEFAULT_CONFIG)
        outer = BotConfig(shop_value_tolerance=0.42)
        with bot_config_scope(outer):
            with bot_config_scope(None):
                self.assertIs(active_config(), outer)


class BotConfigAffectsHotPathTests(unittest.TestCase):
    def test_panic_discard_ratio_respects_config_override(self) -> None:
        state = GameState(ante=2, hands_remaining=4, discards_remaining=4)
        default_ratio = strategy._panic_discard_ratio(state)
        tighter = BotConfig(panic_discard_base_ratio=0.20, panic_discard_floor=0.10)
        with bot_config_scope(tighter):
            override_ratio = strategy._panic_discard_ratio(state)
        self.assertNotEqual(default_ratio, override_ratio)
        self.assertEqual(override_ratio, 0.20)

    def test_shop_target_safety_multiplier_respects_config_override(self) -> None:
        state = GameState(ante=5)
        default_mult = strategy._shop_target_safety_multiplier(state)
        steeper = BotConfig(shop_safety_ante5_bonus=0.20, shop_safety_cap=2.00)
        with bot_config_scope(steeper):
            override_mult = strategy._shop_target_safety_multiplier(state)
        # Steeper ante5 bonus should produce a larger multiplier.
        self.assertGreater(override_mult, default_mult)

    def test_basic_strategy_bot_installs_its_config_during_choose_action(self) -> None:
        # Spy that the config is active inside the bot's decision scope.
        captured: list[BotConfig] = []

        original = strategy._first_action_of_type

        def spy(state: GameState, action_type):
            captured.append(active_config())
            return original(state, action_type)

        custom = BotConfig(joker_sample_coefficient=0.99)
        bot = BasicStrategyBot(seed=1, config=custom)

        try:
            strategy._first_action_of_type = spy
            try:
                bot.choose_action(GameState())
            except Exception:
                pass
        finally:
            strategy._first_action_of_type = original

        self.assertTrue(captured, "expected the bot to invoke _first_action_of_type")
        self.assertIs(captured[0], custom)


if __name__ == "__main__":
    unittest.main()
