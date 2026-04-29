from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot


class BasicStrategyBotTests(unittest.TestCase):
    def test_plays_when_best_hand_beats_remaining_score(self) -> None:
        state = GameState(
            required_score=100,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("K", "C"),
                Card("2", "D"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)),
                Action(ActionType.DISCARD, card_indices=(3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1, 2))

    def test_prefers_smallest_hand_that_beats_remaining_score(self) -> None:
        state = GameState(
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("K", "C"),
                Card("2", "D"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))

    def test_adds_junk_kickers_to_scoring_play_to_cycle_deck(self) -> None:
        state = GameState(
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "C"),
                Card("9", "D"),
                Card("2", "C"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1, 2, 3, 4))
        self.assertIn("tactical_play", action.metadata["reason"])

    def test_does_not_cycle_cards_the_current_archetype_wants_to_keep(self) -> None:
        state = GameState(
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "D"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("T", "H"),
            ),
            jokers=(Joker("Droll Joker"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4, 5)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))

    def test_prefers_hand_that_reduces_hands_needed_over_pace_hand(self) -> None:
        state = GameState(
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("K", "C"),
                Card("2", "D"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)),
                Action(ActionType.DISCARD, card_indices=(3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1, 2))

    def test_discards_when_best_hand_is_behind_score_pace(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("9", "C"),
                Card("5", "D"),
                Card("4", "S"),
                Card("2", "C"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(3, 4, 5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertNotIn(0, action.card_indices)
        self.assertNotIn(1, action.card_indices)

    def test_plays_on_pace_when_discard_has_no_clear_speed_edge(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("9", "C"),
                Card("5", "D"),
                Card("4", "S"),
                Card("2", "C"),
            ),
            jokers=(Joker("Joker"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))
        self.assertIn("tactical_play", action.metadata["reason"])

    def test_discards_when_known_draw_reduces_hands_needed(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("9", "C"),
                Card("5", "D"),
            ),
            known_deck=(Card("A", "D"), Card("Q", "C"), Card("J", "S")),
            jokers=(Joker("Joker"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (2, 3, 4))
        self.assertIn("hands_needed=4->2", action.metadata["reason"])

    def test_discard_preserves_strong_flush_draw(self) -> None:
        state = GameState(
            required_score=900,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("K", "S"),
                Card("Q", "S"),
                Card("J", "S"),
                Card("2", "D"),
                Card("3", "C"),
                Card("4", "H"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0,)),
                Action(ActionType.DISCARD, card_indices=(3, 4, 5)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (4, 5, 6))

    def test_flush_build_discards_off_suit_cards_before_suited_core(self) -> None:
        state = GameState(
            required_score=1200,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("A", "S"),
                Card("A", "D"),
                Card("7", "C"),
                Card("8", "D"),
            ),
            jokers=(Joker("Droll Joker"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(4, 5)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 2)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6, 7)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (4, 5, 6, 7))

    def test_plays_on_last_hand_even_when_score_is_low(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=1,
            discards_remaining=4,
            hand=(Card("A", "S"), Card("A", "H")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(0,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)

    def test_buys_first_affordable_joker_before_leaving_shop(self) -> None:
        state = GameState(
            money=6,
            modifiers={
                "shop_cards": (
                    {"label": "Joker", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Mars", "set": "PLANET", "cost": {"buy": 3}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 0)

    def test_shop_prefers_stronger_joker_over_first_card(self) -> None:
        state = GameState(
            ante=1,
            money=10,
            modifiers={
                "shop_cards": (
                    {"label": "Credit Card", "set": "JOKER", "cost": {"buy": 1}},
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 1)
        self.assertIn("shop_audit", action.metadata)
        audit = action.metadata["shop_audit"]
        self.assertEqual(audit["decision"], "take")
        self.assertEqual(audit["chosen_item"]["name"], "Cavendish")
        self.assertGreaterEqual(len(audit["options"]), 2)

    def test_shop_can_buy_third_high_value_joker(self) -> None:
        state = GameState(
            ante=2,
            money=22,
            jokers=(
                Joker("Clever Joker"),
                Joker("Cartomancer"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Blueprint", "set": "JOKER", "cost": {"buy": 10}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 0)
        self.assertIn("reason", action.metadata)

    def test_shop_sells_weakest_joker_for_major_replacement(self) -> None:
        state = GameState(
            ante=3,
            money=22,
            jokers=(
                Joker("Credit Card", sell_value=1),
                Joker("Clever Joker", sell_value=2),
                Joker("Cartomancer", sell_value=3),
                Joker("Golden Joker", sell_value=3),
                Joker("Runner", sell_value=4),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Blueprint", "set": "JOKER", "cost": {"buy": 10}},
                )
            },
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.SELL, target_id="joker", amount=1, metadata={"kind": "joker", "index": 1}),
                Action(ActionType.SELL, target_id="joker", amount=2, metadata={"kind": "joker", "index": 2}),
                Action(ActionType.SELL, target_id="joker", amount=3, metadata={"kind": "joker", "index": 3}),
                Action(ActionType.SELL, target_id="joker", amount=4, metadata={"kind": "joker", "index": 4}),
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.amount, 0)
        self.assertIn("replace Credit Card", action.metadata["reason"])
        self.assertEqual(action.metadata["shop_audit"]["decision"], "replace")

    def test_shop_opens_power_pack_when_next_blind_pressure_is_high(self) -> None:
        state = GameState(
            ante=5,
            blind="Big Blind",
            required_score=16500,
            money=7,
            jokers=(Joker("Credit Card"), Joker("Cartomancer")),
            modifiers={
                "booster_packs": (
                    {"label": "Buffoon Pack", "set": "PACK", "cost": {"buy": 6}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertIn("pressure=", action.metadata["reason"])

    def test_shop_preserves_money_for_marginal_pack_when_build_is_safe(self) -> None:
        state = GameState(
            ante=1,
            blind="Small Blind",
            required_score=300,
            money=7,
            jokers=(Joker("Cavendish"), Joker("Banner")),
            modifiers={
                "booster_packs": (
                    {"label": "Buffoon Pack", "set": "PACK", "cost": {"buy": 6}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "skip")

    def test_early_shop_skips_economy_joker_before_scoring_is_safe(self) -> None:
        state = GameState(
            ante=1,
            money=7,
            modifiers={
                "shop_cards": (
                    {"label": "Mail-In Rebate", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_early_shop_skips_target_required_tarot_until_targets_are_supported(self) -> None:
        state = GameState(
            ante=1,
            money=7,
            jokers=(Joker("Joker"),),
            modifiers={
                "shop_cards": (
                    {"label": "Death", "set": "TAROT", "cost": {"buy": 3}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_early_pack_skips_unsupported_narrow_joker(self) -> None:
        state = GameState(
            ante=1,
            money=6,
            modifiers={
                "pack_cards": (
                    {"label": "Devious Joker", "set": "JOKER", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")

    def test_late_full_build_stops_rerolling_before_bank_gets_too_low(self) -> None:
        state = GameState(
            ante=7,
            blind="Small Blind",
            required_score=35000,
            money=27,
            jokers=(
                Joker("Mad Joker"),
                Joker("Square Joker"),
                Joker("Gros Michel"),
                Joker("Scary Face"),
                Joker("Banner"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "skip")

    def test_late_shop_prioritizes_missing_xmult_role(self) -> None:
        state = GameState(
            ante=5,
            blind="Big Blind",
            required_score=16500,
            money=80,
            jokers=(
                Joker("Banner"),
                Joker("Jolly Joker"),
                Joker("Green Joker"),
                Joker("Golden Joker"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Joker", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "The Duo", "set": "JOKER", "cost": {"buy": 8}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 1)
        audit = action.metadata["shop_audit"]
        self.assertIn("xmult", audit["build_profile"]["missing_roles"])
        self.assertEqual(audit["build_profile"]["archetype"], "rank")
        self.assertEqual(audit["chosen_item"]["name"], "The Duo")

    def test_late_rich_build_rerolls_when_missing_xmult_and_scaling(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=90,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        missing_roles = action.metadata["shop_audit"]["build_profile"]["missing_roles"]
        self.assertIn("xmult", missing_roles)
        self.assertIn("scaling", missing_roles)

    def test_late_shop_caps_role_hunt_rerolls_to_one_when_safe(self) -> None:
        bot = BasicStrategyBot(seed=1)
        state = GameState(
            seed=123,
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=90,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        first_action = bot.choose_action(state)
        second_action = bot.choose_action(state)

        self.assertEqual(first_action.action_type, ActionType.REROLL)
        self.assertEqual(second_action.action_type, ActionType.END_SHOP)
        self.assertEqual(second_action.metadata["shop_audit"]["shop_context"]["rerolls_in_shop"], 1)

    def test_late_shop_opens_planet_pack_when_missing_scaling(self) -> None:
        state = GameState(
            ante=5,
            blind="Small Blind",
            required_score=11000,
            money=80,
            jokers=(
                Joker("Cavendish"),
                Joker("Banner"),
                Joker("Jolly Joker"),
                Joker("Golden Joker"),
            ),
            modifiers={
                "booster_packs": (
                    {"label": "Celestial Pack", "set": "PACK", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertIn("scaling", action.metadata["shop_audit"]["build_profile"]["missing_roles"])

    def test_late_safe_build_skips_pack_without_capacity_gain(self) -> None:
        state = GameState(
            ante=7,
            blind="Big Blind",
            required_score=52500,
            money=80,
            jokers=(
                Joker("Bull"),
                Joker("Bootstraps"),
                Joker("Seeing Double"),
                Joker("Green Joker"),
                Joker("Arrowhead"),
            ),
            modifiers={
                "booster_packs": (
                    {"label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "skip")

    def test_late_rich_shop_opens_arcana_pack_for_incremental_upgrades(self) -> None:
        state = GameState(
            ante=7,
            blind="Big Blind",
            required_score=52500,
            money=120,
            jokers=(
                Joker("Stuntman"),
                Joker("Gros Michel"),
                Joker("Sock and Buskin"),
                Joker("Smiley Face"),
                Joker("Photograph"),
            ),
            modifiers={
                "booster_packs": (
                    {"label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)

    def test_high_ante_boss_discards_when_best_hand_is_barely_on_raw_pace(self) -> None:
        state = GameState(
            ante=7,
            blind="The Goad",
            required_score=3500,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("9", "C"),
                Card("5", "D"),
                Card("4", "S"),
                Card("2", "C"),
            ),
            jokers=(Joker("Joker"), Joker("Banner"), Joker("Blue Joker"), Joker("Green Joker")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4, 5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)

    def test_pack_choice_takes_useful_card_over_skip(self) -> None:
        state = GameState(
            money=6,
            modifiers={
                "pack_cards": (
                    {"label": "Judgement", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")

    def test_pack_choice_skips_target_required_tarot_until_targets_are_supported(self) -> None:
        state = GameState(
            money=6,
            modifiers={
                "pack_cards": (
                    {"label": "Strength", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")

    def test_pack_choice_skips_consumable_when_slots_are_full(self) -> None:
        state = GameState(
            money=6,
            consumables=("The Fool", "Death"),
            modifiers={
                "pack_cards": (
                    {"label": "Death", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")


if __name__ == "__main__":
    unittest.main()
