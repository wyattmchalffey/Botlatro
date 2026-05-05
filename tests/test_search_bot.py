from __future__ import annotations

import unittest
from unittest.mock import patch

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, with_derived_legal_actions
from balatro_ai.bots.registry import create_bot
from balatro_ai.bots.search_bot import SearchBot
from balatro_ai.search.consumable_search import ConsumableSearchConfig, best_consumable_action, consumable_action_value
from balatro_ai.search.discard_search import DiscardSearchConfig
from balatro_ai.search.pack_search import PackSearchConfig


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

    def test_registry_creates_trace_enabled_search_bot_v1(self) -> None:
        bot = create_bot("search_bot_v1_trace", seed=3)

        self.assertIsInstance(bot, SearchBot)
        self.assertEqual(bot.name, "search_bot_v1_trace")
        self.assertTrue(bot.enable_shop_search)
        self.assertIsNotNone(bot.shop_config)
        self.assertEqual(bot.shop_config.trace_top_paths, 8)

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


class _FallbackChooseRaises:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.name = wrapped.name

    def choose_action(self, state: GameState) -> Action:
        raise AssertionError(f"fallback choose_action should not be called for {state.phase.value}")

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


if __name__ == "__main__":
    unittest.main()
