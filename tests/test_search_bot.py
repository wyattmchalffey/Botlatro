from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.bots.registry import create_bot
from balatro_ai.bots.search_bot import SearchBot
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

    def test_search_bot_uses_discard_search_when_discards_are_legal(self) -> None:
        bot = SearchBot(seed=1, discard_config=DiscardSearchConfig(draw_samples=2, leaf_samples=1, seed=1))
        discard_ace = Action(ActionType.DISCARD, card_indices=(0,))
        discard_king = Action(ActionType.DISCARD, card_indices=(1,))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            current_score=0,
            hands_remaining=1,
            discards_remaining=1,
            deck_size=2,
            hand=(Card("A", "S"), Card("K", "H"), Card("2", "D")),
            known_deck=(Card("A", "H"), Card("3", "C")),
            legal_actions=(discard_ace, discard_king),
        )

        action = bot.choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (1,))
        self.assertEqual(action.metadata["search"], "discard_expectimax")

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


if __name__ == "__main__":
    unittest.main()
