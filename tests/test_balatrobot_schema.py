from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.client import JsonRpcBalatroClient
from balatro_ai.api.state import GamePhase, GameState, Stake


class RecordingClient(JsonRpcBalatroClient):
    def __init__(self, deck: str = "RED") -> None:
        super().__init__(deck=deck)
        self.calls: list[tuple[str, dict | None]] = []

    def call(self, method: str, params: dict | None = None):
        self.calls.append((method, params))
        return {
            "state": "BLIND_SELECT",
            "ante_num": 1,
            "money": 4,
            "stake": "WHITE",
            "seed": "123",
            "round": {"hands_left": 4, "discards_left": 3, "chips": 0},
            "blinds": {"small": {"status": "SELECT", "name": "Small Blind", "score": 300}},
            "hand": {"cards": []},
        }


class BalatroBotSchemaTests(unittest.TestCase):
    def test_parse_documented_gamestate_shape(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SELECTING_HAND",
                "ante_num": 2,
                "money": 9,
                "stake": "WHITE",
                "round": {"hands_left": 4, "discards_left": 3, "chips": 12},
                "blinds": {"current": {"name": "Big Blind", "score": 450}},
                "hand": {
                    "cards": [
                        {
                            "key": "H_A",
                            "label": "Ace of Hearts",
                            "value": {"suit": "H", "rank": "A"},
                            "modifier": {"seal": None, "edition": None, "enhancement": None},
                        }
                    ]
                },
                "jokers": {"cards": [{"key": "j_joker", "label": "Joker", "cost": {"sell": 1}}]},
                "hands": {"Pair": {"level": 2, "played_this_round": 1}},
            }
        )

        self.assertEqual(state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(state.stake, Stake.WHITE)
        self.assertEqual(state.ante, 2)
        self.assertEqual(state.required_score, 450)
        self.assertEqual(state.hand[0].short_name, "AH")
        self.assertEqual(state.jokers[0].name, "Joker")
        self.assertEqual(state.jokers[0].metadata["key"], "j_joker")
        self.assertEqual(state.modifiers["joker_cards"][0]["key"], "j_joker")
        self.assertEqual(state.modifiers["blinds"]["current"]["name"], "Big Blind")
        self.assertEqual(state.hand_levels["Pair"], 2)
        self.assertEqual(state.modifiers["hands"]["Pair"]["played_this_round"], 1)
        self.assertIn("shop_cards", state.modifiers)
        self.assertTrue(any(action.action_type == ActionType.PLAY_HAND for action in state.legal_actions))

    def test_ante_nine_state_is_standard_run_win_boundary(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SELECTING_HAND",
                "ante_num": 9,
                "money": 28,
                "round": {"hands_left": 4, "discards_left": 4, "chips": 0},
                "blinds": {"current": {"name": "Small Blind", "score": 110000}},
                "hand": {"cards": [{"value": {"suit": "S", "rank": "A"}}]},
                "legal_actions": [{"type": "play_hand", "cards": [0]}],
            }
        )

        self.assertTrue(state.run_over)
        self.assertTrue(state.won)
        self.assertEqual(state.ante, 8)
        self.assertEqual(state.legal_actions, ())

    def test_ante_nine_won_but_not_run_over_is_standard_run_win_boundary(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SHOP",
                "ante_num": 9,
                "money": 105,
                "won": True,
                "run_over": False,
                "shop": {"cards": [{"label": "Misprint", "set": "JOKER", "cost": {"buy": 4}}]},
            }
        )

        self.assertTrue(state.run_over)
        self.assertTrue(state.won)
        self.assertEqual(state.ante, 8)
        self.assertEqual(state.legal_actions, ())

    def test_card_parser_preserves_debuff_state_and_raw_metadata(self) -> None:
        card = GameState.from_mapping(
            {
                "state": "SELECTING_HAND",
                "hand": {
                    "cards": [
                        {
                            "id": 7,
                            "key": "S_A",
                            "label": "Ace of Spades",
                            "value": {"suit": "S", "rank": "A", "effect": "Ace"},
                            "modifier": {"enhancement": "STEEL"},
                            "state": {"debuff": True},
                        }
                    ]
                },
            }
        ).hand[0]

        self.assertTrue(card.debuffed)
        self.assertEqual(card.enhancement, "STEEL")
        self.assertEqual(card.metadata["key"], "S_A")
        self.assertEqual(card.metadata["value"]["effect"], "Ace")

    def test_card_parser_derives_enhancement_from_balatrobench_label(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SELECTING_HAND",
                "hand": {
                    "cards": [
                        {
                            "key": "C_K",
                            "label": "Glass Card",
                            "value": {"suit": "C", "rank": "K", "effect": "+10 chips X2 Mult"},
                            "modifier": [],
                            "set": "ENHANCED",
                        },
                        {
                            "key": "H_3",
                            "label": "Bonus Card",
                            "value": {"suit": "H", "rank": "3", "effect": "+3 chips +30 extra chips"},
                            "modifier": [],
                            "set": "ENHANCED",
                        },
                    ]
                },
            }
        )

        self.assertEqual(state.hand[0].enhancement, "GLASS")
        self.assertEqual(state.hand[1].enhancement, "BONUS")

    def test_client_uses_documented_methods(self) -> None:
        client = RecordingClient()
        client.start_run(seed=123, stake="white")
        client.send_action(Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)))

        self.assertEqual(client.calls[0], ("menu", None))
        self.assertEqual(client.calls[1], ("start", {"deck": "RED", "stake": "WHITE", "seed": "123"}))
        self.assertEqual(client.calls[2], ("play", {"cards": [0, 1, 2]}))

    def test_client_uses_configured_deck(self) -> None:
        client = RecordingClient(deck="BLUE")
        client.start_run(seed=123, stake="white")

        self.assertEqual(client.calls[1], ("start", {"deck": "BLUE", "stake": "WHITE", "seed": "123"}))

    def test_pack_action_sends_target_cards(self) -> None:
        client = RecordingClient()
        client.send_action(
            Action(
                ActionType.CHOOSE_PACK_CARD,
                card_indices=(1, 3),
                target_id="card",
                amount=0,
                metadata={"kind": "card", "index": 0},
            )
        )

        self.assertEqual(client.calls[0], ("pack", {"card": 0, "cards": [1, 3]}))

    def test_shop_actions_filter_unaffordable_buys(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SHOP",
                "money": 7,
                "shop": {
                    "cards": [
                        {"label": "Cheap Joker", "cost": {"buy": 4}},
                        {"label": "Expensive Joker", "cost": {"buy": 10}},
                    ]
                },
            }
        )

        buy_actions = [action for action in state.legal_actions if action.action_type == ActionType.BUY]

        self.assertEqual(len(buy_actions), 1)
        self.assertEqual(buy_actions[0].amount, 0)

    def test_shop_actions_filter_normal_joker_buys_when_slots_are_full(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SHOP",
                "money": 20,
                "jokers": {
                    "cards": [
                        {"label": "Joker"},
                        {"label": "Jolly Joker"},
                        {"label": "Sly Joker"},
                        {"label": "Half Joker"},
                        {"label": "Banner"},
                    ]
                },
                "shop": {
                    "cards": [
                        {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                        {"label": "Negative Joker", "set": "JOKER", "edition": "NEGATIVE", "cost": {"buy": 4}},
                        {"label": "Mars", "set": "PLANET", "cost": {"buy": 3}},
                    ]
                },
            }
        )

        buy_actions = [action for action in state.legal_actions if action.action_type == ActionType.BUY]

        self.assertEqual(tuple(action.amount for action in buy_actions), (1, 2))

    def test_shop_actions_include_joker_sells(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SHOP",
                "money": 7,
                "jokers": {
                    "cards": [
                        {"label": "Joker", "cost": {"sell": 1}},
                        {"label": "Credit Card", "cost": {"sell": 1}},
                    ]
                },
                "shop": {"cards": []},
            }
        )

        sell_actions = [action for action in state.legal_actions if action.action_type == ActionType.SELL]

        self.assertEqual(tuple(action.amount for action in sell_actions), (0, 1))

    def test_empty_booster_state_does_not_derive_pack_skip_action(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SMODS_BOOSTER_OPENED",
                "money": 117,
                "pack": {"cards": []},
            }
        )

        self.assertEqual(state.phase, GamePhase.BOOSTER_OPENED)
        self.assertEqual(state.pack, ())
        self.assertEqual(tuple(action.action_type for action in state.legal_actions), (ActionType.NO_OP,))

    def test_empty_booster_state_sanitizes_bridge_pack_actions(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SMODS_BOOSTER_OPENED",
                "money": 117,
                "pack": {"cards": []},
                "legal_actions": [{"type": "choose_pack_card", "target_id": "skip", "metadata": {"kind": "skip", "index": True}}],
            }
        )

        self.assertEqual(tuple(action.action_type for action in state.legal_actions), (ActionType.NO_OP,))


if __name__ == "__main__":
    unittest.main()
