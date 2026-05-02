from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState, Joker
from balatro_ai.search.pack_search import best_pack_action, pack_action_value


def straight_level_value(state: GameState) -> float:
    return float(state.hand_levels.get("Straight", 1))


class PackSearchTests(unittest.TestCase):
    def test_best_pack_action_uses_deterministic_pack_value(self) -> None:
        choose_saturn = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            hand_levels={"Straight": 1},
            pack=("Saturn",),
            modifiers={"pack_cards": ({"label": "Saturn", "set": "PLANET"},)},
            legal_actions=(skip, choose_saturn),
        )

        action = best_pack_action(state, value_fn=straight_level_value)

        self.assertIsNotNone(action)
        self.assertEqual(action.target_id, "card")
        self.assertEqual(action.metadata["search"], "pack_value")
        self.assertEqual(action.metadata["search_value"], 2.0)

    def test_best_pack_action_delegates_non_deterministic_tarot_pack(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("The Fool",),
            modifiers={"pack_cards": ({"label": "The Fool", "set": "TAROT"},)},
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            ),
        )

        self.assertIsNone(best_pack_action(state, value_fn=straight_level_value))

    def test_best_pack_action_skips_normal_joker_when_slots_are_full(self) -> None:
        choose_joker = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            jokers=(Joker("One"),),
            modifiers={
                "joker_slots": 1,
                "pack_cards": ({"label": "Riff-raff", "set": "JOKER", "cost": {"buy": 6}},),
            },
            legal_actions=(choose_joker, skip),
        )

        action = best_pack_action(state, value_fn=lambda next_state: float(len(next_state.jokers) * 100))

        self.assertIsNotNone(action)
        self.assertEqual(action.target_id, "skip")

    def test_best_pack_action_sells_then_takes_full_slot_upgrade(self) -> None:
        choose_joker = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        sell_weak = Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            jokers=(Joker("Weak Joker", sell_value=1),),
            modifiers={
                "joker_slots": 1,
                "pack_cards": ({"label": "Great Joker", "set": "JOKER", "cost": {"buy": 6}},),
            },
            legal_actions=(sell_weak, choose_joker, skip),
        )

        action = best_pack_action(
            state,
            value_fn=lambda next_state: 10000.0 if any(joker.name == "Great Joker" for joker in next_state.jokers) else 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.amount, 0)
        self.assertEqual(action.metadata["search"], "pack_value")
        self.assertEqual(action.metadata["search_sequence"][1]["type"], "choose_pack_card")

    def test_best_pack_action_does_not_sell_when_joker_slot_is_open(self) -> None:
        choose_joker = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        sell_weak = Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            jokers=(Joker("Weak Joker", sell_value=1),),
            modifiers={
                "joker_slots": 2,
                "pack_cards": ({"label": "Great Joker", "set": "JOKER", "cost": {"buy": 6}},),
            },
            legal_actions=(sell_weak, choose_joker, skip),
        )

        action = best_pack_action(
            state,
            value_fn=lambda next_state: (
                1000.0 - len(next_state.jokers) * 100.0
                if any(joker.name == "Great Joker" for joker in next_state.jokers)
                else 0.0
            ),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")

    def test_best_pack_action_skips_full_slot_joker_when_upgrade_is_worse(self) -> None:
        choose_joker = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        sell_good = Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            jokers=(Joker("Good Joker", sell_value=1),),
            modifiers={
                "joker_slots": 1,
                "pack_cards": ({"label": "Bad Joker", "set": "JOKER", "cost": {"buy": 6}},),
            },
            legal_actions=(sell_good, choose_joker, skip),
        )

        action = best_pack_action(
            state,
            value_fn=lambda next_state: 1000.0 if any(joker.name == "Good Joker" for joker in next_state.jokers) else 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.target_id, "skip")

    def test_best_pack_action_allows_negative_joker_when_slots_are_full(self) -> None:
        choose_joker = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            jokers=(Joker("One"),),
            modifiers={
                "joker_slots": 1,
                "pack_cards": (
                    {"label": "Negative Riff-raff", "set": "JOKER", "edition": "NEGATIVE", "cost": {"buy": 11}},
                ),
            },
            legal_actions=(skip, choose_joker),
        )

        action = best_pack_action(state, value_fn=lambda next_state: float(len(next_state.jokers) * 100))

        self.assertIsNotNone(action)
        self.assertEqual(action.target_id, "card")

    def test_pack_action_value_rejects_non_pack_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires choose_pack_card"):
            pack_action_value(GameState(), Action(ActionType.DISCARD, card_indices=(0,)))


if __name__ == "__main__":
    unittest.main()
