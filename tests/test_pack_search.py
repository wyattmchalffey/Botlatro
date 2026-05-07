from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.search.forward_sim import PLANET_TO_HAND
from balatro_ai.search.pack_search import PackSearchConfig, best_pack_action, pack_action_value


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

    def test_best_pack_action_models_tarot_pack_choices(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("The Fool",),
            modifiers={"pack_cards": ({"label": "The Fool", "set": "TAROT"},)},
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            ),
        )

        action = best_pack_action(state, value_fn=straight_level_value)

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.metadata["search"], "pack_value")

    def test_pack_value_rewards_justice_targeted_glass_card(self) -> None:
        choose_justice = Action(
            ActionType.CHOOSE_PACK_CARD,
            card_indices=(0,),
            target_id="card",
            amount=0,
            metadata={"kind": "card", "index": 0},
        )
        choose_emperor = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=1, metadata={"kind": "card", "index": 1})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            hand=(Card("A", "S"), Card("2", "C")),
            pack=("Justice", "The Emperor"),
            modifiers={
                "pack_cards": (
                    {"label": "Justice", "set": "TAROT"},
                    {"label": "The Emperor", "set": "TAROT"},
                )
            },
            legal_actions=(choose_justice, choose_emperor),
        )

        justice_value = pack_action_value(
            state,
            choose_justice,
            config=PackSearchConfig(seed=1),
            value_fn=lambda _state: 0.0,
            include_item_value=True,
        )
        emperor_value = pack_action_value(
            state,
            choose_emperor,
            config=PackSearchConfig(seed=1),
            value_fn=lambda _state: 0.0,
            include_item_value=True,
        )

        self.assertGreater(justice_value, emperor_value)

    def test_pack_value_prefers_hermit_to_equal_temperance_payout(self) -> None:
        choose_hermit = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        choose_temperance = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=1, metadata={"kind": "card", "index": 1})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            money=10,
            jokers=(Joker("Sell Value", sell_value=10),),
            pack=("The Hermit", "Temperance"),
            modifiers={
                "pack_cards": (
                    {"label": "The Hermit", "set": "TAROT"},
                    {"label": "Temperance", "set": "TAROT"},
                )
            },
            legal_actions=(choose_hermit, choose_temperance),
        )

        hermit_value = pack_action_value(state, choose_hermit, config=PackSearchConfig(seed=1), value_fn=lambda leaf: float(leaf.money), include_item_value=True)
        temperance_value = pack_action_value(state, choose_temperance, config=PackSearchConfig(seed=1), value_fn=lambda leaf: float(leaf.money), include_item_value=True)

        self.assertGreater(hermit_value, temperance_value)

    def test_best_pack_action_can_sell_consumable_to_improve_pack_high_priestess(self) -> None:
        choose_high_priestess = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        sell_death = Action(ActionType.SELL, target_id="consumable", amount=0, metadata={"kind": "consumable", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            consumables=("Death", "Strength"),
            modifiers={
                "consumable_slots": 2,
                "pack_cards": ({"label": "The High Priestess", "set": "TAROT"},),
            },
            legal_actions=(sell_death, choose_high_priestess, skip),
        )

        action = best_pack_action(
            state,
            config=PackSearchConfig(seed=1, stochastic_samples=4),
            value_fn=lambda next_state: float(sum(100 for item in next_state.consumables if item in PLANET_TO_HAND)),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.target_id, "consumable")
        self.assertEqual(action.metadata["search_sequence"][1]["type"], "choose_pack_card")

    def test_best_pack_action_does_not_use_stored_consumable_during_pack_choice(self) -> None:
        use_high_priestess = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            consumables=("The High Priestess",),
            modifiers={"consumable_slots": 2, "pack_cards": ()},
            legal_actions=(use_high_priestess, skip),
        )

        action = best_pack_action(
            state,
            config=PackSearchConfig(seed=1, stochastic_samples=4),
            value_fn=lambda next_state: float(sum(100 for item in next_state.consumables if item in PLANET_TO_HAND)),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")

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

    def test_best_pack_action_skips_medium_pick_to_scale_red_card(self) -> None:
        choose_juggler = Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        skip = Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            ante=1,
            jokers=(Joker("Red Card", metadata={"value": {"effect": "Currently +0 Mult"}}),),
            modifiers={
                "joker_slots": 5,
                "pack_cards": ({"label": "Juggler", "set": "JOKER", "cost": {"buy": 4}},),
            },
            legal_actions=(choose_juggler, skip),
        )

        action = best_pack_action(state)

        self.assertIsNotNone(action)
        self.assertEqual(action.target_id, "skip")
        self.assertEqual(action.metadata["search"], "pack_value")

    def test_pack_action_value_rejects_non_pack_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires choose_pack_card"):
            pack_action_value(GameState(), Action(ActionType.DISCARD, card_indices=(0,)))


if __name__ == "__main__":
    unittest.main()
