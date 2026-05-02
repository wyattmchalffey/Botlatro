from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState, Joker
from balatro_ai.search.shop_sampler import ShopSampler
from balatro_ai.search.shop_search import ShopSearchConfig, best_shop_action, shop_action_search_value, shop_leaf_value


def shop_card(name: str, *, cost: int = 4, edition: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "key": f"j_{name.lower().replace(' ', '_')}",
        "name": name,
        "label": name,
        "set": "JOKER",
        "cost": {"buy": cost, "base": cost},
    }
    if edition is not None:
        payload["edition"] = edition
        payload["modifier"] = {"edition": edition}
    return payload


def joker_effect(effect: str) -> dict[str, object]:
    return {"value": {"effect": effect}}


def tiny_sampler() -> ShopSampler:
    return ShopSampler(
        {
            "shop_slots": {"base_joker_max": 2, "booster_slots": 0, "voucher_slots": 0},
            "slot_rates": {"Joker": 1},
            "joker_rarity_rates": {"common": 1.0},
            "reroll": {"base_cost": 5, "increase_per_reroll": 1},
            "jokers": [shop_card("Rerolled Joker")],
            "tarots": [],
            "planets": [],
            "spectrals": [],
            "vouchers": [],
            "boosters": [],
        }
    )


class FixedRerollSampler:
    def reroll_cost(self, _state: GameState) -> int:
        return 5

    def reroll_ev(self, _state: GameState, **_kwargs: object) -> float:
        return 20.0


def joker_leaf_value(state: GameState) -> float:
    names = {joker.name for joker in state.jokers}
    if "Great Joker" in names:
        return 20.0
    if "Negative Joker" in names:
        return 15.0
    if "Bad Joker" in names:
        return 2.0
    return 0.0


class ShopSearchTests(unittest.TestCase):
    def test_best_shop_action_buys_best_leaf_item(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            modifiers={
                "shop_cards": (shop_card("Bad Joker"), shop_card("Great Joker")),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=1, beam_width=4, leaf_weight=1.0),
            sampler=tiny_sampler(),
            leaf_value_fn=joker_leaf_value,
            action_value_fn=lambda _state, _action: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 1)
        self.assertEqual(action.metadata["search"], "shop_beam")

    def test_best_shop_action_can_sell_then_buy_upgrade(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=5,
            jokers=(Joker("Weak Joker", sell_value=5),),
            modifiers={
                "joker_slots": 1,
                "shop_cards": (shop_card("Great Joker", cost=10),),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=2, beam_width=4, leaf_weight=1.0),
            sampler=tiny_sampler(),
            leaf_value_fn=joker_leaf_value,
            action_value_fn=lambda _state, _action: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.amount, 0)
        self.assertEqual(action.metadata["search_path"][0]["type"], "sell")
        self.assertEqual(action.metadata["search_path"][1]["type"], "buy")

    def test_negative_joker_can_be_bought_at_full_normal_slots(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            jokers=(Joker("Normal Joker"),),
            modifiers={
                "joker_slots": 1,
                "shop_cards": (shop_card("Negative Joker", edition="NEGATIVE"),),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=1, beam_width=4, leaf_weight=1.0),
            sampler=tiny_sampler(),
            leaf_value_fn=joker_leaf_value,
            action_value_fn=lambda _state, _action: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 0)

    def test_consumable_shop_card_is_not_bought_when_slots_are_full(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            consumables=("Jupiter", "Saturn"),
            modifiers={
                "shop_cards": ({"key": "c_temperance", "name": "Temperance", "set": "TAROT", "cost": {"buy": 3}},),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=1, beam_width=4, leaf_weight=1.0),
            sampler=tiny_sampler(),
            leaf_value_fn=lambda _state: 0.0,
            action_value_fn=lambda _state, candidate: 20.0 if candidate.action_type == ActionType.BUY else 1.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_protected_joker_is_not_sold(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=0,
            jokers=(Joker("Fresh Joker", sell_value=4),),
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=1, beam_width=2, leaf_weight=0.0),
            sampler=tiny_sampler(),
            action_value_fn=lambda _state, candidate: 10.0 if candidate.action_type == ActionType.SELL else 1.0,
            protected_jokers=("Fresh Joker",),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_sell_is_not_used_just_to_fund_reroll_or_pack(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=3,
            jokers=(Joker("Useful Joker", sell_value=4),),
            modifiers={
                "booster_packs": ({"key": "p_arcana_normal_1", "name": "Arcana Pack", "set": "BOOSTER", "cost": {"buy": 4}},),
                "reroll_cost": 5,
            },
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=2, beam_width=4, leaf_weight=0.0),
            sampler=tiny_sampler(),
            action_value_fn=lambda _state, candidate: 20.0 if candidate.action_type == ActionType.SELL else 1.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_sell_must_be_followed_by_visible_joker_buy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=5,
            jokers=(Joker("Useful Joker", sell_value=5),),
            modifiers={
                "joker_slots": 1,
                "shop_cards": (shop_card("Great Joker", cost=10),),
                "booster_packs": ({"key": "p_arcana_normal_1", "name": "Arcana Pack", "set": "BOOSTER", "cost": {"buy": 8}},),
                "reroll_cost": 99,
            },
        )

        def action_value(_state: GameState, candidate: Action) -> float:
            if candidate.action_type == ActionType.OPEN_PACK:
                return 20.0
            if candidate.action_type == ActionType.END_SHOP:
                return 1.0
            return 0.0

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=2, beam_width=4, leaf_weight=0.0),
            sampler=tiny_sampler(),
            action_value_fn=action_value,
        )

        self.assertIsNotNone(action)
        if action.action_type == ActionType.SELL:
            self.assertNotEqual(action.amount, 0)

    def test_default_replacement_gate_allows_large_dynamic_upgrade(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            money=1,
            jokers=(Joker("Credit Card", sell_value=1),),
            modifiers={
                "joker_slots": 1,
                "shop_cards": (shop_card("Cavendish", cost=1),),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=2, beam_width=8),
            sampler=tiny_sampler(),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.amount, 0)

    def test_default_replacement_gate_rejects_low_dynamic_upgrade(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            money=1,
            jokers=(
                Joker("Photograph", sell_value=2, metadata=joker_effect("First played face card gives X2 Mult when scored")),
                Joker("Card Sharp", sell_value=3, metadata=joker_effect("X3 Mult if played poker hand has already been played this round")),
                Joker(
                    "Swashbuckler",
                    sell_value=2,
                    metadata=joker_effect("Adds the sell value of all other owned Jokers to Mult (Currently +11 Mult)"),
                ),
                Joker("Mr. Bones", edition="HOLO", sell_value=4),
                Joker("Popcorn", sell_value=2),
            ),
            modifiers={
                "joker_slots": 5,
                "shop_cards": (shop_card("Credit Card", cost=1),),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=2, beam_width=8),
            sampler=tiny_sampler(),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_open_pack_leaf_value_pays_pack_cost(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=25,
            modifiers={
                "booster_packs": (
                    {"key": "p_arcana_normal_1", "name": "Arcana Pack", "set": "BOOSTER", "cost": {"buy": 5}},
                ),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=1, beam_width=4, leaf_weight=1.0),
            sampler=tiny_sampler(),
            leaf_value_fn=shop_leaf_value,
            action_value_fn=lambda _state, _candidate: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_shop_action_search_value_rejects_non_shop_action(self) -> None:
        value = shop_action_search_value(GameState(), Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(value, 0.0)

    def test_reroll_search_value_charges_interest_opportunity_cost(self) -> None:
        state = GameState(phase=GamePhase.SHOP, ante=2, money=10, required_score=600)

        value = shop_action_search_value(
            state,
            Action(ActionType.REROLL),
            sampler=FixedRerollSampler(),  # type: ignore[arg-type]
            config=ShopSearchConfig(reroll_samples=1),
        )

        self.assertLess(value, 20.0)

    def test_shop_leaf_value_rewards_next_round_interest_breakpoints(self) -> None:
        just_short = GameState(phase=GamePhase.SHOP, ante=2, money=24)
        at_cap = GameState(phase=GamePhase.SHOP, ante=2, money=25)

        self.assertGreater(shop_leaf_value(at_cap) - shop_leaf_value(just_short), 2.0)


if __name__ == "__main__":
    unittest.main()
