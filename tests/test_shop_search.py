from __future__ import annotations

import unittest
from unittest.mock import patch

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState, Joker
from balatro_ai.search.forward_sim import PLANET_TO_HAND
from balatro_ai.search.shop_sampler import ShopSampler
from balatro_ai.bots.basic_strategy.shop_jokers import (
    _rare_hand_commitment_reliability,
    _stencil_slot_discipline_penalty,
)
from balatro_ai.search.shop_search import (
    ShopSearchConfig,
    _BeamNode,
    _action_potential_shaping,
    _minimum_safe_shop_money,
    _shop_money_floor_penalty,
    best_shop_action,
    shop_build_capacity_delta_value,
    shop_action_search_value,
    shop_capacity_headroom_value,
    shop_leaf_value,
    shop_leaf_terms,
    shop_owned_joker_leaf_value,
    shop_survival_value,
)


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


def planet_card(name: str, *, cost: int = 3) -> dict[str, object]:
    return {
        "key": f"c_{name.lower().replace(' ', '_')}",
        "name": name,
        "label": name,
        "set": "PLANET",
        "cost": {"buy": cost, "base": cost},
    }


def voucher_card(name: str, *, cost: int = 10) -> dict[str, object]:
    return {
        "key": f"v_{name.lower().replace(' ', '_')}",
        "name": name,
        "label": name,
        "set": "VOUCHER",
        "cost": {"buy": cost, "base": cost},
    }


def booster_pack(name: str, *, cost: int = 4) -> dict[str, object]:
    return {
        "key": f"p_{name.lower().replace(' ', '_')}",
        "name": name,
        "label": name,
        "set": "BOOSTER",
        "cost": {"buy": cost, "base": cost},
    }


def joker_effect(effect: str) -> dict[str, object]:
    return {"value": {"effect": effect}}


def _path_key(path: tuple[dict[str, object], ...]) -> tuple[tuple[object, object, object], ...]:
    return tuple((step.get("type"), step.get("kind"), step.get("index")) for step in path)


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

    def test_best_shop_action_can_include_trace_metadata(self) -> None:
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
            config=ShopSearchConfig(depth=1, beam_width=4, leaf_weight=1.0, trace_top_paths=2),
            sampler=tiny_sampler(),
            leaf_value_fn=joker_leaf_value,
            action_value_fn=lambda _state, _action: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.metadata["search_shop_snapshot"]["shop_cards"], ("Bad Joker", "Great Joker"))
        candidates = action.metadata["search_candidates"]
        self.assertGreaterEqual(len(candidates), 1)
        self.assertIn("leaf_score", candidates[0])
        self.assertIn("path", candidates[0])

    def test_default_leaf_trace_includes_term_breakdown(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            money=10,
            modifiers={
                "shop_cards": (shop_card("Joker"),),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=1, beam_width=4, leaf_weight=1.0, trace_top_paths=2),
            sampler=tiny_sampler(),
            action_value_fn=lambda _state, _action: 0.0,
        )

        self.assertIsNotNone(action)
        candidates = action.metadata["search_candidates"]
        self.assertGreaterEqual(len(candidates), 1)
        self.assertIn("leaf_terms", candidates[0])
        self.assertIn("survival", candidates[0]["leaf_terms"])
        self.assertIn("clear_gate_penalty", candidates[0]["leaf_terms"])

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

    def test_best_shop_action_can_buy_before_temperance_when_payout_justifies_it(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            consumables=("Temperance",),
            modifiers={
                "shop_cards": (shop_card("Temperance Target", cost=4),),
                "reroll_cost": 99,
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.USE_CONSUMABLE, target_id="consumable", amount=0, metadata={"kind": "consumable", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        def leaf_value(candidate: GameState) -> float:
            return float(candidate.money + (len(candidate.jokers) * 10))

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=2, beam_width=4, leaf_weight=1.0, trace_top_paths=3),
            sampler=tiny_sampler(),
            leaf_value_fn=leaf_value,
            action_value_fn=lambda _state, _candidate: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 0)
        self.assertEqual(action.metadata["search_path"][0]["type"], "buy")
        self.assertEqual(action.metadata["search_path"][1]["type"], "use_consumable")

    def test_best_shop_action_does_not_buy_just_to_raise_temperance_payout(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            consumables=("Temperance",),
            modifiers={
                "shop_cards": (shop_card("Temperance Bait", cost=4),),
                "reroll_cost": 99,
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.USE_CONSUMABLE, target_id="consumable", amount=0, metadata={"kind": "consumable", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=2, beam_width=4, leaf_weight=1.0),
            sampler=tiny_sampler(),
            leaf_value_fn=lambda candidate: float(candidate.money),
            action_value_fn=lambda _state, _candidate: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_best_shop_action_can_sell_consumable_before_high_priestess(self) -> None:
        sell_death = Action(ActionType.SELL, target_id="consumable", amount=0, metadata={"kind": "consumable", "index": 0})
        use_high_priestess = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=1,
            metadata={"kind": "consumable", "index": 1},
        )
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            consumables=("Death", "The High Priestess"),
            modifiers={"consumable_slots": 2, "reroll_cost": 99},
            legal_actions=(sell_death, use_high_priestess, Action(ActionType.END_SHOP)),
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=2, beam_width=4, leaf_weight=1.0),
            sampler=ShopSampler.from_default_data(),
            leaf_value_fn=lambda next_state: float(sum(100 for item in next_state.consumables if item in PLANET_TO_HAND)),
            action_value_fn=lambda _state, _candidate: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.target_id, "consumable")
        self.assertEqual(action.metadata["search_path"][1]["type"], "use_consumable")

    def test_end_shop_models_perkeo_consumable_copy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            consumables=("Mars",),
            jokers=(Joker("Perkeo"),),
            legal_actions=(Action(ActionType.END_SHOP),),
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=1, beam_width=2, leaf_weight=1.0),
            sampler=tiny_sampler(),
            leaf_value_fn=lambda next_state: float(len(next_state.consumables) * 100),
            action_value_fn=lambda _state, _candidate: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.END_SHOP)
        candidates = action.metadata["search_path"]
        self.assertEqual(candidates[0]["type"], "end_shop")

    def test_open_pack_models_hallucination_tarot_creation(self) -> None:
        open_pack = Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            jokers=(Joker("Hallucination"),),
            modifiers={
                "probability_multiplier": 2,
                "booster_packs": ({"key": "p_arcana_normal_1", "name": "Arcana Pack", "set": "BOOSTER", "kind": "Arcana", "cost": {"buy": 0}},),
                "reroll_cost": 99,
            },
            legal_actions=(open_pack, Action(ActionType.END_SHOP)),
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=1, beam_width=2, leaf_weight=1.0),
            sampler=ShopSampler.from_default_data(),
            leaf_value_fn=lambda next_state: float(len(next_state.consumables) * 100),
            action_value_fn=lambda _state, _candidate: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.OPEN_PACK)

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

    def test_consumable_shop_card_uses_buy_and_use_when_slots_are_full(self) -> None:
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
        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertTrue(action.metadata["buy_and_use"])

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

    def test_sell_can_fund_non_joker_line_when_evaluator_prefers_it(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=3,
            jokers=(Joker("Useful Joker", sell_value=4),),
            modifiers={
                "shop_cards": ({"key": "c_hermit", "name": "The Hermit", "set": "TAROT", "cost": {"buy": 4}},),
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
        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.amount, 0)

    def test_sell_can_fund_visible_non_joker_without_forcing_joker_buy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=5,
            jokers=(Joker("Useful Joker", sell_value=5),),
            modifiers={
                "joker_slots": 1,
                "shop_cards": (
                    shop_card("Great Joker", cost=10),
                    {"key": "c_hermit", "name": "The Hermit", "set": "TAROT", "cost": {"buy": 8}},
                ),
                "reroll_cost": 99,
            },
        )

        def action_value(_state: GameState, candidate: Action) -> float:
            if candidate.action_type == ActionType.BUY and candidate.amount == 1:
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
        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.amount, 0)

    def test_default_shop_search_allows_large_dynamic_replacement(self) -> None:
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

    def test_default_shop_search_allows_clear_capacity_replacement(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=5,
            blind="Small Blind",
            required_score=11000,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            money=81,
            jokers=(
                Joker("Green Joker", sell_value=4),
                Joker("Walkie Talkie", sell_value=4),
                Joker("Swashbuckler", sell_value=4),
                Joker("Smiley Face", sell_value=4),
                Joker("Red Card", edition="FOIL", sell_value=5),
            ),
            modifiers={
                "joker_slots": 5,
                "shop_cards": (planet_card("Venus", cost=3), shop_card("Bootstraps", cost=6)),
                "voucher_cards": (voucher_card("Clearance Sale", cost=10),),
                "booster_packs": (booster_pack("Jumbo Celestial Pack", cost=6),),
                "reroll_cost": 5,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=3, beam_width=8),
            sampler=tiny_sampler(),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertIn(action.amount, (0, 1, 2, 3))

    def test_discount_voucher_line_beats_reversed_discounted_shop_card_line(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            hands_remaining=4,
            deck_size=40,
            money=30,
            jokers=(Joker("Joker", sell_value=1),),
            modifiers={
                "joker_slots": 5,
                "shop_cards": (shop_card("Gros Michel", cost=4),),
                "voucher_cards": (voucher_card("Clearance Sale", cost=10),),
                "reroll_cost": 99,
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}),
            ),
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=2, beam_width=8, min_search_value=-1000.0, trace_top_paths=20),
            sampler=tiny_sampler(),
        )

        self.assertIsNotNone(action)
        paths = {_path_key(candidate["path"]): candidate for candidate in action.metadata["search_candidates"]}
        voucher_first = paths[(("buy", "voucher", 0), ("buy", "card", 0))]
        card_first = paths[(("buy", "card", 0), ("buy", "voucher", 0))]

        # Buying the discount voucher first leaves more money (the card is
        # then discounted) and yields the better shop end-state, which the
        # leaf value reflects. We assert on leaf_score rather than the total
        # beam score: the total blends a per-step action-potential heuristic
        # that is myopic about the discount synergy (it penalizes spending on
        # the no-immediate-score voucher more from a higher money position),
        # so it can rank the card-first ordering slightly ahead even though
        # that line's end-state is worse. leaf_score is the substantive
        # end-state measure this test is about.
        self.assertGreater(voucher_first["result"]["money"], card_first["result"]["money"])
        self.assertGreater(voucher_first["leaf_score"], card_first["leaf_score"])

    def test_sell_can_bridge_through_discount_voucher_before_joker_buy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            hands_remaining=4,
            deck_size=40,
            money=9,
            jokers=(Joker("Credit Card", sell_value=5),),
            modifiers={
                "joker_slots": 1,
                "shop_cards": (shop_card("Gros Michel", cost=6),),
                "voucher_cards": (voucher_card("Clearance Sale", cost=10),),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=3, beam_width=8, min_search_value=-1000.0, trace_top_paths=20),
            sampler=tiny_sampler(),
        )

        self.assertIsNotNone(action)
        candidate_paths = {_path_key(candidate["path"]) for candidate in action.metadata["search_candidates"]}
        self.assertIn(
            (("sell", "joker", 0), ("buy", "voucher", 0), ("buy", "card", 0)),
            candidate_paths,
        )

    def test_overstock_revealed_slots_are_not_spent_against_speculatively(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            hands_remaining=4,
            deck_size=40,
            money=15,
            jokers=(Joker("Credit Card", sell_value=5),),
            modifiers={
                "joker_slots": 1,
                "shop_cards": (),
                "voucher_cards": (voucher_card("Overstock", cost=10),),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=3, beam_width=12, min_search_value=-1000.0, trace_top_paths=30),
            sampler=tiny_sampler(),
        )

        self.assertIsNotNone(action)
        for candidate in action.metadata["search_candidates"]:
            path = _path_key(candidate["path"])
            for index, step in enumerate(path):
                if step == ("buy", "voucher", 0):
                    self.assertNotIn(("buy", "card", 0), path[index + 1 :])

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

    def test_safe_shop_blocks_sampler_only_reroll_value(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            money=8,
            required_score=300,
            jokers=(Joker("Gros Michel", metadata=joker_effect("+15 Mult")),),
        )

        value = shop_action_search_value(
            state,
            Action(ActionType.REROLL),
            sampler=FixedRerollSampler(),  # type: ignore[arg-type]
            config=ShopSearchConfig(reroll_samples=1),
        )

        self.assertLessEqual(value, 0.0)

    def test_shop_leaf_value_rewards_next_round_interest_breakpoints(self) -> None:
        just_short = GameState(phase=GamePhase.SHOP, ante=2, money=24)
        at_cap = GameState(phase=GamePhase.SHOP, ante=2, money=25)

        self.assertGreater(shop_leaf_value(at_cap) - shop_leaf_value(just_short), 2.0)

    def test_shop_money_floor_targets_interest_breakpoints(self) -> None:
        targets = [
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=1)),
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=2)),
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=3)),
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=4)),
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=6)),
        ]

        self.assertEqual(targets, [10, 20, 25, 25, 25])

    def test_shop_money_floor_tracks_raised_interest_cap_vouchers(self) -> None:
        seed_money_targets = [
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=4, vouchers=("Seed Money",))),
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=5, vouchers=("Seed Money",))),
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=6, vouchers=("Seed Money",))),
        ]
        money_tree_targets = [
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=4, vouchers=("Money Tree",))),
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=5, vouchers=("Money Tree",))),
            _minimum_safe_shop_money(GameState(phase=GamePhase.SHOP, ante=7, vouchers=("Money Tree",))),
        ]

        self.assertEqual(seed_money_targets, [35, 40, 50])
        self.assertEqual(money_tree_targets, [45, 65, 100])

    def test_safe_shop_money_floor_keeps_raised_interest_cap(self) -> None:
        below_seed_cap = GameState(
            phase=GamePhase.SHOP,
            ante=8,
            blind="Small Blind",
            required_score=5000,
            money=30,
            vouchers=("Seed Money",),
            jokers=(Joker("Gros Michel", metadata=joker_effect("+15 Mult")), Joker("Card Sharp"), Joker("Stuntman")),
        )
        at_seed_cap = GameState(
            phase=GamePhase.SHOP,
            ante=8,
            blind="Small Blind",
            required_score=5000,
            money=50,
            vouchers=("Seed Money",),
            jokers=(Joker("Gros Michel", metadata=joker_effect("+15 Mult")), Joker("Card Sharp"), Joker("Stuntman")),
        )

        self.assertGreater(_shop_money_floor_penalty(below_seed_cap), 0.0)
        self.assertEqual(_shop_money_floor_penalty(at_seed_cap), 0.0)

    def test_shop_leaf_value_penalizes_ending_later_shops_broke(self) -> None:
        broke = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=3,
            jokers=(Joker("Gros Michel", metadata=joker_effect("+15 Mult")),),
        )
        banked = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=16,
            jokers=(Joker("Gros Michel", metadata=joker_effect("+15 Mult")),),
        )

        self.assertGreater(shop_leaf_value(banked) - shop_leaf_value(broke), 35.0)

    def test_strong_economy_joker_discounts_safe_money_floor_shortfall(self) -> None:
        without_econ = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=1200,
            hands_remaining=4,
            deck_size=40,
            money=16,
            jokers=(Joker("Gros Michel", metadata=joker_effect("+15 Mult")), Joker("Card Sharp"), Joker("Stuntman")),
        )
        with_econ = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=1200,
            hands_remaining=4,
            deck_size=40,
            money=16,
            jokers=(
                Joker("Gros Michel", metadata=joker_effect("+15 Mult")),
                Joker("Card Sharp"),
                Joker("Stuntman"),
                Joker("Rocket"),
            ),
        )

        self.assertLess(_shop_money_floor_penalty(with_econ), _shop_money_floor_penalty(without_econ))

    def test_shop_leaf_terms_total_matches_leaf_value(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=16,
            jokers=(Joker("Joker"), Joker("Card Sharp")),
        )

        terms = shop_leaf_terms(state)

        self.assertAlmostEqual(terms.total, shop_leaf_value(state))
        self.assertIn("total", terms.to_trace_dict())
        self.assertIn("build_score", terms.to_trace_dict())
        self.assertIn("capacity_ratio", terms.to_trace_dict())
        self.assertIn("clear_gate_penalty", terms.to_trace_dict())

    def test_shop_leaf_clear_gate_penalizes_value_before_capacity(self) -> None:
        root = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            hands_remaining=4,
            deck_size=40,
            money=15,
            jokers=(Joker("Joker"),),
        )
        value_leaf = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            hands_remaining=4,
            deck_size=40,
            money=12,
            jokers=(Joker("Joker"),),
            consumables=("Justice",),
        )
        scoring_leaf = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            hands_remaining=4,
            deck_size=40,
            money=11,
            jokers=(Joker("Joker"), Joker("Card Sharp")),
        )

        value_terms = shop_leaf_terms(value_leaf, root_state=root)
        scoring_terms = shop_leaf_terms(scoring_leaf, root_state=root)

        self.assertGreater(value_terms.clear_gate_penalty, 50.0)
        self.assertLess(scoring_terms.clear_gate_penalty, value_terms.clear_gate_penalty)
        self.assertGreater(scoring_terms.total, value_terms.total)

    def test_shop_leaf_values_held_justice_above_emperor(self) -> None:
        justice = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=20,
            consumables=("Justice",),
        )
        emperor = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=20,
            consumables=("The Emperor",),
        )

        self.assertGreater(shop_leaf_terms(justice).consumable_value, shop_leaf_terms(emperor).consumable_value)

    def test_shop_leaf_values_held_hermit_above_temperance(self) -> None:
        hermit = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            money=10,
            consumables=("The Hermit",),
        )
        temperance = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            money=10,
            consumables=("Temperance",),
        )

        self.assertGreater(shop_leaf_terms(hermit).consumable_value, shop_leaf_terms(temperance).consumable_value)

    def test_shop_capacity_headroom_rewards_comfortable_cushion(self) -> None:
        # Headroom is cushion *beyond* pressure.target_score, which is the
        # safety-inflated planning target (raw target x shop_target_safety
        # multiplier). The safety base was raised 1.15 -> 1.30 (validated
        # winrate win), lifting the cushion bar: a build now needs capacity
        # above ~2.6x the raw required score before it registers a cushion.
        # The strong build therefore has to be genuinely dominant; a weak
        # single-joker build clears nowhere near the target and gets 0.
        weak = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            hands_remaining=4,
            deck_size=40,
            money=20,
            jokers=(Joker("Joker"),),
        )
        strong = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            hands_remaining=4,
            deck_size=40,
            money=20,
            jokers=(
                Joker("Card Sharp"),
                Joker("Stuntman"),
                Joker("Baron"),
                Joker("Blueprint"),
                Joker("The Duo"),
            ),
        )

        self.assertGreater(shop_capacity_headroom_value(strong), shop_capacity_headroom_value(weak))
        self.assertGreater(shop_leaf_terms(strong).headroom_value, 0.0)

    def test_shop_owned_value_is_pressure_aware(self) -> None:
        weak_broke = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            hands_remaining=4,
            deck_size=40,
            money=2,
            jokers=(Joker("Driver's License"),),
        )
        comfortable = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            hands_remaining=4,
            deck_size=40,
            money=24,
            jokers=(Joker("Driver's License"), Joker("Card Sharp"), Joker("Stuntman")),
        )

        raw_owned = 300.0

        self.assertLess(shop_owned_joker_leaf_value(weak_broke, raw_owned), raw_owned * 0.45)
        self.assertGreater(
            shop_owned_joker_leaf_value(comfortable, raw_owned),
            shop_owned_joker_leaf_value(weak_broke, raw_owned),
        )

    def test_shop_leaf_trace_exposes_raw_owned_value(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=16,
            jokers=(Joker("Joker"), Joker("Card Sharp")),
        )

        trace = shop_leaf_terms(state).to_trace_dict()

        self.assertIn("owned_raw", trace)
        self.assertGreaterEqual(trace["owned_raw"], trace["owned"])

    def test_shop_leaf_penalizes_conditional_chip_stack_without_stable_mult(self) -> None:
        weak = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Big Blind",
            required_score=1200,
            hands_remaining=4,
            discards_remaining=3,
            money=8,
            jokers=(Joker("Clever Joker"), Joker("Square Joker")),
            modifiers={"boss": {"name": "The Water", "score": 1600}},
        )
        supported = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Big Blind",
            required_score=1200,
            hands_remaining=4,
            discards_remaining=3,
            money=8,
            jokers=(Joker("Clever Joker"), Joker("Square Joker"), Joker("Gros Michel")),
            modifiers={"boss": {"name": "The Water", "score": 1600}},
        )

        weak_terms = shop_leaf_terms(weak)
        supported_terms = shop_leaf_terms(supported)

        self.assertIn("conditional_penalty", weak_terms.to_trace_dict())
        self.assertGreater(weak_terms.conditional_penalty, 80.0)
        self.assertLess(supported_terms.conditional_penalty, weak_terms.conditional_penalty)

    def test_shop_leaf_can_penalize_unsupported_rare_hand_commitment(self) -> None:
        weak = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            money=18,
            jokers=(Joker("The Family"),),
        )
        supported = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            money=18,
            jokers=(Joker("The Family"),),
            consumables=("Strength", "Death", "The Hanged Man"),
        )

        with patch("balatro_ai.search.shop_search._RARE_HAND_LEAF_PENALTY_WEIGHT", 1.0):
            weak_terms = shop_leaf_terms(weak)
            supported_terms = shop_leaf_terms(supported)

        self.assertIn("rare_hand_penalty", weak_terms.to_trace_dict())
        self.assertGreater(weak_terms.rare_hand_penalty, 80.0)
        self.assertLess(supported_terms.rare_hand_penalty, weak_terms.rare_hand_penalty)

    def test_rare_hand_commitment_reliability_discounts_uncommitted_family(self) -> None:
        early = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Big Blind",
            required_score=1200,
            jokers=(Joker("The Family"),),
        )
        weak = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            jokers=(Joker("The Family"),),
        )
        supported = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            jokers=(Joker("The Family"),),
            hand_levels={"Four of a Kind": 3},
            consumables=("Strength",),
        )

        with patch("balatro_ai.bots.basic_strategy.shop_jokers._RARE_HAND_COMMITMENT_RELIABILITY_WEIGHT", 1.0):
            self.assertEqual(_rare_hand_commitment_reliability(early, early.jokers[0]), 1.0)
            self.assertLess(_rare_hand_commitment_reliability(weak, weak.jokers[0]), 0.20)
            self.assertEqual(_rare_hand_commitment_reliability(supported, supported.jokers[0]), 1.0)

    def test_stencil_slot_discipline_penalizes_filling_protected_slots(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            jokers=(Joker("Blue Joker"), Joker("Joker Stencil")),
        )

        self.assertEqual(_stencil_slot_discipline_penalty(state, Joker("Greedy Joker")), 0.0)
        with patch("balatro_ai.bots.basic_strategy.shop_jokers._STENCIL_SLOT_DISCIPLINE_WEIGHT", 1.0):
            self.assertGreater(_stencil_slot_discipline_penalty(state, Joker("Greedy Joker")), 20.0)
            self.assertEqual(_stencil_slot_discipline_penalty(state, Joker("Joker Stencil")), 0.0)

    def test_shop_action_value_prefers_stable_mult_to_more_conditional_chips(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=1200,
            hands_remaining=4,
            discards_remaining=3,
            money=10,
            jokers=(Joker("Square Joker"),),
            modifiers={
                "boss": {"name": "The Water", "score": 1600},
                "shop_cards": (shop_card("Crafty Joker", cost=4), shop_card("Gros Michel", cost=4)),
            },
        )
        crafty = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        gros_michel = Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1})

        self.assertLess(
            shop_action_search_value(state, crafty),
            shop_action_search_value(state, gros_michel),
        )

    def test_shop_action_value_penalizes_unresolved_pressure_joker_buy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            hands_remaining=4,
            deck_size=40,
            money=9,
            jokers=(Joker("Chaos the Clown"), Joker("Square Joker")),
            modifiers={
                "joker_slots": 5,
                "shop_cards": (shop_card("Driver's License", cost=7), shop_card("Drunkard", cost=4)),
            },
        )

        driver = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})

        self.assertLess(shop_action_search_value(state, driver), 40.0)

    def test_shop_consumable_action_value_can_avoid_leaf_delta_double_count(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            blind="Small Blind",
            required_score=600,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            money=7,
            consumables=("Saturn",),
            hand_levels={"Straight": 1},
        )
        action = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )

        with (
            patch("balatro_ai.search.shop_search._USE_CONSUMABLE_LEAF_DELTA_WEIGHT", 0.55),
            patch("balatro_ai.search.shop_search._USE_CONSUMABLE_BONUS_WEIGHT", 0.0),
        ):
            full_delta_value = shop_action_search_value(state, action)
        with (
            patch("balatro_ai.search.shop_search._USE_CONSUMABLE_LEAF_DELTA_WEIGHT", 0.0),
            patch("balatro_ai.search.shop_search._USE_CONSUMABLE_BONUS_WEIGHT", 0.55),
        ):
            bonus_only_value = shop_action_search_value(state, action)

        self.assertGreater(full_delta_value, bonus_only_value)
        self.assertGreater(bonus_only_value, 0.0)

    def test_use_consumable_potential_shaping_is_ablatable(self) -> None:
        state = GameState(phase=GamePhase.SHOP, money=7, consumables=("Saturn",))
        node = _BeamNode(
            state=state,
            first_action=None,
            path=(),
            action_score=0.0,
            leaf_score=0.0,
            score=0.0,
        )
        action = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=0,
            metadata={"kind": "consumable", "index": 0},
        )

        with patch("balatro_ai.search.shop_search._USE_CONSUMABLE_POTENTIAL_WEIGHT", 1.0):
            shaped = _action_potential_shaping(
                node,
                action,
                10.0,
                leaf_value_fn=lambda _state: 2.0,
                leaf_terms_fn=None,
                weight=0.65,
            )
        with patch("balatro_ai.search.shop_search._USE_CONSUMABLE_POTENTIAL_WEIGHT", 0.0):
            disabled = _action_potential_shaping(
                node,
                action,
                10.0,
                leaf_value_fn=lambda _state: 2.0,
                leaf_terms_fn=None,
                weight=0.65,
            )

        self.assertAlmostEqual(shaped, 5.2)
        self.assertEqual(disabled, 0.0)

    def test_shop_build_capacity_delta_rewards_score_growth(self) -> None:
        root = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            money=20,
            jokers=(Joker("Joker"),),
        )
        upgraded = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            money=16,
            jokers=(Joker("Joker"), Joker("Card Sharp")),
        )

        self.assertGreater(shop_build_capacity_delta_value(upgraded, root_state=root), 0.0)
        self.assertLess(shop_build_capacity_delta_value(root, root_state=upgraded), 0.0)

    def test_default_shop_leaf_can_prefer_capacity_upgrade(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            money=12,
            jokers=(Joker("Joker"),),
            modifiers={
                "joker_slots": 5,
                "shop_cards": (shop_card("Joker", cost=4), shop_card("Card Sharp", cost=4)),
                "reroll_cost": 99,
            },
        )

        action = best_shop_action(
            state,
            config=ShopSearchConfig(depth=1, beam_width=4, leaf_weight=1.0),
            sampler=tiny_sampler(),
            action_value_fn=lambda _state, _candidate: 0.0,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 1)

    def test_shop_survival_value_rewards_actual_scoring_capacity(self) -> None:
        weak_cash = GameState(phase=GamePhase.SHOP, ante=4, blind="The Wall", required_score=5000, money=50)
        strong_build = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="The Wall",
            required_score=5000,
            money=10,
            jokers=(Joker("Gros Michel", metadata=joker_effect("+15 Mult")), Joker("Blue Joker")),
        )

        self.assertGreater(shop_survival_value(strong_build), shop_survival_value(weak_cash))


if __name__ == "__main__":
    unittest.main()
