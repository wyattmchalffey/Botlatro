from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, with_derived_legal_actions
from balatro_ai.search.shop_sampler import ShopSampler
from balatro_ai.sim.local_runner import LocalBalatroSimulator, LocalSimOptions, run_local_seed


def tiny_sim_data() -> dict[str, object]:
    return {
        "shop_slots": {"base_joker_max": 2, "booster_slots": 1, "voucher_slots": 1},
        "slot_rates": {"Joker": 1},
        "joker_rarity_rates": {"common": 1.0, "uncommon": 0.0, "rare": 0.0, "legendary": 0.0},
        "edition_rates": {"negative": 0.0, "polychrome": 0.0, "holographic": 0.0, "foil": 0.0},
        "reroll": {"base_cost": 5, "increase_per_reroll": 1},
        "jokers": [
            {"key": "j_joker", "name": "Joker", "set": "Joker", "cost": 2, "rarity_name": "common"},
            {"key": "j_greedy_joker", "name": "Greedy Joker", "set": "Joker", "cost": 5, "rarity_name": "common"},
        ],
        "tarots": [{"key": "c_empress", "name": "The Empress", "set": "Tarot", "cost": 3}],
        "planets": [{"key": "c_jupiter", "name": "Jupiter", "set": "Planet", "cost": 3}],
        "spectrals": [{"key": "c_hex", "name": "Hex", "set": "Spectral", "cost": 4}],
        "vouchers": [{"key": "v_blank", "name": "Blank", "set": "Voucher", "cost": 10, "available_default": True}],
        "boosters": [
            {
                "key": "p_buffoon_normal_1",
                "name": "Buffoon Pack",
                "set": "Booster",
                "kind": "Buffoon",
                "config": {"extra": 2, "choose": 1},
                "cost": 4,
                "weight": 1,
            }
        ],
    }


class ScriptedClearBot:
    name = "scripted_clear_bot"

    def choose_action(self, state: GameState) -> Action:
        if state.phase == GamePhase.BLIND_SELECT:
            return _first_action(state, ActionType.SELECT_BLIND)
        if state.phase == GamePhase.SELECTING_HAND:
            return Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4))
        if state.phase == GamePhase.ROUND_EVAL:
            return _first_action(state, ActionType.CASH_OUT)
        if state.phase == GamePhase.SHOP:
            return _first_action(state, ActionType.END_SHOP)
        return state.legal_actions[0]


def _first_action(state: GameState, action_type: ActionType) -> Action:
    for action in state.legal_actions:
        if action.action_type == action_type:
            return action
    raise AssertionError(f"No {action_type.value} action in {state.legal_actions}")


class FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


class LocalRunnerTests(unittest.TestCase):
    def test_reset_starts_at_ante_one_small_blind_with_exact_deck(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))

        state = sim.reset()

        self.assertEqual(state.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(state.ante, 1)
        self.assertEqual(state.blind, "Small Blind")
        self.assertEqual(state.required_score, 300)
        self.assertEqual(state.money, 4)
        self.assertEqual(state.deck_size, 52)
        self.assertEqual(len(state.known_deck), 52)
        self.assertEqual(state.modifiers["current_blind"]["type"], "SMALL")
        self.assertIn(ActionType.SELECT_BLIND, {action.action_type for action in state.legal_actions})

    def test_needle_uses_source_truth_one_x_score_multiplier(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Needle",))

        state = sim.reset()

        self.assertEqual(state.modifiers["blinds"]["boss"]["name"], "The Needle")
        self.assertEqual(state.modifiers["blinds"]["boss"]["score"], 300)

    def test_select_blind_draws_opening_hand_and_reduces_deck(self) -> None:
        sim = LocalBalatroSimulator(seed=2, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        state = sim.reset()

        state = sim.step(_first_action(state, ActionType.SELECT_BLIND))

        self.assertEqual(state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(len(state.hand), 8)
        self.assertEqual(state.deck_size, 44)
        self.assertEqual(len(state.known_deck), 44)
        self.assertGreater(len([action for action in state.legal_actions if action.action_type == ActionType.PLAY_HAND]), 0)

    def test_clear_cash_out_creates_shop_then_end_shop_advances_to_big_blind(self) -> None:
        deck = tuple(Card("A", "S") for _ in range(52))
        sim = LocalBalatroSimulator(
            seed=3,
            sampler=ShopSampler(tiny_sim_data()),
            initial_deck=deck,
            boss_pool=("The Club",),
        )
        state = sim.reset()
        state = sim.step(_first_action(state, ActionType.SELECT_BLIND))

        state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)))
        self.assertEqual(state.phase, GamePhase.ROUND_EVAL)
        self.assertGreaterEqual(state.current_score, state.required_score)

        state = sim.step(_first_action(state, ActionType.CASH_OUT))
        self.assertEqual(state.phase, GamePhase.SHOP)
        self.assertEqual(state.blind, "Small Blind")
        self.assertEqual(state.required_score, 300)
        self.assertGreater(state.money, 4)
        self.assertTrue(state.modifiers["shop_cards"])
        self.assertTrue(state.modifiers["booster_packs"])

        state = sim.step(_first_action(state, ActionType.END_SHOP))
        self.assertEqual(state.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(state.blind, "Big Blind")
        self.assertEqual(state.required_score, 450)

    def test_buffoon_pack_contents_are_sampled_as_jokers(self) -> None:
        sampler = ShopSampler(tiny_sim_data())
        pack = {
            "key": "p_buffoon_normal_1",
            "name": "Buffoon Pack",
            "set": "Booster",
            "kind": "Buffoon",
            "config": {"extra": 2, "choose": 1},
        }

        contents = sampler.sample_pack_contents(GameState(), pack)

        self.assertEqual(len(contents), 2)
        self.assertEqual({item["set"] for item in contents}, {"JOKER"})

    def test_derived_shop_actions_honor_visible_reroll_cost(self) -> None:
        state = with_derived_legal_actions(GameState(phase=GamePhase.SHOP, money=5, modifiers={"reroll_cost": 6}))

        self.assertNotIn(ActionType.REROLL, {action.action_type for action in state.legal_actions})

        free_state = with_derived_legal_actions(
            GameState(phase=GamePhase.SHOP, money=0, modifiers={"reroll_cost": 6, "free_rerolls": 1})
        )
        self.assertIn(ActionType.REROLL, {action.action_type for action in free_state.legal_actions})

    def test_buying_overstock_refills_prior_empty_shop_slots(self) -> None:
        data = tiny_sim_data()
        data["vouchers"] = [{"key": "v_overstock_norm", "name": "Overstock", "set": "Voucher", "cost": 10, "available_default": True}]
        data["jokers"] = [
            {"key": "j_visible", "name": "Visible Joker", "set": "Joker", "cost": 2, "rarity_name": "common"},
            {"key": "j_fill_a", "name": "Fill A", "set": "Joker", "cost": 2, "rarity_name": "common"},
            {"key": "j_fill_b", "name": "Fill B", "set": "Joker", "cost": 2, "rarity_name": "common"},
        ]
        sim = LocalBalatroSimulator(seed=5, sampler=ShopSampler(data), boss_pool=("The Club",))
        visible = {"key": "j_visible", "name": "Visible Joker", "set": "Joker", "cost": {"buy": 2}}
        sim.state = GameState(
            phase=GamePhase.SHOP,
            money=20,
            modifiers={
                "shop_cards": (visible,),
                "voucher_cards": ({"key": "v_overstock_norm", "name": "Overstock", "set": "Voucher", "cost": {"buy": 10}},),
            },
        )

        next_state = sim.step(Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}))

        shop_cards = next_state.modifiers["shop_cards"]
        self.assertEqual(len(shop_cards), 3)
        self.assertIs(shop_cards[0], visible)
        self.assertEqual(next_state.shop[0], "Visible Joker")

    def test_hook_play_samples_held_discards_and_refills_hand(self) -> None:
        sim = LocalBalatroSimulator(seed=6, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Hook",))
        sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Hook",
            required_score=10_000,
            hands_remaining=4,
            deck_size=3,
            known_deck=(Card("9", "S"), Card("8", "D"), Card("7", "C")),
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D"), Card("J", "C")),
            modifiers={"hand_size": 4},
        )

        next_state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(next_state.deck_size, 0)
        self.assertEqual(len(next_state.hand), 4)
        self.assertEqual(tuple(card.short_name for card in next_state.known_deck), ())
        self.assertEqual(len(next_state.modifiers["discard_pile"]), 2)

    def test_misprint_play_samples_random_mult(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"),),
            jokers=(Joker("Misprint"),),
            modifiers={"hand_size": 1},
        )

        next_state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(next_state.current_score, 80)

    def test_local_runner_counts_blueprint_copied_rng_jokers(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"),),
            jokers=(Joker("Blueprint"), Joker("Space Joker")),
            modifiers={"hand_size": 1, "probability_multiplier": 100},
        )

        next_state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.current_score, 108)

    def test_oops_probability_multiplier_makes_wheel_boss_flips_guaranteed(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Wheel",))
        sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The Wheel",
            deck_size=3,
            known_deck=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            hands_remaining=4,
            discards_remaining=4,
            jokers=(Joker("Oops! All 6s"),),
            modifiers={
                "current_blind": {"type": "BOSS", "name": "The Wheel"},
                "hand_size": 3,
                "probability_multiplier": 7,
            },
        )

        next_state = sim.step(Action(ActionType.SELECT_BLIND))

        self.assertEqual(len(next_state.hand), 3)
        self.assertTrue(all(card.metadata.get("face_down") for card in next_state.hand))
        self.assertTrue(all(card.metadata.get("state", {}).get("facing") == "back" for card in next_state.hand))

    def test_oops_probability_multiplier_makes_wheel_of_fortune_guaranteed(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("The Wheel of Fortune",),
            jokers=(Joker("Joker", sell_value=1),),
            modifiers={
                "pack_cards": ({"label": "The Wheel of Fortune", "set": "TAROT"},),
                "probability_multiplier": 4,
            },
        )

        next_state = sim.step(
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        )

        self.assertIn(next_state.jokers[0].edition, {"FOIL", "HOLOGRAPHIC", "POLYCHROME"})
        self.assertEqual(next_state.modifiers["tarot_cards_used"]["The Wheel of Fortune"], 1)

    def test_derived_actions_allow_stored_tarot_use_and_consumable_sell_in_blind_and_pack(self) -> None:
        blind_state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.SELECTING_HAND,
                hand=(Card("A", "S"), Card("K", "H")),
                consumables=("The Empress",),
            )
        )
        blind_use_actions = [action for action in blind_state.legal_actions if action.action_type == ActionType.USE_CONSUMABLE]
        blind_sell_actions = [
            action
            for action in blind_state.legal_actions
            if action.action_type == ActionType.SELL and action.metadata.get("kind") == "consumable"
        ]

        self.assertIn((0, 1), {action.card_indices for action in blind_use_actions})
        self.assertEqual(len(blind_sell_actions), 1)

        pack_state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.BOOSTER_OPENED,
                pack=("8S",),
                hand=(Card("A", "S"), Card("K", "H")),
                consumables=("Death",),
                modifiers={"pack_cards": ({"rank": "8", "suit": "S", "set": "Base"},)},
            )
        )
        pack_use_actions = [action for action in pack_state.legal_actions if action.action_type == ActionType.USE_CONSUMABLE]

        self.assertIn((0, 1), {action.card_indices for action in pack_use_actions})
        self.assertTrue(any(action.action_type == ActionType.CHOOSE_PACK_CARD for action in pack_state.legal_actions))

    def test_wheel_of_fortune_edition_poll_uses_source_truth_guaranteed_odds(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))

        for poll, expected in (
            (0.49, "FOIL"),
            (0.50, "FOIL"),
            (0.51, "HOLOGRAPHIC"),
            (0.85, "HOLOGRAPHIC"),
            (0.86, "POLYCHROME"),
            (0.99, "POLYCHROME"),
        ):
            sim._rng = FixedRandom(poll)  # type: ignore[assignment]
            self.assertEqual(sim._wheel_of_fortune_edition(), expected)

    def test_crimson_heart_rotates_disabled_joker_after_play(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("Crimson Heart",))
        sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Crimson Heart",
            required_score=10_000,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"),),
            jokers=(
                Joker("Joker", metadata={"state": {"debuff": True}, "value": {"effect": "All abilities are disabled"}}),
                Joker("Stuntman", metadata={"value": {"effect": "+250 Chips, -2 hand size"}}),
            ),
            modifiers={"hand_size": 1},
        )

        next_state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.current_score, 266)
        self.assertFalse(next_state.jokers[0].metadata.get("state"))
        self.assertEqual(next_state.jokers[1].metadata["state"]["debuff"], True)

    def test_run_local_seed_records_local_result(self) -> None:
        deck = tuple(Card("A", "S") for _ in range(52))

        result = run_local_seed(
            bot=ScriptedClearBot(),
            options=LocalSimOptions(seed=4, max_steps=4, initial_deck=deck, boss_pool=("The Club",)),
        )

        self.assertFalse(result.won)
        self.assertEqual(result.ante_reached, 1)
        self.assertEqual(result.death_reason, "max_steps")
        self.assertLess(result.runtime_seconds, 1.0)


if __name__ == "__main__":
    unittest.main()
