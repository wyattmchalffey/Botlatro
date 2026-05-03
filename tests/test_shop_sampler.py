from __future__ import annotations

from collections import Counter
from random import Random
import unittest

import context  # noqa: F401
from balatro_ai.api.state import GameState, Joker
from balatro_ai.search.shop_sampler import ShopSampler, load_shop_pool_data


def tiny_shop_data() -> dict[str, object]:
    return {
        "shop_slots": {"base_joker_max": 2, "booster_slots": 2, "voucher_slots": 1},
        "slot_rates": {"Joker": 20, "Tarot": 4, "Planet": 4, "Base": 0, "Spectral": 0},
        "joker_rarity_rates": {"common": 1.0, "uncommon": 0.0, "rare": 0.0, "legendary": 0.0},
        "edition_rates": {"negative": 0.003, "polychrome": 0.006, "holographic": 0.02, "foil": 0.04},
        "reroll": {"base_cost": 5, "increase_per_reroll": 1},
        "jokers": [
            {"key": "j_owned", "name": "Owned Joker", "set": "Joker", "cost": 4, "rarity_name": "common"},
            {"key": "j_other", "name": "Other Joker", "set": "Joker", "cost": 5, "rarity_name": "common"},
        ],
        "tarots": [{"key": "c_empress", "name": "The Empress", "set": "Tarot", "cost": 3}],
        "planets": [{"key": "c_jupiter", "name": "Jupiter", "set": "Planet", "cost": 3}],
        "spectrals": [{"key": "c_hex", "name": "Hex", "set": "Spectral", "cost": 4}],
        "vouchers": [
            {
                "key": "v_overstock_norm",
                "name": "Overstock",
                "set": "Voucher",
                "cost": 10,
                "available_default": True,
            },
            {
                "key": "v_overstock_plus",
                "name": "Overstock Plus",
                "set": "Voucher",
                "cost": 10,
                "available_default": True,
                "requires": ["v_overstock_norm"],
            },
        ],
        "boosters": [
            {"key": "p_buffoon_normal_1", "name": "Buffoon Pack", "set": "Booster", "kind": "Buffoon", "cost": 4, "weight": 1},
            {"key": "p_arcana_normal_1", "name": "Arcana Pack", "set": "Booster", "kind": "Arcana", "cost": 4, "weight": 3},
        ],
    }


class ShopSamplerTests(unittest.TestCase):
    def test_default_data_has_source_pools(self) -> None:
        data = load_shop_pool_data()

        self.assertEqual(len(data["jokers"]), 150)
        self.assertEqual(len(data["tarots"]), 22)
        self.assertEqual(len(data["planets"]), 12)
        self.assertEqual(len(data["spectrals"]), 18)
        self.assertEqual(len(data["vouchers"]), 32)
        self.assertEqual(len(data["boosters"]), 32)
        joker = next(item for item in data["jokers"] if item["key"] == "j_joker")
        self.assertEqual(joker["name"], "Joker")
        self.assertEqual(joker["rarity_name"], "common")
        self.assertEqual(joker["cost"], 2)

    def test_sample_shop_type_matches_default_slot_weights(self) -> None:
        sampler = ShopSampler.from_default_data()
        rng = Random(11)

        counts = Counter(sampler.sample_shop_type(GameState(), rng) for _ in range(5000))

        self.assertGreater(counts["Joker"] / 5000, 0.68)
        self.assertLess(counts["Joker"] / 5000, 0.75)
        self.assertGreater(counts["Tarot"] / 5000, 0.11)
        self.assertGreater(counts["Planet"] / 5000, 0.11)
        self.assertEqual(counts["Spectral"], 0)

    def test_joker_rarity_distribution_matches_source_thresholds(self) -> None:
        sampler = ShopSampler.from_default_data()
        state = GameState(modifiers={"allow_duplicate_jokers": True})
        rng = Random(17)

        counts = Counter(
            sampler.sample_card_of_type(state, "Joker", rng)["rarity_name"]
            for _ in range(6000)
        )

        self.assertGreater(counts["common"] / 6000, 0.66)
        self.assertLess(counts["common"] / 6000, 0.74)
        self.assertGreater(counts["uncommon"] / 6000, 0.21)
        self.assertLess(counts["uncommon"] / 6000, 0.29)
        self.assertGreater(counts["rare"] / 6000, 0.035)
        self.assertLess(counts["rare"] / 6000, 0.065)

    def test_owned_jokers_are_filtered_without_showman(self) -> None:
        sampler = ShopSampler(tiny_shop_data())
        state = GameState(jokers=(Joker("Owned Joker"),))
        rng = Random(5)

        sampled = tuple(sampler.sample_card_of_type(state, "Joker", rng)["name"] for _ in range(20))

        self.assertEqual(set(sampled), {"Other Joker"})

    def test_showman_allows_owned_jokers_to_reappear(self) -> None:
        data = tiny_shop_data()
        data["jokers"] = [{"key": "j_owned", "name": "Owned Joker", "set": "Joker", "cost": 4, "rarity_name": "common"}]
        sampler = ShopSampler(data)
        state = GameState(jokers=(Joker("Owned Joker"), Joker("Showman")))

        sampled = sampler.sample_card_of_type(state, "Joker", Random(3))

        self.assertEqual(sampled["name"], "Owned Joker")

    def test_voucher_requires_prior_voucher(self) -> None:
        sampler = ShopSampler(tiny_shop_data())

        first = sampler.sample_voucher(GameState(), Random(1))
        second = sampler.sample_voucher(GameState(vouchers=("Overstock",)), Random(1))

        self.assertEqual(first["name"], "Overstock")
        self.assertEqual(second["name"], "Overstock Plus")

    def test_voucher_requirements_understand_all_source_voucher_names(self) -> None:
        data = tiny_shop_data()
        data["vouchers"] = [
            {"key": "v_omen_globe", "name": "Omen Globe", "set": "Voucher", "cost": 10, "requires": ["v_crystal_ball"]},
        ]
        sampler = ShopSampler(data)

        self.assertIsNone(sampler.sample_voucher(GameState(), Random(1)))
        self.assertEqual(sampler.sample_voucher(GameState(vouchers=("Crystal Ball",)), Random(1))["name"], "Omen Globe")

    def test_shop_slot_count_honors_overstock_vouchers(self) -> None:
        sampler = ShopSampler(tiny_shop_data())

        self.assertEqual(sampler.shop_slot_count(GameState()), 2)
        self.assertEqual(sampler.shop_slot_count(GameState(vouchers=("Overstock",))), 3)
        self.assertEqual(sampler.shop_slot_count(GameState(vouchers=("Overstock", "Overstock Plus"))), 4)

    def test_fill_shop_to_slot_count_preserves_visible_cards_and_fills_empty_slots(self) -> None:
        data = tiny_shop_data()
        data["slot_rates"] = {"Joker": 1}
        data["jokers"] = [
            {"key": "j_other", "name": "Other Joker", "set": "Joker", "cost": 5, "rarity_name": "common"},
            {"key": "j_filler_a", "name": "Filler A", "set": "Joker", "cost": 4, "rarity_name": "common"},
            {"key": "j_filler_b", "name": "Filler B", "set": "Joker", "cost": 4, "rarity_name": "common"},
        ]
        sampler = ShopSampler(data)
        visible = ({"key": "j_other", "name": "Other Joker", "set": "Joker", "cost": {"buy": 5}},)

        filled = sampler.fill_shop_to_slot_count(GameState(vouchers=("Overstock",)), visible, Random(4))

        self.assertEqual(len(filled), 3)
        self.assertIs(filled[0], visible[0])
        self.assertEqual([item["name"] for item in filled].count("Other Joker"), 1)

    def test_reroll_cost_uses_free_and_visible_costs(self) -> None:
        sampler = ShopSampler(tiny_shop_data())

        self.assertEqual(sampler.reroll_cost(GameState(modifiers={"free_rerolls": 1, "reroll_cost": 7})), 0)
        self.assertEqual(sampler.reroll_cost(GameState(modifiers={"reroll_cost": 7})), 7)
        self.assertEqual(sampler.reroll_cost(GameState(modifiers={"base_reroll_cost": 3, "reroll_cost_increase": 2})), 5)

    def test_reroll_ev_uses_sampled_best_item_minus_reroll_cost(self) -> None:
        data = tiny_shop_data()
        data["slot_rates"] = {"Joker": 1}
        sampler = ShopSampler(data)
        state = GameState(money=10, modifiers={"reroll_cost": 3})

        seen_money: list[int] = []

        def value_after_reroll(post_state: GameState, item: dict[str, object]) -> float:
            seen_money.append(post_state.money)
            return 11.0 if item["name"] == "Other Joker" else 0.0

        ev = sampler.reroll_ev(
            state,
            samples=8,
            rng=Random(1),
            item_value_fn=value_after_reroll,
        )

        self.assertEqual(ev, 8.0)
        self.assertEqual(set(seen_money), {7})

    def test_reroll_ev_values_affordability_after_paying_reroll_cost(self) -> None:
        data = tiny_shop_data()
        data["slot_rates"] = {"Joker": 1}
        sampler = ShopSampler(data)
        state = GameState(money=5, modifiers={"reroll_cost": 5})

        def value_after_reroll(post_state: GameState, item: dict[str, object]) -> float:
            cost = item["cost"]
            assert isinstance(cost, dict)
            return 11.0 if int(cost["buy"]) <= post_state.money else 0.0

        ev = sampler.reroll_ev(
            state,
            samples=8,
            rng=Random(1),
            item_value_fn=value_after_reroll,
        )

        self.assertEqual(ev, -5.0)

    def test_reroll_ev_is_negative_when_no_sampled_item_has_value(self) -> None:
        sampler = ShopSampler(tiny_shop_data())
        state = GameState(modifiers={"reroll_cost": 5})

        ev = sampler.reroll_ev(state, samples=8, rng=Random(1), item_value_fn=lambda _state, _item: 0.0)

        self.assertEqual(ev, -5.0)

    def test_default_reroll_ev_smoke(self) -> None:
        sampler = ShopSampler.from_default_data()

        ev = sampler.reroll_ev(GameState(ante=1, money=20), samples=4, rng=Random(1))

        self.assertIsInstance(ev, float)

    def test_costs_include_discount_inflation_and_edition(self) -> None:
        sampler = ShopSampler.from_default_data()
        state = GameState(
            vouchers=("Clearance Sale",),
            modifiers={"inflation": 2, "allow_duplicate_jokers": True, "edition_rate": 100},
        )

        card = sampler.sample_card_of_type(state, "Joker", Random(647))

        base = int(card["cost"]["base"])
        edition_extra = {"FOIL": 2, "HOLOGRAPHIC": 3, "POLYCHROME": 5, "NEGATIVE": 5}[card["edition"]]
        expected = max(1, int(((base + 2 + edition_extra + 0.5) * 75) // 100))
        self.assertEqual(card["cost"]["buy"], expected)

    def test_omen_globe_can_put_spectral_cards_in_arcana_packs(self) -> None:
        data = tiny_shop_data()
        data["boosters"] = [{"key": "p_arcana_normal_1", "name": "Arcana Pack", "set": "Booster", "kind": "Arcana", "config": {"extra": 1}}]
        sampler = ShopSampler(data)

        contents = sampler.sample_pack_contents(GameState(vouchers=("Omen Globe",)), data["boosters"][0], Random(2))

        self.assertEqual(contents[0]["set"], "SPECTRAL")

    def test_telescope_forces_first_celestial_card_to_most_played_hand_planet(self) -> None:
        data = tiny_shop_data()
        data["boosters"] = [{"key": "p_celestial_normal_1", "name": "Celestial Pack", "set": "Booster", "kind": "Celestial", "config": {"extra": 2}}]
        data["planets"] = [
            {"key": "c_pluto", "name": "Pluto", "set": "Planet", "cost": 3, "config": {"hand_type": "High Card"}},
            {"key": "c_jupiter", "name": "Jupiter", "set": "Planet", "cost": 3, "config": {"hand_type": "Flush"}},
        ]
        state = GameState(
            vouchers=("Telescope",),
            modifiers={
                "hands": {
                    "High Card": {"played": 1, "order": 12},
                    "Flush": {"played": 3, "order": 7},
                }
            },
        )
        sampler = ShopSampler(data)

        contents = sampler.sample_pack_contents(state, data["boosters"][0], Random(1))

        self.assertEqual(contents[0]["name"], "Jupiter")


if __name__ == "__main__":
    unittest.main()
