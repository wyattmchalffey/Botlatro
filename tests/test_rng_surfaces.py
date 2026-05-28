from __future__ import annotations

import json
from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.rng import BalatroRNG
from balatro_ai.rng.surfaces import (
    BOSS_NAMES,
    TAG_POOL,
    poll_edition,
    predict_initial_surface,
    predict_ouija_rank,
    predict_pack_contents,
    predict_shop_cards,
    predict_shop_surface,
    predict_sigil_suit,
    predict_spectral_created_cards,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"
TAG_NAMES = {
    "tag_uncommon": "Uncommon Tag",
    "tag_rare": "Rare Tag",
    "tag_negative": "Negative Tag",
    "tag_foil": "Foil Tag",
    "tag_holo": "Holographic Tag",
    "tag_polychrome": "Polychrome Tag",
    "tag_investment": "Investment Tag",
    "tag_voucher": "Voucher Tag",
    "tag_boss": "Boss Tag",
    "tag_standard": "Standard Tag",
    "tag_charm": "Charm Tag",
    "tag_meteor": "Meteor Tag",
    "tag_buffoon": "Buffoon Tag",
    "tag_handy": "Handy Tag",
    "tag_garbage": "Garbage Tag",
    "tag_ethereal": "Ethereal Tag",
    "tag_coupon": "Coupon Tag",
    "tag_double": "Double Tag",
    "tag_juggle": "Juggle Tag",
    "tag_d_six": "D6 Tag",
    "tag_top_up": "Top-up Tag",
    "tag_skip": "Skip Tag",
    "tag_orbital": "Orbital Tag",
    "tag_economy": "Economy Tag",
}


def _shop_fixture(seed: str) -> dict | None:
    path = FIXTURE_DIR / f"shop_seed_{seed}_red_white.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class FirstShopSurfaceFixtureTests(unittest.TestCase):
    def _check_seed(self, seed: str) -> None:
        fixture = _shop_fixture(seed)
        if fixture is None:
            self.skipTest(f"No fixture for {seed}; run rng.capture_shop --all")
        surface = predict_shop_surface(seed)
        assert surface.setup is not None

        self.assertEqual(BOSS_NAMES[surface.setup.boss_key], fixture["blinds"]["boss"]["name"])
        self.assertEqual(TAG_NAMES[surface.setup.small_tag_key], fixture["blinds"]["small"]["tag_name"])
        self.assertEqual(TAG_NAMES[surface.setup.big_tag_key], fixture["blinds"]["big"]["tag_name"])
        self.assertEqual(surface.voucher_key, fixture["vouchers"]["cards"][0]["key"])
        self.assertEqual(tuple(card.key for card in surface.shop_cards), tuple(card["key"] for card in fixture["shop"]["cards"]))
        self.assertEqual(surface.booster_keys, tuple(card["key"] for card in fixture["packs"]["cards"]))

    def test_seed_aaaaaaa(self) -> None:
        self._check_seed("AAAAAAA")

    def test_seed_bbbbbbb(self) -> None:
        self._check_seed("BBBBBBB")

    def test_seed_ccccccc(self) -> None:
        self._check_seed("CCCCCCC")

    def test_seed_1234567(self) -> None:
        self._check_seed("1234567")


class SurfacePredictionTests(unittest.TestCase):
    def test_initial_surface_consumes_tag_stream_in_order(self) -> None:
        setup = predict_initial_surface("AAAAAAA")
        self.assertEqual(setup.small_tag_key, "tag_boss")
        self.assertEqual(setup.big_tag_key, "tag_holo")
        self.assertIn(setup.small_tag_key, TAG_POOL)
        self.assertIn(setup.big_tag_key, TAG_POOL)

    def test_ante_two_shop_can_reuse_run_rng_after_ante_one(self) -> None:
        seed = "AAAAAAA"
        rng = BalatroRNG(seed)
        _ = predict_initial_surface(rng, ante=1)
        _ = predict_shop_cards(rng, ante=1)

        carried_ante_two = predict_shop_cards(rng, ante=2)
        fresh_ante_two = predict_shop_cards(seed, ante=2)

        self.assertEqual(carried_ante_two, fresh_ante_two)

    def test_repeated_same_ante_shop_predictions_advance_same_keys(self) -> None:
        rng = BalatroRNG("AAAAAAA")
        first = predict_shop_cards(rng, ante=2)
        second = predict_shop_cards(rng, ante=2)
        self.assertNotEqual(first, second)

    def test_shop_sticker_polls_respect_joker_compatibility(self) -> None:
        rng = BalatroRNG("AAAAAAA")
        _ = predict_initial_surface(rng, ante=1)
        _ = predict_shop_cards(rng, ante=1, enable_eternals=True, enable_perishables=True, enable_rentals=True)
        _ = predict_shop_cards(rng, ante=1, enable_eternals=True, enable_perishables=True, enable_rentals=True)

        cards = predict_shop_cards(rng, ante=2, enable_eternals=True, enable_perishables=True, enable_rentals=True)
        self.assertEqual(cards[1].key, "j_runner")
        self.assertFalse(cards[1].perishable)

    def test_pack_prediction_supports_all_pack_kinds(self) -> None:
        rng = BalatroRNG("AAAAAAA")
        cases = (
            ("p_buffoon_normal_1", "Joker"),
            ("p_celestial_normal_1", "Planet"),
            ("p_arcana_normal_1", "Tarot"),
            ("p_spectral_normal_1", "Spectral"),
            ("p_standard_normal_1", "Default"),
        )
        for pack_key, expected_set in cases:
            with self.subTest(pack_key=pack_key):
                cards = predict_pack_contents(rng, ante=2, pack_key=pack_key)
                self.assertGreater(len(cards), 0)
                self.assertEqual(cards[0].set, expected_set if expected_set != "Default" else cards[0].set)
                if pack_key == "p_standard_normal_1":
                    self.assertIsNotNone(cards[0].front_key)

    def test_edition_poll_and_per_card_helpers_are_deterministic(self) -> None:
        self.assertEqual(poll_edition("AAAAAAA", "fixed", guaranteed=True), poll_edition("AAAAAAA", "fixed", guaranteed=True))
        self.assertEqual(predict_sigil_suit("AAAAAAA"), predict_sigil_suit("AAAAAAA"))
        self.assertEqual(predict_ouija_rank("AAAAAAA"), predict_ouija_rank("AAAAAAA"))
        self.assertEqual(
            predict_spectral_created_cards("AAAAAAA", "c_familiar"),
            predict_spectral_created_cards("AAAAAAA", "c_familiar"),
        )


if __name__ == "__main__":
    unittest.main()
