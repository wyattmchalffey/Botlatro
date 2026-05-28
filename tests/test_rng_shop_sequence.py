from __future__ import annotations

from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.rng.capture_shop_sequence import shop_sequence_fixture_path
from balatro_ai.rng.validate_shop_sequence import (
    _shop_rates_for_vouchers,
    _shop_sticker_options,
    check_shop_sequence_fixture,
    load_shop_sequence_fixture,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"


class ShopSequenceValidationHelperTests(unittest.TestCase):
    def test_shop_sequence_fixture_path_uses_seed_deck_and_stake(self) -> None:
        path = shop_sequence_fixture_path("AAAAAAA", deck="RED", stake="white")
        self.assertEqual(path.name, "shop_sequence_seed_AAAAAAA_red_white.json")
        voucher_path = shop_sequence_fixture_path("AAAAAAA", deck="RED", stake="white", vouchers=("v_magic_trick",))
        self.assertEqual(voucher_path.name, "shop_sequence_seed_AAAAAAA_with_v_magic_trick_red_white.json")

    def test_missing_seed_is_unsupported(self) -> None:
        results = check_shop_sequence_fixture({"shops": []})
        self.assertEqual(results[0].status, "unsupported")

    def test_stake_enables_shop_sticker_polls(self) -> None:
        self.assertEqual(
            _shop_sticker_options("white"),
            {"enable_eternals": False, "enable_perishables": False, "enable_rentals": False},
        )
        self.assertEqual(
            _shop_sticker_options("black"),
            {"enable_eternals": True, "enable_perishables": False, "enable_rentals": False},
        )
        self.assertEqual(
            _shop_sticker_options("orange"),
            {"enable_eternals": True, "enable_perishables": True, "enable_rentals": False},
        )
        self.assertEqual(
            _shop_sticker_options("gold"),
            {"enable_eternals": True, "enable_perishables": True, "enable_rentals": True},
        )

    def test_voucher_shop_rates_enable_magic_trick_playing_cards(self) -> None:
        self.assertEqual(_shop_rates_for_vouchers(frozenset({"v_magic_trick"}))["playing_card"], 4)
        self.assertEqual(_shop_rates_for_vouchers(frozenset({"v_tarot_merchant"}))["Tarot"], 9.6)


class CapturedShopSequenceFixtureTests(unittest.TestCase):
    def test_captured_shop_sequences_match_predictions(self) -> None:
        paths = sorted(FIXTURE_DIR.glob("shop_sequence_seed_*.json"))
        if not paths:
            self.skipTest("No shop sequence fixtures; run `python -m balatro_ai.rng.capture_shop_sequence --all --shops 4`")
        for path in paths:
            with self.subTest(path=path.name):
                fixture = load_shop_sequence_fixture(path)
                results = check_shop_sequence_fixture(fixture)
                self.assertTrue(results)
                for result in results:
                    self.assertEqual(result.status, "ok", f"{path.name} {result.name}: {result.detail}")


if __name__ == "__main__":
    unittest.main()
