from __future__ import annotations

from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.rng.capture_consumable_shop import consumable_shop_fixture_path
from balatro_ai.rng.validate_consumable_shop import (
    check_consumable_shop_fixture,
    load_consumable_shop_fixture,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"


class ConsumableShopHelperTests(unittest.TestCase):
    def test_fixture_path_uses_seed_deck_and_stake(self) -> None:
        path = consumable_shop_fixture_path("AAAAAAA", deck="RED", stake="white")
        self.assertEqual(path.name, "consumable_shop_seed_AAAAAAA_red_white.json")

    def test_missing_seed_is_unsupported(self) -> None:
        results = check_consumable_shop_fixture({"trials": []})
        self.assertEqual(results[0].status, "unsupported")


class CapturedConsumableShopFixtureTests(unittest.TestCase):
    """Using a consumable must not desync the shop-card RNG stream: the
    post-use reroll equals the persistent rng's second shop-card roll."""

    def test_consumable_use_does_not_shift_shop_stream(self) -> None:
        paths = sorted(FIXTURE_DIR.glob("consumable_shop_seed_*.json"))
        if not paths:
            self.skipTest("No consumable-shop fixtures; run `python -m balatro_ai.rng.capture_consumable_shop --all`")
        for path in paths:
            with self.subTest(path=path.name):
                fixture = load_consumable_shop_fixture(path)
                results = check_consumable_shop_fixture(fixture)
                self.assertTrue(results)
                for result in results:
                    self.assertEqual(result.status, "ok", f"{path.name} {result.name}: {result.detail}")


if __name__ == "__main__":
    unittest.main()
