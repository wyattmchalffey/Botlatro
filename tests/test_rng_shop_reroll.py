from __future__ import annotations

from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.rng.capture_reroll import reroll_fixture_path
from balatro_ai.rng.validate_reroll import check_reroll_fixture, load_reroll_fixture


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"


class RerollValidationHelperTests(unittest.TestCase):
    def test_reroll_fixture_path_uses_seed_deck_and_stake(self) -> None:
        path = reroll_fixture_path("AAAAAAA", deck="RED", stake="white")
        self.assertEqual(path.name, "shop_reroll_seed_AAAAAAA_red_white.json")

    def test_missing_seed_is_unsupported(self) -> None:
        results = check_reroll_fixture({"rerolls": []})
        self.assertEqual(results[0].status, "unsupported")


class CapturedShopRerollFixtureTests(unittest.TestCase):
    def test_captured_rerolls_match_predictions(self) -> None:
        paths = sorted(FIXTURE_DIR.glob("shop_reroll_seed_*.json"))
        if not paths:
            self.skipTest("No reroll fixtures; run `python -m balatro_ai.rng.capture_reroll --all --rerolls 4`")
        for path in paths:
            with self.subTest(path=path.name):
                fixture = load_reroll_fixture(path)
                results = check_reroll_fixture(fixture)
                self.assertTrue(results)
                for result in results:
                    self.assertEqual(result.status, "ok", f"{path.name} {result.name}: {result.detail}")


if __name__ == "__main__":
    unittest.main()
