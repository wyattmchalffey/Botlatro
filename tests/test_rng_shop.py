"""Validate shop predictions against captured bridge fixtures.

Skips automatically if fixture files are missing. The fixtures are produced
by `python -m balatro_ai.rng.capture_shop --all` with the bridge running.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.rng.shop import predict_first_shop


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"


def _shop_fixture(seed: str) -> dict | None:
    path = FIXTURE_DIR / f"shop_seed_{seed}_red_white.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class ShopPredictionTests(unittest.TestCase):
    """Each canonical seed predicts the actual first-shop pool exactly."""

    def _check(self, seed: str) -> None:
        fixture = _shop_fixture(seed)
        if fixture is None:
            self.skipTest(f"No fixture for {seed}; run rng.capture_shop --all")
        actual = tuple((c.get("set", "?").upper(), c.get("key", "?")) for c in fixture["shop"]["cards"])
        predicted = predict_first_shop(seed)
        pred_tuple = tuple((slot.category.upper(), slot.item_key) for slot in predicted)
        self.assertEqual(
            pred_tuple,
            actual,
            f"Shop mismatch for {seed}: predicted={pred_tuple} actual={actual}",
        )

    def test_seed_aaaaaaa(self) -> None:
        self._check("AAAAAAA")

    def test_seed_bbbbbbb(self) -> None:
        self._check("BBBBBBB")

    def test_seed_ccccccc(self) -> None:
        self._check("CCCCCCC")

    def test_seed_1234567(self) -> None:
        self._check("1234567")


if __name__ == "__main__":
    unittest.main()
