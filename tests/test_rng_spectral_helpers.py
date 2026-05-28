from __future__ import annotations

from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.rng.capture_spectral_helpers import SPECTRAL_HELPER_KEYS, spectral_helper_fixture_path
from balatro_ai.rng.surfaces import predict_spectral_created_cards
from balatro_ai.rng.validate_spectral_helpers import (
    actual_created_cards,
    check_spectral_helper_fixture,
    load_spectral_helper_fixture,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"


class SpectralHelperValidationTests(unittest.TestCase):
    def test_spectral_helper_fixture_path_uses_key(self) -> None:
        path = spectral_helper_fixture_path("AAAAAAA", "c_familiar")
        self.assertEqual(path.name, "spectral_seed_AAAAAAA_c_familiar_red_white.json")

    def test_actual_created_cards_extracts_enhanced_hand_cards(self) -> None:
        state = {
            "hand": {
                "cards": [
                    {"key": "H_2", "set": "DEFAULT"},
                    {"key": "S_K", "set": "ENHANCED", "label": "Gold Card"},
                ]
            }
        }
        self.assertEqual(actual_created_cards(state), (("K", "S", "m_gold"),))

    def test_spectral_created_cards_never_use_stone_enhancement(self) -> None:
        for key in SPECTRAL_HELPER_KEYS:
            with self.subTest(key=key):
                created = predict_spectral_created_cards("AAAAAAA", key, count=12)
                self.assertTrue(created)
                self.assertNotIn("m_stone", {enhancement for _, _, enhancement in created})


class CapturedSpectralHelperFixtureTests(unittest.TestCase):
    def test_captured_spectral_helper_fixtures_match_predictions(self) -> None:
        paths = sorted(FIXTURE_DIR.glob("spectral_seed_*.json"))
        if not paths:
            self.skipTest("No spectral helper fixtures; run `python -m balatro_ai.rng.capture_spectral_helpers --all-helpers`")
        for path in paths:
            with self.subTest(path=path.name):
                fixture = load_spectral_helper_fixture(path)
                result = check_spectral_helper_fixture(fixture)
                self.assertEqual(result.status, "ok", f"{path.name}: {result.detail}")


if __name__ == "__main__":
    unittest.main()
