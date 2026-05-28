from __future__ import annotations

from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.rng.capture_play_rng import play_rng_fixture_path
from balatro_ai.rng.validate_play_rng import check_play_rng_fixture, load_play_rng_fixture


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"


class PlayRngHelperTests(unittest.TestCase):
    def test_fixture_path_uses_seed_deck_and_stake(self) -> None:
        path = play_rng_fixture_path("AAAAAAA", deck="RED", stake="white")
        self.assertEqual(path.name, "play_rng_seed_AAAAAAA_red_white.json")

    def test_missing_seed_is_unsupported(self) -> None:
        results = check_play_rng_fixture({"effects": []})
        self.assertEqual(results[0].status, "unsupported")


class CapturedPlayRngFixtureTests(unittest.TestCase):
    """Mid-hand probability rolls (lucky_mult/lucky_money/glass/bloodstone/
    business/parking/space/misprint) must reproduce on the persistent rng."""

    def test_captured_play_rolls_match_predictions(self) -> None:
        paths = sorted(FIXTURE_DIR.glob("play_rng_seed_*.json"))
        if not paths:
            self.skipTest("No play-rng fixtures; run `python -m balatro_ai.rng.capture_play_rng --all`")
        for path in paths:
            with self.subTest(path=path.name):
                fixture = load_play_rng_fixture(path)
                results = check_play_rng_fixture(fixture)
                self.assertTrue(results)
                for result in results:
                    self.assertEqual(result.status, "ok", f"{path.name} {result.name}: {result.detail}")


if __name__ == "__main__":
    unittest.main()
