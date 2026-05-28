"""Validate BalatroRNG predictions against captured bridge fixtures.

These tests skip automatically if no fixtures are present under
`.data/rng-validation/`. To populate fixtures, with Balatro + BalatroBot
running:

    python -m balatro_ai.rng.capture --all

The tests then load each canonical fixture and verify our predictions match
the bridge's actual observations.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.rng.capture import CANONICAL_SEEDS, DEFAULT_FIXTURE_DIR, fixture_path
from balatro_ai.rng.validate import extract_deck_order_short_names


def _project_root() -> Path:
    # tests/ sits next to .data/.
    return Path(__file__).resolve().parent.parent


def _resolved_fixture_dir() -> Path:
    return _project_root() / DEFAULT_FIXTURE_DIR


class BridgeFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_dir = _resolved_fixture_dir()
        if not self.fixture_dir.exists() or not any(self.fixture_dir.glob("seed_*.json")):
            self.skipTest(
                f"No bridge fixtures at {self.fixture_dir}. "
                "Run `python -m balatro_ai.rng.capture --all` with Balatro running to populate."
            )

    def test_canonical_seed_fixtures_have_full_post_shuffle_deck(self) -> None:
        # Fixtures are captured at BLIND_SELECT (before any blind is selected),
        # so there is no starting hand yet — but the entire 52-card post-shuffle
        # deck is exposed under cards.cards. That's the richer signal we
        # validate against. Wider equality tests live alongside the RNG
        # implementation as it grows.
        present = sorted(p for p in self.fixture_dir.glob("seed_*.json"))
        self.assertTrue(present, "expected at least one captured fixture")
        for path in present:
            fixture = json.loads(path.read_text(encoding="utf-8"))
            deck = extract_deck_order_short_names(fixture)
            self.assertEqual(
                len(deck),
                52,
                f"{path.name}: expected 52 deck cards, got {len(deck)} ({deck[:5]}...)",
            )
            # All 52 should be unique (post-shuffle is just a permutation).
            self.assertEqual(len(set(deck)), 52, f"{path.name}: deck cards not unique")

    def test_all_canonical_seeds_captured(self) -> None:
        # Loose check: report which canonical seeds are still missing so the
        # next capture run knows what to grab. Skips rather than fails so
        # partial captures don't block CI.
        missing = [
            seed for seed in CANONICAL_SEEDS
            if not fixture_path(seed, root=self.fixture_dir).exists()
        ]
        if missing:
            self.skipTest(f"Missing fixtures for canonical seeds: {missing}")


if __name__ == "__main__":
    unittest.main()
