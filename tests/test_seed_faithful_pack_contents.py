from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import context  # noqa: F401
import balatro_ai.rng.validate_surfaces as v
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.sim.seed_faithful_shop import seed_faithful_pack_contents


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"


class SeedFaithfulPackContentsTests(unittest.TestCase):
    """The sim's pack-content adapter must reproduce captured opened packs,
    including Standard (playing-card), Telescope, and Omen-Globe packs."""

    def test_pack_contents_match_captured_fixtures(self) -> None:
        paths = sorted(FIXTURE_DIR.glob("pack_seed_*.json"))
        if not paths:
            self.skipTest("No pack fixtures; run `python -m balatro_ai.rng.capture_surfaces --all --all-pack-kinds`")

        sim = LocalBalatroSimulator(seed=1, stake="white", balatro_seed="AAAAAAA")
        base = sim.reset()

        for path in paths:
            with self.subTest(path=path.name):
                fixture = v.load_pack_fixture(path)
                seed = v.fixture_seed(fixture)
                pack_key = v.fixture_pack_key(fixture)
                ante = v.fixture_ante(fixture)
                vouchers = tuple(v.fixture_owned_vouchers(fixture))
                hands_mod = {name: {"played": count} for name, count in v.fixture_played_hand_counts(fixture).items()}
                modifiers = dict(base.modifiers)
                modifiers["hands"] = hands_mod
                state = replace(base, ante=ante, vouchers=vouchers, modifiers=modifiers)

                contents = seed_faithful_pack_contents(sim.sampler, state, {"key": pack_key}, seed)
                self.assertIsNotNone(contents, f"{path.name}: sim bailed on pack {pack_key}")

                got = tuple(v.actual_card_signature(card).compact() for card in contents)
                expected = tuple(v.actual_card_signature(card).compact() for card in v.fixture_pack_cards(fixture))
                self.assertEqual(got, expected, f"{path.name} pack={pack_key}")


if __name__ == "__main__":
    unittest.main()
