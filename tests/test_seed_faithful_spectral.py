from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.rng.balatro_rng import BalatroRNG
from balatro_ai.rng.surfaces import predict_spectral_created_cards
from balatro_ai.search.shop_sampler import ENHANCEMENT_BY_CENTER_KEY
from balatro_ai.sim.local_runner import LocalBalatroSimulator


class SeedFaithfulSpectralCardsTests(unittest.TestCase):
    """Familiar/Grim/Incantation create the cards predict_spectral_created_cards
    predicts (itself bridge-validated by validate_spectral_helpers)."""

    def test_created_cards_match_predictor(self) -> None:
        for name, key in (("Familiar", "c_familiar"), ("Grim", "c_grim"), ("Incantation", "c_incantation")):
            with self.subTest(spectral=name):
                sim = LocalBalatroSimulator(seed=1, stake="white", balatro_seed="AAAAAAA")
                sim.reset()
                got = sim._seed_faithful_spectral_cards(name)
                self.assertIsNotNone(got)
                got_sig = tuple((c.rank, c.suit, c.enhancement) for c in got)
                expected = tuple(
                    ("10" if rank == "T" else rank, suit, ENHANCEMENT_BY_CENTER_KEY.get(enh, enh))
                    for rank, suit, enh in predict_spectral_created_cards(BalatroRNG("AAAAAAA"), key)
                )
                self.assertEqual(got_sig, expected)

    def test_falls_back_to_generic_without_balatro_seed(self) -> None:
        sim = LocalBalatroSimulator(seed=1, stake="white")
        sim.reset()
        self.assertIsNone(sim._seed_faithful_spectral_cards("Familiar"))


if __name__ == "__main__":
    unittest.main()
