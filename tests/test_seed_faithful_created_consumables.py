from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.rng.balatro_rng import BalatroRNG
from balatro_ai.rng.surfaces import predict_card
from balatro_ai.sim.local_runner import LocalBalatroSimulator


class SeedFaithfulCreatedConsumableTests(unittest.TestCase):
    """The Emperor / High Priestess create the consumables predict_card
    predicts for their create_card keys ('emp' Tarot, 'pri' Planet).
    Bridge-validated 3/3 seeds; this guards the key wiring offline."""

    def _check(self, card_type: str, key_append: str) -> None:
        for seed in ("AAAAAAA", "BBBBBBB", "CCCCCCC"):
            with self.subTest(seed=seed, key=key_append):
                sim = LocalBalatroSimulator(seed=1, stake="white", balatro_seed=seed)
                state = sim.reset()
                created = sim._sample_created_consumables_of_type(
                    state, card_type, 2, used_consumables=set(), key_append=key_append
                )
                got = tuple(str(c.get("key")) for c in created)

                rng = BalatroRNG(seed)
                used: set[str] = set()
                expected = []
                for _ in range(2):
                    pc = predict_card(rng, card_type, ante=1, key_append=key_append, used_consumables=used)
                    expected.append(pc.key)
                    used.add(pc.key)
                self.assertEqual(got, tuple(expected))

    def test_emperor_tarots(self) -> None:
        self._check("Tarot", "emp")

    def test_high_priestess_planets(self) -> None:
        self._check("Planet", "pri")

    def test_falls_back_to_generic_without_balatro_seed(self) -> None:
        sim = LocalBalatroSimulator(seed=1, stake="white")
        state = sim.reset()
        self.assertIsNone(sim._seed_faithful_created_card(state, "Tarot", "emp", used_consumables=set()))


if __name__ == "__main__":
    unittest.main()
