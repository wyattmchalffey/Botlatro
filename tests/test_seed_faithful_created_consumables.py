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

    def test_judgement_and_soul_jokers(self) -> None:
        # Judgement ('jud') and The Soul ('sou', legendary) create jokers.
        # Bridge-validated 3/3 seeds each.
        for seed in ("AAAAAAA", "BBBBBBB", "CCCCCCC"):
            sim = LocalBalatroSimulator(seed=1, stake="white", balatro_seed=seed)
            state = sim.reset()
            with self.subTest(seed=seed, effect="Judgement"):
                got = sim._seed_faithful_created_card(state, "Joker", "jud", used_consumables=set())
                pc = predict_card(BalatroRNG(seed), "Joker", ante=1, key_append="jud")
                self.assertEqual(str(got.get("key")), pc.key)
            sim2 = LocalBalatroSimulator(seed=1, stake="white", balatro_seed=seed)
            state2 = sim2.reset()
            with self.subTest(seed=seed, effect="The Soul"):
                got = sim2._seed_faithful_created_card(state2, "Joker", "sou", used_consumables=set(), legendary=True)
                pc = predict_card(BalatroRNG(seed), "Joker", ante=1, key_append="sou", legendary=True)
                self.assertEqual(str(got.get("key")), pc.key)

    def test_falls_back_to_generic_without_balatro_seed(self) -> None:
        sim = LocalBalatroSimulator(seed=1, stake="white")
        state = sim.reset()
        self.assertIsNone(sim._seed_faithful_created_card(state, "Tarot", "emp", used_consumables=set()))


class JudgementPackOverCapTests(unittest.TestCase):
    """Judgement / The Soul / Wraith selected FROM A BOOSTER PACK create a
    joker OVER the slot cap — the pack's can_select_card allows any non-Joker
    card regardless of joker room and the deferred create_card emplaces with
    no room check (button_callbacks.lua:2112-2113, card.lua:1418-1420).
    Bridge-proven on seed 0000014 (a pack Judgement made Odd Todd at 5/5
    jokers; the play 4 steps later under-scored by exactly Odd Todd's chips).
    Used from the consumable SLOT they keep the room gate (card.lua:1557)."""

    def _full_slot_state(self, sim):
        from dataclasses import replace

        from balatro_ai.api.state import Joker
        state = sim.reset()
        jokers = tuple(Joker(name) for name in ("Joker", "Joker", "Joker", "Joker", "Joker"))
        return replace(state, jokers=jokers)

    def test_pack_judgement_creates_joker_when_slots_full(self) -> None:
        sim = LocalBalatroSimulator(seed=1, stake="white", balatro_seed="AAAAAAA")
        state = self._full_slot_state(sim)
        inj = sim._consumable_injections(state, "Judgement", storage_use=False, from_pack=True)
        created = inj.get("created_jokers")
        self.assertTrue(created, "pack Judgement at full slots must still create a joker")
        # the created payload carries the over-cap flag the forward_sim guard honours
        payload = created[0]
        meta = payload.get("metadata") if isinstance(payload, dict) else getattr(payload, "metadata", {})
        self.assertTrue(meta.get("pack_created_over_cap"))

    def test_slot_use_judgement_blocked_when_slots_full(self) -> None:
        sim = LocalBalatroSimulator(seed=1, stake="white", balatro_seed="AAAAAAA")
        state = self._full_slot_state(sim)
        inj = sim._consumable_injections(state, "Judgement", storage_use=True, from_pack=False)
        self.assertNotIn("created_jokers", inj)

    def test_overcap_joker_passes_forward_sim_slot_guard(self) -> None:
        from balatro_ai.api.state import GameState, Joker
        from balatro_ai.search.forward_sim import _jokers_and_modifiers_after_created_jokers
        jokers = tuple(Joker(name) for name in ("Joker", "Joker", "Joker", "Joker", "Joker"))
        state = GameState(jokers=jokers)
        over = Joker("Odd Todd", metadata={"pack_created_over_cap": True})
        result, _ = _jokers_and_modifiers_after_created_jokers(state, jokers, {}, (over,))
        self.assertEqual(len(result), 6)
        self.assertEqual(result[-1].name, "Odd Todd")


if __name__ == "__main__":
    unittest.main()
