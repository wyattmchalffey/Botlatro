from __future__ import annotations

import math
import unittest

import context  # noqa: F401
from balatro_ai.rng import (
    BalatroRNG,
    PSEUDOSEED_ADD,
    PSEUDOSEED_MUL,
    pseudohash,
    pseudorandom_element,
    pseudorandom_shuffle,
    pseudoseed_step,
)


class PseudohashTests(unittest.TestCase):
    def test_empty_string_returns_one(self) -> None:
        # Lua's loop body never executes for an empty string, so the
        # initial num = 1.0 is returned untouched.
        self.assertEqual(pseudohash(""), 1.0)

    def test_single_character_matches_explicit_formula(self) -> None:
        # Single-character hash is the formula evaluated once at position 1
        # with the byte value reachable directly. This pins the function to
        # the algorithm we documented.
        text = "A"
        byte = ord(text)
        expected = ((1.1239285023 / 1.0) * byte * math.pi + math.pi * 1) % 1.0
        self.assertEqual(pseudohash(text), expected)

    def test_is_deterministic(self) -> None:
        self.assertEqual(pseudohash("AAAAAAA"), pseudohash("AAAAAAA"))

    def test_different_seeds_produce_different_hashes(self) -> None:
        self.assertNotEqual(pseudohash("AAAAAAA"), pseudohash("AAAAAAB"))

    def test_walks_backwards_so_swapped_characters_change_output(self) -> None:
        # If the implementation accidentally walked forward, "AB" and "BA"
        # would produce hashes that differ only by which byte appears where
        # in a non-positional way; the actual Balatro algorithm uses
        # position i as a multiplier, so reversed characters land at
        # different positions and produce different values.
        self.assertNotEqual(pseudohash("AB"), pseudohash("BA"))

    def test_multi_character_matches_step_by_step_iteration(self) -> None:
        text = "AB"
        # Lua's `for i = #str, 1, -1` keeps i equal to the visited
        # character's original 1-indexed position. So we visit 'B' (position
        # 2) first using position multiplier 2, then 'A' (position 1) using
        # multiplier 1 — NOT the iteration counter.
        num = 1.0
        num = ((1.1239285023 / num) * ord("B") * math.pi + math.pi * 2) % 1.0
        num = ((1.1239285023 / num) * ord("A") * math.pi + math.pi * 1) % 1.0
        self.assertEqual(pseudohash(text), num)

    def test_result_is_always_in_unit_interval(self) -> None:
        for sample in ("AAAAAAA", "ZZZZZZZ", "1234567", "shop_pack", "Mr.Bones"):
            result = pseudohash(sample)
            self.assertGreaterEqual(result, 0.0)
            self.assertLess(result, 1.0)


class PseudoseedStepTests(unittest.TestCase):
    def test_step_matches_lua_iteration_with_13f_rounding(self) -> None:
        # Balatro's pseudoseed applies string.format("%.13f", ...) + abs to
        # the raw float result; the rounded value drifts from the
        # un-rounded one after a few decimal places. We test the rounded
        # result is what comes back.
        state = 0.42
        raw = (PSEUDOSEED_ADD + state * PSEUDOSEED_MUL) % 1.0
        expected = abs(float(f"{raw:.13f}"))
        self.assertEqual(pseudoseed_step(state), expected)

    def test_constants_match_balatro_source(self) -> None:
        # Pinning the constants prevents accidental drift if someone
        # refactors. Source: Balatro 1.0.x functions.lua, pseudoseed().
        self.assertEqual(PSEUDOSEED_MUL, 1.72431234)
        self.assertEqual(PSEUDOSEED_ADD, 2.134453429141)


class BalatroRNGTests(unittest.TestCase):
    def test_same_seed_produces_same_sequence(self) -> None:
        a = BalatroRNG(seed="AAAAAAA")
        b = BalatroRNG(seed="AAAAAAA")
        for _ in range(5):
            self.assertEqual(a.random("shop_pack"), b.random("shop_pack"))

    def test_different_seeds_diverge_immediately(self) -> None:
        a = BalatroRNG(seed="AAAAAAA")
        b = BalatroRNG(seed="BBBBBBB")
        self.assertNotEqual(a.random("shop_pack"), b.random("shop_pack"))

    def test_different_keys_have_independent_streams(self) -> None:
        rng = BalatroRNG(seed="AAAAAAA")
        # First call for "shop_pack" and "boss" should give different values
        # because they bootstrap from different pseudohashes.
        self.assertNotEqual(rng.random("shop_pack"), rng.random("boss"))

    def test_consecutive_calls_for_same_key_advance(self) -> None:
        rng = BalatroRNG(seed="AAAAAAA")
        first = rng.random("shop_pack")
        second = rng.random("shop_pack")
        self.assertNotEqual(first, second)

    def test_random_is_in_unit_interval(self) -> None:
        rng = BalatroRNG(seed="AAAAAAA")
        for _ in range(50):
            value = rng.random("any_key")
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, 1.0)

    def test_random_matches_explicit_first_call_formula(self) -> None:
        seed = "AAAAAAA"
        key = "shop_pack"
        # Bootstrap: state = pseudohash(key + seed); then one step;
        # then combine with hashed_seed. Default mix_hashed_seed=True
        # because Balatro always applies the (state + hashed_seed) / 2
        # transform.
        state = pseudohash(key + seed)
        state = pseudoseed_step(state)
        expected = (state + pseudohash(seed)) / 2

        rng = BalatroRNG(seed=seed)
        self.assertEqual(rng.random(key), expected)

    def test_mix_hashed_seed_applies_the_coupled_formula(self) -> None:
        seed = "AAAAAAA"
        key = "shop_pack"
        state = pseudohash(key + seed)
        state = pseudoseed_step(state)
        expected = (state + pseudohash(seed)) / 2

        rng = BalatroRNG(seed=seed, mix_hashed_seed=True)
        self.assertEqual(rng.random(key), expected)

    def test_seed_key_is_rejected(self) -> None:
        rng = BalatroRNG(seed="AAAAAAA")
        with self.assertRaises(ValueError):
            rng.random("seed")

    def test_snapshot_is_independent_of_rng_state(self) -> None:
        rng = BalatroRNG(seed="AAAAAAA")
        rng.random("shop_pack")
        snapshot = dict(rng.snapshot())
        snapshot["shop_pack"] = 0.999
        self.assertNotEqual(rng.snapshot()["shop_pack"], 0.999)


class PseudorandomElementTests(unittest.TestCase):
    def test_picks_same_element_for_same_seed_and_key(self) -> None:
        rng_a = BalatroRNG(seed="AAAAAAA")
        rng_b = BalatroRNG(seed="AAAAAAA")
        items = ["a", "b", "c", "d", "e"]
        self.assertEqual(
            pseudorandom_element(items, rng_a, "shop_pack"),
            pseudorandom_element(items, rng_b, "shop_pack"),
        )

    def test_picked_index_matches_ceil_of_value_times_length(self) -> None:
        # The single-element case is the only one where we can predict the
        # outcome without re-implementing the algorithm in the test.
        rng = BalatroRNG(seed="AAAAAAA")
        self.assertEqual(pseudorandom_element(["only"], rng, "k"), "only")

    def test_empty_input_raises(self) -> None:
        rng = BalatroRNG(seed="AAAAAAA")
        with self.assertRaises(ValueError):
            pseudorandom_element([], rng, "k")

    def test_distribution_covers_most_positions_over_many_draws(self) -> None:
        # With mix_hashed_seed=True the output is in [h/2, (1+h)/2) where
        # h = hashed_seed for this seed string, so the index distribution
        # is biased and may not cover every position depending on h. We
        # only assert that at least a majority of positions get hit — this
        # still catches off-by-one bugs that would lock out big swaths of
        # the index space.
        items = list(range(5))
        seen: set[int] = set()
        rng = BalatroRNG(seed="AAAAAAA")
        for i in range(500):
            seen.add(pseudorandom_element(items, rng, f"key_{i}"))
        self.assertGreaterEqual(len(seen), 3)


class PseudorandomShuffleTests(unittest.TestCase):
    def test_shuffle_is_deterministic_for_same_seed(self) -> None:
        deck_a = list(range(52))
        deck_b = list(range(52))
        pseudorandom_shuffle(deck_a, BalatroRNG(seed="AAAAAAA"), "shuffle")
        pseudorandom_shuffle(deck_b, BalatroRNG(seed="AAAAAAA"), "shuffle")
        self.assertEqual(deck_a, deck_b)

    def test_shuffle_preserves_multiset(self) -> None:
        deck = list(range(52))
        original = sorted(deck)
        pseudorandom_shuffle(deck, BalatroRNG(seed="AAAAAAA"), "shuffle")
        self.assertEqual(sorted(deck), original)

    def test_shuffle_actually_changes_order_for_long_enough_input(self) -> None:
        deck = list(range(52))
        pseudorandom_shuffle(deck, BalatroRNG(seed="AAAAAAA"), "shuffle")
        self.assertNotEqual(deck, list(range(52)))

    def test_different_seeds_produce_different_shuffles(self) -> None:
        deck_a = list(range(52))
        deck_b = list(range(52))
        pseudorandom_shuffle(deck_a, BalatroRNG(seed="AAAAAAA"), "shuffle")
        pseudorandom_shuffle(deck_b, BalatroRNG(seed="BBBBBBB"), "shuffle")
        self.assertNotEqual(deck_a, deck_b)


if __name__ == "__main__":
    unittest.main()
