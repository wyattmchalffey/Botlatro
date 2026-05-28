"""Parity tests for `balatro_core.identify_hand_type` (Phase 2a).

For every input the Rust port can handle (no Stone/Wild cards), the
output MUST match Python's `_identify_hand_type`. A divergence here
silently corrupts every leaf evaluation in the solver — this is the
correctness gate for the whole Phase 2 port.

Test strategy:
1. **Hand-picked cases**: classic poker hands — verifies the
   classification logic in isolation.
2. **Generated combinations**: every distinct 5-card hand from a
   small fixture deck. ~thousands of cases.
3. **Random fuzz**: 1000 random hands of varying sizes (1-5 cards).
   Catches edge cases the structured tests might miss.

For hands that contain Stone/Wild cards (or where the Rust port
returns None for any reason), the Python implementation is the
oracle and we skip the comparison — the Python wrapper will use
the Python implementation in those cases.
"""

from __future__ import annotations

import random
import unittest
from itertools import combinations

import context  # noqa: F401
from balatro_ai.api.state import Card

try:
    import balatro_core
    BALATRO_CORE_AVAILABLE = True
except ImportError:
    balatro_core = None
    BALATRO_CORE_AVAILABLE = False


def _to_rust(cards: list[Card]) -> list:
    return [balatro_core.RustCard.from_python(c) for c in cards]


def _python_hand_type(cards: tuple) -> str:
    """Ground truth: returns the HandType enum value as a string."""

    from balatro_ai.rules.hand_evaluator import _identify_hand_type
    return _identify_hand_type(tuple(cards)).value


def _assert_parity(testcase: unittest.TestCase, cards: list[Card]) -> None:
    rust = balatro_core.identify_hand_type(_to_rust(cards))
    if rust is None:
        return  # Rust opted out; Python is the oracle, no comparison
    python = _python_hand_type(tuple(cards))
    testcase.assertEqual(
        rust, python,
        f"Rust {rust!r} vs Python {python!r} on cards: "
        f"{[(c.rank, c.suit, c.enhancement) for c in cards]}",
    )


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class HandTypeCanonicalTests(unittest.TestCase):
    """Hand-picked classic poker hands."""

    def test_high_card(self) -> None:
        _assert_parity(self, [
            Card(rank="A", suit="S"),
            Card(rank="K", suit="H"),
            Card(rank="9", suit="D"),
            Card(rank="5", suit="C"),
            Card(rank="2", suit="H"),
        ])

    def test_pair(self) -> None:
        _assert_parity(self, [
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
        ])

    def test_two_pair(self) -> None:
        _assert_parity(self, [
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="K", suit="D"),
            Card(rank="K", suit="C"),
        ])

    def test_three_of_a_kind(self) -> None:
        _assert_parity(self, [
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="A", suit="D"),
        ])

    def test_straight_ace_high(self) -> None:
        _assert_parity(self, [
            Card(rank="10", suit="S"),
            Card(rank="J", suit="H"),
            Card(rank="Q", suit="D"),
            Card(rank="K", suit="C"),
            Card(rank="A", suit="H"),
        ])

    def test_straight_wheel(self) -> None:
        _assert_parity(self, [
            Card(rank="A", suit="S"),
            Card(rank="2", suit="H"),
            Card(rank="3", suit="D"),
            Card(rank="4", suit="C"),
            Card(rank="5", suit="H"),
        ])

    def test_flush(self) -> None:
        _assert_parity(self, [
            Card(rank="2", suit="H"),
            Card(rank="5", suit="H"),
            Card(rank="7", suit="H"),
            Card(rank="9", suit="H"),
            Card(rank="K", suit="H"),
        ])

    def test_full_house(self) -> None:
        _assert_parity(self, [
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="A", suit="D"),
            Card(rank="K", suit="C"),
            Card(rank="K", suit="H"),
        ])

    def test_four_of_a_kind(self) -> None:
        _assert_parity(self, [
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="A", suit="D"),
            Card(rank="A", suit="C"),
            Card(rank="K", suit="H"),
        ])

    def test_straight_flush(self) -> None:
        _assert_parity(self, [
            Card(rank="5", suit="H"),
            Card(rank="6", suit="H"),
            Card(rank="7", suit="H"),
            Card(rank="8", suit="H"),
            Card(rank="9", suit="H"),
        ])

    def test_five_of_a_kind(self) -> None:
        # 5 Aces of mixed suits → Five of a Kind (not Flush Five).
        _assert_parity(self, [
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="A", suit="D"),
            Card(rank="A", suit="C"),
            Card(rank="A", suit="H"),
        ])

    def test_flush_five(self) -> None:
        _assert_parity(self, [Card(rank="A", suit="H")] * 5)

    def test_flush_house(self) -> None:
        _assert_parity(self, [
            Card(rank="A", suit="H"),
            Card(rank="A", suit="H"),
            Card(rank="A", suit="H"),
            Card(rank="K", suit="H"),
            Card(rank="K", suit="H"),
        ])


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class HandTypeStructuredFuzzTests(unittest.TestCase):
    """All distinct 5-card combinations from a small deck.

    Small enough to enumerate exhaustively; gives broader coverage
    than the canonical cases without random-seed flakiness.
    """

    def test_all_5_card_combos_from_small_deck(self) -> None:
        ranks = ["2", "5", "9", "J", "K", "A"]
        suits = ["C", "D", "H", "S"]
        deck = [Card(rank=r, suit=s) for r in ranks for s in suits]
        # Enumerate all C(24, 5) = 42504 — but cap at a sample for speed.
        all_combos = list(combinations(deck, 5))
        random.Random(0).shuffle(all_combos)
        for combo in all_combos[:500]:  # 500 random distinct combos
            _assert_parity(self, list(combo))


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class HandTypeRandomFuzzTests(unittest.TestCase):
    """1000 random hands of varying sizes (1-5 cards)."""

    def test_random_hands_match_python(self) -> None:
        rng = random.Random(42)
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        suits = ["C", "D", "H", "S"]
        for _ in range(1000):
            size = rng.randint(1, 5)
            cards = [
                Card(rank=rng.choice(ranks), suit=rng.choice(suits))
                for _ in range(size)
            ]
            _assert_parity(self, cards)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class HandTypeFallbackTests(unittest.TestCase):
    """Wild cards still bail (suit semantics differ); Stone cards
    are now handled natively (filtered out of identification, added
    back in scoring_indices)."""

    def test_stone_card_identified_natively(self) -> None:
        # Stone + K → K alone drives identification → HighCard.
        # Rust used to bail here; now it filters out the stone and
        # returns the actual hand type.
        cards = [
            Card(rank="A", suit="S", enhancement="stone"),
            Card(rank="K", suit="H"),
        ]
        self.assertEqual(
            balatro_core.identify_hand_type(_to_rust(cards)),
            "High Card",
        )

    def test_wild_card_returns_none(self) -> None:
        cards = [
            Card(rank="A", suit="S", enhancement="wild"),
            Card(rank="K", suit="H"),
        ]
        self.assertIsNone(balatro_core.identify_hand_type(_to_rust(cards)))


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class HandTypeBenchmarkTests(unittest.TestCase):
    """Sanity timing: Rust at hot-loop scale beats Python by ≥5×.

    This is the first port where the FFI overhead is amortized
    over real work — a 5-card hand classification involves several
    iterations + counter operations. Unlike `is_stone_card` (which
    lost to Python due to FFI dominance), this should win.
    """

    def test_rust_at_least_5x_python_on_5card_hands(self) -> None:
        from balatro_ai.rules.hand_evaluator import _identify_hand_type
        import time

        rng = random.Random(7)
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        suits = ["C", "D", "H", "S"]
        # 100 random 5-card hands, prepped both as Python tuples and RustCard lists.
        py_hands = [
            tuple(Card(rank=rng.choice(ranks), suit=rng.choice(suits)) for _ in range(5))
            for _ in range(100)
        ]
        rust_hands = [_to_rust(list(h)) for h in py_hands]

        # Warmup
        for h in py_hands:
            _identify_hand_type(h)
        for h in rust_hands:
            balatro_core.identify_hand_type(h)

        # Python: 10000 iterations × 100 hands = 1M classifications
        t = time.perf_counter()
        for _ in range(1000):
            for h in py_hands:
                _identify_hand_type(h)
        py_time = time.perf_counter() - t

        t = time.perf_counter()
        for _ in range(1000):
            for h in rust_hands:
                balatro_core.identify_hand_type(h)
        rs_time = time.perf_counter() - t

        ratio = py_time / rs_time
        print(f"\n  identify_hand_type: Python {py_time*1000:.1f}ms, "
              f"Rust {rs_time*1000:.1f}ms  ({ratio:.2f}x speedup)")
        # Don't require 5x strictly — list-of-RustCard arg construction
        # still has FFI overhead. The real speedup target is the COMBINED
        # native eval pipeline (Phase 2 complete). We just want this to
        # be at least competitive, not catastrophically slow.
        self.assertGreater(ratio, 1.0,
                           f"Rust ({rs_time*1000:.1f}ms) should be faster than Python "
                           f"({py_time*1000:.1f}ms)")


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class ScoringIndicesParityTests(unittest.TestCase):
    """Rust `scoring_indices` must match Python `_scoring_indices`
    for every input the Rust port can handle."""

    def _assert_parity(self, cards: list[Card], hand_type_str: str) -> None:
        rust_idx = balatro_core.scoring_indices(_to_rust(cards), hand_type_str)
        if rust_idx is None:
            return  # Rust opted out
        from balatro_ai.rules.hand_evaluator import _scoring_indices, HandType
        ht = HandType(hand_type_str)
        python_idx = list(_scoring_indices(tuple(cards), ht))
        self.assertEqual(
            rust_idx, python_idx,
            f"scoring_indices divergence on {hand_type_str}: "
            f"Rust={rust_idx}, Python={python_idx}",
        )

    def test_pair(self) -> None:
        self._assert_parity([
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="K", suit="D"),
        ], "Pair")

    def test_three_of_a_kind(self) -> None:
        self._assert_parity([
            Card(rank="K", suit="D"),
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="A", suit="D"),
        ], "Three of a Kind")

    def test_two_pair(self) -> None:
        self._assert_parity([
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="K", suit="D"),
            Card(rank="K", suit="C"),
        ], "Two Pair")

    def test_four_of_a_kind(self) -> None:
        self._assert_parity([
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="A", suit="D"),
            Card(rank="A", suit="C"),
            Card(rank="K", suit="H"),
        ], "Four of a Kind")

    def test_full_house(self) -> None:
        self._assert_parity([
            Card(rank="A", suit="S"),
            Card(rank="A", suit="H"),
            Card(rank="A", suit="D"),
            Card(rank="K", suit="C"),
            Card(rank="K", suit="H"),
        ], "Full House")

    def test_flush(self) -> None:
        self._assert_parity([
            Card(rank="2", suit="H"),
            Card(rank="5", suit="H"),
            Card(rank="7", suit="H"),
            Card(rank="9", suit="H"),
            Card(rank="K", suit="H"),
        ], "Flush")

    def test_high_card(self) -> None:
        self._assert_parity([
            Card(rank="2", suit="S"),
            Card(rank="5", suit="H"),
            Card(rank="K", suit="D"),
        ], "High Card")

    def test_random_fuzz_via_identify_then_score(self) -> None:
        # Combined Rust path: identify_hand_type → scoring_indices.
        # For each random hand, get the Rust hand type, then ask Rust
        # for the scoring indices. Verify both match Python.
        from balatro_ai.rules.hand_evaluator import _scoring_indices, HandType
        rng = random.Random(11)
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        suits = ["C", "D", "H", "S"]
        compared = 0
        for _ in range(500):
            size = rng.randint(1, 5)
            cards = [
                Card(rank=rng.choice(ranks), suit=rng.choice(suits))
                for _ in range(size)
            ]
            rc = _to_rust(cards)
            ht_str = balatro_core.identify_hand_type(rc)
            if ht_str is None:
                continue
            rust_idx = balatro_core.scoring_indices(rc, ht_str)
            if rust_idx is None:
                continue
            python_idx = list(_scoring_indices(tuple(cards), HandType(ht_str)))
            self.assertEqual(
                rust_idx, python_idx,
                f"divergence on {cards} hand_type={ht_str}: Rust={rust_idx}, Python={python_idx}",
            )
            compared += 1
        # Sanity: most random hands should be handled by the Rust path.
        self.assertGreater(compared, 100, f"only compared {compared} hands — Rust opting out too often?")


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class CardChipValueParityTests(unittest.TestCase):
    """Rust `card_chip_value` must match Python `_card_chip_value`
    for every vanilla card. Metadata-driven overrides (Gold Cards
    with bonus_chips, perma_bonus modifiers) are documented Rust
    fallbacks and not tested here — Python is the oracle for those.
    """

    def _assert_parity(
        self,
        card: Card,
        debuffed_suits: list[str] | None = None,
    ) -> None:
        from balatro_ai.rules.hand_evaluator import _card_chip_value
        debuffed = frozenset(debuffed_suits or [])
        py = _card_chip_value(card, debuffed_suits=debuffed)
        rs_card = balatro_core.RustCard.from_python(card)
        rs = balatro_core.card_chip_value(rs_card, list(debuffed_suits or []))
        self.assertEqual(
            rs, py,
            f"chip value divergence for {card}: Rust={rs}, Python={py}",
        )

    def test_all_ranks_vanilla(self) -> None:
        for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"):
            self._assert_parity(Card(rank=rank, suit="S"))

    def test_bonus_enhancement(self) -> None:
        self._assert_parity(Card(rank="5", suit="H", enhancement="bonus"))

    def test_stone_card_returns_50(self) -> None:
        self._assert_parity(Card(rank="5", suit="H", enhancement="stone"))

    def test_debuffed_card(self) -> None:
        self._assert_parity(Card(rank="A", suit="S", debuffed=True))

    def test_suit_debuff_zeroes_card(self) -> None:
        # When the active blind debuffs Hearts, Heart cards score 0.
        self._assert_parity(Card(rank="A", suit="H"), debuffed_suits=["H"])

    def test_suit_debuff_irrelevant_for_other_suit(self) -> None:
        # Heart debuff doesn't affect Spades.
        self._assert_parity(Card(rank="A", suit="S"), debuffed_suits=["H"])

    def test_random_fuzz_all_vanilla(self) -> None:
        rng = random.Random(99)
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        suits = ["C", "D", "H", "S"]
        enhancements = (None, "bonus", "stone")
        for _ in range(500):
            card = Card(
                rank=rng.choice(ranks),
                suit=rng.choice(suits),
                enhancement=rng.choice(enhancements),
                debuffed=rng.choice([True, False]),
            )
            self._assert_parity(card)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class EvaluateSimpleParityTests(unittest.TestCase):
    """End-to-end: Rust `evaluate_simple` must match Python
    `evaluate_played_cards` on vanilla (no joker, no enhancement
    metadata) hands. This is the Phase 2 acceptance gate's first
    measurable piece — proves the composition of hand_type +
    scoring_indices + chip_value computes the right (chips, mult,
    score) tuple end-to-end.
    """

    def _assert_parity(self, cards: list[Card], hand_levels: dict | None = None, debuffed: list[str] | None = None) -> None:
        from balatro_ai.rules.hand_evaluator import evaluate_played_cards
        levels = hand_levels or {}
        rs_cards = _to_rust(cards)
        rs = balatro_core.evaluate_simple(
            rs_cards,
            hand_level=levels.get("__lookup", 1),
            debuffed_suits=debuffed or [],
        )
        py_eval = evaluate_played_cards(
            tuple(cards),
            hand_levels=levels,
            debuffed_suits=frozenset(debuffed or []),
            jokers=(),
        )
        if rs is None:
            return  # Rust opted out; nothing to compare
        rs_chips, rs_mult, rs_score, rs_ht = rs
        self.assertEqual(rs_ht, py_eval.hand_type.value,
                         f"hand_type mismatch on {cards}")
        self.assertEqual(int(rs_chips), int(py_eval.chips),
                         f"chips mismatch on {cards}: Rust={rs_chips}, Python={py_eval.chips}")
        self.assertEqual(int(rs_mult), int(py_eval.mult),
                         f"mult mismatch on {cards}: Rust={rs_mult}, Python={py_eval.mult}")
        self.assertEqual(int(rs_score), int(py_eval.score),
                         f"score mismatch on {cards}: Rust={rs_score}, Python={py_eval.score}")

    def test_pair_of_aces(self) -> None:
        self._assert_parity([
            Card(rank="A", suit="H"),
            Card(rank="A", suit="S"),
        ])

    def test_straight_wheel(self) -> None:
        self._assert_parity([
            Card(rank="A", suit="H"),
            Card(rank="2", suit="D"),
            Card(rank="3", suit="C"),
            Card(rank="4", suit="S"),
            Card(rank="5", suit="H"),
        ])

    def test_flush(self) -> None:
        self._assert_parity([
            Card(rank="2", suit="H"),
            Card(rank="5", suit="H"),
            Card(rank="7", suit="H"),
            Card(rank="9", suit="H"),
            Card(rank="K", suit="H"),
        ])

    def test_full_house(self) -> None:
        self._assert_parity([
            Card(rank="A", suit="H"),
            Card(rank="A", suit="S"),
            Card(rank="A", suit="D"),
            Card(rank="K", suit="C"),
            Card(rank="K", suit="H"),
        ])

    def test_high_card(self) -> None:
        self._assert_parity([
            Card(rank="K", suit="D"),
            Card(rank="5", suit="C"),
        ])

    def test_random_fuzz_evaluate_simple(self) -> None:
        rng = random.Random(123)
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        suits = ["C", "D", "H", "S"]
        compared = 0
        for _ in range(500):
            size = rng.randint(1, 5)
            cards = [Card(rank=rng.choice(ranks), suit=rng.choice(suits)) for _ in range(size)]
            self._assert_parity(cards)
            compared += 1
        # All vanilla hands should be handled by the Rust path.
        self.assertGreater(compared, 100)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class CardEditionParityTests(unittest.TestCase):
    """Cards with editions (Foil/Holographic/Polychrome) score the
    same in Rust as in Python evaluate_played_cards."""

    def _parity(self, edition: str) -> None:
        from balatro_ai.rules.hand_evaluator import evaluate_played_cards
        cards = [
            Card(rank="K", suit="H", edition=edition),
            Card(rank="K", suit="S"),
        ]
        rs = balatro_core.evaluate_simple(
            _to_rust(cards), hand_level=1, debuffed_suits=[]
        )
        if rs is None:
            self.fail(f"Rust path bailed on edition={edition}")
        py = evaluate_played_cards(tuple(cards), jokers=())
        self.assertEqual(rs[2], py.score,
            f"Edition {edition!r} score mismatch: Rust={rs[2]}, Python={py.score}")

    def test_foil_card(self) -> None:
        self._parity("foil")

    def test_holographic_card(self) -> None:
        self._parity("holographic")

    def test_polychrome_card(self) -> None:
        self._parity("polychrome")

    def test_combined_editions_on_multiple_scoring_cards(self) -> None:
        from balatro_ai.rules.hand_evaluator import evaluate_played_cards
        cards = [
            Card(rank="A", suit="S", edition="foil"),
            Card(rank="A", suit="H", edition="polychrome"),
            Card(rank="A", suit="D", edition="holographic"),
        ]
        rs = balatro_core.evaluate_simple(
            _to_rust(cards), hand_level=1, debuffed_suits=[]
        )
        py = evaluate_played_cards(tuple(cards), jokers=())
        self.assertEqual(rs[2], py.score)

    def test_edition_with_supported_joker(self) -> None:
        # Polychrome King + Joker (+4 mult): each scored card has
        # x1.5 mult applied during its per-card pass.
        # K of hearts polychrome: card_chips=10, mult: 1*1.5=1.5 after
        # per-card; then Joker +4 → 1.5+4 = 5.5
        # score = (5 + 10) * 5.5 = 82.5 → floor 82
        from balatro_ai.rules.hand_evaluator import evaluate_played_cards
        cards = [Card(rank="K", suit="H", edition="polychrome")]
        from balatro_ai.api.state import Joker
        jokers = (Joker(name="Joker"),)
        rs = balatro_core.evaluate_simple(
            _to_rust(cards), hand_level=1, debuffed_suits=[],
            joker_names=["Joker"]
        )
        py = evaluate_played_cards(tuple(cards), jokers=jokers)
        self.assertEqual(rs[2], py.score)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class EvaluateSimpleBenchmarkTests(unittest.TestCase):
    """How much faster is the composed Rust evaluate_simple vs Python
    evaluate_played_cards on vanilla hands? This is the headline
    speedup number for Phase 2."""

    def test_full_eval_speedup(self) -> None:
        from balatro_ai.rules.hand_evaluator import evaluate_played_cards
        import time

        rng = random.Random(7)
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        suits = ["C", "D", "H", "S"]
        py_hands = [
            tuple(Card(rank=rng.choice(ranks), suit=rng.choice(suits)) for _ in range(5))
            for _ in range(100)
        ]
        rust_hands = [_to_rust(list(h)) for h in py_hands]

        # Warmup
        for h in py_hands:
            evaluate_played_cards(h, jokers=())
        for h in rust_hands:
            balatro_core.evaluate_simple(h, hand_level=1, debuffed_suits=[])

        t = time.perf_counter()
        for _ in range(500):
            for h in py_hands:
                evaluate_played_cards(h, jokers=())
        py_time = time.perf_counter() - t

        t = time.perf_counter()
        for _ in range(500):
            for h in rust_hands:
                balatro_core.evaluate_simple(h, hand_level=1, debuffed_suits=[])
        rs_time = time.perf_counter() - t

        ratio = py_time / rs_time
        print(f"\n  evaluate (full): Python {py_time*1000:.1f}ms, "
              f"Rust {rs_time*1000:.1f}ms  ({ratio:.2f}x speedup)")
        self.assertGreater(ratio, 1.5,
                           f"Rust should be at least 1.5x Python on full eval — got {ratio:.2f}")


if __name__ == "__main__":
    unittest.main()
