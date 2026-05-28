"""Parity tests for `balatro_core.RustCard` (Phase 1 of RUST_PORT_PLAN.md).

Validates that the Rust `Card` representation:
1. Accepts every valid Python `Card` via `from_python(py_card)`.
2. Round-trips back to Python preserving visible fields (rank, suit,
   enhancement, edition, seal, debuffed). Metadata is NOT preserved —
   it's deliberately dropped on the Rust side (documented in
   `state/card.rs`).
3. Has correct enum normalization (e.g. "Stone Card" → "stone",
   "holo" → "holographic").

The hot-path performance test for RustCard (vs Python Card) comes
later when we have hot-path Rust operations to compare. Card
construction alone is FFI-overhead-dominated (same trap that hit
the `is_stone_card` PoC) so we don't bench it standalone.
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.state import Card

try:
    import balatro_core
    BALATRO_CORE_AVAILABLE = True
except ImportError:
    balatro_core = None
    BALATRO_CORE_AVAILABLE = False


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class RustCardRoundTripTests(unittest.TestCase):
    """A round-tripped Card must equal the original on every visible
    field. Metadata is the documented exception."""

    # Sentinel: "caller didn't override the expected value, use the
    # input as-is". Distinct from None which means "I expect None".
    _USE_INPUT = object()

    def _assert_round_trip(
        self,
        rank: str,
        suit: str,
        *,
        enhancement: str | None = None,
        edition: str | None = None,
        seal: str | None = None,
        debuffed: bool = False,
        expected_enhancement: object = _USE_INPUT,
        expected_edition: object = _USE_INPUT,
        expected_seal: object = _USE_INPUT,
    ) -> None:
        """Run rank/suit/enhancement etc through the Rust round trip.

        `expected_*` lets the caller assert on a normalized form
        (e.g. input "Stone Card" → expected "stone") when input
        and Rust-canonical form differ. Pass `None` to assert the
        round-trip drops the field; omit to assert it matches input.
        """

        original = Card(
            rank=rank,
            suit=suit,
            enhancement=enhancement,
            edition=edition,
            seal=seal,
            debuffed=debuffed,
        )
        rs = balatro_core.RustCard.from_python(original)
        rt = rs.to_python()

        self.assertEqual(rt.rank, rank, "rank changed")
        self.assertEqual(rt.suit, suit, "suit changed")
        self.assertEqual(
            rt.enhancement,
            enhancement if expected_enhancement is self._USE_INPUT else expected_enhancement,
            "enhancement mismatch",
        )
        self.assertEqual(
            rt.edition,
            edition if expected_edition is self._USE_INPUT else expected_edition,
            "edition mismatch",
        )
        self.assertEqual(
            rt.seal,
            seal if expected_seal is self._USE_INPUT else expected_seal,
            "seal mismatch",
        )
        self.assertEqual(rt.debuffed, debuffed)

    def test_plain_card(self) -> None:
        self._assert_round_trip("A", "S")

    def test_all_ranks(self) -> None:
        for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"):
            self._assert_round_trip(rank, "H")

    def test_all_suits(self) -> None:
        for suit in ("C", "D", "H", "S"):
            self._assert_round_trip("9", suit)

    def test_enhancement_stone_normalizes(self) -> None:
        # "Stone Card" → "stone" (Rust canonical form).
        self._assert_round_trip(
            "9", "H",
            enhancement="Stone Card",
            expected_enhancement="stone",
        )

    def test_enhancement_m_prefix_normalizes(self) -> None:
        # "m_stone" → "stone"
        self._assert_round_trip(
            "9", "H",
            enhancement="m_stone",
            expected_enhancement="stone",
        )

    def test_enhancement_all_known(self) -> None:
        for name in ("bonus", "mult", "wild", "glass", "steel", "stone", "gold", "lucky"):
            self._assert_round_trip("9", "H", enhancement=name, expected_enhancement=name)

    def test_edition_holo_alias_normalizes(self) -> None:
        self._assert_round_trip(
            "9", "H",
            edition="holo",
            expected_edition="holographic",
        )

    def test_edition_all_known(self) -> None:
        for name in ("foil", "holographic", "polychrome", "negative"):
            self._assert_round_trip("9", "H", edition=name, expected_edition=name)

    def test_seal_all_known(self) -> None:
        for name in ("red", "blue", "gold", "purple"):
            self._assert_round_trip("9", "H", seal=name, expected_seal=name)

    def test_debuffed_preserved(self) -> None:
        self._assert_round_trip("A", "C", debuffed=True)

    def test_unknown_enhancement_drops_to_none(self) -> None:
        # Rust treats unknown enhancements as None (consistent with
        # how the Python hand_evaluator handles unrecognized values
        # via _normalize_effect_name returning a non-matching string).
        self._assert_round_trip(
            "5", "D",
            enhancement="not_a_real_enhancement",
            expected_enhancement=None,
        )


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class RustCardGetterTests(unittest.TestCase):
    """The PyO3 `#[getter]` accessors return Python-canonical strings."""

    def test_getters_return_canonical_strings(self) -> None:
        py = Card(
            rank="K",
            suit="D",
            enhancement="Gold Card",
            edition="Foil",
            seal="Blue",
            debuffed=True,
        )
        rs = balatro_core.RustCard.from_python(py)
        self.assertEqual(rs.rank, "K")
        self.assertEqual(rs.suit, "D")
        self.assertEqual(rs.enhancement, "gold")
        self.assertEqual(rs.edition, "foil")
        self.assertEqual(rs.seal, "blue")
        self.assertTrue(rs.debuffed)

    def test_getters_return_none_for_empty(self) -> None:
        py = Card(rank="2", suit="S")
        rs = balatro_core.RustCard.from_python(py)
        self.assertIsNone(rs.enhancement)
        self.assertIsNone(rs.edition)
        self.assertIsNone(rs.seal)
        self.assertFalse(rs.debuffed)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class RustCardErrorTests(unittest.TestCase):
    """Bad input from Python should raise rather than silently corrupt."""

    def test_unknown_rank_raises(self) -> None:
        py = Card(rank="Z", suit="S")
        with self.assertRaises(ValueError):
            balatro_core.RustCard.from_python(py)

    def test_unknown_suit_raises(self) -> None:
        py = Card(rank="9", suit="X")
        with self.assertRaises(ValueError):
            balatro_core.RustCard.from_python(py)


if __name__ == "__main__":
    unittest.main()
