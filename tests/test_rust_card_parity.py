"""Parity test for `balatro_core` Rust port (proof-of-concept).

For every function ported to Rust, the corresponding Python
implementation is treated as ground truth. The test runs both on
the same input corpus and asserts identical output.

This is the correctness gate for the Rust port — a port that drifts
silently from the Python implementation is much worse than no port
at all, because it produces subtly-wrong trajectories that we
can't trace back to the cause.

Skipped silently if `balatro_core` isn't installed (Python-only
contributors aren't blocked by the Rust toolchain).
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


def _card(rank: str = "A", suit: str = "S", enhancement: str | None = None) -> Card:
    return Card(rank=rank, suit=suit, enhancement=enhancement)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class IsStoneCardParityTests(unittest.TestCase):
    """`balatro_core.is_stone_card` must match
    `balatro_ai.rules.hand_evaluator._is_stone_card` on every input."""

    def _assert_parity(self, card: Card) -> None:
        from balatro_ai.rules.hand_evaluator import _is_stone_card
        py = _is_stone_card(card)
        rs = balatro_core.is_stone_card(card)
        self.assertEqual(
            rs, py,
            f"is_stone_card divergence on rank={card.rank} suit={card.suit} "
            f"enhancement={card.enhancement!r}: Rust={rs}, Python={py}",
        )

    def test_no_enhancement(self) -> None:
        self._assert_parity(_card())

    def test_stone_lowercase(self) -> None:
        self._assert_parity(_card(enhancement="stone"))

    def test_stone_card_with_space(self) -> None:
        self._assert_parity(_card(enhancement="stone card"))

    def test_stone_mixed_case(self) -> None:
        self._assert_parity(_card(enhancement="Stone Card"))

    def test_stone_m_prefix(self) -> None:
        self._assert_parity(_card(enhancement="m_stone"))

    def test_stone_m_prefix_with_underscore(self) -> None:
        self._assert_parity(_card(enhancement="m_stone_card"))

    def test_bonus_enhancement_is_not_stone(self) -> None:
        self._assert_parity(_card(enhancement="bonus"))

    def test_wild_enhancement_is_not_stone(self) -> None:
        self._assert_parity(_card(enhancement="wild"))

    def test_glass_enhancement_is_not_stone(self) -> None:
        self._assert_parity(_card(enhancement="m_glass"))

    def test_all_ranks_with_stone(self) -> None:
        # The is_stone check doesn't depend on rank/suit but we
        # verify it's invariant just in case.
        for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"):
            for suit in ("C", "D", "H", "S"):
                self._assert_parity(_card(rank=rank, suit=suit, enhancement="stone"))

    def test_all_ranks_with_no_enhancement(self) -> None:
        for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"):
            for suit in ("C", "D", "H", "S"):
                self._assert_parity(_card(rank=rank, suit=suit))


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class IsStoneCardTimingTests(unittest.TestCase):
    """Sanity timing check — Rust should be at least as fast as Python."""

    def test_rust_at_least_competitive_with_python(self) -> None:
        from balatro_ai.rules.hand_evaluator import _is_stone_card
        import time

        cards = [
            _card(enhancement=enh)
            for enh in (None, "stone", "Stone Card", "bonus", "m_stone")
        ] * 200  # 1000 cards, mixed

        # Warmup
        for c in cards:
            _is_stone_card(c)
            balatro_core.is_stone_card(c)

        t = time.perf_counter()
        for _ in range(100):
            for c in cards:
                _is_stone_card(c)
        py_time = time.perf_counter() - t

        t = time.perf_counter()
        for _ in range(100):
            for c in cards:
                balatro_core.is_stone_card(c)
        rs_time = time.perf_counter() - t

        # Don't require a specific speedup — FFI overhead + the
        # tightness of the original Python function (single dict
        # lookup) means Rust might not be faster on this specific
        # function. We just want to confirm it's not catastrophically
        # slower. The real speedup target is the COMPOSITE evaluation
        # path (evaluate_played_cards), not standalone is_stone_card.
        print(f"\n  is_stone_card: Python {py_time*1000:.1f}ms, Rust {rs_time*1000:.1f}ms"
              f" (ratio {py_time/rs_time:.2f}x)")
        self.assertLess(rs_time, py_time * 5.0,
                        "Rust shouldn't be more than 5x slower than Python")


if __name__ == "__main__":
    unittest.main()
