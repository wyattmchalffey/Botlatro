"""Parity tests for `balatro_core.RustJoker` (Phase 1 of RUST_PORT_PLAN.md).

Joker round-trip preserves: name, edition, sell_value.
NOT preserved (documented): metadata, derived `effect`. Code that
reads `joker.metadata` or `joker.effect` should never go through a
Rust round-trip.
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.state import Joker

try:
    import balatro_core
    BALATRO_CORE_AVAILABLE = True
except ImportError:
    balatro_core = None
    BALATRO_CORE_AVAILABLE = False


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class RustJokerRoundTripTests(unittest.TestCase):
    def test_name_only(self) -> None:
        original = Joker(name="Jolly Joker")
        rt = balatro_core.RustJoker.from_python(original).to_python()
        self.assertEqual(rt.name, "Jolly Joker")
        self.assertIsNone(rt.edition)
        self.assertIsNone(rt.sell_value)

    def test_with_edition(self) -> None:
        original = Joker(name="Smeared Joker", edition="foil")
        rt = balatro_core.RustJoker.from_python(original).to_python()
        self.assertEqual(rt.name, "Smeared Joker")
        self.assertEqual(rt.edition, "foil")

    def test_with_sell_value(self) -> None:
        original = Joker(name="Joker", sell_value=2)
        rt = balatro_core.RustJoker.from_python(original).to_python()
        self.assertEqual(rt.sell_value, 2)

    def test_all_editions(self) -> None:
        for edition in ("foil", "holographic", "polychrome", "negative"):
            original = Joker(name="X", edition=edition)
            rt = balatro_core.RustJoker.from_python(original).to_python()
            self.assertEqual(rt.edition, edition, f"edition {edition} mismatch")

    def test_unknown_edition_drops_to_none(self) -> None:
        original = Joker(name="X", edition="totally_made_up")
        rt = balatro_core.RustJoker.from_python(original).to_python()
        self.assertIsNone(rt.edition)

    def test_round_trip_drops_metadata(self) -> None:
        # Documented divergence — metadata is intentionally NOT
        # preserved across the Rust round trip. If you need metadata
        # post-round-trip, don't go through Rust.
        original = Joker(name="X", metadata={"counter": 5})
        rt = balatro_core.RustJoker.from_python(original).to_python()
        self.assertEqual(rt.metadata, {})


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class RustJokerGetterTests(unittest.TestCase):
    def test_getters_return_canonical(self) -> None:
        original = Joker(name="Brainstorm", edition="Polychrome", sell_value=5)
        rs = balatro_core.RustJoker.from_python(original)
        self.assertEqual(rs.name, "Brainstorm")
        self.assertEqual(rs.edition, "polychrome")
        self.assertEqual(rs.sell_value, 5)


if __name__ == "__main__":
    unittest.main()
