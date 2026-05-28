"""Construct Balatro starting decks before the initial shuffle.

The order cards are inserted into the deck matters: ``pseudorandom_shuffle``
applies Fisher-Yates over that order, so swapping the insertion order
produces a completely different shuffle output for the same seed. This module
encodes Balatro's actual insertion order, recovered empirically from
bridge-captured fixtures (`.data/rng-validation/seed_*.json`).

VERIFIED order (2026-05-24, from card IDs 78–129 in captured Red decks):
  * Suit traversal: C, D, H, S (alphabetical, from Lua table iteration).
  * Rank traversal within suit: 2, 3, 4, 5, 6, 7, 8, 9, A, J, K, Q, T —
    lexicographic by rank string, NOT poker-ascending. Ace comes before
    Jack, King precedes Queen precedes Ten.
  * Suit-outer, rank-inner: C2..CT (13 cards), D2..DT, H2..HT, S2..ST.

The lexicographic rank order is suspicious enough to call out: it comes from
Lua's ``pairs()`` iterating the rank-keyed table, which on string keys is
hash order — which on this tiny static table empirically lands at lex order.
If a future Balatro version changes the iteration, this assumption breaks
and the audit will catch it.

Other decks (Blue gives +1 hand, Yellow starts with +$10, etc.) differ in
modifiers, not card composition for the Red baseline. Special decks like
Stuffed-Suit will need their own constructors; out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass


# Order matters — these are the iteration orders observed in captured
# fixtures. See module docstring for the recovery method.
STANDARD_SUITS_CDHS_ORDER = ("C", "D", "H", "S")
STANDARD_RANKS_LEXICOGRAPHIC = ("2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q", "T")

# Kept under the old name for any caller that still references it; both
# point at the new verified order.
STANDARD_SUITS_SHCD_ORDER = STANDARD_SUITS_CDHS_ORDER
STANDARD_RANKS_ASCENDING = STANDARD_RANKS_LEXICOGRAPHIC


@dataclass(frozen=True, slots=True)
class StandardCard:
    rank: str
    suit: str

    @property
    def short(self) -> str:
        return f"{self.rank}{self.suit}"


def build_standard_red_deck() -> tuple[StandardCard, ...]:
    """Return the 52-card Red-deck baseline in verified insertion order.

    Suit-outer (C,D,H,S), rank-inner (2..9, A, J, K, Q, T) — see module
    docstring for the empirical recovery. Returns a tuple to make the
    canonical order obvious; the caller converts to a list for shuffling.
    """

    return tuple(
        StandardCard(rank=rank, suit=suit)
        for suit in STANDARD_SUITS_CDHS_ORDER
        for rank in STANDARD_RANKS_LEXICOGRAPHIC
    )


def build_standard_red_deck_short_names() -> tuple[str, ...]:
    return tuple(card.short for card in build_standard_red_deck())
