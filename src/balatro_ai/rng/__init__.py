"""Balatro-faithful pseudorandom number generation.

Balatro does not use a standard PRNG for game choices. It uses two layers:

1. ``pseudohash(str)`` — deterministic float-walk over a string, returning a
   value in [0, 1). Used to derive initial per-event state.
2. ``pseudoseed(key)`` — per-key iterated float walk that yields a value in
   [0, 1) per call. First call for a given key bootstraps from
   ``pseudohash(key + seed)``; subsequent calls iterate
   ``x = (x * 1.72431234 + 2.134453429141) % 1`` and return
   ``(x + hashed_seed) / 2``.

This package provides Python re-implementations of those primitives plus the
helpers ``pseudorandom_element`` and ``pseudorandom_shuffle`` so the offline
solver can predict shop rolls, deck shuffles, and pack contents from a run
seed without consulting the live game.

See `src/balatro_ai/rng/pseudohash.py` for primitives and
`src/balatro_ai/rng/balatro_rng.py` for the stateful ``BalatroRNG``.
"""

from balatro_ai.rng.pseudohash import (
    PSEUDOSEED_ADD,
    PSEUDOSEED_MUL,
    pseudohash,
    pseudoseed_step,
)
from balatro_ai.rng.balatro_rng import BalatroRNG, pseudorandom_element, pseudorandom_shuffle
from balatro_ai.rng.deck import StandardCard, build_standard_red_deck, build_standard_red_deck_short_names
from balatro_ai.rng.surfaces import (
    InitialRunSurface,
    PredictedCard,
    ShopSurface,
    poll_edition,
    predict_boss,
    predict_booster_pack,
    predict_initial_surface,
    predict_ouija_rank,
    predict_pack_contents,
    predict_shop_boosters,
    predict_shop_cards,
    predict_shop_surface,
    predict_sigil_suit,
    predict_spectral_created_cards,
    predict_tag,
    predict_voucher,
)

__all__ = [
    "BalatroRNG",
    "InitialRunSurface",
    "PSEUDOSEED_ADD",
    "PSEUDOSEED_MUL",
    "PredictedCard",
    "ShopSurface",
    "StandardCard",
    "build_standard_red_deck",
    "build_standard_red_deck_short_names",
    "poll_edition",
    "predict_boss",
    "predict_booster_pack",
    "predict_initial_surface",
    "predict_ouija_rank",
    "predict_pack_contents",
    "predict_shop_boosters",
    "predict_shop_cards",
    "predict_shop_surface",
    "predict_sigil_suit",
    "predict_spectral_created_cards",
    "predict_tag",
    "predict_voucher",
    "pseudohash",
    "pseudorandom_element",
    "pseudorandom_shuffle",
    "pseudoseed_step",
]
