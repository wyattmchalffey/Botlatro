"""Content-keyed state signatures for solver memoization (Tier 1 #3).

This module builds a hashable digest of the GameState fields that
affect the solver's leaf-value functions. The digest is the cache
key for `search_v2.memo` wrappers around `clear_probability` and
`planning_value`.

Why we can't reuse `state_value._search_state_cache_key`:
that key (correctly) excludes `current_score` and `required_score`
because the helpers it wraps (`_best_immediate_score`,
`_best_greedy_play_action`) don't care about those fields — they
operate over the hand + jokers + blind shape, not the run state.
But `clear_probability` and `planning_value` BOTH depend on
`(required_score - current_score)` for their answer. Caching them
on the existing key produces silently wrong values when two
states share hand/jokers but differ in score.

The `StateSignature` here is intentionally LARGER than the existing
key. We err on the side of "include every field that could plausibly
matter" because a stale cache hit returns a wrong answer that
propagates into trajectory decisions, and there's no easy way to
detect the regression post-hoc. Better to take a smaller hit rate
than to silently lose correctness.

Fields included:
- All scalar run state: ante, blind, scores, hands/discards remaining,
  money, deck_size, won, run_over, phase
- Hand multiset (sorted, content-keyed)
- Jokers (key + edition + counter)
- Consumables, vouchers (as frozen tuples)
- Hand levels
- Modifiers (frozen dict)
- Known deck multiset (drawable cards)

Verified empirically: see `test_solver_search_v2_memo.py` for the
parity test that runs cached vs uncached on representative states
and asserts identical outputs.
"""

from __future__ import annotations

from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.search.state_value import (
    _freeze_for_cache,
    _hand_multiset_cache_key,
    _jokers_cache_key,
)


def state_signature(state: GameState) -> tuple:
    """Content-keyed digest of `state` for memoization.

    Returns a hashable, deeply-content-based tuple. Two states with
    identical relevant fields produce identical signatures; states
    that differ on any of the included fields produce different
    signatures.

    Performance: O(|hand| + |jokers| + |known_deck| + |consumables| +
    |modifiers|) per call. For typical mid-blind states that's
    roughly 30-50 hashable items — fast enough that we call this
    on every leaf state without measurable overhead vs the
    leaf-eval cost saved.

    Returns
    -------
    A tuple suitable for use as a dict / lru_cache key.
    """

    return (
        # Scalar run-state fields (each one can change the leaf value).
        state.phase,
        state.ante,
        state.blind,
        state.required_score,
        state.current_score,
        state.hands_remaining,
        state.discards_remaining,
        state.money,
        state.deck_size,
        state.won,
        state.run_over,
        # Hand multiset — sorted so different orderings hash the same.
        _hand_multiset_cache_key(state.hand),
        # Known deck multiset (drawable cards in play).
        _hand_multiset_cache_key(state.known_deck),
        # Jokers in order (order matters for triggers).
        _jokers_cache_key(state.jokers),
        # Hand levels (planet-card upgrades affect scoring).
        _freeze_for_cache(state.hand_levels),
        # Consumables (held tarots/spectrals/planets).
        tuple(state.consumables),
        # Vouchers (passive effects affecting scoring + rollouts).
        tuple(state.vouchers),
        # Modifiers (joker_slot_limit, debuffed_suits, round counters).
        _freeze_for_cache(state.modifiers),
    )
