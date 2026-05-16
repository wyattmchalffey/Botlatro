"""Decision-scoped caches for the basic strategy bot."""

from __future__ import annotations

from contextlib import contextmanager
from threading import local

from balatro_ai.api.state import Card, GameState, Joker


_DECISION_CACHE_LOCAL = local()


def _decision_cached(key: tuple[object, ...], factory):
    cache = _current_decision_cache()
    if cache is None:
        return factory()
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _current_decision_cache() -> dict[tuple[object, ...], object] | None:
    return getattr(_DECISION_CACHE_LOCAL, "cache", None)


@contextmanager
def decision_cache_scope():
    previous_cache = _current_decision_cache()
    _DECISION_CACHE_LOCAL.cache = {}
    try:
        yield
    finally:
        if previous_cache is None:
            try:
                del _DECISION_CACHE_LOCAL.cache
            except AttributeError:
                pass
        else:
            _DECISION_CACHE_LOCAL.cache = previous_cache


def _sample_build_score_cache_key(state: GameState, jokers: tuple[Joker, ...]) -> tuple[object, ...]:
    state_key = _identity_cached_value(
        "sample_build_score_state_key",
        state,
        lambda: (
            state.ante,
            state.blind,
            state.hands_remaining,
            state.discards_remaining,
            state.money,
            state.deck_size,
            _freeze_for_cache(state.hand_levels),
            _freeze_for_cache(state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))),
            _freeze_for_cache(state.hand),
            _freeze_for_cache(state.known_deck),
            tuple(state.consumables),
        ),
    )
    jokers_key = tuple(_joker_cache_key(joker) for joker in jokers)
    return ("sample_build_score", state_key, jokers_key)


def _joker_cache_key(joker: Joker) -> object:
    return _identity_cached_value("joker_content_key", joker, lambda: _freeze_for_cache(joker))


def _card_cache_key(card: Card) -> object:
    return _identity_cached_value(
        "card_content_key",
        card,
        lambda: (
            "Card",
            card.rank,
            card.suit,
            card.enhancement,
            card.seal,
            card.edition,
            card.debuffed,
            _freeze_for_cache(card.metadata),
        ),
    )


def _identity_cached_value(kind: str, obj: object, factory):
    cache = _current_decision_cache()
    if cache is None:
        return factory()
    key = (kind, id(obj))
    cached = cache.get(key)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] is obj:
        return cached[1]
    value = factory()
    cache[key] = (obj, value)
    return value


def _state_scoped_cache(kind: str, state: GameState) -> dict[tuple[object, ...], object] | None:
    cache = _current_decision_cache()
    if cache is None:
        return None
    return _identity_cached_value(kind, state, dict)


def _decision_scoped_cache(kind: str) -> dict[tuple[object, ...], object] | None:
    cache = _current_decision_cache()
    if cache is None:
        return None
    key = ("decision_scoped_cache", kind)
    bucket = cache.get(key)
    if isinstance(bucket, dict):
        return bucket
    bucket = {}
    cache[key] = bucket
    return bucket


def _freeze_for_cache(value: object) -> object:
    if isinstance(value, Card):
        return _card_cache_key(value)
    if isinstance(value, Joker):
        return (
            "Joker",
            value.name,
            value.edition,
            value.sell_value,
            _freeze_for_cache(value.metadata),
        )
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_for_cache(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_for_cache(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted(_freeze_for_cache(item) for item in value))
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)
