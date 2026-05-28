"""Content-keyed memo wrappers for solver leaf evaluation (Tier 1 #3).

Wraps `clear_probability` and `planning_value` with a
per-process dict cache keyed on `state_signature`. The cache is
opt-in: callers use the `solver_search_cache_scope()` context
manager to activate it for the duration of one decision (or one
top-level call) and the cache is torn down on exit. Per-worker
caches in the multiprocessing pool stay isolated from each other.

Why a dict + scope, not `functools.lru_cache`:
- The cache lives only for one decision; we don't want cross-decision
  pollution (states evolve, the cache would grow without bound).
- The cache scope context manager makes the activation explicit at
  the call site — no surprising "this slow call is actually cached
  somewhere I forgot about" behavior.

API:
    with solver_search_cache_scope():
        action = solver_beam_play_action(state, ...)

Inside the scope, `cached_clear_probability(state, samples, seed)`
and `cached_planning_value(state, samples, seed)` hit a content-keyed
cache. Outside the scope they call through to the underlying
`state_value` functions directly. The leaf evaluators in
`solver.search_v2.leaf_value` use these wrappers automatically.

Correctness guarantee: the `StateSignature` is a content-keyed
digest of EVERY field that affects the wrapped function's output.
A parity test in `test_solver_search_v2_memo.py` runs cached vs
uncached on representative states and asserts identical results.
If a new field starts to matter for clear_probability (e.g. a new
joker effect), update `state_signature.py` and re-run the parity
test.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import local

from balatro_ai.api.state import GameState
from balatro_ai.search.state_value import (
    clear_probability as _uncached_clear_probability,
    planning_value as _uncached_planning_value,
)
from balatro_ai.solver.search_v2.state_signature import state_signature


_CACHE_LOCAL = local()


@contextmanager
def solver_search_cache_scope():
    """Activate the content-keyed cache for one solver decision.

    Use as a context manager wrapping a single `solver_beam_play_action`
    call (or any other consumer of `cached_clear_probability` /
    `cached_planning_value`). The cache is torn down on exit so
    state-evolution between decisions doesn't poison the next call.
    """

    previous = getattr(_CACHE_LOCAL, "cache", None)
    _CACHE_LOCAL.cache = {}
    try:
        yield
    finally:
        if previous is None:
            try:
                del _CACHE_LOCAL.cache
            except AttributeError:
                pass
        else:
            _CACHE_LOCAL.cache = previous


def _current_cache() -> dict | None:
    return getattr(_CACHE_LOCAL, "cache", None)


def cached_clear_probability(
    state: GameState,
    *,
    samples: int = 64,
    seed: int = 0,
) -> float:
    """Content-keyed cache wrapper around `state_value.clear_probability`.

    When called inside a `solver_search_cache_scope()`, identical-content
    states share a cached result. Outside the scope, this is a
    transparent pass-through to the underlying function (no caching).
    """

    cache = _current_cache()
    if cache is None:
        return _uncached_clear_probability(state, samples=samples, seed=seed)
    key = ("clear", state_signature(state), samples, seed)
    if key not in cache:
        cache[key] = _uncached_clear_probability(state, samples=samples, seed=seed)
    return cache[key]


def cached_planning_value(
    state: GameState,
    *,
    samples: int = 16,
    seed: int = 0,
) -> float:
    """Content-keyed cache wrapper around `state_value.planning_value`.

    Same semantics as `cached_clear_probability` — caches inside a
    `solver_search_cache_scope()`, transparent pass-through outside.
    """

    cache = _current_cache()
    if cache is None:
        return _uncached_planning_value(state, samples=samples, seed=seed)
    key = ("planning", state_signature(state), samples, seed)
    if key not in cache:
        cache[key] = _uncached_planning_value(state, samples=samples, seed=seed)
    return cache[key]


def cache_stats() -> tuple[int, int]:
    """Return (entries, scoped_active) for introspection in tests."""

    cache = _current_cache()
    if cache is None:
        return (0, 0)
    return (len(cache), 1)
