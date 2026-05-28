"""Stateful Balatro pseudorandom number generator.

Wraps the primitives in `pseudohash.py` into the same API the live game
exposes to gameplay code: a single ``random(key)`` call returns one value per
event key, with state bootstrapped on first use and iterated on subsequent
calls.

The state mirrors Lua's ``G.GAME.pseudorandom`` table:
  * ``seed``: the original run seed string (e.g. "AAAAAAA").
  * ``hashed_seed``: ``pseudohash(seed)``, added to every returned value.
  * per-key floats: ``pseudohash(key + seed)`` on first call, iterated after.

Reset semantics match Balatro: a new ``BalatroRNG`` instance represents a
fresh run; per-key state persists for the run's lifetime so consecutive
``random("shop_pack")`` calls produce different values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSequence
from dataclasses import dataclass, field
import math
from typing import TypeVar

from balatro_ai.rng.pseudohash import pseudohash, pseudoseed_step


T = TypeVar("T")


@dataclass(slots=True)
class BalatroRNG:
    """Per-run pseudorandom state with per-key streams.

    Construct with ``BalatroRNG(seed="AAAAAAA")``. Call ``random(key)`` to
    consume the next value for ``key``. State is mutated in place. Two
    instances with the same seed produce identical streams.

    The ``mix_hashed_seed`` flag controls Balatro's later-patch coupling
    where the returned value is ``(state + hashed_seed) / 2`` rather than
    just ``state``. Default is False (original algorithm) because the
    coupling halves the output range and prevents uniform index selection.
    Toggle to True if cross-validation against a Balatro build that uses
    the coupled formula requires it.
    """

    seed: str
    # Kept for back-compat with old tests; default is now True because the
    # Balatro source (verified in functions/misc_functions.lua:312) always
    # applies the ``(state + hashed_seed) / 2`` final transform.
    mix_hashed_seed: bool = True
    hashed_seed: float = field(init=False)
    _state: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.hashed_seed = pseudohash(self.seed)

    def random(self, key: str) -> float:
        """Return the next pseudoseed value for ``key`` in [0, 1).

        Bootstraps from ``pseudohash(key + seed)`` on first use, then
        iterates ``pseudoseed_step`` on every call. With
        ``mix_hashed_seed=True`` (the verified Balatro behavior) the
        returned value is ``(state + hashed_seed) / 2``. The False variant
        is preserved only for back-compat tests / debugging.
        """

        if key == "seed":
            raise ValueError("'seed' is reserved by Balatro and would return math.random()")
        if key not in self._state:
            self._state[key] = pseudohash(key + self.seed)
        self._state[key] = pseudoseed_step(self._state[key])
        if self.mix_hashed_seed:
            return (self._state[key] + self.hashed_seed) / 2
        return self._state[key]

    def snapshot(self) -> Mapping[str, float]:
        """Return a read-only view of the per-key state (for debugging/tests)."""

        return dict(self._state)


def pseudorandom_element(items: Iterable[T], rng: BalatroRNG, key: str) -> T:
    """Pick one element of ``items`` deterministically using ``rng[key]``.

    Mirrors Balatro's ``pseudorandom_element`` (functions.lua): iterates the
    container, sorts string keys to match Lua's ``ipairs/pairs`` semantics
    for tables, and picks ``ceil(rng.random(key) * len(items))``.
    """

    items_list = list(items)
    if not items_list:
        raise ValueError("Cannot pick a pseudorandom element from an empty iterable")
    value = rng.random(key)
    index = math.ceil(value * len(items_list))
    # ceil maps (0, 1] -> {1..N}. The vanishingly rare value=0.0 maps to 0;
    # Lua handles this with `math.ceil(0*N) == 0` and indexes [0] returning nil,
    # which game code never observes because the value is bootstrapped > 0.
    index = max(1, min(index, len(items_list))) - 1
    return items_list[index]


def pseudorandom_shuffle(items: MutableSequence[T], rng: BalatroRNG, key: str) -> None:
    """Fisher-Yates shuffle ``items`` in place using the same key sequence Balatro does.

    Matches Lua's:

        for i = #_t, 2, -1 do
            local j = math.ceil(pseudoseed(seed)*i)
            _t[i], _t[j] = _t[j], _t[i]
        end

    The shuffle calls ``rng.random(key)`` once per swap, so the same key may
    be consumed many times in a single shuffle. Subsequent uses of the same
    key will pick up where the shuffle left off — this is the same coupling
    Balatro has, and the solver depends on it.
    """

    for i in range(len(items) - 1, 0, -1):
        value = rng.random(key)
        # i here is 0-indexed end; Lua's i is 1-indexed and the multiplication
        # is by the 1-indexed length, so use (i + 1) to match.
        target = math.ceil(value * (i + 1))
        j = max(1, min(target, i + 1)) - 1
        items[i], items[j] = items[j], items[i]
