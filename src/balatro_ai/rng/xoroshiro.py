"""LÖVE 11.x's PRNG: xoroshiro128+ seeded via Splitmix64.

Balatro's initial deck shuffle calls ``math.randomseed(pseudoseed('nr1'))`` and
then ``math.random(i)`` per Fisher-Yates swap. In LÖVE 11.x the Lua-level
``math.random`` is overridden to call ``love.math``'s C++ PRNG, which is
xoroshiro128+ with Splitmix64 used to expand a single seed value into the
two-uint64 state.

References:
  - love/src/modules/math/RandomGenerator.cpp
  - https://prng.di.unimi.it/xoroshiro128plus.c
  - https://prng.di.unimi.it/splitmix64.c

The float-to-seed-uint64 conversion that LÖVE applies to a Lua number is the
unresolved piece — multiple plausible interpretations exist and the empirical
match against captured fixtures will pick the right one. ``SeedStrategy``
enumerates the candidates so the validator can sweep them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import struct


U64_MASK = (1 << 64) - 1
SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15
SPLITMIX64_MUL_1 = 0xBF58476D1CE4E5B9
SPLITMIX64_MUL_2 = 0x94D049BB133111EB


def _u64(value: int) -> int:
    return value & U64_MASK


def _rotl(x: int, k: int) -> int:
    return _u64((x << k) | (x >> (64 - k)))


def splitmix64(state: int) -> tuple[int, int]:
    """Advance a Splitmix64 state by one step; return (new_state, output)."""

    state = _u64(state + SPLITMIX64_INCREMENT)
    z = state
    z = _u64((z ^ (z >> 30)) * SPLITMIX64_MUL_1)
    z = _u64((z ^ (z >> 27)) * SPLITMIX64_MUL_2)
    z = _u64(z ^ (z >> 31))
    return state, z


def splitmix64_init_state(seed_u64: int) -> tuple[int, int]:
    """Produce a 128-bit xoroshiro state from a 64-bit seed via Splitmix64.

    Matches LÖVE's setSeed: two consecutive Splitmix64 outputs become s[0]
    and s[1] of the xoroshiro state.
    """

    sm_state = seed_u64
    sm_state, s0 = splitmix64(sm_state)
    sm_state, s1 = splitmix64(sm_state)
    # xoroshiro requires non-zero state; if both halves are 0, perturb once.
    if s0 == 0 and s1 == 0:
        sm_state, s0 = splitmix64(sm_state)
        sm_state, s1 = splitmix64(sm_state)
    return s0, s1


@dataclass(slots=True)
class Xoroshiro128Plus:
    """In-place xoroshiro128+ generator.

    Same algorithm LÖVE 11.x ships in RandomGenerator.cpp::rand:

        s0 = state[0]; s1 = state[1]
        result = s0 + s1
        s1 ^= s0
        state[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14)
        state[1] = rotl(s1, 36)
        return result
    """

    s0: int = 0
    s1: int = 0

    @classmethod
    def from_seed_uint64(cls, seed_u64: int) -> "Xoroshiro128Plus":
        s0, s1 = splitmix64_init_state(_u64(seed_u64))
        return cls(s0=s0, s1=s1)

    def next_u64(self) -> int:
        s0 = self.s0
        s1 = self.s1
        result = _u64(s0 + s1)
        s1 ^= s0
        self.s0 = _u64(_rotl(s0, 55) ^ s1 ^ _u64(s1 << 14))
        self.s1 = _rotl(s1, 36)
        return result

    def next_double(self) -> float:
        """Match LÖVE's RandomGenerator::random() — map u64 to [0, 1).

        LÖVE uses the top 53 bits divided by 2^53 to produce a uniform double.
        """

        return (self.next_u64() >> 11) / (1 << 53)

    def random_int_inclusive(self, low: int, high: int) -> int:
        """Match LÖVE's ``math.random(low, high)`` mapping (inclusive).

        Uses LÖVE's exact construction: ``low + floor(random_double * span)``
        where ``span = high - low + 1``. ``math.random(n)`` is the same as
        ``math.random(1, n)``.
        """

        if high < low:
            raise ValueError(f"random_int_inclusive: high ({high}) < low ({low})")
        span = high - low + 1
        return low + int(self.next_double() * span)


class SeedStrategy(Enum):
    """How to derive a uint64 xoroshiro seed from a Lua float.

    LÖVE's wrap_RandomGenerator.cpp casts ``(uint64)luaL_checknumber(L, 2)``,
    which truncates a Lua double's fractional part — meaning a value in
    [0, 1) becomes 0. That would make all run seeds shuffle identically,
    which the captured fixtures contradict, so one of these alternatives
    must be what the game actually does.
    """

    # Direct (uint64) cast — matches LÖVE source literally, expected to fail.
    TRUNCATE_TO_UINT64 = "truncate_uint64"
    # Reinterpret the float's 8 bytes as a uint64 (preserves all bits).
    REINTERPRET_BITS = "reinterpret_bits"
    # Scale to 2^32 (common in older LÖVE / community decoders).
    SCALE_BY_2_POW_32 = "scale_2e32"
    # Scale to 2^53 (the full mantissa range).
    SCALE_BY_2_POW_53 = "scale_2e53"
    # Scale to 2^64 (full uint64 range, with wraparound from rounding).
    SCALE_BY_2_POW_64 = "scale_2e64"


def derive_seed_uint64(value: float, strategy: SeedStrategy) -> int:
    if strategy == SeedStrategy.TRUNCATE_TO_UINT64:
        if value < 0:
            return 0
        return _u64(int(value))
    if strategy == SeedStrategy.REINTERPRET_BITS:
        packed = struct.pack("<d", float(value))
        return int.from_bytes(packed, byteorder="little", signed=False)
    if strategy == SeedStrategy.SCALE_BY_2_POW_32:
        return _u64(int(value * (1 << 32)))
    if strategy == SeedStrategy.SCALE_BY_2_POW_53:
        return _u64(int(value * (1 << 53)))
    if strategy == SeedStrategy.SCALE_BY_2_POW_64:
        # Multiplying a [0,1) float by 2^64 overflows Python int range fine
        # but on conversion the wrap is captured by _u64.
        return _u64(int(value * (1 << 64)))
    raise ValueError(f"Unknown seed strategy: {strategy}")


def pseudoshuffle_with_xoroshiro(
    items: list,
    seed_float: float,
    strategy: SeedStrategy,
) -> None:
    """Balatro's ``pseudoshuffle`` ported to Python.

    Seeds xoroshiro128+ from ``seed_float`` using ``strategy``, then does
    Fisher-Yates with ``math.random(i)`` semantics (inclusive [1, i]).
    Mutates ``items`` in place.
    """

    seed_u64 = derive_seed_uint64(seed_float, strategy)
    rng = Xoroshiro128Plus.from_seed_uint64(seed_u64)
    for i in range(len(items) - 1, 0, -1):
        # LÖVE's math.random(i) returns [1, i]; convert to 0-indexed swap.
        j = rng.random_int_inclusive(1, i + 1) - 1
        items[i], items[j] = items[j], items[i]
