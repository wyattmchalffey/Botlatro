"""Bit-exact port of LuaJIT 2.1's math.random / math.randomseed.

Balatro is a LOVE 11.x game, and LOVE uses LuaJIT 2.1 for Lua. Importantly,
LOVE does NOT override the stock ``math.random`` / ``math.randomseed`` — only
its own ``love.math.random``. So Balatro's ``math.randomseed(seed_float)``
+ ``math.random(i)`` (which is what ``pseudoshuffle`` calls) routes through
LuaJIT, not LOVE.

LuaJIT's PRNG is **Tausworthe-223 (TW223)** with four 64-bit state words.
Seeding does NOT cast the input double to an integer — it reinterprets the
bits after multiplying by pi and adding e:

    static void random_seed(PRNGState *rs, double d) {
      uint32_t r = 0x11090601;
      for (int i = 0; i < 4; i++) {
        uint32_t m = 1u << (r&255);
        r >>= 8;
        u.d = d = d * 3.14159265358979323846 + 2.7182818284590452354;
        if (u.u64 < m) u.u64 += m;
        rs->u[i] = u.u64;
      }
      for (int i = 0; i < 10; i++) (void)lj_prng_u64(rs);
    }

This means small fractional inputs DO produce distinct PRNG seeds, which is
how Balatro's per-seed deck shuffles end up different even though they all
feed math.randomseed a float in [0, 1).

References:
  - LuaJIT/src/lib_math.c (math_random, math_randomseed, random_seed)
  - LuaJIT/src/lj_prng.c (TW223_GEN, TW223_STEP, lj_prng_u64, lj_prng_u64d)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct


U64_MASK = (1 << 64) - 1
LUAJIT_SEED_R_INIT = 0x11090601
LUAJIT_SEED_WARMUP = 10

# TW223_STEP from lj_prng.c: four TW223_GEN calls with (k, q, s) tuples.
TW223_PARAMS: tuple[tuple[int, int, int], ...] = (
    (63, 31, 18),
    (58, 19, 28),
    (55, 24, 7),
    (47, 21, 8),
)

# IEEE-754 double constants for the "double in [1.0, 2.0)" trick in
# lj_prng_u64d: top 12 bits = sign (0) + exponent (0x3FF) for 1.0.
DOUBLE_MANTISSA_MASK = 0x000FFFFFFFFFFFFF
DOUBLE_ONE_EXPONENT = 0x3FF0000000000000


def _u64(value: int) -> int:
    return value & U64_MASK


def _double_to_bits(d: float) -> int:
    """Reinterpret a Python float's 8 bytes as a uint64 (little-endian).

    LuaJIT's ``U64double`` union does the equivalent on the platform's
    native byte order; on x86/x64 (every Balatro target) that's little.
    """

    return struct.unpack("<Q", struct.pack("<d", float(d)))[0]


def _bits_to_double(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", _u64(bits)))[0]


def _tw223_step(state: list[int]) -> int:
    """One step of TW223; mutates ``state`` and returns the running xor.

    TW223_GEN macro from lj_prng.c expanded:
        z = state[i]
        z = (((z << q) ^ z) >> (k - s)) ^ ((z & MASK_TOP_K_BITS) << s)
        r ^= z
        state[i] = z
    """

    r = 0
    for i, (k, q, s) in enumerate(TW223_PARAMS):
        z = state[i]
        # ((z << q) ^ z) >> (k - s)
        shifted = _u64((_u64(z << q)) ^ z) >> (k - s)
        # (uint64_t)(int64_t)-1 << (64 - k) — mask of top k bits.
        top_k_mask = _u64(U64_MASK << (64 - k))
        high = _u64((z & top_k_mask) << s)
        z = _u64(shifted ^ high)
        r ^= z
        state[i] = z
    return _u64(r)


@dataclass(slots=True)
class LuaJITPRNG:
    """TW223 PRNG matching LuaJIT 2.1's ``math.random`` exactly."""

    state: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    @classmethod
    def seeded(cls, seed_double: float) -> "LuaJITPRNG":
        """Seed using LuaJIT's ``random_seed`` algorithm and run 10 warmup steps."""

        rs = cls()
        d = float(seed_double)
        r = LUAJIT_SEED_R_INIT
        for i in range(4):
            m = 1 << (r & 0xFF)
            r >>= 8
            d = d * math.pi + math.e
            u = _double_to_bits(d)
            if u < m:
                u = _u64(u + m)
            rs.state[i] = u
        for _ in range(LUAJIT_SEED_WARMUP):
            _tw223_step(rs.state)
        return rs

    def next_u64(self) -> int:
        """Equivalent to LuaJIT's ``lj_prng_u64``."""

        return _tw223_step(self.state)

    def next_double(self) -> float:
        """Equivalent to LuaJIT's ``math.random()`` (no args) — returns [0, 1)."""

        r = _tw223_step(self.state)
        # lj_prng_u64d builds a double in [1.0, 2.0) bit-pattern, then math_random
        # subtracts 1.0 to get [0, 1). Important: the same TW223 step output is
        # used, not a separate step — both lj_prng_u64 and lj_prng_u64d advance
        # exactly once per call.
        bits = (r & DOUBLE_MANTISSA_MASK) | DOUBLE_ONE_EXPONENT
        return _bits_to_double(bits) - 1.0

    def random_int_1_to_n(self, n: int) -> int:
        """LuaJIT's ``math.random(n)`` — returns int in [1, n]."""

        d = self.next_double()
        return int(math.floor(d * n)) + 1


def luajit_pseudoshuffle(items: list, seed_double: float) -> None:
    """Port of Balatro's ``pseudoshuffle`` using LuaJIT TW223 internally.

    Equivalent to the Lua:

        function pseudoshuffle(list, seed)
            if seed then math.randomseed(seed) end
            for i = #list, 2, -1 do
                local j = math.random(i)
                list[i], list[j] = list[j], list[i]
            end
        end

    Note: ``pseudoshuffle`` also has a ``sort_id`` pre-sort branch
    (functions/misc_functions.lua:209-211); for the initial deck shuffle the
    cards already arrive in sort_id order, so a stable sort is a no-op. The
    caller is expected to pre-sort if needed.
    """

    if not items:
        return
    rng = LuaJITPRNG.seeded(seed_double)
    for i in range(len(items) - 1, 0, -1):
        # Lua 1-indexed i; math.random(i) returns [1, i]; convert to 0-indexed.
        j = rng.random_int_1_to_n(i + 1) - 1
        items[i], items[j] = items[j], items[i]
