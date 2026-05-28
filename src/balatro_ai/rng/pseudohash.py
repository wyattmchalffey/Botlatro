"""Balatro pseudohash and pseudoseed primitives.

Direct ports of Balatro's `functions.lua` definitions. Implementations stay
exact to the source: same constants, same operator order, same iteration
direction. Lua and Python both use IEEE-754 doubles, so the float arithmetic
matches as long as we don't introduce operator-precedence drift.

Source reference (Balatro 1.0.x functions.lua):

    function pseudohash(str)
        local num = 1
        for i = #str, 1, -1 do
            num = ((1.1239285023/num)*string.byte(str,i)*math.pi + math.pi*i) % 1
        end
        return num
    end

    function pseudoseed(key, predict_seed)
        if key == 'seed' then return math.random() end
        if G.GAME.pseudorandom[key] then
            G.GAME.pseudorandom[key] = (G.GAME.pseudorandom[key] * 1.72431234 + 2.134453429141) % 1
            return (G.GAME.pseudorandom[key] + (G.GAME.pseudorandom.hashed_seed or 0))/2
        end
        G.GAME.pseudorandom[key] = pseudohash(key..(G.GAME.pseudorandom.seed or ''))
        G.GAME.pseudorandom[key] = (G.GAME.pseudorandom[key] * 1.72431234 + 2.134453429141) % 1
        return (G.GAME.pseudorandom[key] + (G.GAME.pseudorandom.hashed_seed or 0))/2
    end
"""

from __future__ import annotations

import math


PSEUDOHASH_DIV_NUMERATOR = 1.1239285023
PSEUDOSEED_MUL = 1.72431234
PSEUDOSEED_ADD = 2.134453429141


def pseudohash(text: str) -> float:
    """Return Balatro's float-walk hash of ``text`` as a value in [0, 1).

    Walks the string from last character to first, applying the same
    iteration as ``pseudohash`` in Lua. The position multiplier is the
    character's ORIGINAL 1-indexed position in the string, not the
    iteration counter — Lua's ``for i = #str, 1, -1`` makes ``i`` decrease
    while always equalling the original position of the visited character.
    Empty strings return 1.0 (the loop body never executes; matches Lua).
    """

    num = 1.0
    for position in range(len(text), 0, -1):
        byte = ord(text[position - 1])  # text is 0-indexed; Lua is 1-indexed
        num = ((PSEUDOHASH_DIV_NUMERATOR / num) * byte * math.pi + math.pi * position) % 1.0
    return num


def pseudoseed_step(state: float) -> float:
    """Advance a per-key pseudoseed state one iteration.

    Matches Balatro's exact Lua:

        math.abs(tonumber(string.format("%.13f",
            (2.134453429141 + state * 1.72431234) % 1)))

    The ``%.13f`` round-trip is load-bearing — without it, pure float
    arithmetic drifts after a few iterations and the per-key streams stop
    matching the game. ``math.abs`` defends against negative zero / tiny
    negatives from the format round-trip. Caller is responsible for the
    final ``(state + hashed_seed) / 2`` combination.
    """

    raw = (PSEUDOSEED_ADD + state * PSEUDOSEED_MUL) % 1.0
    # string.format("%.13f", x) rounds to 13 decimal places and re-parses;
    # Python's f-string with .13f matches Lua's printf semantics.
    rounded = float(f"{raw:.13f}")
    return abs(rounded)
