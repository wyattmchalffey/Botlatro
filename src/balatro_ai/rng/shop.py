"""Shop pool prediction for Balatro's initial shop.

Verified for ante 1 white-stake red-deck against 4 captured fixtures
(`.data/rng-validation/shop_seed_*_red_white.json`): all 8/8 categories and
all 8/8 specific items match exactly.

Algorithm reference (from `functions/UI_definitions.lua:create_card_for_shop`
and `functions/common_events.lua:create_card`):

  For each shop slot (in order):
    1. type     = pseudorandom(pseudoseed('cdt' + ante)) * total_rate
                  → category by cumulative-threshold
    2. If type == 'Joker':
         rarity = pseudorandom('rarity' + ante + 'sho')
                  > 0.95 rare, > 0.7 uncommon, else common
         center = pseudorandom_element(rarity_pool,
                                       pseudoseed('Joker' + rarity + 'sho' + ante))
       Else (Tarot / Planet / etc.):
         center = pseudorandom_element(type_pool,
                                       pseudoseed(type + 'sho' + ante))
    3. If center == UNAVAILABLE, retry with key + '_resample' + (attempt+1)

The pseudorandom_element call is `math.randomseed(pseudoseed_float)` then
`math.random(#pool)`. The `math.random` is LuaJIT TW223 (see
`luajit_prng.py`).

This module only covers the basic ante-1 white-stake case. It does NOT yet
handle: vouchers (Illusion / Tarot Merchant / Magic Trick etc.), tags
(Buffoon / Standard / Charm / Meteor / Ethereal that force shop types),
edition polls, eternal/perishable/rental polls. Each of those adds another
pseudoseed advance per slot that the predictor would need to account for.
"""

from __future__ import annotations

from dataclasses import dataclass

from balatro_ai.rng.balatro_rng import BalatroRNG
from balatro_ai.rng.luajit_prng import LuaJITPRNG
from balatro_ai.rng.pools import (
    JOKER_POOL_BY_RARITY,
    PLANET_POOL,
    TAROT_POOL,
    joker_pool_for_rarity,
    planet_pool_for_ante,
)


# Default ante-1 shop rates (game.lua: joker_rate=20, tarot_rate=4,
# planet_rate=4, playing_card_rate=0, spectral_rate=0).
DEFAULT_RATES = {
    "Joker": 20,
    "Tarot": 4,
    "Planet": 4,
    "playing_card": 0,
    "Spectral": 0,
}


# Steamodded reorders the type iteration (`SMODS.poll_object_type` in
# `Mods/smods/src/utils/weights.lua:418`). The cumulative thresholds use
# this exact iteration order.
STEAMODDED_TYPE_ORDER = ("Joker", "playing_card", "Tarot", "Planet", "Spectral")


@dataclass(frozen=True, slots=True)
class ShopSlot:
    category: str           # "Joker" | "Tarot" | "Planet" | "Spectral" | etc.
    item_key: str           # e.g. "j_burglar", "c_neptune"
    joker_rarity: int | None = None  # 1..4 for jokers, None otherwise


def _pick_category(polled: float, rates: dict[str, int]) -> str:
    check = 0
    for type_name in STEAMODDED_TYPE_ORDER:
        val = rates.get(type_name, 0)
        if polled > check and polled <= check + val:
            return type_name
        check += val
    return "Joker"  # fallback if polled lands exactly on total (shouldn't happen)


def _pick_from_pool(rng: BalatroRNG, pool: tuple[str, ...], base_key: str, *, max_resample: int = 20) -> str | None:
    """Pick from a pool, retrying with _resample suffix on UNAVAILABLE entries."""

    pool_key = base_key
    for attempt in range(1, max_resample + 1):
        seed_float = rng.random(pool_key)
        luajit = LuaJITPRNG.seeded(seed_float)
        idx = luajit.random_int_1_to_n(len(pool)) - 1
        chosen = pool[idx]
        if chosen != "UNAVAILABLE":
            return chosen
        pool_key = base_key + "_resample" + str(attempt + 1)
    return None


def predict_first_shop(
    seed: str,
    *,
    ante: int = 1,
    n_slots: int = 2,
    rates: dict[str, int] | None = None,
    unlocked_jokers: frozenset[str] | None = None,
    used_jokers: frozenset[str] = frozenset(),
    planet_played_hands: frozenset[str] = frozenset(),
) -> tuple[ShopSlot, ...]:
    """Predict the contents of the first shop after the small blind.

    Returns one ``ShopSlot`` per slot, in the order Balatro fills them.
    Joker slots also record the rolled rarity for downstream debugging.

    Limitations: see module docstring. Defaults assume a fresh white-stake
    red-deck run with no vouchers, no tags, no modifiers.
    """

    rates = rates or DEFAULT_RATES
    total_rate = sum(rates.values())
    rng = BalatroRNG(seed=seed, mix_hashed_seed=True)

    slots: list[ShopSlot] = []
    for slot_index in range(n_slots):
        cdt_float = rng.random("cdt" + str(ante))
        polled = LuaJITPRNG.seeded(cdt_float).next_double() * total_rate
        category = _pick_category(polled, rates)

        if category == "Joker":
            rarity_float = rng.random("rarity" + str(ante) + "sho")
            r = LuaJITPRNG.seeded(rarity_float).next_double()
            rarity = 3 if r > 0.95 else 2 if r > 0.7 else 1
            pool = joker_pool_for_rarity(rarity, unlocked=unlocked_jokers, used=used_jokers)
            base_key = "Joker" + str(rarity) + "sho" + str(ante)
            item = _pick_from_pool(rng, pool, base_key)
            slots.append(ShopSlot(category="Joker", item_key=item or "?", joker_rarity=rarity))
        elif category == "Tarot":
            pool = TAROT_POOL
            base_key = "Tarotsho" + str(ante)
            item = _pick_from_pool(rng, pool, base_key)
            slots.append(ShopSlot(category="Tarot", item_key=item or "?"))
        elif category == "Planet":
            pool = planet_pool_for_ante(planet_played_hands)
            base_key = "Planetsho" + str(ante)
            item = _pick_from_pool(rng, pool, base_key)
            slots.append(ShopSlot(category="Planet", item_key=item or "?"))
        else:
            # playing_card / Spectral — not commonly observed at default rates;
            # leave unmodelled for now.
            slots.append(ShopSlot(category=category, item_key="?"))

    return tuple(slots)
