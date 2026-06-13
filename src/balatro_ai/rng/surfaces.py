"""Seed-faithful predictors for Balatro run surfaces.

This module builds on the low-level ``BalatroRNG`` and LuaJIT PRNG ports to
predict the game surfaces the offline solver needs without opening the bridge:
bosses, vouchers, skip tags, shop cards, booster packs, pack contents, joker
editions/stickers, and standard-pack card fronts/seals.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from balatro_ai.rng.balatro_rng import BalatroRNG
from balatro_ai.rng.luajit_prng import LuaJITPRNG
from balatro_ai.rng.pools import (
    ETERNAL_INCOMPATIBLE_JOKERS,
    PERISHABLE_INCOMPATIBLE_JOKERS,
    TAROT_POOL,
    joker_pool_for_rarity,
    planet_pool_for_ante,
)


SHOP_POOL_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "shop_pools.json"

DEFAULT_SHOP_RATES: Mapping[str, int] = {
    "Joker": 20,
    "Tarot": 4,
    "Planet": 4,
    "playing_card": 0,
    "Spectral": 0,
}

VANILLA_SHOP_TYPE_ORDER = ("Joker", "Tarot", "Planet", "playing_card", "Spectral")
STEAMODDED_SHOP_TYPE_ORDER = ("Joker", "playing_card", "Tarot", "Planet", "Spectral")

P_CARD_KEYS = tuple(
    f"{suit}_{rank}"
    for suit in ("C", "D", "H", "S")
    for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q", "T")
)
ENHANCED_POOL = ("m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky")
SPECTRAL_CREATED_ENHANCED_POOL = tuple(key for key in ENHANCED_POOL if key != "m_stone")

TAG_POOL = (
    "tag_uncommon",
    "tag_rare",
    "tag_negative",
    "tag_foil",
    "tag_holo",
    "tag_polychrome",
    "tag_investment",
    "tag_voucher",
    "tag_boss",
    "tag_standard",
    "tag_charm",
    "tag_meteor",
    "tag_buffoon",
    "tag_handy",
    "tag_garbage",
    "tag_ethereal",
    "tag_coupon",
    "tag_double",
    "tag_juggle",
    "tag_d_six",
    "tag_top_up",
    "tag_skip",
    "tag_orbital",
    "tag_economy",
)
TAG_MIN_ANTES = {
    "tag_negative": 2,
    "tag_standard": 2,
    "tag_meteor": 2,
    "tag_buffoon": 2,
    "tag_handy": 2,
    "tag_garbage": 2,
    "tag_ethereal": 2,
    "tag_top_up": 2,
    "tag_orbital": 2,
}
TAG_REQUIRES = {
    "tag_rare": "j_blueprint",
    "tag_negative": "e_negative",
    "tag_foil": "e_foil",
    "tag_holo": "e_holo",
    "tag_polychrome": "e_polychrome",
}

BOSS_MIN_ANTES = {
    "bl_ox": 6,
    "bl_hook": 1,
    "bl_mouth": 2,
    "bl_fish": 2,
    "bl_club": 1,
    "bl_manacle": 1,
    "bl_tooth": 3,
    "bl_wall": 2,
    "bl_house": 2,
    "bl_mark": 2,
    "bl_wheel": 2,
    "bl_arm": 2,
    "bl_psychic": 1,
    "bl_goad": 1,
    "bl_water": 2,
    "bl_eye": 3,
    "bl_plant": 4,
    "bl_needle": 2,
    "bl_head": 1,
    "bl_window": 1,
    "bl_serpent": 5,
    "bl_pillar": 1,
    "bl_flint": 2,
}
SHOWDOWN_BOSSES = ("bl_final_acorn", "bl_final_bell", "bl_final_heart", "bl_final_leaf", "bl_final_vessel")
BOSS_NAMES = {
    "bl_club": "The Club",
    "bl_goad": "The Goad",
    "bl_head": "The Head",
    "bl_hook": "The Hook",
    "bl_manacle": "The Manacle",
    "bl_pillar": "The Pillar",
    "bl_psychic": "The Psychic",
    "bl_window": "The Window",
    "bl_fish": "The Fish",
    "bl_house": "The House",
    "bl_wall": "The Wall",
    "bl_wheel": "The Wheel",
    "bl_arm": "The Arm",
    "bl_water": "The Water",
    "bl_needle": "The Needle",
    "bl_flint": "The Flint",
    "bl_mark": "The Mark",
    "bl_mouth": "The Mouth",
    "bl_tooth": "The Tooth",
    "bl_eye": "The Eye",
    "bl_plant": "The Plant",
    "bl_serpent": "The Serpent",
    "bl_ox": "The Ox",
    "bl_final_acorn": "Amber Acorn",
    "bl_final_bell": "Cerulean Bell",
    "bl_final_heart": "Crimson Heart",
    "bl_final_leaf": "Verdant Leaf",
    "bl_final_vessel": "Violet Vessel",
}


@dataclass(frozen=True, slots=True)
class PredictedCard:
    """A predicted center/front plus random modifiers Balatro may attach."""

    set: str
    key: str
    front_key: str | None = None
    rarity: int | None = None
    edition: str | None = None
    seal: str | None = None
    eternal: bool = False
    perishable: bool = False
    rental: bool = False


@dataclass(frozen=True, slots=True)
class InitialRunSurface:
    boss_key: str
    voucher_key: str
    small_tag_key: str
    big_tag_key: str

    @property
    def boss_name(self) -> str:
        return BOSS_NAMES.get(self.boss_key, self.boss_key)


@dataclass(frozen=True, slots=True)
class ShopSurface:
    setup: InitialRunSurface | None
    shop_cards: tuple[PredictedCard, ...]
    voucher_key: str | None
    booster_keys: tuple[str, ...]


def predict_initial_surface(seed_or_rng: str | BalatroRNG, *, ante: int = 1) -> InitialRunSurface:
    """Predict the boss, current voucher, and Small/Big blind tags."""

    rng = _rng(seed_or_rng)
    boss_counts: dict[str, int] = {}
    boss_key = predict_boss(rng, ante=ante, boss_use_counts=boss_counts)
    voucher_key = predict_voucher(rng, ante=ante)
    small_tag = predict_tag(rng, ante=ante)
    big_tag = predict_tag(rng, ante=ante)
    return InitialRunSurface(
        boss_key=boss_key,
        voucher_key=voucher_key,
        small_tag_key=small_tag,
        big_tag_key=big_tag,
    )


def predict_shop_surface(
    seed: str,
    *,
    ante: int = 1,
    n_shop_slots: int = 2,
    n_boosters: int = 2,
    include_initial_setup: bool = True,
) -> ShopSurface:
    """Predict a fresh shop surface for a seed.

    For ante 1 this matches the captured first-shop bridge fixtures: initial
    boss/voucher/tags, rerollable shop cards, voucher slot, and booster packs.
    """

    rng = BalatroRNG(seed=seed)
    setup = predict_initial_surface(rng, ante=ante) if include_initial_setup else None
    voucher_key = setup.voucher_key if setup is not None else predict_voucher(rng, ante=ante)
    return ShopSurface(
        setup=setup,
        shop_cards=predict_shop_cards(rng, ante=ante, n_slots=n_shop_slots),
        voucher_key=voucher_key,
        booster_keys=predict_shop_boosters(rng, ante=ante, n_slots=n_boosters, first_shop_buffoon=ante == 1),
    )


def predict_boss(
    seed_or_rng: str | BalatroRNG,
    *,
    ante: int,
    win_ante: int = 8,
    boss_use_counts: dict[str, int] | None = None,
    banned_keys: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> str:
    """Predict ``get_new_boss`` for the current ante."""

    rng = _rng(seed_or_rng)
    counts = boss_use_counts if boss_use_counts is not None else {}
    banned = set(banned_keys)
    excluded = set(exclude)
    if ante % win_ante == 0 and ante >= 2:
        eligible = [key for key in SHOWDOWN_BOSSES if key not in banned and key not in excluded]
    else:
        eligible = [
            key
            for key, min_ante in BOSS_MIN_ANTES.items()
            if min_ante <= max(1, ante) and key not in banned and key not in excluded
        ]
    if not eligible:
        raise ValueError("No eligible boss blinds for prediction")

    min_use = min(counts.get(key, 0) for key in eligible)
    pool = tuple(sorted(key for key in eligible if counts.get(key, 0) == min_use))
    boss = _pick_from_pool(rng, pool, "boss")
    counts[boss] = counts.get(boss, 0) + 1
    return boss


def predict_voucher(
    seed_or_rng: str | BalatroRNG,
    *,
    ante: int,
    used_vouchers: Iterable[str] = (),
    visible_vouchers: Iterable[str] = (),
    unlocked_vouchers: frozenset[str] | None = None,
    from_tag: bool = False,
) -> str:
    """Predict ``get_next_voucher_key``."""

    rng = _rng(seed_or_rng)
    pool = _voucher_pool(
        used_vouchers=set(used_vouchers),
        visible_vouchers=set(visible_vouchers),
        unlocked_vouchers=unlocked_vouchers,
    )
    pool_key = "Voucher_fromtag" if from_tag else "Voucher" + str(ante)
    return _pick_available(rng, pool, pool_key)


def predict_tag(
    seed_or_rng: str | BalatroRNG,
    *,
    ante: int,
    append: str = "",
    discovered_centers: frozenset[str] | None = None,
    banned_keys: Iterable[str] = (),
) -> str:
    """Predict ``get_next_tag_key``."""

    rng = _rng(seed_or_rng)
    banned = set(banned_keys)
    pool = []
    for key in TAG_POOL:
        required = TAG_REQUIRES.get(key)
        unavailable = (
            TAG_MIN_ANTES.get(key, 1) > ante
            or key in banned
            or (discovered_centers is not None and required is not None and required not in discovered_centers)
        )
        pool.append("UNAVAILABLE" if unavailable else key)
    return _pick_available(rng, tuple(pool), "Tag" + append + str(ante))


def predict_shop_cards(
    seed_or_rng: str | BalatroRNG,
    *,
    ante: int,
    n_slots: int = 2,
    rates: Mapping[str, int] | None = None,
    shop_type_order: tuple[str, ...] = STEAMODDED_SHOP_TYPE_ORDER,
    used_jokers: Iterable[str] = (),
    available_enhancements: frozenset[str] = frozenset(),
    unlocked_jokers: frozenset[str] | None = None,
    played_hand_types: frozenset[str] = frozenset(),
    used_consumables: Iterable[str] = (),
    vouchers: Iterable[str] = (),
    enable_eternals: bool = False,
    enable_perishables: bool = False,
    enable_rentals: bool = False,
    edition_rate: float = 1.0,
    pool_flags: frozenset[str] = frozenset(),
) -> tuple[PredictedCard, ...]:
    """Predict rerollable shop-card slots."""

    rng = _rng(seed_or_rng)
    rates = dict(rates or DEFAULT_SHOP_RATES)
    total_rate = sum(rates.values())
    cards: list[PredictedCard] = []
    used_joker_set = set(used_jokers)
    used_consumable_set = set(used_consumables)
    voucher_set = set(vouchers)

    for _slot_index in range(n_slots):
        polled = _pseudorandom_float(rng, "cdt" + str(ante)) * total_rate
        card_type = _pick_shop_type(polled, rates, shop_type_order)
        if card_type == "playing_card":
            if _has_voucher(voucher_set, "v_illusion", "Illusion") and _pseudorandom_float(rng, "illusion") > 0.6:
                card_type = "Enhanced"
            else:
                card_type = "Base"
        card = predict_card(
            rng,
            card_type,
            ante=ante,
            key_append="sho",
            area="shop",
            used_jokers=used_joker_set,
            available_enhancements=available_enhancements,
            unlocked_jokers=unlocked_jokers,
            played_hand_types=played_hand_types,
            used_consumables=used_consumable_set,
            enable_eternals=enable_eternals,
            enable_perishables=enable_perishables,
            enable_rentals=enable_rentals,
            edition_rate=edition_rate,
            pool_flags=pool_flags,
        )
        if card.set == "Joker":
            used_joker_set.add(card.key)
        elif card.set in {"Tarot", "Planet", "Spectral"}:
            used_consumable_set.add(card.key)
        if card_type in {"Base", "Enhanced"} and _has_voucher(voucher_set, "v_illusion", "Illusion"):
            card = _with_illusion_playing_card_edition(rng, card)
        cards.append(card)

    return tuple(cards)


def predict_shop_boosters(
    seed_or_rng: str | BalatroRNG,
    *,
    ante: int,
    n_slots: int = 2,
    first_shop_buffoon: bool = False,
    banned_keys: Iterable[str] = (),
) -> tuple[str, ...]:
    """Predict booster-pack slots for a shop."""

    rng = _rng(seed_or_rng)
    banned = set(banned_keys)
    boosters: list[str] = []
    for index in range(n_slots):
        if index == 0 and first_shop_buffoon and "p_buffoon_normal_1" not in banned:
            boosters.append(_predict_first_buffoon_pack(rng, ante=ante))
        else:
            boosters.append(predict_booster_pack(rng, ante=ante, key="shop_pack", banned_keys=banned))
    return tuple(boosters)


def predict_booster_pack(
    seed_or_rng: str | BalatroRNG,
    *,
    ante: int,
    key: str = "pack_generic",
    kind: str | None = None,
    banned_keys: Iterable[str] = (),
) -> str:
    """Predict ``get_pack`` for a weighted booster pool."""

    rng = _rng(seed_or_rng)
    banned = set(banned_keys)
    records = [
        record
        for record in _booster_records()
        if record["key"] not in banned and (kind is None or record.get("kind") == kind)
    ]
    if not records:
        raise ValueError("No eligible booster packs for prediction")
    cume = sum(float(record.get("weight", 1)) for record in records)
    poll = _pseudorandom_float(rng, key + str(ante)) * cume
    tally = 0.0
    for record in records:
        weight = float(record.get("weight", 1))
        tally += weight
        if tally >= poll and tally - weight <= poll:
            return str(record["key"])
    return str(records[-1]["key"])


def predict_pack_contents(
    seed_or_rng: str | BalatroRNG,
    *,
    ante: int,
    pack_key: str | None = None,
    kind: str | None = None,
    size: int | None = None,
    vouchers: Iterable[str] = (),
    played_hand_types: frozenset[str] = frozenset(),
    used_jokers: Iterable[str] = (),
    available_enhancements: frozenset[str] = frozenset(),
    used_consumables: Iterable[str] = (),
    telescope_planet_key: str | None = None,
    enable_eternals: bool = False,
    enable_perishables: bool = False,
    enable_rentals: bool = False,
    edition_rate: float = 1.0,
) -> tuple[PredictedCard, ...]:
    """Predict cards revealed by an opened booster pack."""

    rng = _rng(seed_or_rng)
    record = _booster_record(pack_key) if pack_key is not None else None
    pack_kind = kind or (str(record["kind"]) if record is not None else None)
    if pack_kind is None:
        raise ValueError("predict_pack_contents requires pack_key or kind")
    extra = size if size is not None else int((record or {}).get("config", {}).get("extra", 0))
    if extra <= 0:
        return ()

    voucher_set = set(vouchers)
    used_joker_set = set(used_jokers)
    used_consumable_set = set(used_consumables)
    cards: list[PredictedCard] = []
    for index in range(extra):
        if pack_kind == "Arcana":
            if _has_voucher(voucher_set, "v_omen_globe", "Omen Globe") and _pseudorandom_float(rng, "omen_globe") > 0.8:
                card_type, key_append = "Spectral", "ar2"
            else:
                card_type, key_append = "Tarot", "ar1"
            card = predict_card(
                rng,
                card_type,
                ante=ante,
                key_append=key_append,
                area="pack",
                soulable=True,
                used_jokers=used_joker_set,
                available_enhancements=available_enhancements,
                used_consumables=used_consumable_set,
                played_hand_types=played_hand_types,
            )
        elif pack_kind == "Celestial":
            forced = telescope_planet_key if index == 0 and _has_voucher(voucher_set, "v_telescope", "Telescope") else None
            card = predict_card(
                rng,
                "Planet",
                ante=ante,
                key_append="pl1",
                area="pack",
                soulable=True,
                forced_key=forced,
                played_hand_types=played_hand_types,
                used_consumables=used_consumable_set,
            )
        elif pack_kind == "Spectral":
            card = predict_card(
                rng,
                "Spectral",
                ante=ante,
                key_append="spe",
                area="pack",
                soulable=True,
                used_consumables=used_consumable_set,
            )
        elif pack_kind == "Standard":
            card_type = "Enhanced" if _pseudorandom_float(rng, "stdset" + str(ante)) > 0.6 else "Base"
            card = predict_card(rng, card_type, ante=ante, key_append="sta", area="pack")
            card = _with_standard_pack_modifiers(rng, card, ante=ante, edition_rate=edition_rate)
        elif pack_kind == "Buffoon":
            card = predict_card(
                rng,
                "Joker",
                ante=ante,
                key_append="buf",
                area="pack",
                soulable=True,
                used_jokers=used_joker_set,
                available_enhancements=available_enhancements,
                unlocked_jokers=None,
                enable_eternals=enable_eternals,
                enable_perishables=enable_perishables,
                enable_rentals=enable_rentals,
                edition_rate=edition_rate,
            )
        else:
            raise ValueError(f"Unsupported booster kind: {pack_kind}")
        cards.append(card)
        if card.set == "Joker":
            used_joker_set.add(card.key)
        elif card.set in {"Tarot", "Planet", "Spectral"}:
            used_consumable_set.add(card.key)
    return tuple(cards)


def predict_card(
    seed_or_rng: str | BalatroRNG,
    card_type: str,
    *,
    ante: int,
    key_append: str = "",
    area: str = "shop",
    soulable: bool = False,
    forced_key: str | None = None,
    rarity: float | int | None = None,
    legendary: bool = False,
    used_jokers: Iterable[str] = (),
    available_enhancements: frozenset[str] = frozenset(),
    unlocked_jokers: frozenset[str] | None = None,
    played_hand_types: frozenset[str] = frozenset(),
    used_consumables: Iterable[str] = (),
    enable_eternals: bool = False,
    enable_perishables: bool = False,
    enable_rentals: bool = False,
    edition_rate: float = 1.0,
    pool_flags: frozenset[str] = frozenset(),
) -> PredictedCard:
    """Predict ``create_card`` enough for solver-visible RNG surfaces."""

    rng = _rng(seed_or_rng)
    effective_type = card_type
    forced = forced_key
    if forced is None and soulable:
        forced = _soul_for_type(rng, effective_type, ante=ante, used_consumables=set(used_consumables))
    if effective_type == "Base":
        forced = "c_base"

    rolled_rarity: int | None = None
    if forced is not None:
        center_key = forced
        center_set = _set_for_center(center_key, fallback=effective_type)
    else:
        pool, pool_key, rolled_rarity = _current_pool(
            rng,
            effective_type,
            ante=ante,
            key_append=key_append,
            rarity=rarity,
            legendary=legendary,
            used_jokers=set(used_jokers),
            available_enhancements=available_enhancements,
            unlocked_jokers=unlocked_jokers,
            played_hand_types=played_hand_types,
            used_consumables=set(used_consumables),
            pool_flags=pool_flags,
        )
        center_key = _pick_available(rng, pool, pool_key)
        center_set = _set_for_center(center_key, fallback=effective_type)

    front_key = None
    if effective_type in {"Base", "Enhanced"}:
        front_key = _pick_from_pool(rng, P_CARD_KEYS, "front" + key_append + str(ante))

    predicted = PredictedCard(set=center_set, key=center_key, front_key=front_key, rarity=rolled_rarity)
    if center_set == "Joker":
        predicted = _with_joker_shop_stickers(
            rng,
            predicted,
            ante=ante,
            area=area,
            enable_eternals=enable_eternals,
            enable_perishables=enable_perishables,
            enable_rentals=enable_rentals,
        )
        edition = poll_edition(rng, "edi" + key_append + str(ante), edition_rate=edition_rate)
        predicted = _replace_card(predicted, edition=edition)
    return predicted


def poll_edition(
    seed_or_rng: str | BalatroRNG,
    key: str = "edition_generic",
    *,
    modifier: float = 1.0,
    edition_rate: float = 1.0,
    no_negative: bool = False,
    guaranteed: bool = False,
) -> str | None:
    """Predict Balatro's ``poll_edition`` result as an edition type string."""

    rng = _rng(seed_or_rng)
    poll = _pseudorandom_float(rng, key)
    if guaranteed:
        if poll > 1 - 0.003 * 25 and not no_negative:
            return "negative"
        if poll > 1 - 0.006 * 25:
            return "polychrome"
        if poll > 1 - 0.02 * 25:
            return "holographic"
        if poll > 1 - 0.04 * 25:
            return "foil"
        return None
    if poll > 1 - 0.003 * modifier and not no_negative:
        return "negative"
    if poll > 1 - 0.006 * edition_rate * modifier:
        return "polychrome"
    if poll > 1 - 0.02 * edition_rate * modifier:
        return "holographic"
    if poll > 1 - 0.04 * edition_rate * modifier:
        return "foil"
    return None


def predict_spectral_created_cards(
    seed_or_rng: str | BalatroRNG,
    spectral_key: str,
    *,
    count: int | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Predict random playing cards created by Familiar/Grim/Incantation."""

    rng = _rng(seed_or_rng)
    if spectral_key == "c_familiar":
        n = 3 if count is None else count
        return tuple(
            (
                _pick_from_pool(rng, ("J", "Q", "K"), "familiar_create"),
                _pick_from_pool(rng, ("S", "H", "D", "C"), "familiar_create"),
                _pick_from_pool(rng, SPECTRAL_CREATED_ENHANCED_POOL, "spe_card"),
            )
            for _ in range(n)
        )
    if spectral_key == "c_grim":
        n = 2 if count is None else count
        return tuple(
            (
                "A",
                _pick_from_pool(rng, ("S", "H", "D", "C"), "grim_create"),
                _pick_from_pool(rng, SPECTRAL_CREATED_ENHANCED_POOL, "spe_card"),
            )
            for _ in range(n)
        )
    if spectral_key == "c_incantation":
        n = 4 if count is None else count
        return tuple(
            (
                _pick_from_pool(rng, ("2", "3", "4", "5", "6", "7", "8", "9", "T"), "incantation_create"),
                _pick_from_pool(rng, ("S", "H", "D", "C"), "incantation_create"),
                _pick_from_pool(rng, SPECTRAL_CREATED_ENHANCED_POOL, "spe_card"),
            )
            for _ in range(n)
        )
    return ()


def pseudorandom_float(seed_or_rng: str | BalatroRNG, key: str) -> float:
    """Balatro's ``pseudorandom(key)`` result in [0, 1).

    Advances ``key``'s independent stream by one and returns the float used
    for probability checks (e.g. ``pseudorandom('lucky_mult') < normal/5``).
    Mirrors the validated shop/pack float path (``_pseudorandom_float``)."""

    return _pseudorandom_float(_rng(seed_or_rng), key)


def pseudorandom_int(seed_or_rng: str | BalatroRNG, key: str, lo: int, hi: int) -> int:
    """Balatro's ``pseudorandom(key, lo, hi)`` — an integer in [lo, hi].

    Matches LuaJIT ``math.random(lo, hi)`` seeded by ``pseudoseed(key)``
    (e.g. Misprint's ``pseudorandom('misprint', 0, 23)``)."""

    rng = _rng(seed_or_rng)
    span = hi - lo + 1
    return lo + LuaJITPRNG.seeded(rng.random(key)).random_int_1_to_n(span) - 1


def planet_key_for_hand(hand_name: str | None) -> str | None:
    """Map a poker-hand name (e.g. "Full House") to its Planet card key
    (e.g. "c_earth"), or None. Used for the Telescope voucher, which forces
    the first Celestial-pack card to the most-played hand's planet."""

    if not hand_name:
        return None
    return _planet_key_by_hand().get(str(hand_name))


@lru_cache(maxsize=1)
def _planet_key_by_hand() -> Mapping[str, str]:
    planets: dict[str, str] = {}
    for record in _shop_pool_data().get("planets", ()):
        config = record.get("config")
        if isinstance(config, Mapping):
            hand_type = config.get("hand_type")
            key = record.get("key")
            if hand_type is not None and key is not None:
                planets[str(hand_type)] = str(key)
    return planets


def predict_ancient_suit(
    seed_or_rng: str | BalatroRNG, *, ante: int, previous_suit: str | None = None
) -> str:
    """Predict Ancient Joker's chosen suit for a round.

    Balatro's ``reset_ancient_card`` picks
    ``pseudorandom_element(['Spades','Hearts','Clubs','Diamonds'] minus the
    previous suit, pseudoseed('anc'+ante))``. Single-char suits here preserve
    that S,H,C,D order. ``previous_suit`` (single char, or None at run start)
    is excluded; the result chains into the next round's exclusion."""

    pool = tuple(suit for suit in ("S", "H", "C", "D") if suit != previous_suit)
    return _pick_from_pool(_rng(seed_or_rng), pool, "anc" + str(ante))


def predict_sigil_suit(seed_or_rng: str | BalatroRNG) -> str:
    return _pick_from_pool(_rng(seed_or_rng), ("S", "H", "D", "C"), "sigil")


def predict_ouija_rank(seed_or_rng: str | BalatroRNG) -> str:
    return _pick_from_pool(_rng(seed_or_rng), ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"), "ouija")


# Mirror get_current_pool (Balatro common_events.lua:2038-2050): when EVERY entry in
# the culled pool is UNAVAILABLE (e.g. all consumables of a type already used with no
# Showman, common at late antes), Balatro replaces the whole pool with a SINGLE default
# item rather than letting the pick resample -- otherwise the resample loop exhausts.
# Validated against the decompiled source. Without this, predict_shop_cards raised
# "Unable to pick available item from Planet" on late-ante shops with many used
# consumables and fell back to the (non-faithful) generic sampler (~35% of faithful-mode
# runs diverged here, concentrated at antes 6-8). The 1-item default pool yields exactly
# one rng roll on the pick, matching Balatro's pseudorandom_element over [default].
_POOL_EMPTY_DEFAULT = {"Tarot": "c_strength", "Tarot_Planet": "c_strength",
                       "Planet": "c_pluto", "Spectral": "c_incantation", "Joker": "j_joker"}


def _pool_or_default(pool: tuple[str, ...], default: str) -> tuple[str, ...]:
    if any(k != "UNAVAILABLE" for k in pool):
        return pool
    return (default,)


def _current_pool(
    rng: BalatroRNG,
    card_type: str,
    *,
    ante: int,
    key_append: str,
    rarity: float | int | None,
    legendary: bool,
    used_jokers: set[str],
    available_enhancements: frozenset[str],
    unlocked_jokers: frozenset[str] | None,
    played_hand_types: frozenset[str],
    used_consumables: set[str],
    pool_flags: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], str, int | None]:
    if card_type == "Joker":
        if legendary:
            rarity_int = 4
        elif isinstance(rarity, int):
            rarity_int = rarity
        else:
            rarity_poll = float(rarity) if rarity is not None else _pseudorandom_float(rng, "rarity" + str(ante) + key_append)
            rarity_int = 3 if rarity_poll > 0.95 else 2 if rarity_poll > 0.7 else 1
        return (
            _pool_or_default(
                joker_pool_for_rarity(
                    rarity_int,
                    unlocked=unlocked_jokers,
                    used=frozenset(used_jokers),
                    available_enhancements=available_enhancements,
                    pool_flags=pool_flags,
                ),
                "j_joker",
            ),
            "Joker" + str(rarity_int) + (key_append if not legendary else "") + (str(ante) if not legendary else ""),
            rarity_int,
        )
    if card_type == "Tarot":
        return (_pool_or_default(_without_used(TAROT_POOL, used_consumables), "c_strength"), "Tarot" + key_append + str(ante), None)
    if card_type == "Planet":
        return (_pool_or_default(_without_used(planet_pool_for_ante(played_hand_types), used_consumables), "c_pluto"), "Planet" + key_append + str(ante), None)
    if card_type == "Spectral":
        pool = tuple("UNAVAILABLE" if key in {"c_soul", "c_black_hole"} else key for key in _record_keys("spectrals"))
        return (_pool_or_default(_without_used(pool, used_consumables), "c_incantation"), "Spectral" + key_append + str(ante), None)
    if card_type == "Enhanced":
        return (ENHANCED_POOL, "Enhanced" + key_append + str(ante), None)
    if card_type == "Base":
        return (("c_base",), "Base" + key_append + str(ante), None)
    raise ValueError(f"Unsupported card type: {card_type}")


def _voucher_pool(
    *,
    used_vouchers: set[str],
    visible_vouchers: set[str],
    unlocked_vouchers: frozenset[str] | None,
) -> tuple[str, ...]:
    pool: list[str] = []
    for record in _voucher_records():
        key = str(record["key"])
        requires = tuple(str(item) for item in record.get("requires", ()) or ())
        locked = unlocked_vouchers is not None and not bool(record.get("unlocked_default", True)) and key not in unlocked_vouchers
        unavailable = (
            key in used_vouchers
            or key in visible_vouchers
            or locked
            or any(req not in used_vouchers for req in requires)
        )
        pool.append("UNAVAILABLE" if unavailable else key)
    return tuple(pool) or ("v_blank",)


def _soul_for_type(rng: BalatroRNG, card_type: str, *, ante: int, used_consumables: set[str]) -> str | None:
    # Both checks ALWAYS roll when their gate passes (common_events.lua
    # 2088-2100 runs the two ifs sequentially with no early exit), so for
    # Spectral cards the 'soul_Spectral<ante>' stream advances TWICE per card
    # and a Black Hole hit overrides an earlier Soul hit.
    forced: str | None = None
    if card_type in {"Tarot", "Spectral", "Tarot_Planet"} and "c_soul" not in used_consumables:
        if _pseudorandom_float(rng, "soul_" + card_type + str(ante)) > 0.997:
            forced = "c_soul"
    if card_type in {"Planet", "Spectral"} and "c_black_hole" not in used_consumables:
        if _pseudorandom_float(rng, "soul_" + card_type + str(ante)) > 0.997:
            forced = "c_black_hole"
    return forced


def _pick_shop_type(polled: float, rates: Mapping[str, int], shop_type_order: tuple[str, ...]) -> str:
    check = 0.0
    for type_name in shop_type_order:
        value = float(rates.get(type_name, 0))
        if polled > check and polled <= check + value:
            return type_name
        check += value
    return "Joker"


def _pick_available(rng: BalatroRNG, pool: tuple[str, ...], pool_key: str, *, max_resample: int = 50) -> str:
    if not pool:
        raise ValueError(f"Cannot pick from empty pool for {pool_key}")
    key = pool_key
    for attempt in range(1, max_resample + 1):
        picked = _pick_from_pool(rng, pool, key)
        if picked != "UNAVAILABLE":
            return picked
        key = pool_key + "_resample" + str(attempt + 1)
    raise ValueError(f"Unable to pick available item from {pool_key} after {max_resample} attempts")


def _pick_from_pool(rng: BalatroRNG, pool: tuple[str, ...], key: str) -> str:
    seed_float = rng.random(key)
    idx = LuaJITPRNG.seeded(seed_float).random_int_1_to_n(len(pool)) - 1
    return pool[idx]


def _pseudorandom_float(rng: BalatroRNG, key: str) -> float:
    return LuaJITPRNG.seeded(rng.random(key)).next_double()


def _predict_first_buffoon_pack(rng: BalatroRNG, *, ante: int, selectable_cards: int = 52) -> str:
    # Balatro's first shop forces a Buffoon Pack and chooses normal_1/normal_2
    # with bare math.random(1, 2). At shop open, global math.random was last
    # seeded by reset_castle_card's pseudorandom_element('cas'..ante).
    prng = LuaJITPRNG.seeded(rng.random("cas" + str(ante)))
    prng.random_int_1_to_n(selectable_cards)
    return "p_buffoon_normal_" + str(prng.random_int_1_to_n(2))


def _with_joker_shop_stickers(
    rng: BalatroRNG,
    card: PredictedCard,
    *,
    ante: int,
    area: str,
    enable_eternals: bool,
    enable_perishables: bool,
    enable_rentals: bool,
) -> PredictedCard:
    if area not in {"shop", "pack"}:
        return card
    et_key = ("packetper" if area == "pack" else "etperpoll") + str(ante)
    rental_key = ("packssjr" if area == "pack" else "ssjr") + str(ante)
    eternal = card.eternal
    perishable = card.perishable
    if enable_eternals or enable_perishables:
        poll = _pseudorandom_float(rng, et_key)
        eternal = enable_eternals and poll > 0.7 and card.key not in ETERNAL_INCOMPATIBLE_JOKERS
        perishable = (
            (not eternal)
            and enable_perishables
            and poll > 0.4
            and poll <= 0.7
            and card.key not in PERISHABLE_INCOMPATIBLE_JOKERS
        )
    rental = card.rental
    if enable_rentals:
        rental = _pseudorandom_float(rng, rental_key) > 0.7
    return _replace_card(card, eternal=eternal, perishable=perishable, rental=rental)


def _with_standard_pack_modifiers(
    rng: BalatroRNG,
    card: PredictedCard,
    *,
    ante: int,
    edition_rate: float,
) -> PredictedCard:
    edition = poll_edition(
        rng,
        "standard_edition" + str(ante),
        modifier=2,
        edition_rate=edition_rate,
        no_negative=True,
    )
    seal = None
    if _pseudorandom_float(rng, "stdseal" + str(ante)) > 1 - 0.02 * 10:
        seal_type = _pseudorandom_float(rng, "stdsealtype" + str(ante))
        if seal_type > 0.75:
            seal = "Red"
        elif seal_type > 0.5:
            seal = "Blue"
        elif seal_type > 0.25:
            seal = "Gold"
        else:
            seal = "Purple"
    return _replace_card(card, edition=edition, seal=seal)


def _with_illusion_playing_card_edition(rng: BalatroRNG, card: PredictedCard) -> PredictedCard:
    if _pseudorandom_float(rng, "illusion") <= 0.8:
        return card
    edition_poll = _pseudorandom_float(rng, "illusion")
    if edition_poll > 1 - 0.15:
        edition = "polychrome"
    elif edition_poll > 0.5:
        edition = "holographic"
    else:
        edition = "foil"
    return _replace_card(card, edition=edition)


def _replace_card(card: PredictedCard, **changes: object) -> PredictedCard:
    values = {
        "set": card.set,
        "key": card.key,
        "front_key": card.front_key,
        "rarity": card.rarity,
        "edition": card.edition,
        "seal": card.seal,
        "eternal": card.eternal,
        "perishable": card.perishable,
        "rental": card.rental,
    }
    values.update(changes)
    return PredictedCard(**values)  # type: ignore[arg-type]


def _without_used(pool: tuple[str, ...], used: set[str]) -> tuple[str, ...]:
    return tuple("UNAVAILABLE" if key in used else key for key in pool)


def _set_for_center(center_key: str, *, fallback: str) -> str:
    if center_key.startswith("j_"):
        return "Joker"
    if center_key.startswith("v_"):
        return "Voucher"
    if center_key in ENHANCED_POOL:
        return "Enhanced"
    if center_key == "c_base":
        return "Default"
    data = _shop_pool_data()
    for data_key, set_name in (("tarots", "Tarot"), ("planets", "Planet"), ("spectrals", "Spectral")):
        if any(str(record["key"]) == center_key for record in data[data_key]):
            return set_name
    return fallback


def _has_voucher(vouchers: set[str], key: str, name: str) -> bool:
    return key in vouchers or name in vouchers


def _rng(seed_or_rng: str | BalatroRNG) -> BalatroRNG:
    if isinstance(seed_or_rng, BalatroRNG):
        return seed_or_rng
    return BalatroRNG(seed=str(seed_or_rng))


@lru_cache(maxsize=1)
def _shop_pool_data() -> Mapping[str, Any]:
    return json.loads(SHOP_POOL_DATA_PATH.read_text(encoding="utf-8"))


def _record_keys(data_key: str) -> tuple[str, ...]:
    return tuple(str(record["key"]) for record in _shop_pool_data()[data_key])


def _voucher_records() -> tuple[Mapping[str, Any], ...]:
    return tuple(_shop_pool_data()["vouchers"])


def _booster_records() -> tuple[Mapping[str, Any], ...]:
    return tuple(_shop_pool_data()["boosters"])


def _booster_record(pack_key: str | None) -> Mapping[str, Any]:
    for record in _booster_records():
        if str(record["key"]) == pack_key:
            return record
    raise ValueError(f"Unknown booster pack key: {pack_key}")
