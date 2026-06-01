"""Learnable state encoding for the neural planner (Step 0.1).

Converts a `GameState` into a **structured, versioned** representation a
set-encoder / NNUE-style network can consume: global scalars, small fixed
categoricals (phase, blind type), per-poker-hand levels, deck composition, and
three variable-length token sets — jokers, hand cards, and shop items — each
carrying *identity* (vocabulary indices), editions, counters, and flags.

This is the deliberate successor to `env/observations.py` and the deprecated
coarse-aggregate probe in `ml/features.py` (whose own docstring notes the neural
value head needs the *richer raw inputs* this module provides — joker
identities/editions/counters, deck composition, shop contents — not the
heuristic's collapsed signals that made the 2026-05-29 probe fail).

Design:
- **Dependency-free** (stdlib only) so it tests under `unittest` without torch.
  The model layer turns `EncodedState` into tensors and does padding/batching.
- **UNK-safe.** Every embedding vocabulary reserves index 0 = PAD, 1 = UNK, so
  an unseen joker/consumable/boss never crashes encoding — it maps to UNK.
- **Versioned** (`ENCODING_VERSION`). Bump on any layout/vocab change.

Grounded in the real `GameState` API: jokers carry display `name` (not a key),
`hand_levels` is a direct dict field, blind type lives in
`modifiers["current_blind"]["type"]`, and shop items are the dicts in
`modifiers["shop_cards"]`. Vocabularies derive from canonical data
(`data/shop_pools.json`, `api.state` name maps).
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from balatro_ai.api.state import (
    CONSUMABLE_KEY_TO_NAME,
    VOUCHER_KEY_TO_NAME,
    Card,
    GamePhase,
    GameState,
    Joker,
)

ENCODING_VERSION = 1

_DATA = Path(__file__).resolve().parent.parent / "data"

PAD = 0
UNK = 1
_RESERVED = 2  # PAD + UNK occupy indices 0 and 1


# --------------------------------------------------------------------------- #
# Small fixed categoricals (dense 0-based; each has an explicit "none" slot).
# --------------------------------------------------------------------------- #

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
_RANK_INDEX = {r: i for i, r in enumerate(RANKS)}

SUITS = ("Spades", "Hearts", "Diamonds", "Clubs")
_SUIT_INDEX = {s: i for i, s in enumerate(SUITS)}
SUIT_NONE = len(SUITS)  # stone cards / unknown

# index 0 = none, then the editions (after stripping the "e_" prefix).
EDITIONS = ("foil", "holographic", "polychrome", "negative")
_EDITION_INDEX = {e: i + 1 for i, e in enumerate(EDITIONS)}

# index 0 = none, then enhancements (UPPERCASE labels / "m_" keys normalized).
ENHANCEMENTS = ("bonus", "mult", "wild", "glass", "steel", "stone", "gold", "lucky")
_ENHANCEMENT_INDEX = {e: i + 1 for i, e in enumerate(ENHANCEMENTS)}

# index 0 = none, then seals.
SEALS = ("gold", "red", "blue", "purple")
_SEAL_INDEX = {s: i + 1 for i, s in enumerate(SEALS)}

PHASES = tuple(p.value for p in GamePhase)
_PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}

# index 0 = none/unknown, then the blind kinds.
BLIND_TYPES = ("small", "big", "boss")
_BLIND_TYPE_INDEX = {b: i + 1 for i, b in enumerate(BLIND_TYPES)}

# Poker hands in a fixed order for the hand-level vector. Aliases tolerate the
# casings/spacing that show up in `state.hand_levels` keys.
POKER_HANDS = (
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush",
    "Full House", "Four of a Kind", "Straight Flush", "Five of a Kind",
    "Flush House", "Flush Five",
)
_HAND_ALIASES = {h.lower().replace(" ", ""): i for i, h in enumerate(POKER_HANDS)}

# Boss blind display names (24 normal + 4 finisher). UNK covers any miss.
BOSS_NAMES = (
    "The Arm", "The Club", "The Eye", "The Fish", "The Flint", "The Goad",
    "The Head", "The Hook", "The House", "The Manacle", "The Mark", "The Mouth",
    "The Needle", "The Ox", "The Pillar", "The Plant", "The Psychic",
    "The Serpent", "The Tooth", "The Wall", "The Water", "The Window",
    "The Wheel", "Amber Acorn", "Cerulean Bell", "Crimson Heart",
    "Verdant Leaf", "Violet Vessel",
)
_BOSS_INDEX = {b: i + _RESERVED for i, b in enumerate(BOSS_NAMES)}
_BOSS_VOCAB_SIZE = len(BOSS_NAMES) + _RESERVED

# Shop item types (from a shop_cards "set" or key prefix).
ITEM_TYPES = ("joker", "planet", "tarot", "spectral", "voucher", "card", "pack")
_ITEM_TYPE_INDEX = {t: i for i, t in enumerate(ITEM_TYPES)}
ITEM_TYPE_NONE = len(ITEM_TYPES)


# --------------------------------------------------------------------------- #
# Embedding-backed vocabularies (PAD=0, UNK=1, then sorted entries).
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=1)
def _shop_pools() -> dict:
    return json.loads((_DATA / "shop_pools.json").read_text())


@functools.lru_cache(maxsize=1)
def _joker_name_vocab() -> dict[str, int]:
    """Joker identity by display name (matches `Joker.name`)."""
    names = sorted({j["name"] for j in _shop_pools().get("jokers", []) if j.get("name")})
    return {n: i + _RESERVED for i, n in enumerate(names)}


@functools.lru_cache(maxsize=1)
def _consumable_name_vocab() -> dict[str, int]:
    """Consumable identity by display name (matches `state.consumables`)."""
    names = sorted(set(CONSUMABLE_KEY_TO_NAME.values()))
    return {n: i + _RESERVED for i, n in enumerate(names)}


@functools.lru_cache(maxsize=1)
def _voucher_name_vocab() -> dict[str, int]:
    names = sorted(set(VOUCHER_KEY_TO_NAME.values()))
    return {n: i + _RESERVED for i, n in enumerate(names)}


@functools.lru_cache(maxsize=1)
def _item_key_vocab() -> dict[str, int]:
    """Shop-item identity by canonical key (j_/c_/v_), from `shop_cards` dicts."""
    d = _shop_pools()
    keys: set[str] = set()
    for group in ("jokers", "tarots", "planets", "spectrals", "vouchers"):
        keys.update(item["key"] for item in d.get(group, []) if item.get("key"))
    return {k: i + _RESERVED for i, k in enumerate(sorted(keys))}


# --------------------------------------------------------------------------- #
# Lookup + normalization helpers.
# --------------------------------------------------------------------------- #

def _vocab_index(vocab: dict[str, int], key: str | None) -> int:
    if not key:
        return PAD
    return vocab.get(key, UNK)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _signed_log(x: float) -> float:
    v = math.copysign(math.log1p(abs(x)), x)
    return _clamp(v / 10.0, -2.0, 2.0)


def _norm_token(value: Any, *prefixes: str) -> str:
    """Lowercase and strip known prefixes (e.g. 'e_', 'm_') for robust matching."""
    s = str(value or "").strip().lower()
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):]
    return s


def _edition_index(edition: Any) -> int:
    return _EDITION_INDEX.get(_norm_token(edition, "e_"), 0)


def _enhancement_index(enhancement: Any) -> int:
    return _ENHANCEMENT_INDEX.get(_norm_token(enhancement, "m_"), 0)


def _seal_index(seal: Any) -> int:
    return _SEAL_INDEX.get(_norm_token(seal).replace("_seal", ""), 0)


def _hand_level_index(name: str) -> int | None:
    return _HAND_ALIASES.get(name.lower().replace(" ", ""))


def _meta_flag(joker: Joker, *keys: str) -> float:
    meta = joker.metadata or {}
    sources = [meta]
    for sub in ("state", "ability"):
        v = meta.get(sub)
        if isinstance(v, dict):
            sources.append(v)
    for src in sources:
        for k in keys:
            if k in src and bool(src[k]):
                return 1.0
    return 0.0


def _joker_counter(joker: Joker) -> float:
    eff = joker.effect
    if eff.current_xmult_visible is not None:
        return _signed_log(eff.current_xmult_visible)
    if eff.current_int is not None:
        return _signed_log(eff.current_int)
    return 0.0


# --------------------------------------------------------------------------- #
# Tokens + encoded state.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class JokerToken:
    key_index: int        # _joker_name_vocab index (identity)
    edition_index: int
    eternal: float
    perishable: float
    rental: float
    debuffed: float
    counter: float        # signed-log normalized visible counter
    sell_value: float


@dataclass(frozen=True, slots=True)
class CardToken:
    rank_index: int
    suit_index: int
    enhancement_index: int
    edition_index: int
    seal_index: int
    debuffed: float


@dataclass(frozen=True, slots=True)
class ShopToken:
    item_type_index: int
    key_index: int        # _item_key_vocab index (identity)
    edition_index: int
    cost: float           # normalized


SCALAR_NAMES = (
    "ante_norm",                 # ante / 8
    "money_signed_log",
    "money_norm",                # money / 50, clamped
    "score_ratio",               # current / required, clamped
    "required_log",              # log10(required+1) / 6
    "hands_remaining_norm",
    "discards_remaining_norm",
    "hand_count_norm",
    "n_jokers_norm",
    "n_consumables_norm",
    "n_vouchers_norm",
    "deck_size_norm",
    "deck_known",
    "is_boss",
    "shop_size_norm",
)


@dataclass(frozen=True, slots=True)
class EncodedState:
    version: int
    scalars: tuple[float, ...]
    phase_onehot: tuple[float, ...]
    blind_type_onehot: tuple[float, ...]
    boss_index: int
    hand_levels: tuple[float, ...]   # per POKER_HANDS, normalized
    deck_counts: tuple[float, ...]   # rank(13)+suit(4)+enhanced(1), normalized
    jokers: tuple[JokerToken, ...]
    hand: tuple[CardToken, ...]
    shop: tuple[ShopToken, ...]
    consumables: tuple[int, ...]     # _consumable_name_vocab indices
    vouchers: tuple[int, ...]        # _voucher_name_vocab indices


# --------------------------------------------------------------------------- #
# Per-element encoders.
# --------------------------------------------------------------------------- #

def _encode_card(card: Card) -> CardToken:
    return CardToken(
        rank_index=_RANK_INDEX.get(card.rank, 0),
        suit_index=_SUIT_INDEX.get(card.suit, SUIT_NONE),
        enhancement_index=_enhancement_index(card.enhancement),
        edition_index=_edition_index(card.edition),
        seal_index=_seal_index(card.seal),
        debuffed=1.0 if card.debuffed else 0.0,
    )


def _encode_joker(joker: Joker) -> JokerToken:
    return JokerToken(
        key_index=_vocab_index(_joker_name_vocab(), joker.name),
        edition_index=_edition_index(joker.edition),
        eternal=_meta_flag(joker, "eternal"),
        perishable=_meta_flag(joker, "perishable"),
        rental=_meta_flag(joker, "rental"),
        debuffed=_meta_flag(joker, "debuff", "debuffed"),
        counter=_joker_counter(joker),
        sell_value=_clamp((joker.sell_value or 0) / 10.0, 0.0, 5.0),
    )


def _shop_item_type(card: dict) -> int:
    card_set = str(card.get("set", "")).lower()
    if card_set in _ITEM_TYPE_INDEX:
        return _ITEM_TYPE_INDEX[card_set]
    key = str(card.get("key", ""))
    prefix_map = {"j_": "joker", "v_": "voucher", "p_": "pack"}
    for prefix, kind in prefix_map.items():
        if key.startswith(prefix):
            return _ITEM_TYPE_INDEX[kind]
    if key.startswith("c_"):
        return _ITEM_TYPE_INDEX["tarot"]  # generic consumable bucket
    return ITEM_TYPE_NONE


def _encode_shop_item(card: dict) -> ShopToken:
    cost = card.get("cost")
    buy = cost.get("buy") if isinstance(cost, dict) else cost
    try:
        buy_val = float(buy)
    except (TypeError, ValueError):
        buy_val = 0.0
    return ShopToken(
        item_type_index=_shop_item_type(card),
        key_index=_vocab_index(_item_key_vocab(), card.get("key")),
        edition_index=_edition_index(card.get("edition")),
        cost=_clamp(buy_val / 10.0, 0.0, 5.0),
    )


def _encode_hand_levels(state: GameState) -> tuple[float, ...]:
    out = [1.0] * len(POKER_HANDS)  # Balatro hands start at level 1
    for name, lvl in (state.hand_levels or {}).items():
        idx = _hand_level_index(str(name))
        if idx is not None:
            try:
                out[idx] = float(lvl)
            except (TypeError, ValueError):
                pass
    return tuple(_clamp((v - 1.0) / 10.0, 0.0, 5.0) for v in out)


def _encode_deck_counts(state: GameState) -> tuple[float, ...]:
    counts = [0.0] * (len(RANKS) + len(SUITS) + 1)
    deck = state.known_deck
    n = len(deck)
    if n:
        for c in deck:
            ri = _RANK_INDEX.get(c.rank)
            if ri is not None:
                counts[ri] += 1.0
            si = _SUIT_INDEX.get(c.suit)
            if si is not None:
                counts[len(RANKS) + si] += 1.0
            if c.enhancement:
                counts[-1] += 1.0
        counts = [v / n for v in counts]
    return tuple(counts)


def _blind_type(state: GameState) -> str:
    current = state.modifiers.get("current_blind")
    if isinstance(current, dict):
        return str(current.get("type", "")).lower()
    return ""


def _boss_index(state: GameState) -> int:
    if _blind_type(state) == "boss":
        name = state.blind or ""
        current = state.modifiers.get("current_blind")
        if not name and isinstance(current, dict):
            name = str(current.get("name", ""))
        if name:
            return _BOSS_INDEX.get(name, UNK)
    return PAD


def _shop_card_dicts(state: GameState) -> tuple[dict, ...]:
    raw = state.modifiers.get("shop_cards", ())
    return tuple(c for c in raw if isinstance(c, dict))


def encode_state(state: GameState) -> EncodedState:
    """Encode a `GameState` into the structured, versioned representation."""
    required = max(state.required_score, 1)
    is_boss = 1.0 if _blind_type(state) == "boss" else 0.0
    shop_cards = _shop_card_dicts(state)

    scalars = (
        _clamp(state.ante / 8.0, 0.0, 2.0),
        _signed_log(state.money),
        _clamp(state.money / 50.0, -1.0, 4.0),
        _clamp(state.current_score / required, 0.0, 3.0),
        _clamp(math.log10(state.required_score + 1.0) / 6.0, 0.0, 2.0),
        _clamp(state.hands_remaining / 5.0, 0.0, 2.0),
        _clamp(state.discards_remaining / 5.0, 0.0, 2.0),
        _clamp(len(state.hand) / 12.0, 0.0, 2.0),
        _clamp(len(state.jokers) / 5.0, 0.0, 2.0),
        _clamp(len(state.consumables) / 2.0, 0.0, 3.0),
        _clamp(len(state.vouchers) / 4.0, 0.0, 3.0),
        _clamp(state.deck_size / 52.0, 0.0, 3.0),
        1.0 if state.known_deck else 0.0,
        is_boss,
        _clamp(len(shop_cards) / 4.0, 0.0, 3.0),
    )

    phase_onehot = [0.0] * len(PHASES)
    pi = _PHASE_INDEX.get(state.phase.value)
    if pi is not None:
        phase_onehot[pi] = 1.0

    blind_onehot = [0.0] * (len(BLIND_TYPES) + 1)
    blind_onehot[_BLIND_TYPE_INDEX.get(_blind_type(state), 0)] = 1.0

    return EncodedState(
        version=ENCODING_VERSION,
        scalars=scalars,
        phase_onehot=tuple(phase_onehot),
        blind_type_onehot=tuple(blind_onehot),
        boss_index=_boss_index(state),
        hand_levels=_encode_hand_levels(state),
        deck_counts=_encode_deck_counts(state),
        jokers=tuple(_encode_joker(j) for j in state.jokers),
        hand=tuple(_encode_card(c) for c in state.hand),
        shop=tuple(_encode_shop_item(c) for c in shop_cards),
        consumables=tuple(
            _vocab_index(_consumable_name_vocab(), n) for n in state.consumables
        ),
        vouchers=tuple(
            _vocab_index(_voucher_name_vocab(), n) for n in state.vouchers
        ),
    )


def encoding_spec() -> dict:
    """Vocabulary sizes + dims, for sizing model embeddings.

    All `*_vocab_size` values include the reserved PAD/UNK slots, so an
    embedding table sized to the spec is safe for every index `encode_state`
    can emit.
    """
    return {
        "version": ENCODING_VERSION,
        "scalar_dim": len(SCALAR_NAMES),
        "phase_dim": len(PHASES),
        "blind_type_dim": len(BLIND_TYPES) + 1,
        "hand_levels_dim": len(POKER_HANDS),
        "deck_counts_dim": len(RANKS) + len(SUITS) + 1,
        "joker_vocab_size": len(_joker_name_vocab()) + _RESERVED,
        "consumable_vocab_size": len(_consumable_name_vocab()) + _RESERVED,
        "voucher_vocab_size": len(_voucher_name_vocab()) + _RESERVED,
        "item_vocab_size": len(_item_key_vocab()) + _RESERVED,
        "boss_vocab_size": _BOSS_VOCAB_SIZE,
        "rank_vocab_size": len(RANKS),
        "suit_vocab_size": len(SUITS) + 1,
        "enhancement_vocab_size": len(ENHANCEMENTS) + 1,
        "edition_vocab_size": len(EDITIONS) + 1,
        "seal_vocab_size": len(SEALS) + 1,
        "item_type_vocab_size": len(ITEM_TYPES) + 1,
    }
