"""Seed-faithful shop sourcing for the local simulator.

Sources a shop's cards / voucher / booster packs from the validated
`rng/` predictors instead of the simulator's generic `Random`, so the
solver sees the same shop a real Balatro run would for a given seed.

Payloads are built by REUSING `ShopSampler`'s existing record->payload
machinery (`_pool_records` + `_payload_from_record`); we only swap in
the seed-faithful KEYS chosen by the predictors. Anything unexpected
(unknown key, playing-card shop slot, predictor error) returns None so
the caller falls back to generic sampling — never a wrong shop.

Scope: currently the ANTE-1 FIRST shop only. `predict_shop_surface`
reproduces a fresh seed's first shop exactly; later shops require
persistent per-run RNG state that isn't wired yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from balatro_ai.api.state import GameState
    from balatro_ai.search.shop_sampler import ShopSampler

# PredictedCard.set -> ShopSampler pool_type for shop-card slots.
_SET_TO_POOL = {
    "Joker": "Joker",
    "Tarot": "Tarot",
    "Planet": "Planet",
    "Spectral": "Spectral",
}


def _payload_for_key(
    sampler: "ShopSampler",
    state: "GameState",
    pool_type: str,
    key: str,
    *,
    edition: str | None = None,
) -> dict[str, Any] | None:
    """Build a shop payload for a specific pool key, or None if the key
    isn't in the (state-filtered) pool."""

    from balatro_ai.search.shop_sampler import _payload_from_record

    records = sampler._pool_records(pool_type, state)
    record = next((r for r in records if str(r.get("key")) == key), None)
    if record is None:
        return None
    return _payload_from_record(record, state, edition=edition)


def seed_faithful_shop(
    sampler: "ShopSampler",
    state: "GameState",
    rng: Any,
    *,
    first_shop: bool,
    voucher_key: str | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None, tuple[dict[str, Any], ...]] | None:
    """Return ``(shop_cards, voucher, boosters)`` for the current shop,
    advancing the persistent per-run ``rng`` exactly as the validated
    shop-sequence walk does (predict_shop_cards + predict_shop_boosters).
    Returns ``None`` to signal fallback to generic sampling.

    Voucher: ``voucher_key`` is the voucher chosen for the CURRENT ante,
    resolved once per ante by the caller (ante 1 from
    predict_initial_surface; ante 2+ from predict_voucher) and displayed
    on every shop of that ante. Validated 24/24 against the no-purchase
    shop-sequence fixtures (antes 1-3, all canonical seeds). Building the
    payload does not touch ``rng`` — the voucher roll happens in the
    caller so it advances exactly once per ante."""

    try:
        from balatro_ai.rng.surfaces import predict_shop_boosters
        from balatro_ai.search.shop_sampler import _booster_slot_count
    except ImportError:
        return None

    # Shop cards first (advances the cdt/sho streams), then boosters —
    # matching the validated shop-sequence walk order.
    shop_cards = _shop_card_payloads(sampler, state, rng)
    if shop_cards is None:
        return None

    try:
        n_boosters = _booster_slot_count(sampler.data)
        predicted_boosters = predict_shop_boosters(
            rng, ante=state.ante, n_slots=n_boosters, first_shop_buffoon=first_shop,
        )
    except Exception:  # noqa: BLE001 — never crash the sim path
        return None

    voucher: dict[str, Any] | None = None
    if voucher_key:
        voucher = _payload_for_key(sampler, state, "Voucher", voucher_key)
        if voucher is None:
            return None

    boosters: list[dict[str, Any]] = []
    for booster_key in predicted_boosters:
        payload = _payload_for_key(sampler, state, "Booster", booster_key)
        if payload is None:
            return None
        boosters.append(payload)

    return shop_cards, voucher, tuple(boosters)


def _shop_card_payloads(
    sampler: "ShopSampler",
    state: "GameState",
    rng: Any,
) -> tuple[dict[str, Any], ...] | None:
    """Predict the rerollable shop-card slots and map them to ShopSampler
    payloads, advancing ``rng`` by exactly one shop-card roll. Returns None
    (after advancing the rng) on any unsourced slot — playing-card / unknown
    key — so the caller can fall back to generic sampling."""

    try:
        from balatro_ai.rng.surfaces import predict_shop_cards
    except ImportError:
        return None
    try:
        n_slots = sampler.shop_slot_count(state)
        predicted_cards = predict_shop_cards(rng, ante=state.ante, n_slots=n_slots)
    except Exception:  # noqa: BLE001 — never crash the sim path
        return None

    shop_cards: list[dict[str, Any]] = []
    for predicted in predicted_cards:
        pool_type = _SET_TO_POOL.get(predicted.set)
        if pool_type is None:
            # Playing-card / enhanced shop slots aren't sourced yet.
            return None
        payload = _payload_for_key(
            sampler, state, pool_type, predicted.key, edition=predicted.edition,
        )
        if payload is None:
            return None
        shop_cards.append(payload)
    return tuple(shop_cards)


def seed_faithful_reroll(
    sampler: "ShopSampler",
    state: "GameState",
    rng: Any,
) -> tuple[dict[str, Any], ...] | None:
    """Return the new rerollable shop-card slots after a shop reroll, or
    None to fall back to generic sampling.

    A reroll re-rolls ONLY the shop cards (voucher and booster packs are
    untouched), consuming exactly one ``predict_shop_cards`` worth of the
    persistent rng — the same advance as opening the next shop. Validated
    20/20 against captured reroll fixtures (4 seeds x initial + 4 rerolls)."""

    return _shop_card_payloads(sampler, state, rng)


def seed_faithful_pack_contents(
    sampler: "ShopSampler",
    state: "GameState",
    pack: Any,
    seed: str,
) -> tuple[dict[str, Any], ...] | None:
    """Return seed-faithful contents for an opened booster pack, or None
    to fall back to generic sampling.

    Pack contents are keyed by (seed, ante, pack_key) independently of
    the shop-card RNG stream (validated 24/24 against pack fixtures), so
    this uses a fresh BalatroRNG and does NOT advance the persistent
    shop rng. Consumable/joker contents reuse the shop record->payload
    builder; Standard-pack playing cards build a payload from the
    predicted rank/suit/enhancement/edition/seal. Telescope (forces the
    first Celestial card to the most-played hand's planet) and Omen Globe
    (Arcana->Spectral) are now threaded via played-hand context."""

    if not isinstance(pack, dict):
        return None
    pack_key = pack.get("key")
    if not pack_key:
        return None
    try:
        from balatro_ai.rng.surfaces import predict_pack_contents
    except ImportError:
        return None

    ctx = _pack_prediction_context(state)
    try:
        predicted = predict_pack_contents(
            seed,
            ante=state.ante,
            pack_key=str(pack_key),
            vouchers=ctx["vouchers"],
            played_hand_types=ctx["played_hand_types"],
            telescope_planet_key=ctx["telescope_planet_key"],
            edition_rate=ctx["edition_rate"],
        )
    except Exception:  # noqa: BLE001 — never crash the sim path
        return None

    contents: list[dict[str, Any]] = []
    for card in predicted:
        payload = _pack_card_payload(sampler, state, card)
        if payload is None:
            return None
        contents.append(payload)
    return tuple(contents)


# Poker-hand display order (best-first), used to break ties when picking
# the most-played hand for the Telescope voucher — mirrors the validator.
_HAND_ORDER = (
    "Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
    "Four of a Kind", "Full House", "Flush", "Straight",
    "Three of a Kind", "Two Pair", "Pair", "High Card",
)


def _pack_prediction_context(state: "GameState") -> dict[str, Any]:
    """Played-hand / voucher context for pack-content prediction, read from
    the same GameState fields the validator reads from bridge fixtures."""

    vouchers = tuple(state.vouchers)
    counts = _run_played_hand_counts(state)
    played_hand_types = frozenset(name for name, count in counts.items() if count > 0)

    telescope_planet_key: str | None = None
    if any(_has(vouchers, v) for v in ("v_telescope", "Telescope")):
        try:
            from balatro_ai.rng.surfaces import planet_key_for_hand
            telescope_planet_key = planet_key_for_hand(_most_played_hand(counts))
        except Exception:  # noqa: BLE001
            telescope_planet_key = None

    edition_rate = 1.0
    if any(_has(vouchers, v) for v in ("v_glow_up", "Glow Up")):
        edition_rate = 4.0
    elif any(_has(vouchers, v) for v in ("v_hone", "Hone")):
        edition_rate = 2.0

    return {
        "vouchers": vouchers,
        "played_hand_types": played_hand_types,
        "telescope_planet_key": telescope_planet_key,
        "edition_rate": edition_rate,
    }


def _has(vouchers: tuple[Any, ...], value: str) -> bool:
    return value in vouchers


def _run_played_hand_counts(state: "GameState") -> dict[str, int]:
    modifiers = getattr(state, "modifiers", None)
    hands = modifiers.get("hands") if isinstance(modifiers, dict) else None
    counts: dict[str, int] = {}
    if isinstance(hands, dict):
        for name, raw in hands.items():
            if isinstance(raw, dict):
                try:
                    counts[str(name)] = int(raw.get("played", 0) or 0)
                except (TypeError, ValueError):
                    pass
    return counts


def _most_played_hand(counts: dict[str, int]) -> str | None:
    best: str | None = None
    tally = 0
    for hand_name in _HAND_ORDER:
        count = counts.get(hand_name, 0)
        if count > tally:
            best = hand_name
            tally = count
    return best


def _pack_card_payload(
    sampler: "ShopSampler",
    state: "GameState",
    card: Any,
) -> dict[str, Any] | None:
    """Map a PredictedCard to a ShopSampler-style payload, or None if the
    card's set isn't sourceable yet."""

    pool_type = _SET_TO_POOL.get(card.set)
    if pool_type is not None:
        return _payload_for_key(sampler, state, pool_type, card.key, edition=card.edition)
    if card.set in ("Default", "Enhanced") and card.front_key and "_" in card.front_key:
        return _playing_card_payload(sampler, state, card)
    return None


def _playing_card_payload(
    sampler: "ShopSampler",
    state: "GameState",
    card: Any,
) -> dict[str, Any] | None:
    try:
        from balatro_ai.search.shop_sampler import ENHANCEMENT_BY_CENTER_KEY, build_playing_card_payload
    except ImportError:
        return None
    suit, rank_token = card.front_key.split("_", 1)
    rank = "10" if rank_token == "T" else rank_token
    enhancement = ENHANCEMENT_BY_CENTER_KEY.get(card.key) if card.set == "Enhanced" else None
    return build_playing_card_payload(
        sampler.data,
        state,
        rank=rank,
        suit=suit,
        enhancement=enhancement,
        edition=card.edition,
        seal=card.seal,
    )
