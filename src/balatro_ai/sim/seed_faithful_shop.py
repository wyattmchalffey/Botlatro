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
    initial_voucher_key: str | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None, tuple[dict[str, Any], ...]] | None:
    """Return ``(shop_cards, voucher, boosters)`` for the current shop,
    advancing the persistent per-run ``rng`` exactly as the validated
    shop-sequence walk does (predict_shop_cards + predict_shop_boosters).
    Returns ``None`` to signal fallback to generic sampling.

    Voucher: on the first shop we use the ante-1 voucher already chosen
    by predict_initial_surface (``initial_voucher_key``). Per-ante
    voucher *timing* on later shops isn't modeled yet, so later shops
    carry no voucher (a documented gap; shop CARDS + BOOSTERS — the
    build-relevant content validated 51/51 — stay seed-faithful)."""

    try:
        from balatro_ai.rng.surfaces import predict_shop_cards, predict_shop_boosters
        from balatro_ai.search.shop_sampler import _booster_slot_count
    except ImportError:
        return None

    try:
        n_slots = sampler.shop_slot_count(state)
        n_boosters = _booster_slot_count(sampler.data)
        predicted_cards = predict_shop_cards(rng, ante=state.ante, n_slots=n_slots)
        predicted_boosters = predict_shop_boosters(
            rng, ante=state.ante, n_slots=n_boosters, first_shop_buffoon=first_shop,
        )
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

    voucher: dict[str, Any] | None = None
    if first_shop and initial_voucher_key:
        voucher = _payload_for_key(sampler, state, "Voucher", initial_voucher_key)
        if voucher is None:
            return None

    boosters: list[dict[str, Any]] = []
    for booster_key in predicted_boosters:
        payload = _payload_for_key(sampler, state, "Booster", booster_key)
        if payload is None:
            return None
        boosters.append(payload)

    return tuple(shop_cards), voucher, tuple(boosters)


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
    builder; playing-card (Standard pack) contents aren't sourced yet
    and bail. Telescope/Omen-Globe need played-hand context we don't
    thread, so bail when those vouchers are owned."""

    if not isinstance(pack, dict):
        return None
    pack_key = pack.get("key")
    if not pack_key:
        return None
    vouchers = tuple(state.vouchers)
    if any(v in vouchers for v in ("Telescope", "v_telescope", "Omen Globe", "v_omen_globe")):
        return None
    try:
        from balatro_ai.rng.surfaces import predict_pack_contents
        predicted = predict_pack_contents(
            seed, ante=state.ante, pack_key=str(pack_key), vouchers=vouchers,
        )
    except Exception:  # noqa: BLE001 — never crash the sim path
        return None

    contents: list[dict[str, Any]] = []
    for card in predicted:
        pool_type = _SET_TO_POOL.get(card.set)
        if pool_type is None:
            # Standard-pack playing cards aren't sourced yet.
            return None
        payload = _payload_for_key(sampler, state, pool_type, card.key, edition=card.edition)
        if payload is None:
            return None
        contents.append(payload)
    return tuple(contents)
