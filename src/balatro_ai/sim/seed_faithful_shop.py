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


def seed_faithful_first_shop(
    sampler: "ShopSampler",
    state: "GameState",
    seed: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None, tuple[dict[str, Any], ...]] | None:
    """Return ``(shop_cards, voucher, boosters)`` payload tuples for the
    ante-1 first shop sourced from the seed-faithful predictors, or
    ``None`` to signal the caller to fall back to generic sampling."""

    try:
        from balatro_ai.rng.surfaces import predict_shop_surface
        from balatro_ai.search.shop_sampler import _booster_slot_count
    except ImportError:
        return None

    try:
        n_slots = sampler.shop_slot_count(state)
        n_boosters = _booster_slot_count(sampler.data)
        surface = predict_shop_surface(
            seed, ante=state.ante, n_shop_slots=n_slots, n_boosters=n_boosters,
        )
    except Exception:  # noqa: BLE001 — never crash the sim path
        return None

    shop_cards: list[dict[str, Any]] = []
    for predicted in surface.shop_cards:
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
    if surface.voucher_key:
        voucher = _payload_for_key(sampler, state, "Voucher", surface.voucher_key)
        if voucher is None:
            return None

    boosters: list[dict[str, Any]] = []
    for booster_key in surface.booster_keys:
        payload = _payload_for_key(sampler, state, "Booster", booster_key)
        if payload is None:
            return None
        boosters.append(payload)

    return tuple(shop_cards), voucher, tuple(boosters)
