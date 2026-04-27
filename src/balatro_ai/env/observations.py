"""Observation encoding for early baseline bots and tests."""

from __future__ import annotations

from balatro_ai.api.state import GameState

BASIC_FEATURES = (
    "ante",
    "required_score",
    "current_score",
    "hands_remaining",
    "discards_remaining",
    "money",
    "deck_size",
    "hand_size",
    "joker_count",
    "consumable_count",
    "voucher_count",
    "shop_count",
    "pack_count",
    "hand_level_sum",
)


def vector_from_state(state: GameState) -> tuple[float, ...]:
    """Return a small numeric observation vector.

    This is intentionally simple. Later phases should replace or augment this
    with a richer tensor encoding for cards, jokers, shops, and deck state.
    """

    hand_level_sum = sum(int(value) for value in state.hand_levels.values())
    values = (
        state.ante,
        state.required_score,
        state.current_score,
        state.hands_remaining,
        state.discards_remaining,
        state.money,
        state.deck_size,
        len(state.hand),
        len(state.jokers),
        len(state.consumables),
        len(state.vouchers),
        len(state.shop),
        len(state.pack),
        hand_level_sum,
    )
    return tuple(float(value) for value in values)

