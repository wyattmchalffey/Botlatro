"""Phase 8 value-model guidance for shop buys.

Adds a learned 1-step-lookahead bonus to joker/planet buy values:
    delta = V(state after acquiring the card) - V(state)
scaled into the heuristic's value units. V is the outcome-grounded
win-probability model (ml.value_model), so this nudges the bot toward
buys that actually raise win probability -- targeting the proven
build-power bottleneck without rewriting the shop heuristics.

Gated on BALATRO_VALUE_MODEL=1 (off by default) so it's a clean A/B
against the heuristic baseline. BALATRO_VALUE_SCALE tunes the weight.
"""

from __future__ import annotations

import os
from dataclasses import replace

from balatro_ai.api.state import GameState


def _enabled() -> bool:
    return os.environ.get("BALATRO_VALUE_MODEL") == "1"


def _scale() -> float:
    try:
        return float(os.environ.get("BALATRO_VALUE_SCALE", "250"))
    except ValueError:
        return 250.0


def value_bonus_for_card(state: GameState, card: object) -> float:
    """Scaled win-prob delta for acquiring ``card`` (joker/planet), or 0.0
    when disabled / no model / unsupported card. Never raises."""

    if not _enabled():
        return 0.0
    try:
        from balatro_ai.ml.features import features_from_state
        from balatro_ai.ml.value_model import get_value_model
        model = get_value_model()
        if model is None:
            return 0.0
        resulting = _state_after_acquire(state, card)
        if resulting is None:
            return 0.0
        base = model.predict(features_from_state(state))
        after = model.predict(features_from_state(resulting))
        return (after - base) * _scale()
    except Exception:  # noqa: BLE001 — guidance must never break the bot
        return 0.0


def _state_after_acquire(state: GameState, card: object) -> GameState | None:
    from balatro_ai.bots.basic_strategy.cards import (
        _card_cost,
        _card_label,
        _is_joker_card,
        _is_planet_card,
        _joker_from_shop_card,
    )
    from balatro_ai.bots.basic_strategy.data import PLANET_TO_HAND

    money = max(0, int(state.money) - _card_cost(card))
    if _is_joker_card(card):
        joker = _joker_from_shop_card(card)
        return replace(state, jokers=tuple(state.jokers) + (joker,), money=money)
    if _is_planet_card(card):
        hand_type = PLANET_TO_HAND.get(_card_label(card))
        if hand_type is None:
            return None
        levels = dict(state.hand_levels)
        levels[hand_type.value] = levels.get(hand_type.value, 1) + 1
        return replace(state, hand_levels=levels, money=money)
    return None
