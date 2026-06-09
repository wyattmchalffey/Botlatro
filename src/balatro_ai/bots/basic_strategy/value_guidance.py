"""Phase 8 value-model guidance for shop buys.

Adds a learned 1-step-lookahead bonus to joker/planet buy values:
    delta = V(state after acquiring the card) - V(state)
scaled into the heuristic's value units. V is the outcome-grounded value model,
so this nudges the bot toward buys that raise win/ante value without rewriting
the shop heuristics.

Off by default so it remains a clean A/B against the heuristic baseline. Set
BALATRO_VALUE_MODEL_CKPT to use a current torch ValueNet checkpoint, or
BALATRO_VALUE_MODEL=1 to use the legacy pure-numpy value model.
BALATRO_VALUE_SCALE tunes the weight.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from balatro_ai.api.state import GameState


def _enabled() -> bool:
    return bool(_value_net_checkpoint_path() or os.environ.get("BALATRO_VALUE_MODEL") == "1")


def _scale() -> float:
    try:
        return float(os.environ.get("BALATRO_VALUE_SCALE", "250"))
    except ValueError:
        return 250.0


def _value_net_checkpoint_path() -> Path | None:
    raw = os.environ.get("BALATRO_VALUE_MODEL_CKPT", "").strip()
    return Path(raw) if raw else None


def _value_net_head() -> str:
    head = os.environ.get("BALATRO_VALUE_MODEL_HEAD", "ante").strip().lower()
    return head if head in {"ante", "win", "clear"} else "ante"


_VALUE_NET_CACHE: tuple[Path, object] | None = None


def _value_net_predict(state: GameState) -> float | None:
    from balatro_ai.bots.basic_strategy.cache import _identity_cached_value

    return _identity_cached_value("value_guidance_value_net_predict", state, lambda: _value_net_predict_uncached(state))


def _value_net_predict_uncached(state: GameState) -> float | None:
    global _VALUE_NET_CACHE

    path = _value_net_checkpoint_path()
    if path is None:
        return None
    if _VALUE_NET_CACHE is None or _VALUE_NET_CACHE[0] != path:
        from balatro_ai.ml.train import load_checkpoint

        model = load_checkpoint(path)
        model.eval()
        _VALUE_NET_CACHE = (path, model)
    else:
        model = _VALUE_NET_CACHE[1]

    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.model import collate_states
    import torch

    torch.set_num_threads(1)
    batch = collate_states([encode_state(state)])
    head = _value_net_head()
    with torch.no_grad():
        if head == "win":
            value = model.win_prob(batch)
        elif head == "clear":
            value = model.clear_value(batch)
        else:
            value = model.ante_value(batch)
    return float(value[0])


def _legacy_value_predict(state: GameState) -> float | None:
    if os.environ.get("BALATRO_VALUE_MODEL") != "1":
        return None
    from balatro_ai.ml.features import features_from_state
    from balatro_ai.ml.value_model import get_value_model

    model = get_value_model()
    if model is None:
        return None
    return float(model.predict(features_from_state(state)))


def value_bonus_for_card(state: GameState, card: object) -> float:
    """Scaled value delta for acquiring ``card`` (joker/planet), or 0.0
    when disabled / no model / unsupported card. Never raises."""

    if not _enabled():
        return 0.0
    try:
        resulting = _state_after_acquire(state, card)
        if resulting is None:
            return 0.0
        base = _value_net_predict(state)
        after = _value_net_predict(resulting)
        if base is None or after is None:
            base = _legacy_value_predict(state)
            after = _legacy_value_predict(resulting)
        if base is None or after is None:
            return 0.0
        return (after - base) * _scale()
    except Exception:  # noqa: BLE001 - guidance must never break the bot
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
