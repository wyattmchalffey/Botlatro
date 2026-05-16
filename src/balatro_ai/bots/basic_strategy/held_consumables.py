"""Held consumable use policy."""

from __future__ import annotations

import sys

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.actions import _action_index_for_strategy, _annotated_action
from balatro_ai.bots.basic_strategy.cards import (
    _consumable_card_for_name,
    _consumable_open_slots_after_storage_use,
    _is_black_hole_card,
    _is_planet_card,
    _is_spectral_card,
    _is_tarot_card,
    _normal_joker_open_slots,
    _pack_card_requires_targets,
)


def _held_consumable_action(state: GameState) -> Action | None:
    best: tuple[float, int, Action, str] | None = None
    for action in state.legal_actions:
        if action.action_type != ActionType.USE_CONSUMABLE:
            continue
        index = _action_index_for_strategy(action)
        if index is None or index >= len(state.consumables):
            continue
        name = state.consumables[index]
        value = _held_consumable_value(state, name, action)
        if value <= 0.0:
            continue
        candidate = (value, len(action.card_indices), action, name)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
            best = candidate
    if best is None:
        return None
    value, _, action, name = best
    return _annotated_action(action, reason=f"use_consumable name={name} value={value:.1f}")


def _held_consumable_value(state: GameState, name: str, action: Action) -> float:
    card = _consumable_card_for_name(name)
    if _is_planet_card(card):
        return _planet_card_value(state, card) + 30.0
    if _is_black_hole_card(card):
        return _black_hole_card_value(state) + 100.0
    if _is_tarot_card(card):
        if _pack_card_requires_targets(card) and not action.card_indices:
            return 0.0
        if name == "Judgement" and _normal_joker_open_slots(state) <= 0:
            return 0.0
        if name in {"The Emperor", "The High Priestess", "The Fool"} and _consumable_open_slots_after_storage_use(state) <= 0:
            return 0.0
        return _tarot_card_value(state, card) + 8.0
    if _is_spectral_card(card):
        value = _spectral_card_value(state, card)
        return value + 8.0 if value > 0.0 else 0.0
    return 0.0


def _black_hole_card_value(*args, **kwargs) -> float:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._black_hole_card_value``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_black_hole_card_value", None) if strategy_module is not None else None
    if patched is None or patched is _black_hole_card_value:
        raise RuntimeError("basic_strategy_bot._black_hole_card_value is not available")
    return patched(*args, **kwargs)


def _planet_card_value(*args, **kwargs) -> float:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._planet_card_value``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_planet_card_value", None) if strategy_module is not None else None
    if patched is None or patched is _planet_card_value:
        raise RuntimeError("basic_strategy_bot._planet_card_value is not available")
    return patched(*args, **kwargs)


def _spectral_card_value(*args, **kwargs) -> float:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._spectral_card_value``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_spectral_card_value", None) if strategy_module is not None else None
    if patched is None or patched is _spectral_card_value:
        raise RuntimeError("basic_strategy_bot._spectral_card_value is not available")
    return patched(*args, **kwargs)


def _tarot_card_value(*args, **kwargs) -> float:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._tarot_card_value``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_tarot_card_value", None) if strategy_module is not None else None
    if patched is None or patched is _tarot_card_value:
        raise RuntimeError("basic_strategy_bot._tarot_card_value is not available")
    return patched(*args, **kwargs)
