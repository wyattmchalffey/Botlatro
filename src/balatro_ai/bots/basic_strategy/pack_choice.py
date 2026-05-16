"""Booster pack choice and skip policy."""

from __future__ import annotations

from dataclasses import replace
import sys

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.basic_strategy.actions import _annotated_action, _first_action_of_type, _with_target_indices
from balatro_ai.bots.basic_strategy.cards import (
    _is_joker_card,
    _is_spectral_card,
    _joker_from_shop_card,
    _pack_card_requires_targets,
    _uses_normal_joker_slot,
)
from balatro_ai.bots.basic_strategy.jokers import _joker_with_added_current_plus
from balatro_ai.bots.basic_strategy.pack_targets import _pack_card_is_pickable, _pack_card_target_indices
from balatro_ai.bots.basic_strategy.profile import _ShopContext


def _pack_choice_action(state: GameState, context: _ShopContext | None = None) -> Action | None:
    context = context or _ShopContext()
    pack_cards = state.modifiers.get("pack_cards", ())
    if not pack_cards:
        return None
    best_action: Action | None = None
    best_value = float("-inf")
    best_target_count = -1
    best_should_take_without_edge = False
    skip_action = _pack_skip_action(state, has_pack_cards=bool(pack_cards))
    skip_value = _pack_skip_value(state)

    for action in state.legal_actions:
        if action.action_type != ActionType.CHOOSE_PACK_CARD or action.target_id == "skip":
            continue
        index = int(action.metadata.get("index", action.amount or 0))
        if index >= len(pack_cards):
            continue
        pack_card = pack_cards[index]
        if context.filled_last_joker_slot and _normal_slot_joker_card(pack_card):
            continue
        if not _pack_card_is_pickable(state, pack_card):
            continue
        target_indices = _pack_card_target_indices(state, pack_card) or tuple(action.card_indices)
        if _pack_card_requires_targets(pack_card) and not target_indices:
            continue
        value = _pack_card_value(state, pack_card)
        target_count = len(target_indices)
        if value > best_value or (value == best_value and target_count > best_target_count):
            best_action = _with_target_indices(action, target_indices)
            best_value = value
            best_target_count = target_count
            best_should_take_without_edge = not _is_joker_card(pack_card) and not _is_spectral_card(pack_card)

    if skip_action is not None and skip_value > 0.0 and skip_value >= best_value:
        return _annotated_action(skip_action, reason=f"pack_skip value={skip_value:.1f}")

    if best_action is not None and (best_value > skip_value or best_should_take_without_edge):
        return _annotated_action(best_action, reason=f"pack_pick value={best_value:.1f}")

    if skip_action is not None:
        return skip_action
    return None


def _pack_skip_action(state: GameState, *, has_pack_cards: bool) -> Action | None:
    for action in state.legal_actions:
        if action.action_type == ActionType.CHOOSE_PACK_CARD and action.target_id == "skip":
            return action
    if has_pack_cards:
        return Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
    return None


def _stale_empty_pack_action(state: GameState) -> Action | None:
    if state.phase != GamePhase.BOOSTER_OPENED:
        return None
    if state.pack or state.modifiers.get("pack_cards"):
        return None
    return _first_action_of_type(state, ActionType.NO_OP) or Action(ActionType.NO_OP)


def _pack_skip_value(state: GameState) -> float:
    return _red_card_pack_skip_value(state)


def _red_card_pack_skip_value(state: GameState) -> float:
    if not any(joker.name == "Red Card" for joker in state.jokers):
        return 0.0

    boosted_jokers = tuple(
        _joker_with_added_current_plus(joker, 3, suffix="mult") if joker.name == "Red Card" else joker
        for joker in state.jokers
    )
    current_score = _sample_build_score(state, state.jokers)
    boosted_score = _sample_build_score(replace(state, jokers=boosted_jokers), boosted_jokers)
    score_delta = max(0.0, boosted_score - current_score)
    value = 18.0 + min(90.0, score_delta / 8.0)
    if state.ante <= 2:
        value += 8.0
    return value


def _normal_slot_joker_card(card: object) -> bool:
    return _is_joker_card(card) and _uses_normal_joker_slot(_joker_from_shop_card(card))


def _pack_card_value(*args, **kwargs) -> float:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._pack_card_value``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_pack_card_value", None) if strategy_module is not None else None
    if patched is None or patched is _pack_card_value:
        raise RuntimeError("basic_strategy_bot._pack_card_value is not available")
    return patched(*args, **kwargs)


def _sample_build_score(*args, **kwargs) -> float:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._sample_build_score``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_sample_build_score", None) if strategy_module is not None else None
    if patched is None or patched is _sample_build_score:
        raise RuntimeError("basic_strategy_bot._sample_build_score is not available")
    return patched(*args, **kwargs)
