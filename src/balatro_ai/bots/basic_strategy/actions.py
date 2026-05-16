"""Small action helpers for the basic strategy bot."""

from __future__ import annotations

from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState


def _first_action_of_type(state: GameState, action_type: ActionType) -> Action | None:
    for action in state.legal_actions:
        if action.action_type == action_type:
            return action
    return None


def _blind_select_action(state: GameState) -> Action | None:
    select = _first_action_of_type(state, ActionType.SELECT_BLIND)
    if select is None:
        return None
    return select


def _annotated_action(action: Action, *, reason: str, audit: dict[str, Any] | None = None) -> Action:
    metadata = {**action.metadata, "reason": reason}
    if audit is not None:
        metadata["shop_audit"] = audit
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata=metadata,
    )


def _shop_memory_key(state: GameState) -> tuple[int | None, int, str, int]:
    return (state.seed, state.ante, state.blind, state.required_score)


def _blind_memory_key(state: GameState) -> tuple[int | None, int, str, int]:
    return (state.seed, state.ante, state.blind, state.required_score)


def _with_target_indices(action: Action, target_indices: tuple[int, ...]) -> Action:
    if not target_indices:
        return action
    return Action(
        action.action_type,
        card_indices=target_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata=action.metadata,
    )


def _action_index_for_strategy(action: Action) -> int | None:
    raw = action.metadata.get("index", action.amount)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
