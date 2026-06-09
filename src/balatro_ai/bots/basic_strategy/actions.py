"""Small action helpers for the basic strategy bot."""

from __future__ import annotations

from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.data import DANGEROUS_BOSS_BLINDS, FINAL_BOSS_BLINDS
from balatro_ai.bots.basic_strategy.shop_forecast import _upcoming_boss_blind_name
from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure

# NOTE: a static blind-SKIP policy was tested 2026-06-08 and REVERTED as
# net-negative (see PROGRESS.md). The bot never skips Small/Big blinds; skipping
# trades a blind's cash-out + the following shop for a skip tag. A static
# tag/ante rule (skip Small on free-value tags) regressed winrate 21->11/100
# because the same tag at the same ante helps on some seeds and hurts on others
# -- the value of a skip depends on the specific forgone shop + build state, i.e.
# it needs counterfactual foresight (the whole-run-planning gap), not a heuristic.


def _first_action_of_type(state: GameState, action_type: ActionType) -> Action | None:
    for action in state.legal_actions:
        if action.action_type == action_type:
            return action
    return None


def _blind_select_action(state: GameState) -> Action | None:
    select = _first_action_of_type(state, ActionType.SELECT_BLIND)
    if select is None:
        return None
    boss_reroll = _boss_reroll_action(state)
    if boss_reroll is not None and _should_reroll_boss_blind(state):
        boss_name = _upcoming_boss_blind_name(state) or state.blind
        pressure = _shop_pressure(state)
        return _annotated_action(
            boss_reroll,
            reason=(
                f"boss_reroll boss={boss_name} "
                f"pressure={pressure.ratio:.2f} raw_pressure={pressure.raw_ratio:.2f}"
            ),
        )
    return select


def _boss_reroll_action(state: GameState) -> Action | None:
    for action in state.legal_actions:
        if action.action_type != ActionType.REROLL:
            continue
        if str(action.metadata.get("kind", "")) == "boss":
            return action
    return None


def _should_reroll_boss_blind(state: GameState) -> bool:
    boss_name = _upcoming_boss_blind_name(state) or state.blind
    if boss_name not in DANGEROUS_BOSS_BLINDS and boss_name not in FINAL_BOSS_BLINDS:
        return False
    if state.ante < 5:
        return False

    pressure = _shop_pressure(state)
    if boss_name != "Violet Vessel":
        return False
    return pressure.ratio >= 1.0 or pressure.raw_ratio >= 0.75


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
