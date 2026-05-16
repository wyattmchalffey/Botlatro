"""Winning discard lines that improve round-end economy."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import sys

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.actions import _annotated_action
from balatro_ai.bots.basic_strategy.blind_reasons import _winning_economy_hunt_discard_reason
from balatro_ai.bots.basic_strategy.discard_state import (
    _conditional_discard_money_delta_for_economy_hunt,
    _known_draw_for_discard,
    _state_after_known_discard_for_economy_hunt,
)
from balatro_ai.bots.basic_strategy.economy_hunt import (
    _card_is_economy_hunt_target,
    _discard_sensitive_cash_out_value,
    _drawn_economy_hunt_value,
    _held_round_end_economy_value,
)
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.play_scoring import (
    _action_index_sum,
    _boss_adjusted_score as _default_boss_adjusted_score,
    _evaluate_play_action as _default_evaluate_play_action,
)
from balatro_ai.bots.config import active_config


WINNING_ECONOMY_HUNT_MIN_GAIN = 2.5


def _winning_economy_hunt_discard_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    if (
        remaining_score <= 0
        or score < remaining_score
        or state.discards_remaining <= 0
        or not state.known_deck
    ):
        return None

    baseline_value = _clear_economy_value_for_play(state, best_play, context)
    ranked: list[tuple[tuple[float, float, int, int, int], Action, float, int]] = []
    for action in state.legal_actions:
        if action.action_type != ActionType.DISCARD or not action.card_indices:
            continue
        drawn_cards = _known_draw_for_discard(state, action)
        if not drawn_cards or not any(_card_is_economy_hunt_target(state, card) for card in drawn_cards):
            continue
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        discard_value = _conditional_discard_money_delta_for_economy_hunt(state, discarded_cards, context)
        projected_state = _state_after_known_discard_for_economy_hunt(state, action, drawn_cards, context)
        projected_context = replace(context, discards_taken=context.discards_taken + 1)
        projected_value, projected_score = _best_clear_economy_value(projected_state, remaining_score, projected_context)
        projected_value += discard_value
        if projected_score < remaining_score:
            continue
        gain = projected_value - baseline_value
        if gain + active_config().shop_value_tolerance < WINNING_ECONOMY_HUNT_MIN_GAIN:
            continue
        ranked.append(
            (
                (
                    gain,
                    projected_value,
                    _drawn_economy_hunt_value(state, drawn_cards),
                    -len(action.card_indices),
                    -_action_index_sum(action),
                ),
                action,
                gain,
                projected_score,
            )
        )

    if not ranked:
        return None

    _, action, gain, projected_score = max(ranked, key=lambda item: item[0])
    return _annotated_action(
        action,
        reason=_winning_economy_hunt_discard_reason(state, best_play, action, gain, projected_score, context),
    )


def _best_clear_economy_value(
    state: GameState,
    remaining_score: int,
    context: _BlindContext,
) -> tuple[float, int]:
    best_value = float("-inf")
    best_score = 0
    max_cards = min(5, len(state.hand))
    for count in range(1, max_cards + 1):
        for indices in combinations(range(len(state.hand)), count):
            action = Action(ActionType.PLAY_HAND, card_indices=tuple(indices))
            evaluation = _evaluate_play_action(state, action, context)
            score = _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)
            if score < remaining_score:
                continue
            value = _clear_economy_value_for_evaluated_play(state, action, evaluation, context)
            if value > best_value or (value == best_value and score > best_score):
                best_value = value
                best_score = score
    return best_value, best_score


def _clear_economy_value_for_play(
    state: GameState,
    action: Action,
    context: _BlindContext,
) -> float:
    evaluation = _evaluate_play_action(state, action, context)
    return _clear_economy_value_for_evaluated_play(state, action, evaluation, context)


def _clear_economy_value_for_evaluated_play(
    state: GameState,
    action: Action,
    evaluation,
    context: _BlindContext,
) -> float:
    played = set(action.card_indices)
    held_cards = tuple(card for index, card in enumerate(state.hand) if index not in played)
    return (
        float(evaluation.money_delta)
        + _held_round_end_economy_value(state, held_cards)
        + _discard_sensitive_cash_out_value(state, context)
    )


def _boss_adjusted_score(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._boss_adjusted_score``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_boss_adjusted_score", None) if strategy_module is not None else None
    if patched is not None and patched is not _boss_adjusted_score:
        return patched(*args, **kwargs)
    return _default_boss_adjusted_score(*args, **kwargs)


def _evaluate_play_action(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._evaluate_play_action``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_evaluate_play_action", None) if strategy_module is not None else None
    if patched is not None and patched is not _evaluate_play_action:
        return patched(*args, **kwargs)
    return _default_evaluate_play_action(*args, **kwargs)
