"""Blind setup actions that spend tempo for joker value."""

from __future__ import annotations

from dataclasses import replace
import sys

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.actions import _annotated_action
from balatro_ai.bots.basic_strategy.blind_reasons import (
    _joker_discard_reason,
    _mystic_summit_setup_discard_reason,
    _play_reason,
)
from balatro_ai.bots.basic_strategy.cards import _has_consumable_room
from balatro_ai.bots.basic_strategy.discard_policy import (
    _best_discard_action as _default_best_discard_action,
    _discard_action_playstyle_bonus,
    _estimated_hands_needed as _default_estimated_hands_needed,
)
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type
from balatro_ai.bots.basic_strategy.hand_value import _card_keep_scores, _card_long_term_value
from balatro_ai.bots.basic_strategy.jokers import _joker_names
from balatro_ai.bots.basic_strategy.play_scoring import _score_play_action as _default_score_play_action
from balatro_ai.bots.basic_strategy.score_projection import (
    _projected_score_after_discard as _default_projected_score_after_discard,
)


def _opening_joker_setup_play_action(
    state: GameState,
    best_play: Action,
    best_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    if (
        state.current_score != 0
        or context.played_hand_types
        or state.hands_remaining <= 2
        or remaining_score <= 0
    ):
        return None

    single_play_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.PLAY_HAND and len(action.card_indices) == 1
    ]
    if not single_play_actions:
        return None

    names = _joker_names(state)
    candidates: list[tuple[float, Action]] = []
    for action in single_play_actions:
        card = state.hand[action.card_indices[0]]
        value = 0.0
        if "DNA" in names:
            value += 450.0 + _card_long_term_value(state, card)
        if "Sixth Sense" in names and card.rank == "6" and _has_consumable_room(state):
            value += 520.0
        if value <= 0:
            continue
        score = _score_play_action(state, action, context)
        if not _opening_setup_is_safe(state, remaining_score, setup_score=score, fallback_score=best_score):
            continue
        candidates.append((value + score, action))

    if not candidates:
        return None
    _, action = max(candidates, key=lambda item: item[0])
    return _annotated_action(action, reason=f"joker_setup {_play_reason(state, action, context)}")


def _opening_setup_is_safe(
    state: GameState,
    remaining_score: int,
    *,
    setup_score: int,
    fallback_score: int,
) -> bool:
    after_remaining = max(0, remaining_score - setup_score)
    if after_remaining <= 0:
        return True
    if state.ante <= 2 and state.hands_remaining >= 3:
        return _estimated_hands_needed(after_remaining, max(1, fallback_score)) <= state.hands_remaining
    return _estimated_hands_needed(after_remaining, max(1, fallback_score)) <= max(1, state.hands_remaining - 1)


def _mystic_summit_setup_discard_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    names = _joker_names(state)
    if (
        "Mystic Summit" not in names
        or "Banner" in names
        or "Delayed Gratification" in names
        or "Green Joker" in names
        or "Ramen" in names
        or state.ante > 2
        or state.current_score > 0
        or remaining_score <= 0
        or score >= remaining_score
        or state.hands_remaining <= 1
        or state.discards_remaining <= 0
    ):
        return None

    active_state = replace(state, discards_remaining=0)
    active_score = _score_play_action(active_state, best_play, context)
    if active_score <= score:
        return None

    current_hands_needed = _estimated_hands_needed(remaining_score, score)
    active_hands_needed = _estimated_hands_needed(remaining_score, active_score)
    if active_score < remaining_score and active_hands_needed >= current_hands_needed:
        return None

    best_discard = _best_discard_action(state, current_best_score=score, context=context)
    if best_discard is None:
        return None

    if state.discards_remaining <= 1:
        projected_score = _projected_score_after_discard(state, best_discard, context)
        projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
        if projected_score < remaining_score and projected_hands_needed >= current_hands_needed:
            return None

    return _annotated_action(
        best_discard,
        reason=_mystic_summit_setup_discard_reason(
            state,
            best_play,
            best_discard,
            score,
            active_score,
            context,
        ),
    )


def _strategic_joker_discard_action(
    state: GameState,
    best_play: Action,
    best_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    if state.discards_remaining <= 0 or state.hands_remaining <= 1 or remaining_score <= 0:
        return None

    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        return None

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    ranked: list[tuple[float, int, Action]] = []
    for action in discard_actions:
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        bonus = _discard_action_playstyle_bonus(
            state,
            action,
            discarded_cards=discarded_cards,
            keep_scores=keep_scores,
            context=context,
        )
        if bonus < 400.0:
            continue
        projected_score = _projected_score_after_discard(state, action, context)
        if not _strategic_discard_is_safe(state, remaining_score, projected_score, best_score):
            continue
        ranked.append((bonus, projected_score, action))

    if not ranked:
        return None
    _, _, action = max(ranked, key=lambda item: (item[0], item[1], len(item[2].card_indices)))
    return _annotated_action(action, reason=_joker_discard_reason(state, best_play, action, context))


def _strategic_discard_is_safe(
    state: GameState,
    remaining_score: int,
    projected_score: int,
    best_score: int,
) -> bool:
    if state.ante <= 2 and state.hands_remaining >= 3:
        return _estimated_hands_needed(remaining_score, max(1, projected_score, best_score)) <= state.hands_remaining
    if projected_score <= 0:
        return False
    pace_score = remaining_score / max(1, state.hands_remaining)
    if projected_score < pace_score * 0.65:
        return False
    return _estimated_hands_needed(remaining_score, projected_score) <= state.hands_remaining


def _best_discard_action(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._best_discard_action``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_best_discard_action", None) if strategy_module is not None else None
    if patched is not None and patched is not _best_discard_action:
        return patched(*args, **kwargs)
    return _default_best_discard_action(*args, **kwargs)


def _estimated_hands_needed(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._estimated_hands_needed``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_estimated_hands_needed", None) if strategy_module is not None else None
    if patched is not None and patched is not _estimated_hands_needed:
        return patched(*args, **kwargs)
    return _default_estimated_hands_needed(*args, **kwargs)


def _projected_score_after_discard(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._projected_score_after_discard``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_projected_score_after_discard", None) if strategy_module is not None else None
    if patched is not None and patched is not _projected_score_after_discard:
        return patched(*args, **kwargs)
    return _default_projected_score_after_discard(*args, **kwargs)


def _score_play_action(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._score_play_action``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_score_play_action", None) if strategy_module is not None else None
    if patched is not None and patched is not _score_play_action:
        return patched(*args, **kwargs)
    return _default_score_play_action(*args, **kwargs)
