"""Banner-specific discard veto policy."""

from __future__ import annotations

from dataclasses import replace
import sys

from balatro_ai.api.actions import Action
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.actions import _annotated_action
from balatro_ai.bots.basic_strategy.blind_state import _is_boss_blind
from balatro_ai.bots.basic_strategy.discard_policy import (
    _estimated_hands_needed as _default_estimated_hands_needed,
)
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.jokers import _active_joker_names
from balatro_ai.bots.basic_strategy.play_scoring import _score_play_action as _default_score_play_action
from balatro_ai.bots.basic_strategy.score_projection import (
    _projected_score_after_discard as _default_projected_score_after_discard,
)


BANNER_DISCARD_FUTURE_TAX_WEIGHT = 0.65


def _banner_vetoed_play_action(
    state: GameState,
    best_play: Action,
    discard: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext,
) -> Action | None:
    reason = _banner_preserve_play_reason(state, best_play, discard, current_score, remaining_score, context)
    if reason is None:
        return None
    return _annotated_action(best_play, reason=reason)


def _banner_preserve_play_reason(
    state: GameState,
    best_play: Action,
    best_discard: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext,
) -> str | None:
    if (
        "Banner" not in _active_joker_names(state)
        or state.hands_remaining <= 0
        or state.discards_remaining <= 0
        or current_score <= 0
        or remaining_score <= 0
        or not best_discard.card_indices
    ):
        return None

    reduced_state = replace(state, discards_remaining=max(0, state.discards_remaining - 1))
    reduced_current_score = _score_play_action(reduced_state, best_play, context)
    banner_tax = max(0, current_score - reduced_current_score)
    if banner_tax <= 0:
        return None

    projected_score = _projected_score_after_discard(state, best_discard, context)
    if projected_score >= remaining_score:
        return None

    current_hands_needed = _estimated_hands_needed(remaining_score, current_score)
    projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)

    if state.hands_remaining <= 1:
        return None

    raw_pace = remaining_score / max(1, state.hands_remaining)
    if current_score < raw_pace:
        if state.ante >= 8 and _is_boss_blind(state) and projected_score < current_score:
            immediate_gain = projected_score - current_score
            future_hands = max(0, min(state.hands_remaining - 1, projected_hands_needed - 1))
            future_banner_tax = banner_tax * future_hands
            required_gain = banner_tax * BANNER_DISCARD_FUTURE_TAX_WEIGHT
            return _banner_ev_reason(
                state,
                current_score=current_score,
                projected_score=projected_score,
                immediate_gain=immediate_gain,
                banner_tax=banner_tax,
                future_banner_tax=future_banner_tax,
                required_gain=required_gain,
                current_hands_needed=current_hands_needed,
                projected_hands_needed=projected_hands_needed,
            )
        return None

    if projected_hands_needed < current_hands_needed and projected_hands_needed <= state.hands_remaining:
        return None

    immediate_gain = projected_score - current_score
    future_hands = max(0, min(state.hands_remaining - 1, current_hands_needed - 1, projected_hands_needed - 1))
    future_banner_tax = banner_tax * future_hands
    required_gain = future_banner_tax * BANNER_DISCARD_FUTURE_TAX_WEIGHT
    if projected_score > current_score:
        if state.ante < 8:
            return None
        if immediate_gain >= required_gain:
            return None
        return _banner_ev_reason(
            state,
            current_score=current_score,
            projected_score=projected_score,
            immediate_gain=immediate_gain,
            banner_tax=banner_tax,
            future_banner_tax=future_banner_tax,
            required_gain=required_gain,
            current_hands_needed=current_hands_needed,
            projected_hands_needed=projected_hands_needed,
        )

    return _banner_ev_reason(
        state,
        current_score=current_score,
        projected_score=projected_score,
        immediate_gain=immediate_gain,
        banner_tax=banner_tax,
        future_banner_tax=future_banner_tax,
        required_gain=required_gain,
        current_hands_needed=current_hands_needed,
        projected_hands_needed=projected_hands_needed,
    )


def _banner_ev_reason(
    state: GameState,
    *,
    current_score: int,
    projected_score: int,
    immediate_gain: int,
    banner_tax: int,
    future_banner_tax: float,
    required_gain: float,
    current_hands_needed: int,
    projected_hands_needed: int,
) -> str:
    return (
        "preserve_banner_ev "
        f"current_score={current_score} projected_score={projected_score} "
        f"gain={immediate_gain} banner_tax={banner_tax} "
        f"future_tax={future_banner_tax:.1f} required_gain={required_gain:.1f} "
        f"hands_needed={current_hands_needed}->{projected_hands_needed} "
        f"hands_left={state.hands_remaining} discards_left={state.discards_remaining}"
    )


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
