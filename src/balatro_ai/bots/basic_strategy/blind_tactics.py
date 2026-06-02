"""Top-level blind play/discard routing."""

from __future__ import annotations

import sys

from balatro_ai.api.actions import Action
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.actions import _annotated_action
from balatro_ai.bots.basic_strategy.ante_one_hunt import _first_blind_one_hand_hunt_action
from balatro_ai.bots.basic_strategy.banner_policy import _banner_vetoed_play_action
from balatro_ai.bots.basic_strategy.blind_reasons import (
    _discard_reason,
    _last_hand_hunt_discard_reason,
    _panic_discard_reason,
    _play_reason,
    _safety_discard_reason,
)
from balatro_ai.bots.basic_strategy.blind_setup import (
    _mystic_summit_setup_discard_action,
    _opening_joker_setup_play_action,
    _strategic_joker_discard_action,
)
from balatro_ai.bots.basic_strategy.blind_solver import (
    _BlindSolution,
    _ClearLine,
    _best_clear_line,
    _best_discard_for_solution,
    _dig_via_play_action,
    _solve_blind,
)
from balatro_ai.bots.basic_strategy.discard_policy import (
    _best_discard_action as _default_best_discard_action,
    _discard_can_reduce_hands_needed,
    _estimated_hands_needed as _default_estimated_hands_needed,
    _score_is_on_pace,
    _should_chase_discard,
    _should_last_hand_hunt_discard,
    _should_panic_discard,
    _should_safety_discard,
)
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.play_scoring import _score_play_action as _default_score_play_action
from balatro_ai.bots.basic_strategy.preferred_hunt import (
    _preferred_hand_hunt_discard_action,
    _preferred_hand_hunt_redraw_play_action,
)
from balatro_ai.bots.basic_strategy.winning_economy import _winning_economy_hunt_discard_action


def _tactical_blind_action(
    state: GameState,
    best_play: Action,
    context: _BlindContext | None = None,
    solution: _BlindSolution | None = None,
) -> Action:
    context = context or _BlindContext()
    solution = solution or _solve_blind(state, context, best_play=best_play)
    score = solution.best_play_score
    remaining_score = solution.remaining_score
    if score >= remaining_score > 0:
        economy_hunt = _winning_economy_hunt_discard_action(state, best_play, score, remaining_score, context)
        if economy_hunt is not None:
            return economy_hunt

    first_blind_discard = _first_blind_one_hand_hunt_action(state, best_play, score, remaining_score, context)
    if first_blind_discard is not None:
        return first_blind_discard

    opening_setup = _opening_joker_setup_play_action(state, best_play, score, remaining_score, context)
    if opening_setup is not None:
        return opening_setup

    strategic_discard = _strategic_joker_discard_action(state, best_play, score, remaining_score, context)
    if strategic_discard is not None:
        return strategic_discard

    mystic_setup_discard = _mystic_summit_setup_discard_action(state, best_play, score, remaining_score, context)
    if mystic_setup_discard is not None:
        return mystic_setup_discard

    clear_line_hunt = _preferred_hand_hunt_discard_action(
        state,
        best_play,
        score,
        remaining_score,
        context,
        clear_line_only=True,
    )
    if clear_line_hunt is not None:
        banner_play = _banner_vetoed_play_action(state, best_play, clear_line_hunt, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return clear_line_hunt

    clear_line_redraw = _preferred_hand_hunt_redraw_play_action(state, best_play, score, remaining_score, context)
    if clear_line_redraw is not None:
        return clear_line_redraw

    clear_line = _best_clear_line(state, score, remaining_score, context)
    if clear_line is not None:
        banner_play = _banner_vetoed_play_action(state, best_play, clear_line.action, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(clear_line.action, reason=_clear_line_reason(state, clear_line, score))

    # When the greedy line can't clear (a guaranteed loss), dig for a clearing hand by
    # playing throwaways to draw — using spare HANDS as discards. Env-gated; only fires
    # when greedy can't win, so it can't regress a winnable blind.
    dig_play = _dig_via_play_action(state, score, remaining_score, context)
    if dig_play is not None:
        return _annotated_action(
            dig_play,
            reason=(f"dig_via_play hands_left={state.hands_remaining} "
                    f"remaining={remaining_score} best_play_score={score}"),
        )

    if (
        remaining_score == 0
        or score >= remaining_score
        or state.discards_remaining <= 0
        or _estimated_hands_needed(remaining_score, score) < state.hands_remaining
    ):
        return _annotated_action(best_play, reason=_play_reason(state, best_play, context))

    best_discard = _best_discard_for_solution(state, solution, context)
    if (
        best_discard is not None
        and state.hands_remaining <= 1
        and _should_last_hand_hunt_discard(state, best_discard, score, remaining_score, context)
    ):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_last_hand_hunt_discard_reason(state, best_play, best_discard, context))

    preferred_hunt = _preferred_hand_hunt_discard_action(state, best_play, score, remaining_score, context)
    if preferred_hunt is not None:
        banner_play = _banner_vetoed_play_action(state, best_play, preferred_hunt, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return preferred_hunt

    if best_discard is not None and _should_panic_discard(state, best_discard, score, remaining_score, context):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_panic_discard_reason(state, best_play, best_discard, context))

    if best_discard is not None and _discard_can_reduce_hands_needed(state, best_discard, score, remaining_score, context):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_discard_reason(state, best_play, best_discard, context))

    if best_discard is not None and _should_safety_discard(state, best_discard, score, remaining_score, context):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_safety_discard_reason(state, best_play, best_discard, context))

    if _score_is_on_pace(state, score, remaining_score):
        return _annotated_action(best_play, reason=_play_reason(state, best_play, context))

    if best_discard is not None and _should_chase_discard(state, best_discard, score, remaining_score, context):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_discard_reason(state, best_play, best_discard, context))
    return _annotated_action(best_play, reason=_play_reason(state, best_play, context))


def _clear_line_reason(state: GameState, line: _ClearLine, current_score: int) -> str:
    remaining_score = max(0, state.required_score - state.current_score)
    return (
        f"blind_solver_clear_line target={line.hand_type.value} current_score={current_score} "
        f"projected_score={line.projected_score} remaining={remaining_score} "
        f"p={line.completion_probability:.2f} completion={line.completion_score} {line.detail}"
    )


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


def _score_play_action(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._score_play_action``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_score_play_action", None) if strategy_module is not None else None
    if patched is not None and patched is not _score_play_action:
        return patched(*args, **kwargs)
    return _default_score_play_action(*args, **kwargs)
