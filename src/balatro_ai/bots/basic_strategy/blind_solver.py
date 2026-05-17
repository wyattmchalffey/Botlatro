"""Shared blind solver for play, discard, and shop-pressure planning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import sys

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy.cache import _state_scoped_cache
from balatro_ai.bots.basic_strategy.cards import _normalize_suit
from balatro_ai.bots.basic_strategy.data import PREFERRED_HAND_HUNT_TYPES
from balatro_ai.bots.basic_strategy.discard_policy import (
    _best_discard_action as _default_best_discard_action,
    _discard_action_playstyle_bonus,
    _discard_detail_limit,
    _estimated_hands_needed as _default_estimated_hands_needed,
    _prefilter_discard_actions,
    _score_is_on_pace,
)
from balatro_ai.bots.basic_strategy.discard_state import _discard_draw_count
from balatro_ai.bots.basic_strategy.draw_evaluation import (
    _preferred_target_draw_evaluation,
    _straight_draw_evaluation,
    _straight_draw_reason_detail,
    _target_draw_reason_detail,
)
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext, _PlayCandidate
from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type
from balatro_ai.bots.basic_strategy.hand_value import _card_keep_scores, _straight_draw_potential
from balatro_ai.bots.basic_strategy.play_scoring import (
    _action_index_sum,
    _best_play_action as _default_best_play_action,
    _play_candidate,
)
from balatro_ai.rules.hand_evaluator import HandType, RANK_VALUES, _prepare_joker_evaluation_context


CLEAR_LINE_HAND_TYPES = (
    HandType.STRAIGHT,
    HandType.FLUSH,
    HandType.FULL_HOUSE,
    HandType.THREE_OF_A_KIND,
    HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND,
)


@dataclass(frozen=True, slots=True)
class _ClearLine:
    action: Action
    hand_type: HandType
    projected_score: int
    completion_score: int
    completion_probability: float
    quality: float
    detail: str


@dataclass(frozen=True, slots=True)
class _BlindSolution:
    best_play: Action | None
    best_play_candidate: _PlayCandidate | None
    best_play_score: int
    remaining_score: int
    hands_needed: int
    on_pace: bool
    hands_remaining: int

    @property
    def can_clear_now(self) -> bool:
        return self.remaining_score <= 0 or self.best_play_score >= self.remaining_score

    @property
    def under_pressure(self) -> bool:
        return self.remaining_score > 0 and (not self.on_pace or self.hands_needed >= max(1, self.hands_remaining))


def _solve_blind(
    state: GameState,
    context: _BlindContext | None = None,
    *,
    best_play: Action | None = None,
) -> _BlindSolution:
    """Return the shared tactical view of the current blind state."""

    context = context or _BlindContext()
    cache = _state_scoped_cache("blind_solution", state)
    cache_key = (
        context.played_hand_types,
        context.discards_taken,
        tuple(best_play.card_indices) if best_play is not None else None,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]  # type: ignore[return-value]
    solution = _solve_blind_uncached(state, context, best_play=best_play)
    if cache is not None:
        cache[cache_key] = solution
    return solution


def _solve_blind_uncached(
    state: GameState,
    context: _BlindContext,
    *,
    best_play: Action | None,
) -> _BlindSolution:
    selected_play = best_play or _best_play_action(state, context)
    best_play_candidate: _PlayCandidate | None = None
    best_play_score = 0
    if selected_play is not None:
        joker_context = _prepare_joker_evaluation_context(state.jokers)
        best_play_candidate = _play_candidate(state, selected_play, context, joker_context=joker_context)
        best_play_score = best_play_candidate.score

    remaining_score = max(0, state.required_score - state.current_score)
    hands_needed = _estimated_hands_needed(remaining_score, best_play_score)
    on_pace = _score_is_on_pace(state, best_play_score, remaining_score)
    return _BlindSolution(
        best_play=selected_play,
        best_play_candidate=best_play_candidate,
        best_play_score=best_play_score,
        remaining_score=remaining_score,
        hands_needed=hands_needed,
        on_pace=on_pace,
        hands_remaining=state.hands_remaining,
    )


def _best_discard_for_solution(
    state: GameState,
    solution: _BlindSolution,
    context: _BlindContext,
) -> Action | None:
    cache = _state_scoped_cache("blind_solution_best_discard", state)
    cache_key = (
        solution.best_play_score,
        solution.remaining_score,
        context.played_hand_types,
        context.discards_taken,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]  # type: ignore[return-value]
    action = _best_discard_action(state, current_best_score=solution.best_play_score, context=context)
    if cache is not None:
        cache[cache_key] = action
    return action


def _best_clear_line(
    state: GameState,
    current_score: int,
    remaining_score: int,
    context: _BlindContext,
) -> _ClearLine | None:
    cache = _state_scoped_cache("blind_solution_clear_line", state)
    cache_key = (
        current_score,
        remaining_score,
        context.played_hand_types,
        context.discards_taken,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]  # type: ignore[return-value]

    if not _clear_line_hunt_allowed(state, current_score, remaining_score):
        if cache is not None:
            cache[cache_key] = None
        return None

    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        if cache is not None:
            cache[cache_key] = None
        return None

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    protected = _clear_line_protected_indices(state, keep_scores)
    detailed_actions = _prefilter_discard_actions(
        state,
        discard_actions,
        keep_scores,
        protected,
        preferred,
        limit=max(_discard_detail_limit(state), 40),
    )

    threshold = _clear_line_probability_threshold(state, current_score, remaining_score)
    ranked: list[tuple[tuple[float, float, float, int, int, int], _ClearLine]] = []
    for action in detailed_actions:
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        draw_count = _discard_draw_count(state, action, len(kept_cards))
        if draw_count <= 0:
            continue
        side_goal_bonus = _discard_action_playstyle_bonus(
            state,
            action,
            discarded_cards=discarded_cards,
            keep_scores=keep_scores,
            context=context,
        )
        for line in _clear_lines_for_action(
            state,
            action,
            kept_cards=kept_cards,
            discarded_cards=discarded_cards,
            draw_count=draw_count,
            context=context,
        ):
            if line.completion_score < remaining_score:
                continue
            if not state.known_deck and line.completion_probability < threshold:
                continue
            ranked.append(
                (
                    (
                        line.completion_probability,
                        line.quality,
                        min(420.0, max(0.0, side_goal_bonus)) * 0.35,
                        min(line.projected_score, remaining_score),
                        len(action.card_indices),
                        -_action_index_sum(action),
                    ),
                    line,
                )
            )

    if not ranked:
        if cache is not None:
            cache[cache_key] = None
        return None
    line = max(ranked, key=lambda item: item[0])[1]
    if cache is not None:
        cache[cache_key] = line
    return line


def _clear_lines_for_action(
    state: GameState,
    action: Action,
    *,
    kept_cards: tuple[Card, ...],
    discarded_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext,
) -> tuple[_ClearLine, ...]:
    lines: list[_ClearLine] = []
    for hand_type in _clear_line_target_hand_types(state, kept_cards):
        if hand_type == HandType.STRAIGHT:
            evaluation = _straight_draw_evaluation(
                state,
                kept_cards,
                discarded_cards=discarded_cards,
                draw_count=draw_count,
                context=context,
            )
            if evaluation is None or evaluation.present_count < _clear_line_min_present(state, hand_type):
                continue
            lines.append(
                _ClearLine(
                    action=action,
                    hand_type=hand_type,
                    projected_score=evaluation.completion_score,
                    completion_score=evaluation.completion_score,
                    completion_probability=evaluation.completion_probability,
                    quality=evaluation.quality,
                    detail=_straight_draw_reason_detail(evaluation),
                )
            )
            continue

        evaluation = _preferred_target_draw_evaluation(
            state,
            hand_type,
            kept_cards,
            discarded_cards=discarded_cards,
            draw_count=draw_count,
            context=context,
        )
        if evaluation is None or evaluation.present_count < _clear_line_min_present(state, hand_type):
            continue
        lines.append(
            _ClearLine(
                action=action,
                hand_type=hand_type,
                projected_score=evaluation.completion_score,
                completion_score=evaluation.completion_score,
                completion_probability=evaluation.completion_probability,
                quality=evaluation.quality,
                detail=_target_draw_reason_detail(evaluation),
            )
        )
    return tuple(lines)


def _clear_line_target_hand_types(state: GameState, kept_cards: tuple[Card, ...]) -> tuple[HandType, ...]:
    preferred = _preferred_hand_type(state)
    ordered: list[HandType] = []
    if preferred in PREFERRED_HAND_HUNT_TYPES and preferred in CLEAR_LINE_HAND_TYPES:
        ordered.append(preferred)
    ordered.extend(hand_type for hand_type in CLEAR_LINE_HAND_TYPES if hand_type not in ordered)
    return tuple(hand_type for hand_type in ordered if _kept_cards_can_support_clear_line(state, kept_cards, hand_type))


def _kept_cards_can_support_clear_line(state: GameState, kept_cards: tuple[Card, ...], hand_type: HandType) -> bool:
    if not kept_cards:
        return False
    min_present = _clear_line_min_present(state, hand_type)
    if hand_type == HandType.STRAIGHT:
        return _straight_draw_potential(kept_cards) >= min_present
    if hand_type == HandType.FLUSH:
        suit_counts = Counter(_normalize_suit(card.suit) for card in kept_cards)
        return max(suit_counts.values(), default=0) >= min_present

    rank_counts = Counter(card.rank for card in kept_cards)
    if hand_type == HandType.FULL_HOUSE:
        clipped = sorted((min(3, count) for count in rank_counts.values()), reverse=True)
        return len(clipped) >= 2 and clipped[0] + min(2, clipped[1]) >= min_present
    if hand_type == HandType.THREE_OF_A_KIND:
        return max(rank_counts.values(), default=0) >= min_present
    if hand_type in {HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        return max(rank_counts.values(), default=0) >= min_present
    return False


def _clear_line_hunt_allowed(state: GameState, current_score: int, remaining_score: int) -> bool:
    if state.ante <= 1 or remaining_score <= 0 or current_score >= remaining_score:
        return False
    if state.discards_remaining <= 0 or state.hands_remaining <= 1:
        return False
    if state.blind == "The Water":
        return False
    current_needed = _estimated_hands_needed(remaining_score, current_score)
    return current_needed >= max(1, state.hands_remaining) or not _score_is_on_pace(state, current_score, remaining_score)


def _clear_line_min_present(state: GameState, hand_type: HandType) -> int:
    if hand_type in {HandType.STRAIGHT, HandType.FLUSH, HandType.FULL_HOUSE}:
        return 4 if state.discards_remaining <= 1 else 3
    if hand_type == HandType.THREE_OF_A_KIND:
        return 2
    return 3


def _clear_line_probability_threshold(state: GameState, current_score: int, remaining_score: int) -> float:
    current_needed = _estimated_hands_needed(remaining_score, current_score)
    threshold = 0.18
    if current_needed >= max(1, state.hands_remaining):
        threshold -= 0.08
    if state.hands_remaining <= 1:
        threshold -= 0.04
    if state.discards_remaining >= 3:
        threshold -= 0.03
    if state.ante >= 6:
        threshold -= 0.03
    return max(0.04, threshold)


def _clear_line_protected_indices(state: GameState, keep_scores: tuple[float, ...]) -> set[int]:
    if not state.hand:
        return set()
    protected_count = max(1, min(3, len(state.hand) // 3))
    return {
        index
        for index, _ in sorted(
            enumerate(keep_scores),
            key=lambda item: (item[1], RANK_VALUES[state.hand[item[0]].rank]),
            reverse=True,
        )[:protected_count]
    }


def _blind_capacity_from_score(
    score: float,
    effective_hands: float,
    *,
    per_hand_discount: float = 0.85,
    capacity_safety_factor: float = 1.0,
) -> float:
    raw_capacity = max(1.0, float(score) * float(effective_hands) * per_hand_discount)
    return max(1.0, raw_capacity * capacity_safety_factor)


def _best_play_action(*args, **kwargs):
    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_best_play_action", None) if strategy_module is not None else None
    if patched is not None and patched is not _best_play_action:
        return patched(*args, **kwargs)
    return _default_best_play_action(*args, **kwargs)


def _best_discard_action(*args, **kwargs):
    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_best_discard_action", None) if strategy_module is not None else None
    if patched is not None and patched is not _best_discard_action:
        return patched(*args, **kwargs)
    return _default_best_discard_action(*args, **kwargs)


def _estimated_hands_needed(*args, **kwargs):
    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_estimated_hands_needed", None) if strategy_module is not None else None
    if patched is not None and patched is not _estimated_hands_needed:
        return patched(*args, **kwargs)
    return _default_estimated_hands_needed(*args, **kwargs)

