"""Preferred-hand hunt policy for blind discard and redraw lines."""

from __future__ import annotations

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy.actions import _annotated_action
from balatro_ai.bots.basic_strategy.blind_reasons import (
    _preferred_hand_hunt_discard_reason,
    _preferred_hand_hunt_redraw_play_reason,
)
from balatro_ai.bots.basic_strategy.blind_state import _is_boss_blind
from balatro_ai.bots.basic_strategy.data import PREFERRED_HAND_HUNT_TYPES
from balatro_ai.bots.basic_strategy.discard_policy import (
    _best_discard_action,
    _discard_action_playstyle_bonus,
    _discard_can_reduce_hands_needed,
    _discard_detail_limit,
    _estimated_hands_needed,
    _prefilter_discard_actions,
    _score_is_on_pace,
)
from balatro_ai.bots.basic_strategy.discard_state import _discard_draw_count
from balatro_ai.bots.basic_strategy.draw_evaluation import (
    _hand_matches_preferred_family,
    _preferred_target_draw_evaluation,
    _straight_draw_evaluation,
    _straight_draw_reason_detail,
    _target_draw_reason_detail,
)
from balatro_ai.bots.basic_strategy.draw_math import _rank_matches_straight_value
from balatro_ai.bots.basic_strategy.hand_models import (
    _BlindContext,
    _PlayCandidate,
    _StraightDrawEvaluation,
)
from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type
from balatro_ai.bots.basic_strategy.hand_value import _card_keep_scores, _straight_draw_potential
from balatro_ai.bots.basic_strategy.play_scoring import (
    _action_index_sum,
    _evaluate_play_action,
    _play_candidates,
    _score_play_action,
)
from balatro_ai.bots.basic_strategy.score_projection import _projected_score_after_discard, _strong_draw_size
from balatro_ai.rules.hand_evaluator import HandType, RANK_VALUES


def _should_play_now(state: GameState, action: Action) -> bool:
    score = _score_play_action(state, action)
    remaining_score = max(0, state.required_score - state.current_score)
    if remaining_score == 0:
        return True
    if score >= remaining_score:
        return True
    if state.hands_remaining <= 1 or state.discards_remaining <= 0:
        return True

    estimated_hands_to_clear = _estimated_hands_needed(remaining_score, score)
    if estimated_hands_to_clear < state.hands_remaining:
        return True

    best_discard = _best_discard_action(state, current_best_score=score)
    if best_discard is None:
        return True

    if _discard_can_reduce_hands_needed(state, best_discard, score, remaining_score):
        return False

    return _score_is_on_pace(state, score, remaining_score)


def _preferred_hand_hunt_discard_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
    *,
    clear_line_only: bool = False,
) -> Action | None:
    context = context or _BlindContext()
    preferred = _preferred_hand_type(state)
    if (
        preferred not in PREFERRED_HAND_HUNT_TYPES
        or remaining_score <= 0
        or score >= remaining_score
        or state.hands_remaining <= 0
        or (state.hands_remaining <= 1 and not clear_line_only)
        or state.discards_remaining <= 0
    ):
        return None

    if not _preferred_hand_hunt_allowed(
        state,
        preferred,
        score,
        remaining_score,
        context,
        clear_line_only=clear_line_only,
    ):
        return None

    current_hand_type = _evaluate_play_action(state, best_play, context).hand_type
    if _hand_matches_preferred_family(current_hand_type, preferred):
        return None

    pace_score = remaining_score / max(1, state.hands_remaining)
    if score >= pace_score * _preferred_hunt_play_ceiling(state, preferred):
        return None

    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        return None

    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    protected = _preferred_hunt_protected_indices(state, preferred, keep_scores)
    detailed_actions = _prefilter_discard_actions(
        state,
        discard_actions,
        keep_scores,
        protected,
        preferred,
        limit=_preferred_hunt_discard_detail_limit(state, preferred, clear_line_only=clear_line_only),
    )

    current_needed = _estimated_hands_needed(remaining_score, score)
    ranked: list[tuple[tuple[float, float, int, float, int, int], Action, int, str]] = []
    for action in detailed_actions:
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        reason_detail = ""
        projected_score = _projected_score_after_discard(state, action, context)
        side_goal_bonus = _discard_action_playstyle_bonus(
            state,
            action,
            discarded_cards=discarded_cards,
            keep_scores=keep_scores,
            context=context,
        )
        if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
            draw_count = _discard_draw_count(state, action, len(kept_cards))
            straight_eval = _straight_draw_evaluation(
                state,
                kept_cards,
                discarded_cards=discarded_cards,
                draw_count=draw_count,
                context=context,
            )
            if straight_eval is None or not _straight_hunt_projection_is_safe(
                state,
                evaluation=straight_eval,
                current_score=score,
                projected_score=projected_score,
                remaining_score=remaining_score,
                pace_score=pace_score,
            ):
                continue
            draw_score = straight_eval.quality
            clears_after_hit = straight_eval.completion_score >= remaining_score
            if clear_line_only and not clears_after_hit:
                continue
            if clears_after_hit:
                draw_score += 900.0 + straight_eval.completion_probability * 500.0
            reason_detail = _straight_draw_reason_detail(straight_eval)
        else:
            draw_count = _discard_draw_count(state, action, len(kept_cards))
            target_eval = _preferred_target_draw_evaluation(
                state,
                preferred,
                kept_cards,
                discarded_cards=discarded_cards,
                draw_count=draw_count,
                context=context,
            )
            if target_eval is None:
                continue
            if target_eval.present_count < _preferred_hunt_min_draw_strength(state, preferred):
                continue
            target_score = target_eval.completion_score
            clears_after_hit = target_eval.completion_score >= remaining_score
            if clear_line_only and not clears_after_hit:
                continue
            if not _preferred_hunt_projection_is_safe(
                state,
                kept_cards=kept_cards,
                current_score=score,
                projected_score=max(projected_score, target_score),
                remaining_score=remaining_score,
                pace_score=pace_score,
            ):
                continue
            draw_score = target_eval.quality
            if clears_after_hit:
                draw_score += 600.0
            projected_score = max(projected_score, target_score)
            reason_detail = _target_draw_reason_detail(target_eval)

        projected_needed = _estimated_hands_needed(remaining_score, projected_score)
        ranked.append(
            (
                (
                    draw_score + min(420.0, max(0.0, side_goal_bonus)) * 0.35,
                    side_goal_bonus,
                    current_needed - projected_needed,
                    min(projected_score, remaining_score),
                    len(action.card_indices),
                    -_action_index_sum(action),
                ),
                action,
                projected_score,
                reason_detail,
            )
        )

    if not ranked:
        return None

    _, action, projected_score, reason_detail = max(ranked, key=lambda item: item[0])
    return _annotated_action(
        action,
        reason=_preferred_hand_hunt_discard_reason(
            state,
            best_play,
            action,
            projected_score,
            preferred,
            current_hand_type,
            context,
            detail=reason_detail,
        ),
    )


def _preferred_hand_hunt_allowed(
    state: GameState,
    preferred: HandType,
    score: int,
    remaining_score: int,
    context: _BlindContext,
    *,
    clear_line_only: bool,
) -> bool:
    if _preferred_hunt_blind_blocks_hunt(state, preferred, context):
        return False
    if state.ante < 4:
        return False
    if state.blind != "Big Blind" and not clear_line_only:
        return False
    if clear_line_only:
        return True

    current_needed = _estimated_hands_needed(remaining_score, score)
    under_pressure = current_needed >= max(1, state.hands_remaining)
    below_pace = not _score_is_on_pace(state, score, remaining_score)

    if state.blind == "Big Blind":
        return state.ante >= 4 or under_pressure or below_pace
    if _is_boss_blind(state):
        return under_pressure or (state.ante >= 4 and below_pace and state.hands_remaining >= 2)
    if state.blind == "Small Blind":
        return under_pressure or (state.ante >= 4 and below_pace and state.hands_remaining >= 3)
    return under_pressure or below_pace


def _preferred_hunt_blind_blocks_hunt(
    state: GameState,
    preferred: HandType,
    context: _BlindContext,
) -> bool:
    if state.blind == "The Water":
        return True
    if state.blind == "The Mouth" and context.played_hand_types:
        return not any(_hand_matches_preferred_family(hand_type, preferred) for hand_type in context.played_hand_types)
    if state.blind == "The Eye":
        return any(_hand_matches_preferred_family(hand_type, preferred) for hand_type in context.played_hand_types)
    return False


def _preferred_hunt_discard_detail_limit(
    state: GameState,
    preferred: HandType,
    *,
    clear_line_only: bool,
) -> int:
    limit = _discard_detail_limit(state)
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        limit = max(limit, 48 if clear_line_only else 32)
        if state.ante >= 7:
            limit = min(limit, 18 if clear_line_only else 14)
        if len(state.hand) >= 10:
            limit = min(limit, 10)
        if len(state.hand) >= 9 and len(state.jokers) >= 6:
            limit = min(limit, 12)
    return limit


def _preferred_hand_hunt_redraw_play_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    preferred = _preferred_hand_type(state)
    if (
        preferred not in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}
        or state.ante < 4
        or remaining_score <= 0
        or score >= remaining_score
        or state.hands_remaining <= 1
        or state.discards_remaining > 0
        or _preferred_hunt_blind_blocks_redraw_play(state, preferred, context)
    ):
        return None

    current_hand_type = _evaluate_play_action(state, best_play, context).hand_type
    if _hand_matches_preferred_family(current_hand_type, preferred):
        return None

    candidates = _play_candidates(state, context)
    if not candidates:
        return None

    candidate_pool: list[tuple[tuple[int, int, int], _PlayCandidate, tuple[Card, ...]]] = []
    for candidate in candidates:
        if _hand_matches_preferred_family(candidate.hand_type, preferred):
            continue
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in candidate.action.card_indices)
        potential = _straight_draw_potential(kept_cards)
        if potential < 3:
            continue
        candidate_pool.append(((potential, candidate.score, -_action_index_sum(candidate.action)), candidate, kept_cards))

    if not candidate_pool:
        return None

    redraw_limit = 14 if len(state.hand) >= 10 else 32
    candidate_pool = sorted(candidate_pool, key=lambda item: item[0], reverse=True)[:redraw_limit]

    ranked: list[tuple[tuple[float, float, int, float, int], _PlayCandidate, _StraightDrawEvaluation]] = []
    for _, candidate, kept_cards in candidate_pool:
        draw_count = _discard_draw_count(state, candidate.action, len(kept_cards))
        straight_eval = _straight_draw_evaluation(state, kept_cards, draw_count=draw_count, context=context)
        if straight_eval is None or straight_eval.present_count < 3:
            continue
        next_remaining = max(1, remaining_score - min(candidate.score, remaining_score - 1))
        if straight_eval.completion_score < next_remaining:
            continue
        if state.known_deck and not straight_eval.completes_from_known_draw:
            continue
        if straight_eval.missing_count >= 2 and straight_eval.completion_probability < 0.08:
            continue
        if straight_eval.missing_count == 1 and straight_eval.completion_probability < 0.12:
            continue
        ranked.append(
            (
                (
                    straight_eval.completion_probability,
                    straight_eval.quality,
                    draw_count,
                    min(candidate.score, remaining_score),
                    -_action_index_sum(candidate.action),
                ),
                candidate,
                straight_eval,
            )
        )

    if not ranked:
        return None

    _, candidate, straight_eval = max(ranked, key=lambda item: item[0])
    return _annotated_action(
        candidate.action,
        reason=_preferred_hand_hunt_redraw_play_reason(
            state,
            best_play,
            candidate,
            straight_eval,
            preferred,
            current_hand_type,
            context,
        ),
    )


def _preferred_hunt_blind_blocks_redraw_play(
    state: GameState,
    preferred: HandType,
    context: _BlindContext,
) -> bool:
    if state.blind in {"The Needle", "The Mouth"}:
        return True
    return _preferred_hunt_blind_blocks_hunt(state, preferred, context)


def _preferred_hunt_play_ceiling(state: GameState, preferred: HandType) -> float:
    ceiling = 1.45
    if preferred in {HandType.STRAIGHT, HandType.FLUSH, HandType.FULL_HOUSE}:
        ceiling += 0.10
    if state.ante >= 6:
        ceiling += 0.05
    if state.discards_remaining <= 1:
        ceiling -= 0.15
    return max(1.20, ceiling)


def _preferred_hunt_min_draw_strength(state: GameState, preferred: HandType) -> int:
    if preferred in {HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND, HandType.FLUSH_FIVE}:
        return 3
    if preferred in {HandType.STRAIGHT, HandType.FLUSH, HandType.FULL_HOUSE, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE}:
        return 4 if state.discards_remaining <= 1 else 3
    return 2


def _preferred_hunt_protected_indices(
    state: GameState,
    preferred: HandType,
    keep_scores: tuple[float, ...],
) -> set[int]:
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        straight_core = _straight_hunt_core_indices(state)
        if straight_core:
            return straight_core

    keep_count = 4 if preferred in {HandType.STRAIGHT, HandType.FLUSH, HandType.STRAIGHT_FLUSH} else 3
    return {
        index
        for index, _ in sorted(
            enumerate(keep_scores),
            key=lambda item: (item[1], RANK_VALUES[state.hand[item[0]].rank]),
            reverse=True,
        )[: min(keep_count, len(state.hand))]
    }


def _straight_hunt_core_indices(state: GameState) -> set[int]:
    best: tuple[int, int, int, int, tuple[int, ...]] | None = None
    for start in range(1, 11):
        window_values = tuple(range(start, start + 5))
        selected: list[int] = []
        used: set[int] = set()
        for value in window_values:
            candidates = [
                index
                for index, card in enumerate(state.hand)
                if index not in used and _rank_matches_straight_value(card, value)
            ]
            if not candidates:
                continue
            index = max(candidates, key=lambda candidate: RANK_VALUES[state.hand[candidate].rank])
            selected.append(index)
            used.add(index)
        present_count = len(selected)
        if present_count < 3:
            continue
        missing_count = 5 - present_count
        open_ended = int(
            missing_count == 1
            and any(not any(_rank_matches_straight_value(card, edge) for card in state.hand) for edge in (window_values[0], window_values[-1]))
        )
        key = (present_count, -missing_count, open_ended, max(window_values), tuple(sorted(selected)))
        if best is None or key > best:
            best = key
    if best is None:
        return set()
    return set(best[-1])


def _preferred_hunt_projection_is_safe(
    state: GameState,
    *,
    kept_cards: tuple[Card, ...],
    current_score: int,
    projected_score: int,
    remaining_score: int,
    pace_score: float,
) -> bool:
    if projected_score >= remaining_score:
        return True
    if state.known_deck:
        return projected_score >= max(current_score * 0.95, pace_score * 0.75)
    if state.discards_remaining <= 1 and projected_score < max(current_score, pace_score):
        return False
    if projected_score >= max(current_score * 1.08, pace_score * 0.90):
        return True
    return _strong_draw_size(kept_cards) >= 4 and projected_score >= current_score * 0.85


def _straight_hunt_projection_is_safe(
    state: GameState,
    *,
    evaluation: _StraightDrawEvaluation,
    current_score: int,
    projected_score: int,
    remaining_score: int,
    pace_score: float,
) -> bool:
    if evaluation.present_count < 3:
        return False
    if evaluation.missing_count <= 0:
        return True
    if state.known_deck and not evaluation.completes_from_known_draw:
        return False
    if evaluation.missing_count >= 2 and evaluation.completion_probability < 0.08:
        return False
    if evaluation.completion_score >= remaining_score:
        if evaluation.present_count >= 4 and evaluation.completion_probability >= 0.10:
            return True
        if evaluation.present_count >= 3 and evaluation.completion_probability >= 0.08:
            return True
    if evaluation.present_count >= 4 and evaluation.completion_probability >= 0.20:
        return projected_score >= current_score * 0.80 or evaluation.completion_score >= pace_score
    if evaluation.completion_probability >= 0.14 and evaluation.completion_score >= max(pace_score, current_score * 1.10):
        return True
    return evaluation.completion_score >= remaining_score and projected_score >= current_score * 0.85
