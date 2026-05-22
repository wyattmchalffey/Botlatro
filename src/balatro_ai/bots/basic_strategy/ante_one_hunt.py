"""Ante-one small-blind discard hunt policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import sys

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy.actions import _annotated_action
from balatro_ai.bots.basic_strategy.blind_reasons import (
    _ante_one_upgrade_discard_reason,
    _first_blind_discard_reason,
)
from balatro_ai.bots.basic_strategy.discard_state import _discard_draw_count
from balatro_ai.bots.basic_strategy.draw_evaluation import (
    _preferred_target_draw_evaluation,
    _straight_draw_evaluation,
    _straight_draw_reason_detail,
    _target_draw_reason_detail,
)
from balatro_ai.bots.basic_strategy.discard_policy import _best_discard_action as _default_best_discard_action
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.hand_value import _straight_draw_potential
from balatro_ai.bots.basic_strategy.cards import _normalize_suit
from balatro_ai.bots.basic_strategy.play_scoring import (
    _action_index_sum,
    _evaluate_play_action as _default_evaluate_play_action,
)
from balatro_ai.bots.basic_strategy.score_projection import (
    _dominant_suit_from_cards,
    _projected_score_after_discard as _default_projected_score_after_discard,
    _strong_draw_size,
)
from balatro_ai.rules.hand_evaluator import HandType, RANK_VALUES, STRAIGHT_VALUES


ANTE_ONE_UPGRADE_NEAR_CLEAR_RATIO = 0.80
ANTE_ONE_UPGRADE_TARGET_RATIO = 0.96
ANTE_ONE_UPGRADE_MIN_GAIN = 16
ANTE_ONE_PLAY_NEAR_CLEAR_RATIO = 0.86
ANTE_ONE_PLAY_MIN_DISCARD_GAIN = 32
ANTE_ONE_HUNT_MIN_CLEAR_PROBABILITY = 0.10
ANTE_ONE_HUNT_TARGETS = (
    HandType.FLUSH,
    HandType.STRAIGHT,
    HandType.FULL_HOUSE,
    HandType.THREE_OF_A_KIND,
    HandType.FOUR_OF_A_KIND,
)


@dataclass(frozen=True, slots=True)
class _AnteOneHuntLine:
    action: Action
    hand_type: HandType
    projected_score: int
    completion_score: int
    completion_probability: float
    quality: float
    detail: str


def _first_blind_one_hand_hunt_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    if (
        state.ante != 1
        or state.blind != "Small Blind"
        or state.current_score != 0
        or state.hands_remaining < 3
        or state.discards_remaining <= 0
        or state.jokers
        or remaining_score <= 0
        or score >= remaining_score
    ):
        return None

    hunt_line = _ante_one_best_one_hand_hunt_line(state, score, remaining_score, context)
    if hunt_line is not None:
        if _ante_one_should_play_near_clear(state, best_play, hunt_line, score, remaining_score, context):
            return _annotated_action(
                best_play,
                reason=(
                    "first_blind_near_clear_play "
                    f"current_score={score} projected_score={hunt_line.projected_score} "
                    f"hunt_p={hunt_line.completion_probability:.2f} remaining={remaining_score}"
                ),
            )
        return _annotated_action(
            hunt_line.action,
            reason=(
                f"first_blind_one_hand_hunt target={hunt_line.hand_type.value} "
                f"current_score={score} projected_score={hunt_line.projected_score} "
                f"completion={hunt_line.completion_score} p={hunt_line.completion_probability:.2f} "
                f"discarding={len(hunt_line.action.card_indices)} {hunt_line.detail}"
            ),
        )

    best_discard = _ante_one_best_unbiased_projected_discard(state, score, context)
    if best_discard is None:
        return None
    upgrade_discard = _ante_one_near_clear_upgrade_discard_action(
        state,
        best_play,
        best_discard,
        score,
        remaining_score,
        context,
    )
    if upgrade_discard is not None:
        return upgrade_discard
    return _annotated_action(best_discard, reason=_first_blind_discard_reason(state, best_play, best_discard, context))


def _ante_one_best_one_hand_hunt_line(
    state: GameState,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> _AnteOneHuntLine | None:
    context = context or _BlindContext()
    if state.discards_remaining <= 0 or remaining_score <= 0:
        return None

    probability_state = replace(state, known_deck=())
    ranked: list[tuple[tuple[float, float, float, int, int], _AnteOneHuntLine]] = []
    for action in _ante_one_discard_actions(state):
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        if not kept_cards:
            continue
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        draw_count = _discard_draw_count(state, action, len(kept_cards))
        if draw_count <= 0:
            continue
        projected_score = _projected_score_after_discard(state, action, context)
        for line in _ante_one_hunt_lines_for_action(
            probability_state,
            action,
            kept_cards=kept_cards,
            discarded_cards=discarded_cards,
            draw_count=draw_count,
            projected_score=projected_score,
            context=context,
        ):
            if line.completion_score < remaining_score:
                continue
            if line.completion_probability < _ante_one_hunt_probability_floor(state, score, remaining_score):
                continue
            ranked.append(
                (
                    (
                        line.completion_probability,
                        min(float(line.completion_score), remaining_score * 1.4),
                        line.quality,
                        len(action.card_indices),
                        -_action_index_sum(action),
                    ),
                    line,
                )
            )

    if not ranked:
        return None
    return max(ranked, key=lambda item: item[0])[1]


def _ante_one_hunt_lines_for_action(
    state: GameState,
    action: Action,
    *,
    kept_cards: tuple[Card, ...],
    discarded_cards: tuple[Card, ...],
    draw_count: int,
    projected_score: int,
    context: _BlindContext,
) -> tuple[_AnteOneHuntLine, ...]:
    lines: list[_AnteOneHuntLine] = []
    for hand_type in ANTE_ONE_HUNT_TARGETS:
        if hand_type == HandType.STRAIGHT:
            straight = _straight_draw_evaluation(
                state,
                kept_cards,
                discarded_cards=discarded_cards,
                draw_count=draw_count,
                context=context,
            )
            if straight is None or straight.present_count < 3:
                continue
            lines.append(
                _AnteOneHuntLine(
                    action=action,
                    hand_type=HandType.STRAIGHT,
                    projected_score=projected_score,
                    completion_score=straight.completion_score,
                    completion_probability=straight.completion_probability,
                    quality=straight.quality,
                    detail=_straight_draw_reason_detail(straight),
                )
            )
            continue

        target = _preferred_target_draw_evaluation(
            state,
            hand_type,
            kept_cards,
            discarded_cards=discarded_cards,
            draw_count=draw_count,
            context=context,
        )
        if target is None or target.present_count < _ante_one_min_present_for_target(hand_type):
            continue
        lines.append(
            _AnteOneHuntLine(
                action=action,
                hand_type=hand_type,
                projected_score=projected_score,
                completion_score=target.completion_score,
                completion_probability=target.completion_probability,
                quality=target.quality,
                detail=_target_draw_reason_detail(target),
            )
        )
    return tuple(lines)


def _ante_one_min_present_for_target(hand_type: HandType) -> int:
    if hand_type in {HandType.FLUSH, HandType.STRAIGHT, HandType.FULL_HOUSE}:
        return 3
    if hand_type == HandType.THREE_OF_A_KIND:
        return 2
    return 3


def _ante_one_hunt_probability_floor(state: GameState, score: int, remaining_score: int) -> float:
    floor = ANTE_ONE_HUNT_MIN_CLEAR_PROBABILITY
    if state.discards_remaining >= 3:
        floor -= 0.03
    if score >= remaining_score * 0.75:
        floor += 0.04
    return max(0.04, floor)


def _ante_one_best_unbiased_projected_discard(
    state: GameState,
    score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    ranked: list[tuple[tuple[int, int, int], Action]] = []
    for action in _ante_one_discard_actions(state):
        projected_score = _projected_score_after_discard(state, action, context)
        if projected_score < score:
            continue
        ranked.append(
            (
                (
                    projected_score,
                    len(action.card_indices),
                    -_action_index_sum(action),
                ),
                action,
            )
        )
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[0])[1]


def _ante_one_discard_actions(state: GameState) -> tuple[Action, ...]:
    return tuple(
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    )


def _ante_one_should_play_near_clear(
    state: GameState,
    best_play: Action,
    hunt_line: _AnteOneHuntLine,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    if (
        state.ante != 1
        or state.blind != "Small Blind"
        or state.current_score != 0
        or state.hands_remaining < 3
        or state.jokers
        or remaining_score <= 0
        or score < remaining_score * ANTE_ONE_PLAY_NEAR_CLEAR_RATIO
    ):
        return False

    if hunt_line.completion_probability >= _ante_one_hunt_probability_floor(state, score, remaining_score):
        return False
    if hunt_line.projected_score >= remaining_score:
        return False
    min_gain = max(ANTE_ONE_PLAY_MIN_DISCARD_GAIN, int(remaining_score * 0.10))
    return hunt_line.projected_score < score + min_gain


def _ante_one_near_clear_upgrade_discard_action(
    state: GameState,
    best_play: Action,
    fallback_discard: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    if (
        state.ante != 1
        or state.blind != "Small Blind"
        or state.current_score != 0
        or context.discards_taken != 1
        or state.discards_remaining <= 0
        or state.jokers
        or remaining_score <= 0
        or score >= remaining_score
        or len(best_play.card_indices) != 5
        or score < remaining_score * ANTE_ONE_UPGRADE_NEAR_CLEAR_RATIO
    ):
        return None

    fallback_projected_score = _projected_score_after_discard(state, fallback_discard, context)
    if fallback_projected_score >= score * 0.75:
        return None

    evaluation = _evaluate_play_action(state, best_play, context)
    keep_candidates = _ante_one_upgrade_keep_candidates(state, best_play, evaluation.hand_type)
    if not keep_candidates:
        return None

    ranked: list[tuple[tuple[float, float, int, int], Action, int]] = []
    for keep_indices in keep_candidates:
        discard_indices = _ante_one_upgrade_discard_indices(state, keep_indices)
        if not discard_indices:
            continue
        discard = _matching_discard_action(state, discard_indices)
        if discard is None:
            continue

        projected_score = _projected_score_after_discard(state, discard, context)
        if not _ante_one_upgrade_projection_is_good(
            score,
            projected_score,
            remaining_score,
        ):
            continue
        core_score = _ante_one_upgrade_core_score(state, keep_indices)
        ranked.append(
            (
                (float(projected_score), core_score, -len(discard.card_indices), -_action_index_sum(discard)),
                discard,
                projected_score,
            )
        )

    if not ranked:
        return None

    _, discard, projected_score = max(ranked, key=lambda item: item[0])
    return _annotated_action(
        discard,
        reason=_ante_one_upgrade_discard_reason(
            state,
            best_play,
            discard,
            evaluation.hand_type,
            projected_score,
            context,
        ),
    )


def _ante_one_upgrade_keep_candidates(
    state: GameState,
    best_play: Action,
    hand_type: HandType,
) -> tuple[tuple[int, ...], ...]:
    selected_indices = tuple(best_play.card_indices)
    candidates: list[tuple[int, ...]] = []

    if hand_type in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        flush_core = _ante_one_flush_upgrade_core_indices(state, selected_indices)
        if flush_core:
            candidates.append(flush_core)

    if hand_type in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        straight_core = _ante_one_straight_upgrade_core_indices(state, selected_indices)
        if straight_core:
            candidates.append(straight_core)

    return _unique_index_tuples(candidates)


def _ante_one_flush_upgrade_core_indices(
    state: GameState,
    selected_indices: tuple[int, ...],
) -> tuple[int, ...] | None:
    selected_cards = tuple(state.hand[index] for index in selected_indices)
    suit = _dominant_suit_from_cards(selected_cards)
    if suit is None:
        return None

    suited_indices = tuple(
        index
        for index in selected_indices
        if _normalize_suit(state.hand[index].suit) == _normalize_suit(suit)
    )
    if len(suited_indices) < 5:
        return None

    keep = sorted(suited_indices, key=lambda index: _ante_one_card_upgrade_key(state.hand[index]), reverse=True)[:4]
    return tuple(sorted(keep))


def _ante_one_straight_upgrade_core_indices(
    state: GameState,
    selected_indices: tuple[int, ...],
) -> tuple[int, ...] | None:
    best: tuple[tuple[int, int, int, int], tuple[int, ...]] | None = None
    for core in combinations(selected_indices, 4):
        cards = tuple(state.hand[index] for index in core)
        if _straight_draw_potential(cards) < 4:
            continue
        key = (
            _straight_core_high_end(cards),
            sum(STRAIGHT_VALUES[card.rank] for card in cards),
            sum(RANK_VALUES[card.rank] for card in cards),
            -sum(core),
        )
        if best is None or key > best[0]:
            best = (key, tuple(sorted(core)))
    return best[1] if best is not None else None


def _ante_one_upgrade_discard_indices(
    state: GameState,
    keep_core_indices: tuple[int, ...],
) -> tuple[int, ...]:
    keep_indices = set(keep_core_indices)
    extra_keeps_needed = max(0, len(state.hand) - len(keep_indices) - 5)
    if extra_keeps_needed > 0:
        extras = sorted(
            (index for index in range(len(state.hand)) if index not in keep_indices),
            key=lambda index: _ante_one_card_upgrade_key(state.hand[index]),
            reverse=True,
        )
        keep_indices.update(extras[:extra_keeps_needed])
    return tuple(index for index in range(len(state.hand)) if index not in keep_indices)


def _matching_discard_action(state: GameState, discard_indices: tuple[int, ...]) -> Action | None:
    wanted = tuple(sorted(discard_indices))
    for action in state.legal_actions:
        if action.action_type == ActionType.DISCARD and tuple(sorted(action.card_indices)) == wanted:
            return action
    return None


def _ante_one_card_upgrade_key(card: Card) -> tuple[int, int]:
    return (RANK_VALUES[card.rank], STRAIGHT_VALUES[card.rank])


def _ante_one_upgrade_core_score(state: GameState, keep_indices: tuple[int, ...]) -> float:
    cards = tuple(state.hand[index] for index in keep_indices)
    return sum(RANK_VALUES[card.rank] for card in cards) + (_strong_draw_size(cards) * 8.0)


def _ante_one_upgrade_projection_is_good(
    score: int,
    projected_score: int,
    remaining_score: int,
) -> bool:
    if projected_score <= score:
        return False
    if projected_score >= remaining_score:
        return True
    target_score = max(
        score + ANTE_ONE_UPGRADE_MIN_GAIN,
        remaining_score * ANTE_ONE_UPGRADE_TARGET_RATIO,
    )
    return projected_score >= target_score


def _straight_core_high_end(cards: tuple[Card, ...]) -> int:
    values = {STRAIGHT_VALUES[card.rank] for card in cards}
    if any(card.rank == "A" for card in cards):
        values.add(1)

    best = 0
    for start in range(1, 11):
        if sum(1 for value in range(start, start + 5) if value in values) >= 4:
            best = max(best, start + 4)
    return best


def _unique_index_tuples(candidates: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        normalized = tuple(sorted(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique)


def _best_discard_action(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._best_discard_action``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_best_discard_action", None) if strategy_module is not None else None
    if patched is not None and patched is not _best_discard_action:
        return patched(*args, **kwargs)
    return _default_best_discard_action(*args, **kwargs)


def _evaluate_play_action(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._evaluate_play_action``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_evaluate_play_action", None) if strategy_module is not None else None
    if patched is not None and patched is not _evaluate_play_action:
        return patched(*args, **kwargs)
    return _default_evaluate_play_action(*args, **kwargs)


def _projected_score_after_discard(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._projected_score_after_discard``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_projected_score_after_discard", None) if strategy_module is not None else None
    if patched is not None and patched is not _projected_score_after_discard:
        return patched(*args, **kwargs)
    return _default_projected_score_after_discard(*args, **kwargs)
