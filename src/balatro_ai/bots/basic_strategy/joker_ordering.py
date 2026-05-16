"""Joker order search for scoring-sensitive blind states."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations
from math import comb

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState, Joker
from balatro_ai.bots.basic_strategy.cards import (
    _edition_chips_value,
    _edition_mult_value,
    _edition_xmult_value,
)
from balatro_ai.bots.basic_strategy.data import (
    JOKER_ORDER_CHIPS,
    JOKER_ORDER_MULT,
    JOKER_ORDER_XMULT,
)
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.jokers import _joker_current_plus_value, _joker_current_xmult_value
from balatro_ai.bots.basic_strategy.play_scoring import _play_candidates, _played_hand_counts
from balatro_ai.rules.hand_evaluator import best_play_from_hand, debuffed_suits_for_blind


MAX_JOKER_REARRANGE_COUNT = 6
JOKER_REARRANGE_EXHAUSTIVE_COUNT = 4
JOKER_REARRANGE_MIN_GAIN = 1


def _joker_rearrange_action(state: GameState, context: _BlindContext | None = None) -> Action | None:
    context = context or _BlindContext()
    if state.phase != GamePhase.SELECTING_HAND:
        return None
    if len(state.jokers) < 2 or len(state.jokers) > MAX_JOKER_REARRANGE_COUNT:
        return None
    if not _joker_order_can_matter(state.jokers):
        return None

    current_order = tuple(range(len(state.jokers)))
    current_score = _best_play_score_for_joker_order(state, state.jokers, context)
    best_order = current_order
    best_score = current_score

    for order in _joker_rearrange_candidate_orders(state.jokers):
        if order == current_order:
            continue
        ordered_jokers = tuple(state.jokers[index] for index in order)
        score = _best_play_score_for_joker_order(state, ordered_jokers, context)
        if score > best_score:
            best_score = score
            best_order = order

    if best_order == current_order or best_score < current_score + JOKER_REARRANGE_MIN_GAIN:
        return None
    return Action(
        ActionType.REARRANGE,
        card_indices=best_order,
        target_id="jokers",
        metadata={
            "kind": "jokers",
            "reason": f"rearrange_jokers score={current_score}->{best_score}",
        },
    )


def _best_play_score_for_joker_order(state: GameState, jokers: tuple[Joker, ...], context: _BlindContext) -> int:
    ordered_state = replace(state, jokers=jokers)
    if _can_use_direct_best_play_score(ordered_state, context):
        try:
            evaluation = best_play_from_hand(
                ordered_state.hand,
                ordered_state.hand_levels,
                debuffed_suits=debuffed_suits_for_blind(ordered_state.blind),
                blind_name=ordered_state.blind,
                jokers=jokers,
                discards_remaining=ordered_state.discards_remaining,
                hands_remaining=ordered_state.hands_remaining,
                deck_size=ordered_state.deck_size,
                money=ordered_state.money,
                played_hand_types_this_round=context.played_hand_types,
                played_hand_counts=_played_hand_counts(ordered_state),
            )
            return evaluation.score
        except ValueError:
            return 0
    return max((candidate.score for candidate in _play_candidates(ordered_state, context)), default=0)


def _can_use_direct_best_play_score(state: GameState, context: _BlindContext) -> bool:
    if context.played_hand_types and state.blind in {"The Eye", "The Mouth"}:
        return False
    legal_play_count = sum(1 for action in state.legal_actions if action.action_type == ActionType.PLAY_HAND and action.card_indices)
    return legal_play_count == _full_play_action_count(len(state.hand))


def _full_play_action_count(card_count: int) -> int:
    total = 0
    for size in range(1, min(5, card_count) + 1):
        total += comb(card_count, size)
    return total


def _joker_rearrange_candidate_orders(jokers: tuple[Joker, ...]) -> tuple[tuple[int, ...], ...]:
    current_order = tuple(range(len(jokers)))
    if len(jokers) <= JOKER_REARRANGE_EXHAUSTIVE_COUNT:
        return tuple(permutations(current_order))

    orders: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(order: tuple[int, ...]) -> None:
        if len(order) != len(jokers) or set(order) != set(current_order) or order in seen:
            return
        seen.add(order)
        orders.append(order)

    role_order = tuple(sorted(current_order, key=lambda index: _joker_order_sort_key(jokers[index], index)))
    add(current_order)
    add(role_order)

    copy_indices = tuple(
        index for index, joker in enumerate(jokers) if joker.name in {"Blueprint", "Brainstorm"}
    )
    if copy_indices:
        for target_index in current_order:
            if jokers[target_index].name in {"Blueprint", "Brainstorm"}:
                continue
            add(_order_with_target_first(role_order, target_index))
            for copy_index in copy_indices:
                add(_order_with_copy_before_target(role_order, copy_index, target_index))

    return tuple(orders)


def _joker_order_sort_key(joker: Joker, original_index: int) -> tuple[int, int]:
    role = _joker_order_role(joker)
    if role in {"chips", "mult"}:
        return (0, original_index)
    if joker.name in {"Blueprint", "Brainstorm"}:
        return (1, original_index)
    if role == "xmult":
        return (2, original_index)
    return (1, original_index)


def _order_with_target_first(order: tuple[int, ...], target_index: int) -> tuple[int, ...]:
    return (target_index, *(index for index in order if index != target_index))


def _order_with_copy_before_target(
    order: tuple[int, ...],
    copy_index: int,
    target_index: int,
) -> tuple[int, ...]:
    without_copy = [index for index in order if index != copy_index]
    try:
        target_position = without_copy.index(target_index)
    except ValueError:
        return tuple(order)
    without_copy.insert(target_position, copy_index)
    return tuple(without_copy)


def _joker_order_can_matter(jokers: tuple[Joker, ...]) -> bool:
    names = {joker.name for joker in jokers}
    if names & {"Blueprint", "Brainstorm", "Baseball Card"}:
        return True
    has_xmult = any(_joker_order_role(joker) == "xmult" for joker in jokers)
    has_pre_xmult_value = any(_joker_order_role(joker) in {"chips", "mult"} for joker in jokers)
    return has_xmult and has_pre_xmult_value


def _joker_order_role(joker: Joker) -> str:
    if _edition_xmult_value(joker.edition) > 1:
        return "xmult"
    if _edition_mult_value(joker.edition) > 0:
        return "mult"
    if _edition_chips_value(joker.edition) > 0:
        return "chips"
    if joker.name in JOKER_ORDER_XMULT:
        return "xmult"
    if joker.name in JOKER_ORDER_CHIPS:
        return "chips"
    if joker.name in JOKER_ORDER_MULT:
        return "mult"
    if _joker_current_xmult_value(joker) > 1:
        return "xmult"
    if _joker_current_plus_value(joker, suffix="mult") > 0:
        return "mult"
    if _joker_current_plus_value(joker, suffix="chips") > 0:
        return "chips"
    return ""
