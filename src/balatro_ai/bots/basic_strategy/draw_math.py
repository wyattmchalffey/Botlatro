"""Pure draw-probability and straight/rank helper math."""

from __future__ import annotations

from itertools import combinations
from math import comb

from balatro_ai.api.state import Card
from balatro_ai.rules.hand_evaluator import STRAIGHT_VALUES


def _straight_values_present(cards: tuple[Card, ...]) -> set[int]:
    values: set[int] = set()
    for card in cards:
        values.update(_straight_values_for_card(card))
    return values


def _straight_values_for_card(card: Card) -> tuple[int, ...]:
    value = STRAIGHT_VALUES[card.rank]
    if card.rank == "A":
        return (1, 14)
    return (value,)


def _draw_at_least_one_probability(deck_size: int, draw_count: int, out_count: int) -> float:
    if out_count <= 0 or draw_count <= 0:
        return 0.0
    if out_count >= deck_size:
        return 1.0
    draw_count = min(draw_count, deck_size)
    total = comb(deck_size, draw_count)
    if total <= 0:
        return 0.0
    misses = comb(max(0, deck_size - out_count), draw_count) if deck_size - out_count >= draw_count else 0
    return max(0.0, min(1.0, 1.0 - (misses / total)))


def _draw_all_groups_probability(deck_size: int, draw_count: int, group_counts: tuple[int, ...]) -> float:
    total = comb(deck_size, draw_count)
    if total <= 0:
        return 0.0
    probability = 1.0
    group_indexes = range(len(group_counts))
    for size in range(1, len(group_counts) + 1):
        sign = -1.0 if size % 2 else 1.0
        for indexes in combinations(group_indexes, size):
            blocked = sum(group_counts[index] for index in indexes)
            remaining = deck_size - blocked
            ways = comb(remaining, draw_count) if remaining >= draw_count else 0
            probability += sign * (ways / total)
    return max(0.0, min(1.0, probability))


def _straight_window_duplicate_penalty(
    value_to_cards: dict[int, list[Card]],
    present_values: tuple[int, ...],
) -> int:
    return sum(max(0, len(value_to_cards.get(value, ())) - 1) for value in present_values)


def _straight_window_is_open_ended(missing_values: tuple[int, ...], window_values: tuple[int, ...]) -> bool:
    return len(missing_values) == 1 and missing_values[0] in {window_values[0], window_values[-1]}


def _rank_matches_straight_value(card: Card, value: int) -> bool:
    return value in _straight_values_for_card(card)


def _draw_at_least_k_probability(deck_size: int, draw_count: int, out_count: int, needed: int) -> float:
    if needed <= 0:
        return 1.0
    if out_count < needed or draw_count < needed:
        return 0.0
    draw_count = min(draw_count, deck_size)
    total = comb(deck_size, draw_count)
    if total <= 0:
        return 0.0
    max_hits = min(out_count, draw_count)
    ways = 0
    rest = max(0, deck_size - out_count)
    for hits in range(needed, max_hits + 1):
        misses = draw_count - hits
        if misses > rest:
            continue
        ways += comb(out_count, hits) * comb(rest, misses)
    return max(0.0, min(1.0, ways / total))


def _draw_group_min_counts_probability(
    deck_size: int,
    draw_count: int,
    requirements: tuple[int, ...],
    out_counts: tuple[int, ...],
) -> float:
    draw_count = min(draw_count, deck_size)
    total = comb(deck_size, draw_count)
    if total <= 0:
        return 0.0
    if len(requirements) != 2 or len(out_counts) != 2:
        return 0.0
    first_needed, second_needed = requirements
    first_out, second_out = out_counts
    if first_out < first_needed or second_out < second_needed:
        return 0.0
    rest = max(0, deck_size - first_out - second_out)
    ways = 0
    for first_hits in range(first_needed, min(first_out, draw_count) + 1):
        for second_hits in range(second_needed, min(second_out, draw_count - first_hits) + 1):
            misses = draw_count - first_hits - second_hits
            if misses > rest:
                continue
            ways += comb(first_out, first_hits) * comb(second_out, second_hits) * comb(rest, misses)
    return max(0.0, min(1.0, ways / total))


def _strategy_rank_order() -> tuple[str, ...]:
    return ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")


def _straight_rank_for_value(value: int) -> str:
    if value == 1:
        return "A"
    for rank, rank_value in STRAIGHT_VALUES.items():
        if rank_value == value and rank != "T":
            return rank
    return "A"
