from __future__ import annotations

import context  # noqa: F401

from balatro_ai.probability.hand_type_odds import (
    HAND_TYPES,
    RANKS,
    CardKey,
    SimulationConfig,
    available_hand_types,
    estimate_hand_type_probabilities,
    standard_deck_counts,
)
from balatro_ai.rules.hand_evaluator import HandType


def test_available_hand_types_uses_exact_balatro_classification() -> None:
    hand = (
        CardKey(rank=8, suit=0),
        CardKey(rank=9, suit=0),
        CardKey(rank=10, suit=0),
        CardKey(rank=11, suit=0),
        CardKey(rank=12, suit=0),
    )

    available = available_hand_types(hand)

    assert HandType.STRAIGHT_FLUSH in available
    assert HandType.STRAIGHT not in available
    assert HandType.FLUSH not in available
    assert HandType.HIGH_CARD in available


def test_available_hand_types_requires_distinct_full_house_ranks() -> None:
    five_same_rank = tuple(CardKey(rank=0, suit=0) for _ in range(5))

    available = available_hand_types(five_same_rank)

    assert HandType.FLUSH_FIVE in available
    assert HandType.FIVE_OF_A_KIND not in available
    assert HandType.FULL_HOUSE not in available


def test_standard_deck_simulation_returns_all_hand_rows() -> None:
    result = estimate_hand_type_probabilities(
        SimulationConfig(
            deck_counts=standard_deck_counts(),
            hand_size=8,
            play_size=5,
            discard_size=5,
            discards=1,
            hands=1,
            trials=100,
            seed=7,
        )
    )

    assert tuple(row.hand_type for row in result.rows) == HAND_TYPES
    assert result.deck_size == 52
    assert result.trials == 100
    assert result.rows[0].hand_type == HandType.HIGH_CARD
    assert result.rows[0].opening == 1.0
    assert all(0.0 <= row.after_discards_and_hands <= 1.0 for row in result.rows)


def test_flush_five_does_not_count_as_five_of_a_kind_without_off_suit_copy() -> None:
    hand = tuple(CardKey(rank=11, suit=0) for _copy in range(5))

    available = available_hand_types(hand)

    assert HandType.FLUSH_FIVE in available
    assert HandType.FIVE_OF_A_KIND not in available


def test_five_of_a_kind_can_use_one_off_suit_copy_when_deck_is_exhausted() -> None:
    counts = [[0 for _rank in RANKS] for _suit in range(4)]
    king = RANKS.index("K")
    counts[0][king] = 5
    counts[1][king] = 1
    counts[2][king] = 1
    counts[3][king] = 1

    added = 0
    for suit in range(4):
        for rank in range(len(RANKS)):
            if rank == king:
                continue
            if added >= 22:
                break
            counts[suit][rank] = 1
            added += 1
        if added >= 22:
            break

    result = estimate_hand_type_probabilities(
        SimulationConfig(
            deck_counts=tuple(tuple(row) for row in counts),
            hand_size=8,
            play_size=5,
            discard_size=5,
            discards=4,
            hands=4,
            trials=100,
            seed=19,
        )
    )
    rows = {row.hand_type: row for row in result.rows}

    assert result.deck_size == 30
    assert rows[HandType.FLUSH_FIVE].after_discards_and_hands == 1.0
    assert rows[HandType.FIVE_OF_A_KIND].after_discards_and_hands == 1.0
    assert rows[HandType.FIVE_OF_A_KIND].after_discards_and_hands >= rows[HandType.FLUSH_FIVE].after_discards_and_hands
