"""Rare-hand planning and support scoring helpers."""

from __future__ import annotations

from collections import Counter

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.data import (
    FLUSH_DECK_MANIPULATION_TAROTS,
    IMPOSSIBLE_HAND_MANIPULATION_GAP,
    IMPOSSIBLE_STARTER_DECK_HANDS,
    RANK_DECK_MANIPULATION_TAROTS,
    RARE_FLUSH_HAND_TYPES,
    RARE_HAND_JOKER_TARGETS,
    RARE_HAND_SUPPORT_TAROTS,
    RARE_HAND_TYPES,
    RARE_RANK_HAND_TYPES,
    TWO_PAIR_SUPPORT_JOKERS,
)
from balatro_ai.bots.basic_strategy.hand_preferences import (
    _has_dedicated_two_pair_plan,
    _preferred_hand_type,
)
from balatro_ai.rules.hand_evaluator import HandType


def _unsupported_rare_joker_extra_penalty(state: GameState, joker_name: str) -> float:
    hand_type = RARE_HAND_JOKER_TARGETS.get(joker_name)
    if hand_type is None:
        return 0.0
    need = _rare_hand_deck_manipulation_need(state, hand_type)
    if need <= 0:
        return 0.0
    return min(32.0, 14.0 + need * 4.0)


def _unsupported_two_pair_joker_penalty(state: GameState, joker_name: str) -> float:
    if joker_name not in TWO_PAIR_SUPPORT_JOKERS:
        return 0.0
    if _has_dedicated_two_pair_plan(state):
        return 0.0
    return 54.0 if state.ante <= 3 else 72.0


def _rare_hand_plan(state: GameState) -> HandType | None:
    preferred = _preferred_hand_type(state)
    if preferred in RARE_HAND_TYPES:
        return preferred

    for joker in state.jokers:
        target = RARE_HAND_JOKER_TARGETS.get(joker.name)
        if target is not None:
            return target

    leveled_rare_hands = [
        hand_type
        for hand_type in RARE_HAND_TYPES
        if state.hand_levels.get(hand_type.value, 1) > 1
    ]
    if leveled_rare_hands:
        return max(leveled_rare_hands, key=lambda hand_type: state.hand_levels.get(hand_type.value, 1))
    return None


def _rare_hand_investment_penalty(state: GameState, hand_type: HandType) -> float:
    if hand_type not in RARE_HAND_TYPES:
        return 0.0

    support_gap = _rare_hand_support_gap(state, hand_type)
    if support_gap <= 0:
        return 0.0
    early_risk = 1.0 if state.ante <= 3 else 0.75
    return min(42.0, (18.0 + support_gap * 10.0) * early_risk)


def _rare_hand_deck_manipulation_need(state: GameState, hand_type: HandType) -> float:
    if hand_type not in RARE_HAND_TYPES:
        return 0.0
    return _rare_hand_support_gap(state, hand_type)


def _rare_hand_support_gap(state: GameState, hand_type: HandType) -> float:
    gap = max(0.0, _rare_hand_support_threshold(hand_type) - _rare_hand_support_score(state, hand_type))
    if hand_type in IMPOSSIBLE_STARTER_DECK_HANDS and not _has_impossible_hand_manipulation_path(state, hand_type):
        gap += IMPOSSIBLE_HAND_MANIPULATION_GAP[hand_type]
    return gap


def _rare_hand_support_threshold(hand_type: HandType) -> float:
    if hand_type == HandType.FOUR_OF_A_KIND:
        return 3.0
    if hand_type in {HandType.FIVE_OF_A_KIND, HandType.FLUSH_FIVE}:
        return 4.0
    if hand_type == HandType.FLUSH_HOUSE:
        return 3.5
    return 0.0


def _rare_hand_support_score(state: GameState, hand_type: HandType) -> float:
    cards = (*state.hand, *state.known_deck)
    rank_counts = Counter(card.rank for card in cards)
    suit_counts = Counter(card.suit for card in cards)
    suited_rank_counts = Counter((card.rank, card.suit) for card in cards)

    max_rank = max(rank_counts.values(), default=0)
    max_suit = max(suit_counts.values(), default=0)
    max_suited_rank = max(suited_rank_counts.values(), default=0)

    if hand_type == HandType.FOUR_OF_A_KIND:
        score = float(max_rank)
    elif hand_type == HandType.FIVE_OF_A_KIND:
        score = float(max_rank)
    elif hand_type == HandType.FLUSH_FIVE:
        score = float(max_suited_rank)
        if max_rank >= 3 and max_suit >= 4:
            score += 0.75
    elif hand_type == HandType.FLUSH_HOUSE:
        score = min(float(max_rank), 3.0) + min(float(max_suit), 5.0) * 0.25
    else:
        score = 0.0

    level = state.hand_levels.get(hand_type.value, 1)
    score += min(1.5, max(0, level - 1) * 0.5)
    score += min(1.5, sum(1 for name in state.consumables if name in RARE_HAND_SUPPORT_TAROTS) * 0.5)
    return score


def _has_impossible_hand_manipulation_path(state: GameState, hand_type: HandType) -> bool:
    if hand_type not in IMPOSSIBLE_STARTER_DECK_HANDS:
        return True
    if state.hand_levels.get(hand_type.value, 1) > 1:
        return True
    if any(_tarot_supports_rare_hand(name, hand_type) for name in state.consumables):
        return True

    cards = (*state.hand, *state.known_deck)
    if not cards:
        return False
    rank_counts = Counter(card.rank for card in cards)
    exact_counts = Counter((card.rank, card.suit) for card in cards)

    max_rank = max(rank_counts.values(), default=0)
    max_exact = max(exact_counts.values(), default=0)
    if hand_type == HandType.FIVE_OF_A_KIND:
        return max_rank >= 5 or max_exact >= 2
    if hand_type == HandType.FLUSH_FIVE:
        return max_exact >= 2
    if hand_type == HandType.FLUSH_HOUSE:
        return max_exact >= 2
    return False


def _tarot_supports_rare_hand(name: str, hand_type: HandType) -> bool:
    if hand_type in RARE_RANK_HAND_TYPES and name in RANK_DECK_MANIPULATION_TAROTS:
        return True
    if hand_type in RARE_FLUSH_HAND_TYPES and name in FLUSH_DECK_MANIPULATION_TAROTS:
        return True
    return False


def _rare_rank_target(state: GameState) -> str | None:
    cards = (*state.hand, *state.known_deck)
    if not cards:
        return None
    rank, count = Counter(card.rank for card in cards).most_common(1)[0]
    return rank if count >= 2 else None


def _visible_rank_count(state: GameState, rank: str) -> int:
    return sum(1 for card in (*state.hand, *state.known_deck) if card.rank == rank)
