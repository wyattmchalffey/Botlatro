"""Preferred hand and archetype helpers for the basic strategy bot."""

from __future__ import annotations

from collections import Counter

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.cache import _identity_cached_value
from balatro_ai.bots.basic_strategy.data import (
    DEDICATED_PAIR_BUILD_JOKERS,
    DEDICATED_TWO_PAIR_BUILD_JOKERS,
    FLUSH_ARCHETYPE_HANDS,
    JOKER_HAND_SYNERGY,
    JOKER_PRIMARY_HAND,
    NARROW_CHIP_PRIMARY_JOKERS,
    RANK_ARCHETYPE_HANDS,
    RARE_HAND_JOKER_TARGETS,
    RARE_HAND_TYPES,
)
from balatro_ai.rules.hand_evaluator import HandType
from balatro_ai.search.hand_viability import ADVANCED_HAND_TYPES, hand_type_is_viable


def _preferred_hand_type(state: GameState) -> HandType | None:
    return _identity_cached_value("preferred_hand_type", state, lambda: _preferred_hand_type_uncached(state))


def _preferred_hand_type_uncached(state: GameState) -> HandType | None:
    joker_votes: Counter[HandType] = Counter()
    for joker in state.jokers:
        primary = JOKER_PRIMARY_HAND.get(joker.name)
        if primary is not None:
            joker_votes[primary] += _primary_hand_vote_weight(state, joker.name, primary)
        for hand_type in JOKER_HAND_SYNERGY.get(joker.name, ()):
            if hand_type == primary or hand_type in RARE_HAND_TYPES:
                continue
            if hand_type == HandType.TWO_PAIR and joker.name not in DEDICATED_TWO_PAIR_BUILD_JOKERS:
                continue
            joker_votes[hand_type] += 1

    level_votes = Counter(
        {
            hand_type: _hand_level_vote(state, hand_type)
            for hand_type in HandType
        }
    )
    combined = joker_votes + level_votes
    if combined:
        if combined.get(HandType.PAIR, 0) > 0 and not _has_dedicated_pair_plan(state) and state.ante >= 3:
            combined[HandType.PAIR] -= 1
        if combined.get(HandType.TWO_PAIR, 0) > 0 and not _has_dedicated_two_pair_plan(state):
            combined[HandType.TWO_PAIR] -= 2
        best, score = combined.most_common(1)[0]
        if score > 0:
            if _single_narrow_chip_signal_is_noise(state, best, score):
                return HandType.PAIR if state.ante <= 3 else None
            return best
    if any(joker.name in {"Smeared Joker", "Four Fingers", "The Tribe", "Droll Joker"} for joker in state.jokers):
        return HandType.FLUSH
    if _has_dedicated_two_pair_plan(state):
        return HandType.TWO_PAIR
    if state.ante <= 2:
        return HandType.PAIR
    return None


def _primary_hand_vote_weight(state: GameState, joker_name: str, hand_type: HandType) -> int:
    if hand_type == HandType.PAIR and not _has_dedicated_pair_plan(state):
        return 1 if state.ante <= 2 else 0
    if hand_type == HandType.TWO_PAIR and joker_name not in DEDICATED_TWO_PAIR_BUILD_JOKERS:
        return 1 if state.ante <= 2 else 0
    if joker_name in NARROW_CHIP_PRIMARY_JOKERS and state.ante <= 3:
        if _hand_archetype_support_count(state, hand_type) <= 1 and _hand_level_vote(state, hand_type) <= 0:
            return 1
        return 2
    return 3


def _hand_level_vote(state: GameState, hand_type: HandType) -> int:
    level = max(0, state.hand_levels.get(hand_type.value, 1) - 1)
    if level > 0 and hand_type in ADVANCED_HAND_TYPES and not _advanced_hand_level_is_playable(state, hand_type):
        return 0
    if hand_type == HandType.PAIR and not _has_dedicated_pair_plan(state):
        return max(0, level - 1)
    if hand_type == HandType.TWO_PAIR and not _has_dedicated_two_pair_plan(state):
        return max(0, level - 1)
    return level


def _advanced_hand_level_is_playable(state: GameState, hand_type: HandType) -> bool:
    if _hand_archetype_support_count(state, hand_type) > 0:
        return True
    if any(RARE_HAND_JOKER_TARGETS.get(joker.name) == hand_type for joker in state.jokers):
        return True
    return hand_type_is_viable(state, hand_type)


def _single_narrow_chip_signal_is_noise(state: GameState, hand_type: HandType, score: int) -> bool:
    if state.ante > 3 or score > 1:
        return False
    if hand_type not in {HandType.STRAIGHT, HandType.FLUSH}:
        return False
    return any(JOKER_PRIMARY_HAND.get(joker.name) == hand_type for joker in state.jokers if joker.name in NARROW_CHIP_PRIMARY_JOKERS)


def _hand_archetype_support_count(state: GameState, hand_type: HandType) -> int:
    count = 0
    for joker in state.jokers:
        if JOKER_PRIMARY_HAND.get(joker.name) == hand_type:
            count += 1
        elif hand_type in JOKER_HAND_SYNERGY.get(joker.name, ()):
            count += 1
    return count


def _has_dedicated_pair_plan(state: GameState) -> bool:
    if any(joker.name in DEDICATED_PAIR_BUILD_JOKERS for joker in state.jokers):
        return True
    return state.hand_levels.get(HandType.PAIR.value, 1) >= 3


def _has_dedicated_two_pair_plan(state: GameState) -> bool:
    if any(joker.name in DEDICATED_TWO_PAIR_BUILD_JOKERS for joker in state.jokers):
        return True
    return state.hand_levels.get(HandType.TWO_PAIR.value, 1) >= 3


def _flexible_hand_types(state: GameState) -> set[HandType]:
    preferred = _preferred_hand_type(state)
    if preferred in RANK_ARCHETYPE_HANDS:
        hands = set(RANK_ARCHETYPE_HANDS)
        if not _has_dedicated_two_pair_plan(state):
            hands.discard(HandType.TWO_PAIR)
        return hands
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}
    if preferred in FLUSH_ARCHETYPE_HANDS:
        return set(FLUSH_ARCHETYPE_HANDS)
    hands = {HandType.PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE, HandType.FLUSH, HandType.STRAIGHT}
    if _has_dedicated_two_pair_plan(state):
        hands.add(HandType.TWO_PAIR)
    return hands


def _dominant_suit(state: GameState) -> str | None:
    cards = state.hand or state.known_deck
    if not cards:
        return None
    return Counter(card.suit for card in cards).most_common(1)[0][0]
