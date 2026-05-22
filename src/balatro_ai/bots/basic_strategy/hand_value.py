"""Card retention and kept-hand value helpers for blind decisions."""

from __future__ import annotations

from collections import Counter

from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy.cache import _freeze_for_cache, _state_scoped_cache
from balatro_ai.bots.basic_strategy.cards import _is_face_card_for_state, _normalize_suit, _rank_matches
from balatro_ai.bots.basic_strategy.data import FLUSH_ARCHETYPE_HANDS
from balatro_ai.bots.basic_strategy.hand_preferences import _dominant_suit, _preferred_hand_type
from balatro_ai.bots.basic_strategy.jokers import (
    _castle_target_suit,
    _current_plus_for_joker,
    _joker_names,
    _mail_in_rebate_rank,
)
from balatro_ai.rules.hand_evaluator import (
    HandType,
    RANK_VALUES,
    STRAIGHT_VALUES,
    best_play_from_hand,
    debuffed_suits_for_blind,
)


def _kept_hand_potential(state: GameState, kept_cards: tuple[Card, ...]) -> float:
    if not kept_cards:
        return 0.0
    cache = _state_scoped_cache("kept_hand_potential", state)
    cache_key = (_freeze_for_cache(kept_cards),)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    cheap_potential = _cheap_kept_hand_potential(kept_cards, _preferred_hand_type(state))
    immediate_score = best_play_from_hand(
        kept_cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=state.jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=state.hands_remaining,
        deck_size=state.deck_size,
        money=state.money,
    ).score

    potential = cheap_potential + (immediate_score * 0.03)
    if cache is not None:
        cache[cache_key] = potential
    return potential


def _cheap_kept_hand_potential(kept_cards: tuple[Card, ...], preferred: HandType | None = None) -> float:
    if not kept_cards:
        return 0.0

    rank_counts = Counter(card.rank for card in kept_cards)
    suit_counts = Counter(card.suit for card in kept_cards)
    rank_weight = 20
    flush_weight = 8
    straight_weight = 6
    if preferred in {HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE}:
        rank_weight = 28
        flush_weight = 5
        straight_weight = 4
    elif preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        rank_weight = 8
        flush_weight = 24
        straight_weight = 4
    elif preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        rank_weight = 10
        flush_weight = 5
        straight_weight = 22

    pair_bonus = sum(count * rank_weight for count in rank_counts.values() if count >= 2)
    flush_draw_bonus = max(suit_counts.values(), default=0) * flush_weight
    straight_draw_bonus = _straight_draw_potential(kept_cards) * straight_weight
    high_card_bonus = sum(RANK_VALUES[card.rank] for card in kept_cards) / max(1, len(kept_cards))
    return pair_bonus + flush_draw_bonus + straight_draw_bonus + high_card_bonus


def _straight_draw_potential(cards: tuple[Card, ...]) -> int:
    values = {STRAIGHT_VALUES[card.rank] for card in cards}
    if "A" in {card.rank for card in cards}:
        values.add(1)
    best = 0
    for start in range(1, 11):
        best = max(best, sum(1 for value in range(start, start + 5) if value in values))
    return best


def _joker_card_keep_bonus(state: GameState, card: Card) -> float:
    names = _joker_names(state)
    bonus = 0.0
    face = _is_face_card_for_state(state, card)
    black = _normalize_suit(card.suit) in {"S", "C"}

    if "Ride the Bus" in names and face and "Pareidolia" not in names:
        bonus -= _ride_the_bus_face_keep_penalty(state)
    if "Wee Joker" in names and card.rank == "2":
        bonus += 130.0
    if "Hack" in names and card.rank in {"2", "3", "4", "5"}:
        bonus += 55.0
    if "Fibonacci" in names and card.rank in {"A", "2", "3", "5", "8"}:
        bonus += 45.0
    if "Scholar" in names and card.rank == "A":
        bonus += 45.0
    if "Even Steven" in names and card.rank in {"2", "4", "6", "8", "10", "T"}:
        bonus += 30.0
    if "Odd Todd" in names and card.rank in {"A", "3", "5", "7", "9"}:
        bonus += 30.0
    if "Walkie Talkie" in names and card.rank in {"10", "T", "4"}:
        bonus += 35.0
    if "Shoot the Moon" in names and card.rank == "Q":
        bonus += 95.0 * _held_effect_multiplier(state)
    if "Baron" in names and card.rank == "K":
        bonus += 110.0 * _held_effect_multiplier(state)
    if "Reserved Parking" in names and face:
        bonus += 35.0 * _held_effect_multiplier(state)
    if "Raised Fist" in names:
        bonus += RANK_VALUES[card.rank] * 4.0 * _held_effect_multiplier(state)
    if "Blackboard" in names:
        bonus += 80.0 if black else -55.0
    castle_suit = _castle_target_suit(state)
    if castle_suit is not None and _normalize_suit(card.suit) == castle_suit:
        bonus -= 35.0
    mail_rank = _mail_in_rebate_rank(state)
    if mail_rank is not None and _rank_matches(card.rank, mail_rank):
        bonus -= 25.0
    return bonus


def _ride_the_bus_face_keep_penalty(state: GameState) -> float:
    current_mult = max(0, _current_plus_for_joker(state, "Ride the Bus", suffix="mult"))
    return 95.0 + min(260.0, current_mult * 45.0)


def _held_effect_multiplier(state: GameState) -> float:
    return 2.0 if any(joker.name == "Mime" for joker in state.jokers) else 1.0


def _card_keep_scores(
    hand: tuple[Card, ...],
    preferred: HandType | None = None,
    *,
    state: GameState | None = None,
) -> tuple[float, ...]:
    rank_counts = Counter(card.rank for card in hand)
    suit_counts = Counter(card.suit for card in hand)
    straight_values = [STRAIGHT_VALUES[card.rank] for card in hand]

    rank_weight = 100
    suit_weight = 8
    straight_weight = 6
    if preferred in {HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE}:
        rank_weight = 140
        suit_weight = 5
        straight_weight = 4
    elif preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        rank_weight = 35
        suit_weight = 28
        straight_weight = 4
    elif preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        rank_weight = 45
        suit_weight = 5
        straight_weight = 24

    scores: list[float] = []
    for card in hand:
        straight_value = STRAIGHT_VALUES[card.rank]
        nearby = sum(
            1
            for value in straight_values
            if value != straight_value and abs(value - straight_value) <= 4
        )
        ace_low_nearby = 0
        if card.rank == "A":
            ace_low_nearby = sum(1 for value in straight_values if value in {2, 3, 4, 5})

        rank_count = rank_counts[card.rank]
        score = RANK_VALUES[card.rank]
        score += suit_counts[card.suit] * suit_weight
        score += max(nearby, ace_low_nearby) * straight_weight
        if rank_count >= 2:
            score += rank_count * rank_weight
        score += _joker_card_keep_bonus(state, card) if state is not None else 0.0
        scores.append(float(score))

    return tuple(scores)


def _card_long_term_value(state: GameState, card: Card) -> float:
    value = float(RANK_VALUES.get(card.rank, 0))
    if card.enhancement:
        value += 18.0
    if card.edition:
        value += 20.0
    if card.seal:
        value += 16.0
    if card.rank in {"A", "K", "Q", "J"}:
        value += 4.0
    if _preferred_hand_type(state) in FLUSH_ARCHETYPE_HANDS:
        dominant = _dominant_suit(state)
        if dominant and card.suit == dominant:
            value += 10.0
    return value
