"""Shop valuation for planets, tarots, spectral cards, and playing cards."""

from __future__ import annotations

from dataclasses import replace

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.build_profile import _build_profile
from balatro_ai.bots.basic_strategy.build_scoring import _sample_build_score
from balatro_ai.bots.basic_strategy.cards import _card_label, _card_modifier, _card_rank, _card_suit
from balatro_ai.bots.basic_strategy.data import (
    JOKER_HAND_SYNERGY,
    JOKER_PRIMARY_HAND,
    PLANET_TO_HAND,
    RARE_FLUSH_HAND_TYPES,
    RARE_HAND_TYPES,
    RARE_RANK_HAND_TYPES,
    TAROT_VALUES,
)
from balatro_ai.bots.basic_strategy.hand_preferences import (
    _dominant_suit,
    _flexible_hand_types,
    _has_dedicated_pair_plan,
    _has_dedicated_two_pair_plan,
    _preferred_hand_type,
)
from balatro_ai.bots.basic_strategy.jokers import _active_joker_names
from balatro_ai.bots.basic_strategy.rare_hands import (
    _rare_hand_deck_manipulation_need,
    _rare_hand_investment_penalty,
    _rare_hand_plan,
    _rare_rank_target,
    _tarot_supports_rare_hand,
    _visible_rank_count,
)
from balatro_ai.rules.hand_evaluator import HandType, RANK_VALUES
from balatro_ai.search.hand_viability import ADVANCED_HAND_TYPES, hand_type_viability_multiplier


def _planet_card_value(state: GameState, card: object) -> float:
    hand_type = PLANET_TO_HAND.get(_card_label(card))
    if hand_type is None:
        return 0.0

    preferred = _preferred_hand_type(state)
    current_level = state.hand_levels.get(hand_type.value, 1)
    value = 14 + min(current_level, 5) * 2
    capacity_gain = _planet_capacity_gain(state, hand_type)
    flexible = _flexible_hand_types(state)
    alignment_score = _hand_joker_alignment_score(state, hand_type)
    if hand_type == preferred:
        value += min(48.0, capacity_gain / 12.0)
        value += min(20.0, alignment_score * 5.0)
    elif hand_type in flexible:
        value += min(24.0, capacity_gain / 18.0)
        value += min(10.0, alignment_score * 3.0)
    else:
        value += min(6.0, capacity_gain / 36.0)
    if hand_type in RARE_HAND_TYPES:
        value -= _rare_hand_investment_penalty(state, hand_type) * 0.8
    if hand_type in ADVANCED_HAND_TYPES:
        viability = hand_type_viability_multiplier(state, hand_type)
        if viability < 1.0:
            value *= 0.20 + (0.80 * viability)
            value -= (1.0 - viability) * 18.0
    value -= _weak_hand_plan_penalty(state, hand_type)
    if hand_type == preferred:
        value += 24
        if hand_type in RARE_HAND_TYPES and _rare_hand_deck_manipulation_need(state, hand_type) > 0:
            value -= 10
        elif state.ante >= 4 and not _build_profile(state).has_scaling:
            value += 8
    elif hand_type in flexible:
        value += 8
        if state.ante <= 2 and alignment_score <= 0 and current_level <= 1:
            value -= 20
    else:
        value -= 8
        if state.ante <= 2:
            value -= 24
    value = max(value, _early_flexible_planet_floor(state, hand_type, current_level))
    if "Constellation" in _active_joker_names(state):
        value += 18.0
    return value


def _early_flexible_planet_floor(state: GameState, hand_type: HandType, current_level: int) -> float:
    if state.ante > 1 or current_level > 1 or hand_type in RARE_HAND_TYPES:
        return 0.0
    if hand_type in {HandType.FLUSH, HandType.STRAIGHT, HandType.FULL_HOUSE, HandType.THREE_OF_A_KIND}:
        return 36.0
    if hand_type == HandType.TWO_PAIR:
        return 24.0
    return 18.0


def _planet_capacity_gain(state: GameState, hand_type: HandType) -> float:
    current = _sample_build_score(state, state.jokers)
    hand_levels = dict(state.hand_levels)
    hand_levels[hand_type.value] = hand_levels.get(hand_type.value, 1) + 1
    leveled = replace(state, hand_levels=hand_levels)
    return max(0.0, _sample_build_score(leveled, leveled.jokers) - current)


def _hand_joker_alignment_score(state: GameState, hand_type: HandType) -> float:
    score = 0.0
    for joker in state.jokers:
        if hand_type == JOKER_PRIMARY_HAND.get(joker.name):
            score += 1.25
        elif hand_type in JOKER_HAND_SYNERGY.get(joker.name, ()):
            score += 1.0
    return score


def _black_hole_card_value(state: GameState) -> float:
    preferred = _preferred_hand_type(state)
    matching_planet_value = 0.0
    for planet, hand_type in PLANET_TO_HAND.items():
        if hand_type == preferred:
            matching_planet_value = _planet_card_value(state, {"label": planet, "set": "PLANET"})
            break
    levels = sum(max(1, int(level)) for level in state.hand_levels.values()) if state.hand_levels else 12
    return max(160.0, matching_planet_value + 90.0 + min(levels, 36) * 1.5)


def _weak_hand_plan_penalty(state: GameState, hand_type: HandType) -> float:
    if hand_type == HandType.PAIR:
        if _has_dedicated_pair_plan(state):
            return 0.0
        penalty = 8.0
        if state.ante >= 4:
            penalty += 8.0
        if state.hand_levels.get(hand_type.value, 1) >= 3:
            penalty -= 8.0
        return max(0.0, penalty)
    if hand_type != HandType.TWO_PAIR or _has_dedicated_two_pair_plan(state):
        return 0.0
    current_level = state.hand_levels.get(hand_type.value, 1)
    penalty = 18.0
    if current_level >= 2:
        penalty -= 6.0
    if state.ante >= 4:
        penalty += 6.0
    return max(8.0, penalty)


def _tarot_card_value(state: GameState, card: object) -> float:
    name = _card_label(card)
    value = TAROT_VALUES.get(name, 18)
    preferred = _preferred_hand_type(state)
    profile = _build_profile(state)
    if preferred == HandType.FLUSH and name in {"The Sun", "The Moon", "The Star", "The World"}:
        value += 14
    if preferred in {HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE} and name in {
        "Strength",
        "Death",
        "The Hanged Man",
    }:
        value += 12
    if state.money < 8 and name in {"Temperance", "The Hermit"}:
        value += 12
    if name == "The Hermit":
        value += 8
        if state.money >= 10:
            value += 5
    elif name == "Temperance" and state.ante <= 2:
        value -= 3
    if profile.rich and profile.ante >= 4 and name in {"Death", "The Hanged Man", "Strength"}:
        value += 10
    if not profile.has_economy and name in {"The Hermit", "Temperance"}:
        value += 8
    value += _rare_hand_deck_manipulation_bonus(state, name)
    return value


def _rare_hand_deck_manipulation_bonus(state: GameState, name: str) -> float:
    hand_type = _rare_hand_plan(state)
    if hand_type is None or not _tarot_supports_rare_hand(name, hand_type):
        return 0.0

    need = _rare_hand_deck_manipulation_need(state, hand_type)
    if need <= 0:
        return 10.0
    return min(34.0, 16.0 + need * 7.0)


def _playing_card_shop_value(state: GameState, card: object) -> float:
    rank = _card_rank(card)
    suit = _card_suit(card)
    value = RANK_VALUES.get(rank, 5)
    if _preferred_hand_type(state) == HandType.FLUSH:
        dominant_suit = _dominant_suit(state)
        if dominant_suit and suit == dominant_suit:
            value += 18
    rare_plan = _rare_hand_plan(state)
    if rare_plan in RARE_RANK_HAND_TYPES:
        target_rank = _rare_rank_target(state)
        if target_rank and rank == target_rank:
            value += 26
        elif rank and _visible_rank_count(state, rank) >= 2:
            value += 14
    if rare_plan in RARE_FLUSH_HAND_TYPES:
        dominant_suit = _dominant_suit(state)
        if dominant_suit and suit == dominant_suit:
            value += 14
    enhancement = str(_card_modifier(card).get("enhancement", card.get("enhancement", ""))) if isinstance(card, dict) else ""
    if enhancement:
        value += 10
        if any(joker.name == "Vampire" for joker in state.jokers):
            value += 22
    if any(joker.name == "Hologram" for joker in state.jokers):
        value += 20
    if any(joker.name == "Midas Mask" for joker in state.jokers) and rank in {"J", "Q", "K"}:
        value += 12
    return float(value)
