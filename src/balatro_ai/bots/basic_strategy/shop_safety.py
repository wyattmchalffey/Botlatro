"""Early shop safety checks and hand-support predicates."""

from __future__ import annotations

from collections import Counter

from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy.cards import (
    _card_cost,
    _is_joker_card,
    _is_planet_card,
    _is_tarot_card,
    _joker_from_shop_card,
    _pack_card_requires_targets,
)
from balatro_ai.bots.basic_strategy.data import (
    JOKER_HAND_SYNERGY,
    MONEY_SCALING_RESERVE_TARGETS,
    NARROW_EARLY_JOKERS,
)
from balatro_ai.bots.basic_strategy.hand_value import _straight_draw_potential
from balatro_ai.bots.basic_strategy.jokers import _joker_roles, _joker_sell_total
from balatro_ai.bots.basic_strategy.pack_targets import _target_required_tarot_is_supported
from balatro_ai.rules.hand_evaluator import HandType


def _early_shop_safety_adjustment(state: GameState, card: object) -> float:
    if state.ante > 2:
        return 0.0

    adjustment = 0.0
    if _is_tarot_card(card) and _pack_card_requires_targets(card) and not _target_required_tarot_is_supported(state, card):
        return -100.0
    if _is_planet_card(card) and not _has_real_scoring_joker(state):
        adjustment -= 18.0
    if _is_joker_card(card):
        joker = _joker_from_shop_card(card)
        roles = _joker_roles(joker)
        if not _has_real_scoring_joker(state):
            if roles == {"economy"}:
                adjustment -= 55.0
            if joker.name == "Swashbuckler" and _joker_sell_total(state) < 8:
                adjustment -= 45.0
            if joker.name in NARROW_EARLY_JOKERS and not _early_hand_type_is_supported(state, joker.name):
                adjustment -= 95.0
        if state.money - _card_cost(card) <= 1 and roles.isdisjoint({"chips", "mult", "xmult", "scaling"}):
            adjustment -= 20.0
    return adjustment


def _has_real_scoring_joker(state: GameState) -> bool:
    return any(_joker_roles(joker) & {"chips", "mult", "xmult", "scaling"} for joker in state.jokers)


def _has_money_scaling_joker(state: GameState) -> bool:
    return any(joker.name in MONEY_SCALING_RESERVE_TARGETS for joker in state.jokers)


def _early_hand_type_is_supported(state: GameState, joker_name: str) -> bool:
    wanted = set(JOKER_HAND_SYNERGY.get(joker_name, ()))
    if not wanted:
        return True
    if state.hand_levels and any(state.hand_levels.get(hand_type.value, 1) > 1 for hand_type in wanted):
        return True
    return _hand_has_natural_support(state.hand, wanted)


def _hand_has_natural_support(hand: tuple[Card, ...], wanted: set[HandType]) -> bool:
    if not hand:
        return False
    rank_counts = Counter(card.rank for card in hand)
    suit_counts = Counter(card.suit for card in hand)
    if wanted & {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return _straight_draw_potential(hand) >= 4
    if wanted & {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        return max(suit_counts.values(), default=0) >= 4
    if wanted & {HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        return max(rank_counts.values(), default=0) >= 3
    if wanted & {HandType.PAIR, HandType.TWO_PAIR}:
        return max(rank_counts.values(), default=0) >= 2
    return False
