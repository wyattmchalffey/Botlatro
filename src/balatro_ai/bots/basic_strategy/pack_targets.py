"""Booster-pack card pickability and target selection."""

from __future__ import annotations

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.cards import (
    _card_label,
    _is_black_hole_card,
    _is_joker_card,
    _joker_from_shop_card,
    _joker_would_overfill_slots,
    _pack_card_requires_targets,
)
from balatro_ai.bots.basic_strategy.data import (
    FLUSH_ARCHETYPE_HANDS,
    FLUSH_DECK_MANIPULATION_TAROTS,
    SUIT_TAROT_TARGET_SUITS,
)
from balatro_ai.bots.basic_strategy.hand_preferences import (
    _dominant_suit,
    _hand_archetype_support_count,
    _preferred_hand_type,
)
from balatro_ai.bots.basic_strategy.hand_value import _card_keep_scores
from balatro_ai.bots.basic_strategy.rare_hands import _rare_hand_plan, _tarot_supports_rare_hand
from balatro_ai.rules.hand_evaluator import HandType, RANK_VALUES


def _pack_card_is_pickable(state: GameState, card: object) -> bool:
    if _is_black_hole_card(card):
        return True
    if _is_joker_card(card) and _joker_would_overfill_slots(state, _joker_from_shop_card(card)):
        return False
    return True


def _target_required_tarot_is_supported(state: GameState, card: object) -> bool:
    name = _card_label(card)
    rare_plan = _rare_hand_plan(state)
    if rare_plan is not None and _tarot_supports_rare_hand(name, rare_plan):
        return True
    preferred = _preferred_hand_type(state)
    if preferred not in FLUSH_ARCHETYPE_HANDS and _hand_archetype_support_count(state, HandType.FLUSH) <= 0:
        return False
    target_suit = SUIT_TAROT_TARGET_SUITS.get(name)
    if target_suit is None:
        return name in FLUSH_DECK_MANIPULATION_TAROTS
    dominant_suit = _dominant_suit(state)
    return dominant_suit in {None, target_suit} or sum(1 for card in state.hand if card.suit == target_suit) >= 2


def _pack_card_target_indices(state: GameState, card: object) -> tuple[int, ...]:
    if not _pack_card_requires_targets(card):
        return ()
    name = _card_label(card)
    target_suit = SUIT_TAROT_TARGET_SUITS.get(name)
    if target_suit is None or not _target_required_tarot_is_supported(state, card):
        return ()
    return _suit_tarot_target_indices(state, target_suit)


def _suit_tarot_target_indices(state: GameState, target_suit: str) -> tuple[int, ...]:
    if not state.hand:
        return ()

    candidates = [
        (index, card)
        for index, card in enumerate(state.hand)
        if card.suit != target_suit and not card.debuffed
    ]
    if not candidates:
        return ()

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    ranked = sorted(
        candidates,
        key=lambda item: (RANK_VALUES.get(item[1].rank, 0), -keep_scores[item[0]]),
        reverse=True,
    )
    return tuple(sorted(index for index, _ in ranked[:3]))
