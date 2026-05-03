"""First-pass Balatro poker hand evaluator.

This evaluator covers poker hand identification, base chips/mult, planet level
scaling, scored card chips, simple boss-blind suit debuffs, basic card
enhancements, and a small set of straightforward joker effects. Editions,
seals, retriggers, and complicated conditional jokers are intentionally left for
later phases.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import combinations
import math
import re

from balatro_ai.api.state import Card, Joker
from balatro_ai.rules.joker_compat import is_blueprint_compatible


class HandType(StrEnum):
    HIGH_CARD = "High Card"
    PAIR = "Pair"
    TWO_PAIR = "Two Pair"
    THREE_OF_A_KIND = "Three of a Kind"
    STRAIGHT = "Straight"
    FLUSH = "Flush"
    FULL_HOUSE = "Full House"
    FOUR_OF_A_KIND = "Four of a Kind"
    STRAIGHT_FLUSH = "Straight Flush"
    FIVE_OF_A_KIND = "Five of a Kind"
    FLUSH_HOUSE = "Flush House"
    FLUSH_FIVE = "Flush Five"


RANK_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}

STRAIGHT_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

BASE_HAND_VALUES = {
    HandType.HIGH_CARD: (5, 1),
    HandType.PAIR: (10, 2),
    HandType.TWO_PAIR: (20, 2),
    HandType.THREE_OF_A_KIND: (30, 3),
    HandType.STRAIGHT: (30, 4),
    HandType.FLUSH: (35, 4),
    HandType.FULL_HOUSE: (40, 4),
    HandType.FOUR_OF_A_KIND: (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.FIVE_OF_A_KIND: (120, 12),
    HandType.FLUSH_HOUSE: (140, 14),
    HandType.FLUSH_FIVE: (160, 16),
}

LEVEL_INCREMENTS = {
    HandType.HIGH_CARD: (10, 1),
    HandType.PAIR: (15, 1),
    HandType.TWO_PAIR: (20, 1),
    HandType.THREE_OF_A_KIND: (20, 2),
    HandType.STRAIGHT: (30, 3),
    HandType.FLUSH: (15, 2),
    HandType.FULL_HOUSE: (25, 2),
    HandType.FOUR_OF_A_KIND: (30, 3),
    HandType.STRAIGHT_FLUSH: (40, 4),
    HandType.FIVE_OF_A_KIND: (35, 3),
    HandType.FLUSH_HOUSE: (40, 4),
    HandType.FLUSH_FIVE: (50, 3),
}

BLIND_DEBUFFED_SUITS = {
    "The Club": frozenset({"C", "Club", "Clubs"}),
    "The Goad": frozenset({"S", "Spade", "Spades"}),
    "The Head": frozenset({"H", "Heart", "Hearts"}),
    "The Window": frozenset({"D", "Diamond", "Diamonds"}),
}

SUIT_ALIASES = {
    "C": "C",
    "Club": "C",
    "Clubs": "C",
    "S": "S",
    "Spade": "S",
    "Spades": "S",
    "H": "H",
    "Heart": "H",
    "Hearts": "H",
    "D": "D",
    "Diamond": "D",
    "Diamonds": "D",
}

SUIT_MULT_JOKERS = {
    "Greedy Joker": ("D", 3),
    "Lusty Joker": ("H", 3),
    "Wrathful Joker": ("S", 3),
    "Gluttonous Joker": ("C", 3),
}


@dataclass(frozen=True, slots=True)
class HandEvaluation:
    hand_type: HandType
    cards: tuple[Card, ...]
    scoring_indices: tuple[int, ...]
    base_chips: int
    base_mult: int
    card_chips: int
    level: int
    level_chips: int
    level_mult: int
    effect_chips: int = 0
    effect_mult: int = 0
    effect_xmult: float = 1.0
    ordered_score: int | None = None
    score_override: int | None = None
    money_delta: int = 0

    @property
    def chips(self) -> int:
        return self.base_chips + self.level_chips + self.card_chips + self.effect_chips

    @property
    def mult(self) -> int:
        return self.base_mult + self.level_mult + self.effect_mult

    @property
    def score(self) -> int:
        if self.score_override is not None:
            return self.score_override
        if self.ordered_score is not None:
            return self.ordered_score
        return _score_floor(self.chips * self.mult * self.effect_xmult)


def evaluate_played_cards(
    cards: tuple[Card, ...] | list[Card],
    hand_levels: dict[str, int] | None = None,
    debuffed_suits: set[str] | frozenset[str] | None = None,
    blind_name: str = "",
    jokers: tuple[Joker, ...] | list[Joker] = (),
    discards_remaining: int = 0,
    hands_remaining: int = 0,
    held_cards: tuple[Card, ...] | list[Card] = (),
    deck_size: int = 0,
    money: int = 0,
    played_hand_types_this_round: tuple[HandType | str, ...] | list[HandType | str] = (),
    played_hand_counts: dict[str, int] | None = None,
    stochastic_outcomes: dict[str, object] | None = None,
) -> HandEvaluation:
    """Evaluate a played hand without joker or enhancement effects."""

    original_cards = tuple(cards)
    outcomes = stochastic_outcomes or {}
    raw_jokers = tuple(jokers)
    ability_jokers = _effective_ability_jokers(raw_jokers)
    active_ability_jokers = tuple(
        ability
        for physical, ability in zip(raw_jokers, ability_jokers, strict=False)
        if not _joker_is_disabled(physical) and not _joker_is_disabled(ability)
    )
    hand_shape_cards = _cards_after_pre_score_transforms(original_cards, active_ability_jokers)
    if not hand_shape_cards:
        raise ValueError("Cannot evaluate an empty played hand")
    if len(hand_shape_cards) > 5:
        raise ValueError("Balatro hands cannot contain more than 5 played cards")

    hand_type = _identify_hand_type(hand_shape_cards, active_ability_jokers)
    score_override = _score_override_for_blind(
        hand_shape_cards,
        blind_name=blind_name,
        hand_type=hand_type,
        played_hand_types_this_round=played_hand_types_this_round,
    )
    scoring_indices = _scoring_indices(hand_shape_cards, hand_type, active_ability_jokers)
    played_cards, vampire_xmult_bonuses = _cards_after_vampire_transforms(
        hand_shape_cards,
        raw_jokers,
        scoring_indices,
    )
    base_level = max(1, int((hand_levels or {}).get(hand_type.value, 1)))
    space_joker_triggers = _outcome_int(outcomes, "space_joker_triggers") if _has_joker(active_ability_jokers, "Space Joker") else 0
    level = _adjusted_hand_level(base_level + space_joker_triggers, blind_name=blind_name)
    raw_base_chips, raw_base_mult = BASE_HAND_VALUES[hand_type]
    chip_increment, mult_increment = LEVEL_INCREMENTS[hand_type]
    level_delta = level - 1
    raw_hand_chips = raw_base_chips + (chip_increment * level_delta)
    raw_hand_mult = raw_base_mult + (mult_increment * level_delta)
    adjusted_hand_chips, adjusted_hand_mult = _adjusted_hand_values(
        raw_hand_chips,
        raw_hand_mult,
        blind_name=blind_name,
    )
    base_chips, base_mult = _adjusted_base_values(hand_type, blind_name=blind_name)
    level_chips = adjusted_hand_chips - base_chips
    level_mult = adjusted_hand_mult - base_mult
    normalized_debuffed_suits = _normalize_suits(debuffed_suits or frozenset())
    card_chips = sum(
        _card_chip_value(played_cards[index], debuffed_suits=normalized_debuffed_suits)
        for index in scoring_indices
    )
    pre_joker_chips = adjusted_hand_chips + card_chips
    pre_joker_mult = adjusted_hand_mult
    effect_chips, effect_mult, effect_xmult, ordered_score, money_delta = _effect_adjustments(
        cards=played_cards,
        hand_type=hand_type,
        scoring_indices=scoring_indices,
        debuffed_suits=normalized_debuffed_suits,
        jokers=tuple(jokers),
        discards_remaining=discards_remaining,
        hands_remaining=hands_remaining,
        held_cards=tuple(held_cards),
        deck_size=deck_size,
        money=money,
        pre_joker_chips=pre_joker_chips,
        pre_joker_mult=pre_joker_mult,
        blind_name=blind_name,
        played_hand_types_this_round=played_hand_types_this_round,
        played_hand_counts=played_hand_counts or {},
        vampire_xmult_bonuses=vampire_xmult_bonuses,
        stochastic_outcomes=outcomes,
    )

    return HandEvaluation(
        hand_type=hand_type,
        cards=played_cards,
        scoring_indices=scoring_indices,
        base_chips=base_chips,
        base_mult=base_mult,
        card_chips=card_chips,
        level=level,
        level_chips=level_chips,
        level_mult=level_mult,
        effect_chips=effect_chips,
        effect_mult=effect_mult,
        effect_xmult=effect_xmult,
        ordered_score=ordered_score,
        score_override=score_override,
        money_delta=money_delta,
    )


def best_play_from_hand(
    hand: tuple[Card, ...] | list[Card],
    hand_levels: dict[str, int] | None = None,
    max_cards: int = 5,
    debuffed_suits: set[str] | frozenset[str] | None = None,
    blind_name: str = "",
    jokers: tuple[Joker, ...] | list[Joker] = (),
    discards_remaining: int = 0,
    hands_remaining: int = 0,
    deck_size: int = 0,
    money: int = 0,
    played_hand_types_this_round: tuple[HandType | str, ...] | list[HandType | str] = (),
    played_hand_counts: dict[str, int] | None = None,
) -> HandEvaluation:
    """Return the highest immediate-score play from the available hand."""

    cards = tuple(hand)
    if not cards:
        raise ValueError("Cannot choose a best play from an empty hand")

    best: HandEvaluation | None = None
    max_size = min(max_cards, len(cards))
    for size in range(1, max_size + 1):
        for indexes in combinations(range(len(cards)), size):
            candidate_cards = tuple(cards[index] for index in indexes)
            evaluation = evaluate_played_cards(
                candidate_cards,
                hand_levels,
                debuffed_suits=debuffed_suits,
                blind_name=blind_name,
                jokers=jokers,
                discards_remaining=discards_remaining,
                hands_remaining=hands_remaining,
                held_cards=tuple(cards[index] for index in range(len(cards)) if index not in indexes),
                deck_size=deck_size,
                money=money,
                played_hand_types_this_round=played_hand_types_this_round,
                played_hand_counts=played_hand_counts,
            )
            if best is None or _evaluation_sort_key(evaluation) > _evaluation_sort_key(best):
                best = evaluation

    if best is None:
        raise RuntimeError("No candidate plays were evaluated")
    return best


def _identify_hand_type(cards: tuple[Card, ...], jokers: tuple[Joker, ...] = ()) -> HandType:
    ranked_cards = tuple(card for card in cards if not _is_stone_card(card))
    if not ranked_cards:
        return HandType.HIGH_CARD

    rank_counts = Counter(card.rank for card in ranked_cards)
    counts = sorted(rank_counts.values())
    straight_size = 4 if _has_joker(jokers, "Four Fingers") else 5
    flush_size = 4 if _has_joker(jokers, "Four Fingers") else 5
    is_flush = len(ranked_cards) >= flush_size and _flush_indices(ranked_cards, flush_size, jokers) != ()
    is_straight = len(ranked_cards) >= straight_size and _straight_indices(ranked_cards, straight_size, jokers) != ()
    max_count = max(counts)
    pair_count = counts.count(2)

    if is_flush and max_count == len(ranked_cards) and len(ranked_cards) >= 5:
        return HandType.FLUSH_FIVE
    if is_flush and len(ranked_cards) == 5 and counts == [2, 3]:
        return HandType.FLUSH_HOUSE
    if max_count == len(ranked_cards) and len(ranked_cards) >= 5:
        return HandType.FIVE_OF_A_KIND
    if is_flush and is_straight:
        return HandType.STRAIGHT_FLUSH
    if max_count == 4:
        return HandType.FOUR_OF_A_KIND
    if len(ranked_cards) == 5 and counts == [2, 3]:
        return HandType.FULL_HOUSE
    if is_flush:
        return HandType.FLUSH
    if is_straight:
        return HandType.STRAIGHT
    if max_count == 3:
        return HandType.THREE_OF_A_KIND
    if pair_count == 2:
        return HandType.TWO_PAIR
    if pair_count == 1:
        return HandType.PAIR
    return HandType.HIGH_CARD


def _is_straight(cards: tuple[Card, ...], *, shortcut: bool = False) -> bool:
    if any(_is_stone_card(card) for card in cards):
        return False
    values = sorted({STRAIGHT_VALUES[card.rank] for card in cards})
    if len(values) != len(cards):
        return False
    if len(values) == 5 and values == [2, 3, 4, 5, 14]:
        return True
    if shortcut:
        return all(1 <= right - left <= 2 for left, right in zip(values, values[1:]))
    return values == list(range(values[0], values[0] + len(values)))


def _scoring_indices(cards: tuple[Card, ...], hand_type: HandType, jokers: tuple[Joker, ...] = ()) -> tuple[int, ...]:
    if _has_joker(jokers, "Splash"):
        return tuple(range(len(cards)))

    stone_indices = tuple(index for index, card in enumerate(cards) if _is_stone_card(card))
    ranked_indices = tuple(index for index, card in enumerate(cards) if not _is_stone_card(card))

    if hand_type in {
        HandType.FULL_HOUSE,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }:
        return _with_stone_indices(ranked_indices, stone_indices, len(cards))
    if hand_type == HandType.STRAIGHT:
        return _with_stone_indices(
            _five_or_four_fingers_indices(cards, jokers, _straight_indices),
            stone_indices,
            len(cards),
        )
    if hand_type == HandType.FLUSH:
        return _with_stone_indices(
            _five_or_four_fingers_indices(cards, jokers, _flush_indices),
            stone_indices,
            len(cards),
        )
    if hand_type == HandType.STRAIGHT_FLUSH:
        return _with_stone_indices(
            _five_or_four_fingers_indices(cards, jokers, _straight_flush_indices),
            stone_indices,
            len(cards),
        )

    rank_counts = Counter(cards[index].rank for index in ranked_indices)

    if hand_type == HandType.FOUR_OF_A_KIND:
        return _with_stone_indices(
            tuple(index for index in ranked_indices if rank_counts[cards[index].rank] == 4),
            stone_indices,
            len(cards),
        )
    if hand_type == HandType.THREE_OF_A_KIND:
        return _with_stone_indices(
            tuple(index for index in ranked_indices if rank_counts[cards[index].rank] == 3),
            stone_indices,
            len(cards),
        )
    if hand_type == HandType.TWO_PAIR:
        return _with_stone_indices(
            tuple(index for index in ranked_indices if rank_counts[cards[index].rank] == 2),
            stone_indices,
            len(cards),
        )
    if hand_type == HandType.PAIR:
        return _with_stone_indices(
            tuple(index for index in ranked_indices if rank_counts[cards[index].rank] == 2),
            stone_indices,
            len(cards),
        )

    if not ranked_indices:
        return stone_indices
    highest_index = max(ranked_indices, key=lambda index: RANK_VALUES[cards[index].rank])
    return _with_stone_indices((highest_index,), stone_indices, len(cards))


def _five_or_four_fingers_indices(
    cards: tuple[Card, ...],
    jokers: tuple[Joker, ...],
    finder,
) -> tuple[int, ...]:
    five_card_indices = finder(cards, 5, jokers)
    if five_card_indices or not _has_joker(jokers, "Four Fingers"):
        return five_card_indices
    return finder(cards, 4, jokers)


def _with_stone_indices(
    indices: tuple[int, ...],
    stone_indices: tuple[int, ...],
    total_cards: int,
) -> tuple[int, ...]:
    selected = set(indices) | set(stone_indices)
    return tuple(index for index in range(total_cards) if index in selected)


def _card_chip_value(card: Card, *, debuffed_suits: frozenset[str] = frozenset()) -> int:
    if card.debuffed:
        return 0
    if _is_stone_card(card):
        return _displayed_card_chip_value(card) or 50
    if _card_is_debuffed_by_suit(card, debuffed_suits):
        return 0
    return RANK_VALUES[card.rank] + _enhancement_chips(card) + _permanent_card_chips(card)


def _card_can_score(card: Card, *, debuffed_suits: frozenset[str]) -> bool:
    if card.debuffed:
        return False
    return _is_stone_card(card) or not _card_is_debuffed_by_suit(card, debuffed_suits)


def _card_has_rank(card: Card, ranks: set[str]) -> bool:
    return not _is_stone_card(card) and card.rank in ranks


def _card_has_suit(card: Card, suit: str, *, jokers: tuple[Joker, ...] = ()) -> bool:
    return _card_is_suit_like_source(card, suit, jokers=jokers)


def _card_is_suit_like_source(
    card: Card,
    suit: str,
    *,
    jokers: tuple[Joker, ...] = (),
    bypass_debuff: bool = False,
    flush_calc: bool = False,
) -> bool:
    if flush_calc:
        if _is_stone_card(card):
            return False
        if _is_wild_enhancement(card) and not card.debuffed:
            return True
        return _card_base_suit_matches(card, suit, jokers=jokers)

    if card.debuffed and not bypass_debuff:
        return False
    if _is_stone_card(card):
        return False
    if _is_wild_enhancement(card):
        return True
    return _card_base_suit_matches(card, suit, jokers=jokers)


def _card_base_suit_matches(card: Card, suit: str, *, jokers: tuple[Joker, ...] = ()) -> bool:
    card_suit = _normalize_suit(card.suit)
    target_suit = _normalize_suit(suit)
    if _has_joker(jokers, "Smeared Joker"):
        return _smeared_suit_key(card_suit) == _smeared_suit_key(target_suit)
    return card_suit == target_suit


def _card_rank_matches(card: Card, target_rank: str) -> bool:
    return not _is_stone_card(card) and _rank_matches(card.rank, target_rank)


def _evaluation_sort_key(evaluation: HandEvaluation) -> tuple[int, int, int]:
    return (evaluation.score, evaluation.chips, evaluation.mult)


def _adjusted_base_values(hand_type: HandType, *, blind_name: str) -> tuple[int, int]:
    base_chips, base_mult = BASE_HAND_VALUES[hand_type]
    if blind_name == "The Flint":
        return math.ceil(base_chips / 2), math.ceil(base_mult / 2)
    return base_chips, base_mult


def _adjusted_hand_values(chips: int, mult: int, *, blind_name: str) -> tuple[int, int]:
    if blind_name == "The Flint":
        return math.ceil(chips / 2), math.ceil(mult / 2)
    return chips, mult


def _adjusted_hand_level(level: int, *, blind_name: str) -> int:
    if blind_name == "The Arm":
        return max(1, level - 1)
    return level


def _money_after_scoring_dollar_effects(
    money: int,
    *,
    played_cards: tuple[Card, ...],
    scored_entries: tuple[tuple[int, Card], ...],
    trigger_counts: dict[int, int],
    ability_jokers: tuple[Joker, ...],
    blind_name: str,
    hand_type: HandType,
    played_hand_types_this_round: tuple[HandType | str, ...] | list[HandType | str] = (),
    stochastic_outcomes: dict[str, object] | None = None,
) -> int:
    outcomes = stochastic_outcomes or {}
    adjusted = money
    if blind_name == "The Tooth":
        adjusted -= len(played_cards)
    scoring_triggered_cards = tuple(
        card
        for index, card in scored_entries
        for _ in range(trigger_counts.get(index, 1))
    )
    adjusted += 3 * sum(1 for card in scoring_triggered_cards if _normalize_effect_name(card.seal) == "gold")

    if _has_joker(ability_jokers, "Golden Ticket"):
        adjusted += 4 * sum(1 for card in scoring_triggered_cards if _normalize_effect_name(card.enhancement) in {"gold", "gold card"})
    if _has_joker(ability_jokers, "Rough Gem"):
        adjusted += sum(1 for card in scoring_triggered_cards if _card_has_suit(card, "D", jokers=ability_jokers))
    adjusted += 20 * max(0, _outcome_int(outcomes, "lucky_card_money_triggers"))
    if _has_joker(ability_jokers, "Business Card"):
        adjusted += 2 * max(0, _outcome_int(outcomes, "business_card_triggers"))
    if _has_joker(ability_jokers, "Reserved Parking"):
        adjusted += max(0, _outcome_int(outcomes, "reserved_parking_triggers"))
    if _has_joker(ability_jokers, "Matador") and _matador_triggered(
        played_cards,
        blind_name=blind_name,
        hand_type=hand_type,
        played_hand_types_this_round=played_hand_types_this_round,
        stochastic_outcomes=outcomes,
    ):
        adjusted += 8 * _joker_name_count(ability_jokers, "Matador")
    return adjusted


def _supernova_current_mult(hand_type: HandType, played_hand_counts: dict[str, int]) -> int:
    return _int_or_zero(played_hand_counts.get(hand_type.value)) + 1


def _matador_triggered(
    cards: tuple[Card, ...],
    *,
    blind_name: str,
    hand_type: HandType,
    played_hand_types_this_round: tuple[HandType | str, ...] | list[HandType | str] = (),
    stochastic_outcomes: dict[str, object] | None = None,
) -> bool:
    explicit = _outcome_bool_or_none(stochastic_outcomes or {}, "matador_triggered")
    if explicit is not None:
        return explicit
    if blind_name in {"The Hook", "The Flint", "The Tooth"}:
        return True
    if blind_name == "The Psychic":
        return len(cards) != 5
    played = _normalized_played_hand_types(played_hand_types_this_round)
    if blind_name == "The Eye":
        return hand_type in played
    if blind_name == "The Mouth":
        return bool(played and hand_type != played[0])
    if blind_name == "The Plant":
        return any(_is_face_card(card, ()) for card in cards)
    debuffed_suits = debuffed_suits_for_blind(blind_name)
    if debuffed_suits:
        return any(not _is_stone_card(card) and _normalize_suit(card.suit) in debuffed_suits for card in cards)
    return False


def _score_override_for_blind(
    cards: tuple[Card, ...],
    *,
    blind_name: str,
    hand_type: HandType,
    played_hand_types_this_round: tuple[HandType | str, ...] | list[HandType | str] = (),
) -> int | None:
    if blind_name == "The Psychic" and len(cards) != 5:
        return 0
    played = _normalized_played_hand_types(played_hand_types_this_round)
    if blind_name == "The Eye" and hand_type in played:
        return 0
    if blind_name == "The Mouth" and played and hand_type != played[0]:
        return 0
    return None


def _normalized_played_hand_types(raw: tuple[HandType | str, ...] | list[HandType | str]) -> tuple[HandType, ...]:
    normalized: list[HandType] = []
    for item in raw:
        if isinstance(item, HandType):
            normalized.append(item)
            continue
        try:
            normalized.append(HandType(str(item)))
        except ValueError:
            continue
    return tuple(normalized)


def _effective_ability_joker_indices(jokers: tuple[Joker, ...]) -> tuple[int, ...]:
    def resolve(index: int, seen: frozenset[int] = frozenset()) -> int:
        joker = jokers[index]
        if index in seen:
            return index
        if joker.name == "Blueprint" and index + 1 < len(jokers) and is_blueprint_compatible(jokers[index + 1]):
            return resolve(index + 1, seen | {index})
        if joker.name == "Brainstorm" and index != 0 and is_blueprint_compatible(jokers[0]):
            return resolve(0, seen | {index})
        return index

    return tuple(resolve(index) for index in range(len(jokers)))


def _effective_ability_jokers(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    return tuple(jokers[index] for index in _effective_ability_joker_indices(jokers))


def _has_joker(jokers: tuple[Joker, ...], name: str) -> bool:
    return any(joker.name == name for joker in jokers)


def _joker_name_count(jokers: tuple[Joker, ...], name: str) -> int:
    return sum(1 for joker in jokers if joker.name == name)


def _straight_indices(cards: tuple[Card, ...], size: int, jokers: tuple[Joker, ...]) -> tuple[int, ...]:
    shortcut = _has_joker(jokers, "Shortcut")
    best: tuple[int, ...] = ()
    for indexes in combinations(range(len(cards)), size):
        candidate = tuple(cards[index] for index in indexes)
        if _is_straight(candidate, shortcut=shortcut):
            if not best or _rank_sum(candidate) > _rank_sum(tuple(cards[index] for index in best)):
                best = tuple(indexes)
    return best


def _flush_indices(cards: tuple[Card, ...], size: int, jokers: tuple[Joker, ...]) -> tuple[int, ...]:
    groups: dict[str, list[int]] = {}
    for index, card in enumerate(cards):
        if _is_stone_card(card):
            continue
        for suit_key in _flush_suit_keys(card, jokers):
            groups.setdefault(suit_key, []).append(index)
    candidates = [tuple(indexes) for indexes in groups.values() if len(indexes) >= size]
    if not candidates:
        return ()
    return max(candidates, key=lambda indexes: _rank_sum(tuple(cards[index] for index in indexes)))


def _straight_flush_indices(cards: tuple[Card, ...], size: int, jokers: tuple[Joker, ...]) -> tuple[int, ...]:
    best: tuple[int, ...] = ()
    for indexes in combinations(range(len(cards)), size):
        candidate = tuple(cards[index] for index in indexes)
        if any(_is_stone_card(card) for card in candidate):
            continue
        same_suit = any(all(suit_key in _flush_suit_keys(card, jokers) for card in candidate) for suit_key in _possible_flush_keys(jokers))
        if same_suit and _is_straight(candidate, shortcut=_has_joker(jokers, "Shortcut")):
            if not best or _rank_sum(candidate) > _rank_sum(tuple(cards[index] for index in best)):
                best = tuple(indexes)
    return best


def _flush_suit_keys(card: Card, jokers: tuple[Joker, ...]) -> tuple[str, ...]:
    if _is_wild_card(card):
        return _possible_flush_keys(jokers)
    return (_flush_suit_key(card.suit, jokers),)


def _possible_flush_keys(jokers: tuple[Joker, ...]) -> tuple[str, ...]:
    if _has_joker(jokers, "Smeared Joker"):
        return ("R", "B")
    return ("S", "H", "C", "D")


def _flush_suit_key(suit: str, jokers: tuple[Joker, ...]) -> str:
    normalized = _normalize_suit(suit)
    if _has_joker(jokers, "Smeared Joker"):
        return _smeared_suit_key(normalized)
    return normalized


def _smeared_suit_key(suit: str) -> str:
    normalized = _normalize_suit(suit)
    if normalized in {"H", "D"}:
        return "R"
    if normalized in {"S", "C"}:
        return "B"
    return normalized


def _rank_sum(cards: tuple[Card, ...]) -> int:
    return sum(STRAIGHT_VALUES[card.rank] for card in cards if not _is_stone_card(card))


def _scored_card_trigger_counts(
    scored_entries: tuple[tuple[int, Card], ...],
    *,
    scoring_indices: tuple[int, ...],
    cards: tuple[Card, ...],
    jokers: tuple[Joker, ...],
    hands_remaining: int,
) -> dict[int, int]:
    counts = {index: 1 for index, _ in scored_entries}
    if not scored_entries:
        return counts

    first_scored_index = scoring_indices[0] if scoring_indices else scored_entries[0][0]
    for index, card in scored_entries:
        extra = 0
        if card.seal and _normalize_effect_name(card.seal) == "red":
            extra += 1
        if _card_has_rank(card, {"2", "3", "4", "5"}):
            extra += _joker_name_count(jokers, "Hack")
        if _is_face_card(card, jokers):
            extra += _joker_name_count(jokers, "Sock and Buskin")
        if hands_remaining == 1:
            extra += _joker_name_count(jokers, "Dusk")
        extra += _joker_name_count(jokers, "Seltzer")
        if index == first_scored_index:
            extra += 2 * _joker_name_count(jokers, "Hanging Chad")
        counts[index] += extra
    return counts


def _sum_triggers_for_cards(
    scored_entries: tuple[tuple[int, Card], ...],
    trigger_counts: dict[int, int],
    predicate,
) -> int:
    return sum(trigger_counts.get(index, 1) for index, card in scored_entries if predicate(card))


def _is_face_card(card: Card, jokers: tuple[Joker, ...]) -> bool:
    if _is_stone_card(card):
        return False
    return card.rank in {"J", "Q", "K"} or _has_joker(jokers, "Pareidolia")


def _effect_adjustments(
    *,
    cards: tuple[Card, ...],
    hand_type: HandType,
    scoring_indices: tuple[int, ...],
    debuffed_suits: frozenset[str],
    jokers: tuple[Joker, ...],
    discards_remaining: int,
    hands_remaining: int,
    held_cards: tuple[Card, ...],
    deck_size: int,
    money: int,
    pre_joker_chips: int,
    pre_joker_mult: int,
    blind_name: str,
    played_hand_types_this_round: tuple[HandType | str, ...] | list[HandType | str] = (),
    played_hand_counts: dict[str, int] | None = None,
    vampire_xmult_bonuses: tuple[float, ...] = (),
    stochastic_outcomes: dict[str, object] | None = None,
) -> tuple[int, int, float, int, int]:
    outcomes = stochastic_outcomes or {}
    ability_sources = _effective_ability_joker_indices(jokers)
    ability_pairs = tuple(zip(jokers, _effective_ability_jokers(jokers), ability_sources, strict=False))
    ability_jokers = tuple(
        ability
        for physical, ability, _source_index in ability_pairs
        if not _joker_is_disabled(physical) and not _joker_is_disabled(ability)
    )
    scored_entries = tuple(
        (index, cards[index])
        for index in scoring_indices
        if _card_can_score(cards[index], debuffed_suits=debuffed_suits)
    )
    scoring_cards = tuple(cards[index] for index in scoring_indices)
    scored_cards = tuple(card for _, card in scored_entries)
    trigger_counts = _scored_card_trigger_counts(
        scored_entries,
        scoring_indices=scoring_indices,
        cards=cards,
        jokers=ability_jokers,
        hands_remaining=hands_remaining,
    )
    effect_chips = 0
    effect_mult = 0
    effect_xmult = 1.0
    ordered_chips = pre_joker_chips
    ordered_mult = float(pre_joker_mult)
    money_for_jokers = _money_after_scoring_dollar_effects(
        money,
        played_cards=cards,
        scored_entries=scored_entries,
        trigger_counts=trigger_counts,
        ability_jokers=ability_jokers,
        blind_name=blind_name,
        hand_type=hand_type,
        played_hand_types_this_round=played_hand_types_this_round,
        stochastic_outcomes=outcomes,
    )
    played_counts = played_hand_counts or {}

    def add_chips(amount: int | float) -> None:
        nonlocal effect_chips, ordered_chips
        chip_delta = int(amount)
        effect_chips += chip_delta
        ordered_chips += chip_delta

    def add_mult(amount: int | float) -> None:
        nonlocal effect_mult, ordered_mult
        mult_delta = int(amount)
        effect_mult += mult_delta
        ordered_mult += mult_delta

    def multiply_mult(amount: float) -> None:
        nonlocal effect_xmult, ordered_mult
        effect_xmult *= amount
        ordered_mult *= amount

    hiker_chip_gain = 5 * sum(1 for joker in ability_jokers if joker.name == "Hiker")
    played_hand_types = _normalized_played_hand_types(played_hand_types_this_round)
    first_face_index = next(
        (index for index, card in scored_entries if _is_face_card(card, ability_jokers)),
        None,
    )

    def apply_scored_card_jokers(index: int, card: Card) -> None:
        for joker in ability_jokers:
            name = joker.name
            if name == "Fibonacci" and _card_has_rank(card, {"A", "2", "3", "5", "8"}):
                add_mult(8)
            elif name == "Scholar" and _card_has_rank(card, {"A"}):
                add_chips(20)
                add_mult(4)
            elif name == "Scary Face" and _is_face_card(card, ability_jokers):
                add_chips(30)
            elif name == "Arrowhead" and _card_has_suit(card, "S", jokers=ability_jokers):
                add_chips(50)
            elif name == "Even Steven" and _card_has_rank(card, {"2", "4", "6", "8", "10", "T"}):
                add_mult(4)
            elif name == "Odd Todd" and _card_has_rank(card, {"A", "3", "5", "7", "9"}):
                add_chips(31)
            elif name == "Smiley Face" and _is_face_card(card, ability_jokers):
                add_mult(5)
            elif name == "Walkie Talkie" and _card_has_rank(card, {"10", "T", "4"}):
                add_chips(10)
                add_mult(4)
            elif name == "Onyx Agate" and _card_has_suit(card, "C", jokers=ability_jokers):
                add_mult(7)
            elif name in SUIT_MULT_JOKERS:
                suit, mult = SUIT_MULT_JOKERS[name]
                if _card_has_suit(card, suit, jokers=ability_jokers):
                    add_mult(mult)
            elif name == "Ancient Joker":
                suit = _joker_target_suit(joker)
                if suit and _card_has_suit(card, suit, jokers=ability_jokers):
                    multiply_mult(1.5)
            elif name == "The Idol":
                target = _joker_target_rank_suit(joker)
                if target is not None:
                    rank, suit = target
                    if _card_rank_matches(card, rank) and _card_has_suit(card, suit, jokers=ability_jokers):
                        multiply_mult(2)
            elif name == "Triboulet" and _card_has_rank(card, {"K", "Q"}):
                multiply_mult(2)
            elif name == "Photograph" and index == first_face_index:
                multiply_mult(2)

    remaining_lucky_mult_triggers = max(0, _outcome_int(outcomes, "lucky_card_mult_triggers"))
    if remaining_lucky_mult_triggers <= 0:
        remaining_lucky_mult_triggers = max(0, _outcome_int(outcomes, "lucky_card_triggers"))
    for index, card in scored_entries:
        triggers = trigger_counts.get(index, 1)
        for trigger_index in range(triggers):
            if trigger_index > 0:
                add_chips(_card_chip_value(card) + (hiker_chip_gain * trigger_index))
            add_chips(_edition_chips(card.edition))
            add_mult(_enhancement_mult(card))
            if remaining_lucky_mult_triggers > 0 and _normalize_effect_name(card.enhancement) in {"lucky", "lucky card"}:
                add_mult(20)
                remaining_lucky_mult_triggers -= 1
            add_mult(_edition_mult(card.edition))
            multiply_mult(_enhancement_xmult(card))
            multiply_mult(_edition_xmult(card.edition))
            apply_scored_card_jokers(index, card)

    if _has_joker(ability_jokers, "Bloodstone"):
        for _ in range(max(0, _outcome_int(outcomes, "bloodstone_triggers"))):
            multiply_mult(1.5)

    mime_retrigger_count = _joker_name_count(ability_jokers, "Mime")
    lowest_held = _raised_fist_card(held_cards)
    for card in held_cards:
        held_effects = _held_card_effects(
            card,
            ability_jokers=ability_jokers,
            debuffed_suits=debuffed_suits,
            lowest_held=lowest_held,
        )
        held_retrigger_count = 1 + mime_retrigger_count
        if _normalize_effect_name(card.seal) == "red":
            held_retrigger_count += 1
        for _ in range(held_retrigger_count):
            for chips_delta, mult_delta, xmult_delta in held_effects:
                add_chips(chips_delta)
                add_mult(mult_delta)
                multiply_mult(xmult_delta)

    for physical_joker, ability_joker, source_index in ability_pairs:
        if _joker_is_disabled(physical_joker) or _joker_is_disabled(ability_joker):
            continue
        name = ability_joker.name
        add_chips(_edition_chips(physical_joker.edition))
        add_mult(_edition_mult(physical_joker.edition))
        if name == "Joker":
            add_mult(4)
        elif name == "Stuntman":
            add_chips(250)
        elif name == "Bull":
            add_chips(2 * max(0, money_for_jokers))
        elif name == "Gros Michel":
            add_mult(15)
        elif name == "Misprint":
            add_mult(_outcome_int(outcomes, "misprint_mult"))
        elif name == "Banner":
            add_chips(30 * max(0, discards_remaining))
        elif name == "Mystic Summit" and discards_remaining == 0:
            add_mult(15)
        elif name == "Abstract Joker":
            add_mult(3 * len(jokers))
        elif name == "Swashbuckler":
            add_mult(sum(other.sell_value or 0 for other in jokers if other is not physical_joker))
        elif name == "Supernova":
            add_mult(_joker_current_plus(ability_joker, suffix="mult") or _supernova_current_mult(hand_type, played_counts))
        elif name == "Bootstraps":
            add_mult(2 * (max(0, money_for_jokers) // 5))
        elif name == "Half Joker" and len(cards) <= 3:
            add_mult(20)
        elif name == "Jolly Joker" and _contains_pair(cards):
            add_mult(8)
        elif name == "Zany Joker" and _contains_three_of_a_kind(cards):
            add_mult(12)
        elif name == "Mad Joker" and _contains_two_pair(cards):
            add_mult(10)
        elif name == "Crazy Joker" and _contains_straight(hand_type):
            add_mult(12)
        elif name == "Droll Joker" and _contains_flush(hand_type):
            add_mult(10)
        elif name == "Sly Joker" and _contains_pair(cards):
            add_chips(50)
        elif name == "Wily Joker" and _contains_three_of_a_kind(cards):
            add_chips(100)
        elif name == "Clever Joker" and _contains_two_pair(cards):
            add_chips(80)
        elif name == "Devious Joker" and _contains_straight(hand_type):
            add_chips(100)
        elif name == "Crafty Joker" and _contains_flush(hand_type):
            add_chips(80)
        elif name == "The Duo" and _contains_pair(cards):
            multiply_mult(2)
        elif name == "The Trio" and _contains_three_of_a_kind(cards):
            multiply_mult(3)
        elif name == "The Family" and _contains_four_of_a_kind(cards):
            multiply_mult(4)
        elif name == "The Order" and _contains_straight(hand_type):
            multiply_mult(3)
        elif name == "The Tribe" and _contains_flush(hand_type):
            multiply_mult(2)
        elif name == "Acrobat" and hands_remaining == 1:
            multiply_mult(3)
        elif name == "Seeing Double" and _contains_scored_club_and_other_suit(scored_cards, ability_jokers):
            multiply_mult(2)
        elif name == "Flower Pot" and _contains_all_suits(scoring_cards, ability_jokers):
            multiply_mult(3)
        elif name == "Blackboard" and _blackboard_active(held_cards, ability_jokers):
            multiply_mult(3)
        elif name == "Blue Joker":
            current_chips = 0 if _joker_is_hidden(ability_joker) else _joker_current_plus(ability_joker, suffix="chips")
            add_chips(current_chips or 2 * max(0, deck_size))
        elif name == "Wee Joker":
            add_chips(_joker_current_plus(ability_joker, suffix="chips"))
            add_chips(8 * _sum_triggers_for_cards(scored_entries, trigger_counts, lambda card: _card_has_rank(card, {"2"})))
        elif name == "Runner":
            add_chips(_joker_current_plus(ability_joker, suffix="chips"))
            if _contains_straight(hand_type):
                add_chips(15)
        elif name == "Green Joker":
            add_mult(_joker_current_plus(ability_joker, suffix="mult") + 1)
        elif name == "Ride the Bus":
            if not any(_is_face_card(card, ability_jokers) for card in scored_cards):
                add_mult(_joker_current_plus(ability_joker, suffix="mult"))
                add_mult(1)
        elif name == "Spare Trousers":
            add_mult(_joker_current_plus(ability_joker, suffix="mult"))
            if _contains_two_pair(cards):
                add_mult(2)
        elif name in {"Fortune Teller", "Red Card", "Flash Card", "Ceremonial Dagger"}:
            add_mult(_joker_current_plus(ability_joker, suffix="mult"))
        elif name == "Popcorn":
            add_mult(_joker_current_plus(ability_joker, suffix="mult") or _joker_leading_plus(ability_joker, suffix="mult"))
        elif name == "Ice Cream":
            add_chips(_joker_current_plus(ability_joker, suffix="chips") or _joker_leading_plus(ability_joker, suffix="chips"))
        elif name in {"Stone Joker", "Castle"}:
            add_chips(_joker_current_plus(ability_joker, suffix="chips"))
        elif name == "Square Joker":
            add_chips(_joker_current_plus(ability_joker, suffix="chips"))
            if len(cards) == 4:
                add_chips(4)
        elif name == "Erosion":
            add_mult(_joker_current_plus(ability_joker, suffix="mult"))
        elif name == "Vampire":
            bonus = vampire_xmult_bonuses[source_index] if source_index < len(vampire_xmult_bonuses) else 0.0
            multiply_mult(_joker_current_xmult(ability_joker) + bonus)
        elif name in {"Constellation", "Madness", "Hologram", "Obelisk"}:
            multiply_mult(_joker_current_xmult(ability_joker))
        elif name == "Lucky Cat":
            current = _joker_current_xmult(ability_joker)
            current += 0.25 * max(0, _outcome_int(outcomes, "lucky_card_triggers"))
            multiply_mult(current)
        elif name in {
            "Canio",
            "Caino",
            "Yorick",
            "Ramen",
            "Campfire",
            "Throwback",
            "Steel Joker",
            "Glass Joker",
            "Joker Stencil",
            "Hit the Road",
        }:
            multiply_mult(_joker_current_xmult(ability_joker))
        elif name in {"Cavendish"}:
            multiply_mult(3)
        elif name == "Loyalty Card" and _loyalty_card_ready(ability_joker):
            multiply_mult(4)
        elif name == "Driver's License" and _drivers_license_active(ability_joker):
            multiply_mult(3)
        elif name == "Card Sharp" and hand_type in played_hand_types:
            multiply_mult(3)
        for baseball_joker in ability_jokers:
            if baseball_joker.name == "Baseball Card" and baseball_joker is not physical_joker and _joker_rarity(physical_joker) == "uncommon":
                multiply_mult(1.5)
        multiply_mult(_edition_xmult(physical_joker.edition))

    return effect_chips, effect_mult, effect_xmult, _score_floor(ordered_chips * ordered_mult), money_for_jokers - money


def _score_floor(value: float) -> int:
    nearest = round(value)
    if abs(value - nearest) <= 1e-12:
        return int(nearest)
    return int(value)


def _cards_after_pre_score_transforms(cards: tuple[Card, ...], ability_jokers: tuple[Joker, ...]) -> tuple[Card, ...]:
    if not _has_joker(ability_jokers, "Midas Mask"):
        return cards
    return tuple(
        _midas_mask_gold_card(card) if _is_face_card(card, ability_jokers) and not _is_stone_card(card) else card
        for card in cards
    )


def _cards_after_vampire_transforms(
    cards: tuple[Card, ...],
    jokers: tuple[Joker, ...],
    scoring_indices: tuple[int, ...],
) -> tuple[tuple[Card, ...], tuple[float, ...]]:
    if not any(joker.name == "Vampire" and not _joker_is_disabled(joker) for joker in jokers):
        return cards, tuple(0.0 for _ in jokers)

    updated = list(cards)
    bonuses = [0.0 for _ in jokers]
    for joker_index, joker in enumerate(jokers):
        if joker.name != "Vampire" or _joker_is_disabled(joker):
            continue
        stripped = 0
        for card_index in scoring_indices:
            card = updated[card_index]
            if _vampire_can_strip(card):
                updated[card_index] = _base_card(card)
                stripped += 1
        bonuses[joker_index] = 0.1 * stripped
    return tuple(updated), tuple(bonuses)


def _midas_mask_gold_card(card: Card) -> Card:
    metadata = dict(card.metadata)
    value = metadata.get("value")
    if isinstance(value, dict):
        value = dict(value)
        value["effect"] = f"+{RANK_VALUES.get(card.rank, 0)} chips $3 if this card is held in hand at end of round"
        metadata["value"] = value
    modifier = metadata.get("modifier")
    if isinstance(modifier, dict):
        modifier = dict(modifier)
        modifier["enhancement"] = "GOLD"
        metadata["modifier"] = modifier
    return replace(card, enhancement="GOLD", metadata=metadata)


def _vampire_can_strip(card: Card) -> bool:
    if card.debuffed:
        return False
    return _normalize_effect_name(card.enhancement) not in {"", "base", "base card"}


def _base_card(card: Card) -> Card:
    metadata = dict(card.metadata)
    modifier = metadata.get("modifier")
    if isinstance(modifier, dict):
        modifier = dict(modifier)
        modifier["enhancement"] = ""
        metadata["modifier"] = modifier
    return replace(card, enhancement=None, metadata=metadata)


def _held_card_effects(
    card: Card,
    *,
    ability_jokers: tuple[Joker, ...],
    debuffed_suits: frozenset[str],
    lowest_held: Card | None,
) -> tuple[tuple[int, int, float], ...]:
    if _is_stone_card(card):
        return ()
    if card.debuffed or _card_is_debuffed_by_suit(card, debuffed_suits):
        return ()

    effects: list[tuple[int, int, float]] = []
    xmult = _held_card_xmult(card)
    if xmult != 1.0:
        effects.append((0, 0, xmult))

    for joker in ability_jokers:
        if joker.name == "Shoot the Moon" and card.rank == "Q":
            effects.append((0, 13, 1.0))
        elif joker.name == "Raised Fist" and lowest_held == card:
            effects.append((0, 2 * RANK_VALUES[card.rank], 1.0))
        elif joker.name == "Baron" and card.rank == "K":
            effects.append((0, 0, 1.5))
    return tuple(effects)


def _raised_fist_card(held_cards: tuple[Card, ...]) -> Card | None:
    candidates = tuple(card for card in held_cards if _normalize_effect_name(card.enhancement) not in {"stone", "stone card"})
    if not candidates:
        return None
    lowest_rank = min(STRAIGHT_VALUES[card.rank] for card in candidates)
    return next(card for card in reversed(candidates) if STRAIGHT_VALUES[card.rank] == lowest_rank)


def _blackboard_active(held_cards: tuple[Card, ...], jokers: tuple[Joker, ...]) -> bool:
    return bool(held_cards) and all(_blackboard_card_is_black(card, jokers) for card in held_cards)


def _blackboard_card_is_black(card: Card, jokers: tuple[Joker, ...]) -> bool:
    if _is_stone_card(card):
        return False
    return _card_is_suit_like_source(card, "S", jokers=jokers, flush_calc=True) or _card_is_suit_like_source(
        card,
        "C",
        jokers=jokers,
        flush_calc=True,
    )


def _is_stone_card(card: Card) -> bool:
    return _normalize_effect_name(card.enhancement) in {"stone", "stone card"}


def _is_wild_card(card: Card) -> bool:
    return not card.debuffed and _is_wild_enhancement(card)


def _is_wild_enhancement(card: Card) -> bool:
    return _normalize_effect_name(card.enhancement) in {"wild", "wild card"}


def _card_is_debuffed_by_suit(card: Card, debuffed_suits: frozenset[str]) -> bool:
    if not debuffed_suits:
        return False
    if _is_wild_card(card):
        return True
    return _normalize_suit(card.suit) in debuffed_suits


def _enhancement_chips(card: Card) -> int:
    enhancement = _normalize_effect_name(card.enhancement)
    if enhancement in {"bonus", "bonus card"}:
        return 30
    return 0


def _permanent_card_chips(card: Card) -> int:
    modifier = card.metadata.get("modifier")
    value = card.metadata.get("value")
    for source in (card.metadata, modifier if isinstance(modifier, dict) else {}, value if isinstance(value, dict) else {}):
        for key in ("perma_bonus", "permanent_chips", "bonus_chips", "extra_chips"):
            if key in source:
                return _int_or_zero(source[key])
    extra = _displayed_extra_chip_value(card)
    if extra is not None:
        return extra
    displayed = _displayed_card_chip_value(card)
    if displayed is not None and card.rank in RANK_VALUES:
        intrinsic = RANK_VALUES[card.rank] + _enhancement_chips(card)
        return max(0, displayed - intrinsic)
    return 0


def _displayed_extra_chip_value(card: Card) -> int | None:
    value = card.metadata.get("value")
    if not isinstance(value, dict):
        return None
    effect = str(value.get("effect", ""))
    match = re.search(r"([+-]?\d+)\s+extra\s+chips\b", effect, flags=re.IGNORECASE)
    if not match:
        return None
    return max(0, int(match.group(1)) - _enhancement_chips(card))


def _displayed_card_chip_value(card: Card) -> int | None:
    value = card.metadata.get("value")
    if not isinstance(value, dict):
        return None
    effect = str(value.get("effect", ""))
    match = re.search(r"^\s*\+(\d+)\s+chips\b", effect, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _outcome_int(outcomes: dict[str, object], key: str, default: int = 0) -> int:
    try:
        return int(outcomes.get(key, default))
    except (TypeError, ValueError):
        return default


def _outcome_bool_or_none(outcomes: dict[str, object], key: str) -> bool | None:
    if key not in outcomes or outcomes[key] is None:
        return None
    value = outcomes[key]
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _enhancement_mult(card: Card) -> int:
    enhancement = _normalize_effect_name(card.enhancement)
    if enhancement in {"mult", "mult card"}:
        return 4
    return 0


def _enhancement_xmult(card: Card) -> float:
    enhancement = _normalize_effect_name(card.enhancement)
    if enhancement in {"glass", "glass card"}:
        return 2.0
    return 1.0


def _held_card_xmult(card: Card) -> float:
    enhancement = _normalize_effect_name(card.enhancement)
    if enhancement in {"steel", "steel card"}:
        return 1.5
    return 1.0


def _edition_chips(edition: str | None) -> int:
    edition_name = _normalize_effect_name(edition)
    if edition_name == "foil":
        return 50
    return 0


def _edition_mult(edition: str | None) -> int:
    edition_name = _normalize_effect_name(edition)
    if edition_name in {"holographic", "holo"}:
        return 10
    return 0


def _edition_xmult(edition: str | None) -> float:
    edition_name = _normalize_effect_name(edition)
    if edition_name in {"polychrome", "poly"}:
        return 1.5
    return 1.0


def _normalize_effect_name(name: str | None) -> str:
    if name is None:
        return ""
    return name.removeprefix("m_").replace("_", " ").lower()


def _joker_current_plus(joker: Joker, *, suffix: str) -> int:
    metadata_value = _metadata_current_plus_or_none(joker, suffix=suffix)
    if metadata_value is not None:
        return metadata_value
    return _current_plus_number(_joker_effect_text(joker), suffix=suffix)


def _joker_current_xmult(joker: Joker) -> float:
    metadata_value = _metadata_current_xmult_or_none(joker)
    if metadata_value is not None:
        return metadata_value
    effect = _joker_effect_text(joker)
    match = re.search(r"currently\s+x\s*([0-9]+(?:\.[0-9]+)?)", effect, flags=re.IGNORECASE)
    if match:
        visible = float(match.group(1))
        return _ramen_internal_xmult_from_visible(visible) if joker.name == "Ramen" else visible
    match = re.search(r"\bx\s*([0-9]+(?:\.[0-9]+)?)\s*mult", effect, flags=re.IGNORECASE)
    if match:
        visible = float(match.group(1))
        return _ramen_internal_xmult_from_visible(visible) if joker.name == "Ramen" else visible
    return _metadata_current_xmult(joker)


def _ramen_internal_xmult_from_visible(visible: float) -> float:
    discards = max(0, round((2.0 - visible) / 0.01))
    current = 2.0
    for _ in range(discards):
        current -= 0.01
    return max(1.0, current)


def _joker_effect_text(joker: Joker) -> str:
    value = joker.metadata.get("value")
    if isinstance(value, dict):
        return str(value.get("effect", ""))
    return str(joker.metadata.get("effect", ""))


def _joker_is_disabled(joker: Joker) -> bool:
    state = joker.metadata.get("state")
    if isinstance(state, dict) and state.get("debuff"):
        return True
    return "all abilities are disabled" in _joker_effect_text(joker).lower()


def _joker_is_hidden(joker: Joker) -> bool:
    state = joker.metadata.get("state")
    return isinstance(state, dict) and bool(state.get("hidden"))


def _current_plus_number(text: str, *, suffix: str) -> int:
    normalized = text.replace("$", " ").replace("(", " ").replace(")", " ")
    match = re.search(
        rf"currently\s+([+-]?\d+)(?:\s+{re.escape(suffix)})?",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return 0
    return int(match.group(1))


def _joker_leading_plus(joker: Joker, *, suffix: str) -> int:
    effect = _joker_effect_text(joker)
    match = re.match(rf"\s*([+-]?\d+)\s+{re.escape(suffix)}\b", effect, flags=re.IGNORECASE)
    if not match:
        return 0
    return int(match.group(1))


def _metadata_current_plus(joker: Joker, *, suffix: str) -> int:
    value = _metadata_current_plus_or_none(joker, suffix=suffix)
    return 0 if value is None else value


def _metadata_current_plus_or_none(joker: Joker, *, suffix: str) -> int | None:
    keys = {
        "chips": ("current_chips", "chips"),
        "mult": ("current_mult", "mult"),
    }.get(suffix, (f"current_{suffix}", suffix))
    for mapping in _metadata_numeric_sources(joker.metadata):
        for key in keys:
            if key in mapping:
                value = _int_or_none(mapping[key])
                if value is not None:
                    return value
    return None


def _metadata_current_xmult(joker: Joker) -> float:
    value = _metadata_current_xmult_or_none(joker)
    return 1.0 if value is None else value


def _metadata_current_xmult_or_none(joker: Joker) -> float | None:
    keys = ("current_xmult", "xmult", "current_x_mult", "x_mult")
    for mapping in _metadata_numeric_sources(joker.metadata):
        for key in keys:
            if key in mapping:
                value = _float_or_none(mapping[key])
                if value is not None:
                    return value
    return None


def _metadata_numeric_sources(metadata: dict[str, object]) -> tuple[dict[str, object], ...]:
    sources: list[dict[str, object]] = [metadata]
    for key in ("ability", "config", "extra"):
        value = metadata.get(key)
        if isinstance(value, dict):
            sources.append(value)
            nested_extra = value.get("extra")
            if isinstance(nested_extra, dict):
                sources.append(nested_extra)
    return tuple(sources)


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _loyalty_card_ready(joker: Joker) -> bool:
    remaining = _metadata_loyalty_remaining(joker)
    if remaining is not None:
        return remaining == 0
    text = _joker_effect_text(joker)
    remaining_match = re.search(r"(\d+)\s+remaining", text, flags=re.IGNORECASE)
    if remaining_match:
        return int(remaining_match.group(1)) == 0
    lowered = text.lower()
    return "ready" in lowered or "active" in lowered


def _metadata_loyalty_remaining(joker: Joker) -> int | None:
    for mapping in _metadata_numeric_sources(joker.metadata):
        for key in ("current_remaining", "remaining", "hands_remaining", "hands_left"):
            if key in mapping:
                value = _int_or_none(mapping[key])
                if value is not None:
                    return value
    return None


def _drivers_license_active(joker: Joker) -> bool:
    for mapping in _metadata_numeric_sources(joker.metadata):
        for key in ("driver_tally", "enhanced_count", "current_enhanced", "enhancement_tally"):
            if key in mapping:
                value = _int_or_none(mapping[key])
                if value is not None:
                    return value >= 16
    text = _joker_effect_text(joker)
    match = re.search(r"currently\s+(\d+)", text, flags=re.IGNORECASE)
    return bool(match and int(match.group(1)) >= 16)


def _joker_target_suit(joker: Joker) -> str | None:
    for mapping in _metadata_numeric_sources(joker.metadata):
        for key in ("current_suit", "target_suit", "suit"):
            value = mapping.get(key)
            if isinstance(value, str) and value:
                return _normalize_suit(value)
    text = _joker_effect_text(joker)
    for name, suit in (
        ("spade", "S"),
        ("spades", "S"),
        ("heart", "H"),
        ("hearts", "H"),
        ("club", "C"),
        ("clubs", "C"),
        ("diamond", "D"),
        ("diamonds", "D"),
    ):
        if re.search(rf"\b{name}\b", text, flags=re.IGNORECASE):
            return suit
    return None


def _joker_target_rank_suit(joker: Joker) -> tuple[str, str] | None:
    metadata_rank: str | None = None
    for mapping in _metadata_numeric_sources(joker.metadata):
        for key in ("current_rank", "target_rank", "rank"):
            value = mapping.get(key)
            if isinstance(value, str) and value:
                metadata_rank = _rank_from_text(value) or _normalize_rank(value)
                break
        if metadata_rank:
            break
    metadata_suit = _joker_target_suit(joker)
    if metadata_rank and metadata_suit:
        return metadata_rank, metadata_suit

    text = _joker_effect_text(joker)
    rank_match = re.search(
        r"\b(ace|king|queen|jack|10|ten|9|nine|8|eight|7|seven|6|six|5|five|4|four|3|three|2|two)\b",
        text,
        flags=re.IGNORECASE,
    )
    suit = _joker_target_suit(joker)
    if not rank_match or not suit:
        return None
    rank = _rank_from_text(rank_match.group(1))
    return (rank, suit) if rank else None


def _rank_from_text(text: str) -> str | None:
    return {
        "ace": "A",
        "king": "K",
        "queen": "Q",
        "jack": "J",
        "10": "10",
        "ten": "10",
        "9": "9",
        "nine": "9",
        "8": "8",
        "eight": "8",
        "7": "7",
        "seven": "7",
        "6": "6",
        "six": "6",
        "5": "5",
        "five": "5",
        "4": "4",
        "four": "4",
        "3": "3",
        "three": "3",
        "2": "2",
        "two": "2",
    }.get(text.lower())


def _normalize_rank(rank: str) -> str:
    value = rank.strip().lower()
    return _rank_from_text(value) or {"a": "A", "k": "K", "q": "Q", "j": "J", "t": "10"}.get(value, value.upper())


def _rank_matches(card_rank: str, target_rank: str) -> bool:
    if target_rank == "10":
        return card_rank in {"10", "T"}
    return card_rank == target_rank


def _uncommon_joker_count(jokers: tuple[Joker, ...]) -> int:
    return sum(1 for joker in jokers if _joker_rarity(joker) == "uncommon")


JOKER_RARITY_FALLBACKS = {
    "8 Ball": "common",
    "Abstract Joker": "common",
    "Acrobat": "uncommon",
    "Ancient Joker": "rare",
    "Arrowhead": "uncommon",
    "Astronomer": "uncommon",
    "Banner": "common",
    "Baron": "rare",
    "Baseball Card": "rare",
    "Blackboard": "uncommon",
    "Bloodstone": "uncommon",
    "Blue Joker": "common",
    "Blueprint": "rare",
    "Bootstraps": "uncommon",
    "Brainstorm": "rare",
    "Bull": "uncommon",
    "Burglar": "uncommon",
    "Burnt Joker": "rare",
    "Business Card": "common",
    "Caino": "legendary",
    "Campfire": "rare",
    "Card Sharp": "uncommon",
    "Cartomancer": "uncommon",
    "Castle": "uncommon",
    "Cavendish": "common",
    "Ceremonial Dagger": "uncommon",
    "Certificate": "uncommon",
    "Chaos the Clown": "common",
    "Chicot": "legendary",
    "Clever Joker": "common",
    "Cloud 9": "uncommon",
    "Constellation": "uncommon",
    "Crafty Joker": "common",
    "Crazy Joker": "common",
    "Credit Card": "common",
    "Delayed Gratification": "common",
    "Devious Joker": "common",
    "Diet Cola": "uncommon",
    "DNA": "rare",
    "Driver's License": "rare",
    "Droll Joker": "common",
    "Drunkard": "common",
    "Dusk": "uncommon",
    "Egg": "common",
    "Erosion": "uncommon",
    "Even Steven": "common",
    "Faceless Joker": "common",
    "Fibonacci": "uncommon",
    "Flash Card": "uncommon",
    "Flower Pot": "uncommon",
    "Fortune Teller": "common",
    "Four Fingers": "uncommon",
    "Gift Card": "uncommon",
    "Glass Joker": "uncommon",
    "Gluttonous Joker": "common",
    "Golden Joker": "common",
    "Golden Ticket": "common",
    "Greedy Joker": "common",
    "Green Joker": "common",
    "Gros Michel": "common",
    "Hack": "uncommon",
    "Half Joker": "common",
    "Hallucination": "common",
    "Hanging Chad": "common",
    "Hiker": "uncommon",
    "Hit the Road": "rare",
    "Hologram": "uncommon",
    "Ice Cream": "common",
    "Invisible Joker": "rare",
    "Joker": "common",
    "Joker Stencil": "uncommon",
    "Jolly Joker": "common",
    "Juggler": "common",
    "Loyalty Card": "uncommon",
    "Luchador": "uncommon",
    "Lucky Cat": "uncommon",
    "Lusty Joker": "common",
    "Mad Joker": "common",
    "Madness": "uncommon",
    "Mail-In Rebate": "common",
    "Marble Joker": "uncommon",
    "Matador": "uncommon",
    "Merry Andy": "uncommon",
    "Midas Mask": "uncommon",
    "Mime": "uncommon",
    "Misprint": "common",
    "Mr. Bones": "uncommon",
    "Mystic Summit": "common",
    "Obelisk": "rare",
    "Odd Todd": "common",
    "Onyx Agate": "uncommon",
    "Oops! All 6s": "uncommon",
    "Pareidolia": "uncommon",
    "Perkeo": "legendary",
    "Photograph": "common",
    "Popcorn": "common",
    "Raised Fist": "common",
    "Ramen": "uncommon",
    "Red Card": "common",
    "Reserved Parking": "common",
    "Ride the Bus": "common",
    "Riff-raff": "common",
    "Rocket": "uncommon",
    "Rough Gem": "uncommon",
    "Runner": "common",
    "Satellite": "uncommon",
    "Scary Face": "common",
    "Scholar": "common",
    "Seance": "uncommon",
    "Seeing Double": "uncommon",
    "Seltzer": "uncommon",
    "Shoot the Moon": "common",
    "Shortcut": "uncommon",
    "Showman": "uncommon",
    "Sixth Sense": "uncommon",
    "Sly Joker": "common",
    "Smeared Joker": "uncommon",
    "Smiley Face": "common",
    "Sock and Buskin": "uncommon",
    "Space Joker": "uncommon",
    "Spare Trousers": "uncommon",
    "Splash": "common",
    "Square Joker": "common",
    "Steel Joker": "uncommon",
    "Stone Joker": "uncommon",
    "Stuntman": "rare",
    "Supernova": "common",
    "Superposition": "common",
    "Swashbuckler": "common",
    "The Duo": "rare",
    "The Family": "rare",
    "The Idol": "uncommon",
    "The Order": "rare",
    "The Tribe": "rare",
    "The Trio": "rare",
    "Throwback": "uncommon",
    "To Do List": "common",
    "To the Moon": "uncommon",
    "Trading Card": "uncommon",
    "Triboulet": "legendary",
    "Troubadour": "uncommon",
    "Turtle Bean": "uncommon",
    "Vagabond": "rare",
    "Vampire": "uncommon",
    "Walkie Talkie": "common",
    "Wee Joker": "rare",
    "Wily Joker": "common",
    "Wrathful Joker": "common",
    "Yorick": "legendary",
    "Zany Joker": "common",
}


def _joker_rarity(joker: Joker) -> str:
    value = joker.metadata.get("value")
    candidates = [
        joker.metadata.get("rarity"),
        joker.metadata.get("rarity_name"),
        value.get("rarity") if isinstance(value, dict) else None,
    ]
    for candidate in candidates:
        normalized = _normalize_joker_rarity(candidate)
        if normalized:
            return normalized
    if joker.name in JOKER_RARITY_FALLBACKS:
        return JOKER_RARITY_FALLBACKS[joker.name]
    return ""


def _normalize_joker_rarity(candidate: object) -> str:
    if candidate is None:
        return ""
    text = str(candidate).strip().lower()
    return {
        "1": "common",
        "1.0": "common",
        "common": "common",
        "2": "uncommon",
        "2.0": "uncommon",
        "uncommon": "uncommon",
        "3": "rare",
        "3.0": "rare",
        "rare": "rare",
        "4": "legendary",
        "4.0": "legendary",
        "legendary": "legendary",
    }.get(text, text)


def _contains_scored_club_and_other_suit(cards: tuple[Card, ...], jokers: tuple[Joker, ...]) -> bool:
    suits = {suit: 0 for suit in ("H", "D", "S", "C")}
    for card in cards:
        if _is_wild_enhancement(card):
            continue
        for suit in suits:
            if _card_is_suit_like_source(card, suit, jokers=jokers):
                suits[suit] += 1
    for card in cards:
        if not _is_wild_enhancement(card):
            continue
        for suit in ("C", "D", "S", "H"):
            if _card_is_suit_like_source(card, suit, jokers=jokers) and suits[suit] == 0:
                suits[suit] += 1
                break
    return (suits["H"] > 0 or suits["D"] > 0 or suits["S"] > 0) and suits["C"] > 0


def _contains_all_suits(cards: tuple[Card, ...], jokers: tuple[Joker, ...]) -> bool:
    suits = {suit: 0 for suit in ("H", "D", "S", "C")}
    for card in cards:
        if _is_wild_enhancement(card):
            continue
        for suit in ("H", "D", "S", "C"):
            if _card_is_suit_like_source(card, suit, jokers=jokers, bypass_debuff=True) and suits[suit] == 0:
                suits[suit] += 1
                break
    for card in cards:
        if not _is_wild_enhancement(card):
            continue
        for suit in ("H", "D", "S", "C"):
            if _card_is_suit_like_source(card, suit, jokers=jokers) and suits[suit] == 0:
                suits[suit] += 1
                break
    return all(count > 0 for count in suits.values())


def _contains_pair(cards: tuple[Card, ...]) -> bool:
    rank_counts = _non_stone_rank_counts(cards)
    return bool(rank_counts) and max(rank_counts.values()) >= 2


def _contains_two_pair(cards: tuple[Card, ...]) -> bool:
    return sum(1 for count in _non_stone_rank_counts(cards).values() if count >= 2) >= 2


def _contains_three_of_a_kind(cards: tuple[Card, ...]) -> bool:
    rank_counts = _non_stone_rank_counts(cards)
    return bool(rank_counts) and max(rank_counts.values()) >= 3


def _contains_four_of_a_kind(cards: tuple[Card, ...]) -> bool:
    rank_counts = _non_stone_rank_counts(cards)
    return bool(rank_counts) and max(rank_counts.values()) >= 4


def _non_stone_rank_counts(cards: tuple[Card, ...]) -> Counter[str]:
    return Counter(card.rank for card in cards if not _is_stone_card(card))


def _contains_straight(hand_type: HandType) -> bool:
    return hand_type in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}


def _contains_flush(hand_type: HandType) -> bool:
    return hand_type in {
        HandType.FLUSH,
        HandType.STRAIGHT_FLUSH,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }


def debuffed_suits_for_blind(blind_name: str) -> frozenset[str]:
    return _normalize_suits(BLIND_DEBUFFED_SUITS.get(blind_name, frozenset()))


def _normalize_suits(suits: set[str] | frozenset[str]) -> frozenset[str]:
    return frozenset(_normalize_suit(suit) for suit in suits)


def _normalize_suit(suit: str) -> str:
    return SUIT_ALIASES.get(suit, suit)
