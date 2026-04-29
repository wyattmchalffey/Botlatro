"""First-pass Balatro poker hand evaluator.

This evaluator covers poker hand identification, base chips/mult, planet level
scaling, scored card chips, simple boss-blind suit debuffs, basic card
enhancements, and a small set of straightforward joker effects. Editions,
seals, retriggers, and complicated conditional jokers are intentionally left for
later phases.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
import re

from balatro_ai.api.state import Card, Joker


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
    score_override: int | None = None

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
        return int(self.chips * self.mult * self.effect_xmult)


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
) -> HandEvaluation:
    """Evaluate a played hand without joker or enhancement effects."""

    played_cards = tuple(cards)
    if not played_cards:
        raise ValueError("Cannot evaluate an empty played hand")
    if len(played_cards) > 5:
        raise ValueError("Balatro hands cannot contain more than 5 played cards")

    ability_jokers = _effective_ability_jokers(tuple(jokers))
    hand_type = _identify_hand_type(played_cards, ability_jokers)
    score_override = _score_override_for_blind(played_cards, blind_name=blind_name)
    scoring_indices = _scoring_indices(played_cards, hand_type, ability_jokers)
    level = max(1, int((hand_levels or {}).get(hand_type.value, 1)))
    base_chips, base_mult = _adjusted_base_values(hand_type, blind_name=blind_name)
    chip_increment, mult_increment = LEVEL_INCREMENTS[hand_type]
    level_delta = level - 1
    normalized_debuffed_suits = _normalize_suits(debuffed_suits or frozenset())
    card_chips = sum(
        _card_chip_value(played_cards[index], debuffed_suits=normalized_debuffed_suits)
        for index in scoring_indices
    )
    effect_chips, effect_mult, effect_xmult = _effect_adjustments(
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
        pre_joker_mult=base_mult + (mult_increment * level_delta),
    )

    return HandEvaluation(
        hand_type=hand_type,
        cards=played_cards,
        scoring_indices=scoring_indices,
        base_chips=base_chips,
        base_mult=base_mult,
        card_chips=card_chips,
        level=level,
        level_chips=chip_increment * level_delta,
        level_mult=mult_increment * level_delta,
        effect_chips=effect_chips,
        effect_mult=effect_mult,
        effect_xmult=effect_xmult,
        score_override=score_override,
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
            )
            if best is None or _evaluation_sort_key(evaluation) > _evaluation_sort_key(best):
                best = evaluation

    if best is None:
        raise RuntimeError("No candidate plays were evaluated")
    return best


def _identify_hand_type(cards: tuple[Card, ...], jokers: tuple[Joker, ...] = ()) -> HandType:
    rank_counts = Counter(card.rank for card in cards)
    counts = sorted(rank_counts.values())
    straight_size = 4 if _has_joker(jokers, "Four Fingers") else 5
    flush_size = 4 if _has_joker(jokers, "Four Fingers") else 5
    is_flush = len(cards) >= flush_size and _flush_indices(cards, flush_size, jokers) != ()
    is_straight = len(cards) >= straight_size and _straight_indices(cards, straight_size, jokers) != ()
    max_count = max(counts)
    pair_count = counts.count(2)

    if is_flush and max_count == len(cards) and len(cards) >= 5:
        return HandType.FLUSH_FIVE
    if is_flush and len(cards) == 5 and counts == [2, 3]:
        return HandType.FLUSH_HOUSE
    if max_count == len(cards) and len(cards) >= 5:
        return HandType.FIVE_OF_A_KIND
    if is_flush and is_straight:
        return HandType.STRAIGHT_FLUSH
    if max_count == 4:
        return HandType.FOUR_OF_A_KIND
    if len(cards) == 5 and counts == [2, 3]:
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

    if hand_type in {
        HandType.FULL_HOUSE,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }:
        return tuple(range(len(cards)))
    if hand_type == HandType.STRAIGHT:
        return _straight_indices(cards, 4 if _has_joker(jokers, "Four Fingers") else 5, jokers)
    if hand_type == HandType.FLUSH:
        return _flush_indices(cards, 4 if _has_joker(jokers, "Four Fingers") else 5, jokers)
    if hand_type == HandType.STRAIGHT_FLUSH:
        size = 4 if _has_joker(jokers, "Four Fingers") else 5
        return _straight_flush_indices(cards, size, jokers)

    rank_counts = Counter(card.rank for card in cards)

    if hand_type == HandType.FOUR_OF_A_KIND:
        return tuple(index for index, card in enumerate(cards) if rank_counts[card.rank] == 4)
    if hand_type == HandType.THREE_OF_A_KIND:
        return tuple(index for index, card in enumerate(cards) if rank_counts[card.rank] == 3)
    if hand_type == HandType.TWO_PAIR:
        return tuple(index for index, card in enumerate(cards) if rank_counts[card.rank] == 2)
    if hand_type == HandType.PAIR:
        return tuple(index for index, card in enumerate(cards) if rank_counts[card.rank] == 2)

    highest_index = max(range(len(cards)), key=lambda index: RANK_VALUES[cards[index].rank])
    return (highest_index,)


def _card_chip_value(card: Card, *, debuffed_suits: frozenset[str] = frozenset()) -> int:
    if card.debuffed or _normalize_suit(card.suit) in debuffed_suits:
        return 0
    return RANK_VALUES[card.rank] + _enhancement_chips(card) + _permanent_card_chips(card)


def _evaluation_sort_key(evaluation: HandEvaluation) -> tuple[int, int, int]:
    return (evaluation.score, evaluation.chips, evaluation.mult)


def _adjusted_base_values(hand_type: HandType, *, blind_name: str) -> tuple[int, int]:
    base_chips, base_mult = BASE_HAND_VALUES[hand_type]
    if blind_name == "The Flint":
        return base_chips // 2, base_mult // 2
    return base_chips, base_mult


def _score_override_for_blind(cards: tuple[Card, ...], *, blind_name: str) -> int | None:
    if blind_name == "The Psychic" and len(cards) != 5:
        return 0
    return None


def _effective_ability_jokers(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    def resolve(index: int, seen: frozenset[int] = frozenset()) -> Joker:
        joker = jokers[index]
        if index in seen:
            return joker
        if joker.name == "Blueprint" and index + 1 < len(jokers):
            return resolve(index + 1, seen | {index})
        if joker.name == "Brainstorm" and index != 0:
            return resolve(0, seen | {index})
        return joker

    return tuple(resolve(index) for index in range(len(jokers)))


def _has_joker(jokers: tuple[Joker, ...], name: str) -> bool:
    return any(joker.name == name for joker in jokers)


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
        groups.setdefault(_flush_suit_key(card.suit, jokers), []).append(index)
    candidates = [tuple(indexes) for indexes in groups.values() if len(indexes) >= size]
    if not candidates:
        return ()
    return max(candidates, key=lambda indexes: _rank_sum(tuple(cards[index] for index in indexes)))


def _straight_flush_indices(cards: tuple[Card, ...], size: int, jokers: tuple[Joker, ...]) -> tuple[int, ...]:
    best: tuple[int, ...] = ()
    for indexes in combinations(range(len(cards)), size):
        candidate = tuple(cards[index] for index in indexes)
        same_suit = len({_flush_suit_key(card.suit, jokers) for card in candidate}) == 1
        if same_suit and _is_straight(candidate, shortcut=_has_joker(jokers, "Shortcut")):
            if not best or _rank_sum(candidate) > _rank_sum(tuple(cards[index] for index in best)):
                best = tuple(indexes)
    return best


def _flush_suit_key(suit: str, jokers: tuple[Joker, ...]) -> str:
    normalized = _normalize_suit(suit)
    if _has_joker(jokers, "Smeared Joker"):
        if normalized in {"H", "D"}:
            return "R"
        if normalized in {"S", "C"}:
            return "B"
    return normalized


def _rank_sum(cards: tuple[Card, ...]) -> int:
    return sum(STRAIGHT_VALUES[card.rank] for card in cards)


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
        if _has_joker(jokers, "Hack") and card.rank in {"2", "3", "4", "5"}:
            extra += 1
        if _has_joker(jokers, "Sock and Buskin") and _is_face_card(card, jokers):
            extra += 1
        if _has_joker(jokers, "Dusk") and hands_remaining == 1:
            extra += 1
        if _has_joker(jokers, "Seltzer"):
            extra += 1
        if _has_joker(jokers, "Hanging Chad") and index == first_scored_index:
            extra += 2
        counts[index] += extra
    return counts


def _sum_triggers_for_cards(
    scored_entries: tuple[tuple[int, Card], ...],
    trigger_counts: dict[int, int],
    predicate,
) -> int:
    return sum(trigger_counts.get(index, 1) for index, card in scored_entries if predicate(card))


def _is_face_card(card: Card, jokers: tuple[Joker, ...]) -> bool:
    return card.rank in {"J", "Q", "K"} or _has_joker(jokers, "Pareidolia")


def _first_scored_face_entry(
    scored_entries: tuple[tuple[int, Card], ...],
    jokers: tuple[Joker, ...],
) -> tuple[int, Card] | None:
    for entry in scored_entries:
        if _is_face_card(entry[1], jokers):
            return entry
    return None


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
    pre_joker_mult: int,
) -> tuple[int, int, float]:
    ability_pairs = tuple(zip(jokers, _effective_ability_jokers(jokers), strict=False))
    ability_jokers = tuple(ability for _, ability in ability_pairs)
    scored_entries = tuple(
        (index, cards[index])
        for index in scoring_indices
        if not cards[index].debuffed and _normalize_suit(cards[index].suit) not in debuffed_suits
    )
    scoring_cards = tuple(cards[index] for index in scoring_indices)
    scored_cards = tuple(card for _, card in scored_entries)
    active_held_cards = tuple(
        card for card in held_cards if not card.debuffed and _normalize_suit(card.suit) not in debuffed_suits
    )
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
    for index, card in scored_entries:
        triggers = trigger_counts.get(index, 1)
        effect_chips += _card_chip_value(card) * (triggers - 1)
        effect_chips += _edition_chips(card.edition) * triggers
        effect_mult += _enhancement_mult(card) * triggers
        effect_mult += _edition_mult(card.edition) * triggers
        effect_xmult *= _enhancement_xmult(card) ** triggers
        effect_xmult *= _edition_xmult(card.edition) ** triggers

    held_retrigger_count = 2 if _has_joker(ability_jokers, "Mime") else 1
    for card in active_held_cards:
        effect_xmult *= _held_card_xmult(card) ** held_retrigger_count

    photograph_consumed = False
    for physical_joker, ability_joker in ability_pairs:
        name = ability_joker.name
        effect_chips += _edition_chips(physical_joker.edition)
        effect_mult += _edition_mult(physical_joker.edition)
        effect_xmult *= _edition_xmult(physical_joker.edition)
        if name == "Joker":
            effect_mult += 4
        elif name == "Stuntman":
            effect_chips += 250
        elif name == "Bull":
            effect_chips += 2 * max(0, money)
        elif name == "Gros Michel":
            effect_mult += 15
        elif name == "Banner":
            effect_chips += 30 * max(0, discards_remaining)
        elif name == "Mystic Summit" and discards_remaining == 0:
            effect_mult += 15
        elif name == "Abstract Joker":
            effect_mult += 3 * len(jokers)
        elif name == "Swashbuckler":
            effect_mult += sum(other.sell_value or 0 for other in jokers if other is not physical_joker)
        elif name == "Supernova":
            effect_mult += _joker_current_plus(ability_joker, suffix="mult")
        elif name == "Bootstraps":
            effect_mult += 2 * (max(0, money) // 5)
        elif name == "Fibonacci":
            effect_mult += _sum_triggers_for_cards(scored_entries, trigger_counts, lambda card: card.rank in {"A", "2", "3", "5", "8"}) * 8
        elif name == "Scholar":
            ace_count = _sum_triggers_for_cards(scored_entries, trigger_counts, lambda card: card.rank == "A")
            effect_chips += 20 * ace_count
            effect_mult += 4 * ace_count
        elif name == "Scary Face":
            effect_chips += 30 * _sum_triggers_for_cards(
                scored_entries, trigger_counts, lambda card: _is_face_card(card, ability_jokers)
            )
        elif name == "Arrowhead":
            effect_chips += 50 * _sum_triggers_for_cards(
                scored_entries, trigger_counts, lambda card: _normalize_suit(card.suit) == "S"
            )
        elif name == "Even Steven":
            effect_mult += 4 * _sum_triggers_for_cards(
                scored_entries, trigger_counts, lambda card: card.rank in {"2", "4", "6", "8", "10", "T"}
            )
        elif name == "Half Joker" and len(cards) <= 3:
            effect_mult += 20
        elif name == "Odd Todd":
            effect_chips += 31 * _sum_triggers_for_cards(
                scored_entries, trigger_counts, lambda card: card.rank in {"A", "3", "5", "7", "9"}
            )
        elif name == "Smiley Face":
            effect_mult += 5 * _sum_triggers_for_cards(
                scored_entries, trigger_counts, lambda card: _is_face_card(card, ability_jokers)
            )
        elif name == "Walkie Talkie":
            walkie_count = _sum_triggers_for_cards(scored_entries, trigger_counts, lambda card: card.rank in {"10", "T", "4"})
            effect_chips += 10 * walkie_count
            effect_mult += 4 * walkie_count
        elif name == "Onyx Agate":
            effect_mult += 7 * _sum_triggers_for_cards(
                scored_entries, trigger_counts, lambda card: _normalize_suit(card.suit) == "C"
            )
        elif name in SUIT_MULT_JOKERS:
            suit, mult = SUIT_MULT_JOKERS[name]
            effect_mult += mult * _sum_triggers_for_cards(
                scored_entries, trigger_counts, lambda card: _normalize_suit(card.suit) == suit
            )
        elif name == "Jolly Joker" and _contains_pair(cards):
            effect_mult += 8
        elif name == "Zany Joker" and _contains_three_of_a_kind(cards):
            effect_mult += 12
        elif name == "Mad Joker" and _contains_two_pair(cards):
            effect_mult += 10
        elif name == "Crazy Joker" and _contains_straight(hand_type):
            effect_mult += 12
        elif name == "Droll Joker" and _contains_flush(hand_type):
            effect_mult += 10
        elif name == "Sly Joker" and _contains_pair(cards):
            effect_chips += 50
        elif name == "Wily Joker" and _contains_three_of_a_kind(cards):
            effect_chips += 100
        elif name == "Clever Joker" and _contains_two_pair(cards):
            effect_chips += 80
        elif name == "Devious Joker" and _contains_straight(hand_type):
            effect_chips += 100
        elif name == "Crafty Joker" and _contains_flush(hand_type):
            effect_chips += 80
        elif name == "The Duo" and _contains_pair(cards):
            effect_xmult *= 2
        elif name == "The Trio" and _contains_three_of_a_kind(cards):
            effect_xmult *= 3
        elif name == "The Family" and _contains_four_of_a_kind(cards):
            effect_xmult *= 4
        elif name == "The Order" and _contains_straight(hand_type):
            effect_xmult *= 3
        elif name == "The Tribe" and _contains_flush(hand_type):
            effect_xmult *= 2
        elif name == "Acrobat" and hands_remaining == 1:
            effect_xmult *= 3
        elif name == "Seeing Double" and _contains_scored_club_and_other_suit(scored_cards):
            effect_xmult *= 2
        elif name == "Flower Pot" and _contains_all_suits(scoring_cards):
            effect_xmult *= 3
        elif name == "Shoot the Moon":
            effect_mult += 13 * held_retrigger_count * sum(1 for card in active_held_cards if card.rank == "Q")
        elif name == "Raised Fist" and held_cards:
            lowest_rank = min(STRAIGHT_VALUES[card.rank] for card in held_cards)
            lowest_held = next(card for card in reversed(held_cards) if STRAIGHT_VALUES[card.rank] == lowest_rank)
            if not lowest_held.debuffed and _normalize_suit(lowest_held.suit) not in debuffed_suits:
                effect_mult += 2 * RANK_VALUES[lowest_held.rank]
        elif name == "Baron":
            effect_xmult *= 1.5 ** (held_retrigger_count * sum(1 for card in active_held_cards if card.rank == "K"))
        elif name == "Blackboard" and active_held_cards and all(
            _normalize_suit(card.suit) in {"S", "C"} for card in active_held_cards
        ):
            effect_xmult *= 3
        elif name == "Blue Joker" and deck_size > 0:
            effect_chips += 2 * deck_size
        elif name == "Wee Joker":
            effect_chips += _joker_current_plus(ability_joker, suffix="chips")
            effect_chips += 8 * _sum_triggers_for_cards(scored_entries, trigger_counts, lambda card: card.rank == "2")
        elif name == "Runner":
            effect_chips += _joker_current_plus(ability_joker, suffix="chips")
            if _contains_straight(hand_type):
                effect_chips += 15
        elif name == "Green Joker":
            effect_mult += _joker_current_plus(ability_joker, suffix="mult") + 1
        elif name == "Ride the Bus":
            if not any(_is_face_card(card, ability_jokers) for card in scored_cards):
                effect_mult += _joker_current_plus(ability_joker, suffix="mult")
                effect_mult += 1
        elif name == "Spare Trousers":
            effect_mult += _joker_current_plus(ability_joker, suffix="mult")
            if hand_type == HandType.TWO_PAIR:
                effect_mult += 2
        elif name in {"Fortune Teller", "Red Card", "Flash Card", "Popcorn", "Ceremonial Dagger"}:
            effect_mult += _joker_current_plus(ability_joker, suffix="mult")
        elif name in {"Ice Cream", "Square Joker", "Stone Joker", "Castle"}:
            effect_chips += _joker_current_plus(ability_joker, suffix="chips")
        elif name == "Erosion":
            effect_mult += _joker_current_plus(ability_joker, suffix="mult")
        elif name in {"Constellation", "Madness", "Vampire", "Hologram", "Obelisk", "Lucky Cat"}:
            effect_xmult *= _joker_current_xmult(ability_joker)
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
            effect_xmult *= _joker_current_xmult(ability_joker)
        elif name in {"Cavendish"}:
            effect_xmult *= 3
        elif name == "Loyalty Card" and _loyalty_card_ready(ability_joker):
            effect_xmult *= 4
        elif name == "Driver's License" and _drivers_license_active(ability_joker):
            effect_xmult *= 3
        elif name == "Baseball Card":
            effect_xmult *= 1.5 ** _uncommon_joker_count(jokers)
        elif name == "Ancient Joker":
            suit = _joker_target_suit(ability_joker)
            if suit:
                effect_xmult *= 1.5 ** _sum_triggers_for_cards(
                    scored_entries,
                    trigger_counts,
                    lambda card: _normalize_suit(card.suit) == suit,
                )
        elif name == "The Idol":
            target = _joker_target_rank_suit(ability_joker)
            if target is not None:
                rank, suit = target
                effect_xmult *= 2 ** _sum_triggers_for_cards(
                    scored_entries,
                    trigger_counts,
                    lambda card: _rank_matches(card.rank, rank) and _normalize_suit(card.suit) == suit,
                )
        elif name == "Triboulet":
            effect_xmult *= 2 ** _sum_triggers_for_cards(
                scored_entries,
                trigger_counts,
                lambda card: card.rank in {"K", "Q"},
            )
        elif name == "Photograph" and not photograph_consumed:
            first_face = _first_scored_face_entry(scored_entries, ability_jokers)
            if first_face is not None:
                # Photograph triggers while cards score, before later flat joker mult is added.
                effect_mult += pre_joker_mult * ((2 ** trigger_counts.get(first_face[0], 1)) - 1)
                photograph_consumed = True

    return effect_chips, effect_mult, effect_xmult


def _enhancement_chips(card: Card) -> int:
    enhancement = _normalize_effect_name(card.enhancement)
    if enhancement in {"bonus", "bonus card"}:
        return 30
    if enhancement in {"stone", "stone card"}:
        return 50
    return 0


def _permanent_card_chips(card: Card) -> int:
    modifier = card.metadata.get("modifier")
    value = card.metadata.get("value")
    for source in (card.metadata, modifier if isinstance(modifier, dict) else {}, value if isinstance(value, dict) else {}):
        for key in ("perma_bonus", "permanent_chips", "bonus_chips", "extra_chips"):
            if key in source:
                return _int_or_zero(source[key])
    return 0


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
    return _current_plus_number(_joker_effect_text(joker), suffix=suffix)


def _joker_current_xmult(joker: Joker) -> float:
    effect = _joker_effect_text(joker)
    match = re.search(r"currently\s+x\s*([0-9]+(?:\.[0-9]+)?)", effect, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"\bx\s*([0-9]+(?:\.[0-9]+)?)\s*mult", effect, flags=re.IGNORECASE)
    return float(match.group(1)) if match else 1.0


def _joker_effect_text(joker: Joker) -> str:
    value = joker.metadata.get("value")
    if isinstance(value, dict):
        return str(value.get("effect", ""))
    return str(joker.metadata.get("effect", ""))


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


def _loyalty_card_ready(joker: Joker) -> bool:
    text = _joker_effect_text(joker)
    remaining_match = re.search(r"(\d+)\s+remaining", text, flags=re.IGNORECASE)
    if remaining_match:
        return int(remaining_match.group(1)) == 0
    return "ready" in text.lower()


def _drivers_license_active(joker: Joker) -> bool:
    text = _joker_effect_text(joker)
    match = re.search(r"currently\s+(\d+)", text, flags=re.IGNORECASE)
    return bool(match and int(match.group(1)) >= 16)


def _joker_target_suit(joker: Joker) -> str | None:
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


def _contains_scored_club_and_other_suit(cards: tuple[Card, ...]) -> bool:
    suits = {_normalize_suit(card.suit) for card in cards}
    return "C" in suits and len(suits - {"C"}) > 0


def _contains_all_suits(cards: tuple[Card, ...]) -> bool:
    return {"S", "H", "C", "D"}.issubset({_normalize_suit(card.suit) for card in cards})


def _contains_pair(cards: tuple[Card, ...]) -> bool:
    return max(Counter(card.rank for card in cards).values()) >= 2


def _contains_two_pair(cards: tuple[Card, ...]) -> bool:
    return sum(1 for count in Counter(card.rank for card in cards).values() if count >= 2) >= 2


def _contains_three_of_a_kind(cards: tuple[Card, ...]) -> bool:
    return max(Counter(card.rank for card in cards).values()) >= 3


def _contains_four_of_a_kind(cards: tuple[Card, ...]) -> bool:
    return max(Counter(card.rank for card in cards).values()) >= 4


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
