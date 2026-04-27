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
    held_cards: tuple[Card, ...] | list[Card] = (),
    deck_size: int = 0,
) -> HandEvaluation:
    """Evaluate a played hand without joker or enhancement effects."""

    played_cards = tuple(cards)
    if not played_cards:
        raise ValueError("Cannot evaluate an empty played hand")
    if len(played_cards) > 5:
        raise ValueError("Balatro hands cannot contain more than 5 played cards")

    hand_type = _identify_hand_type(played_cards)
    score_override = _score_override_for_blind(played_cards, blind_name=blind_name)
    scoring_indices = _scoring_indices(played_cards, hand_type)
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
        held_cards=tuple(held_cards),
        deck_size=deck_size,
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
    deck_size: int = 0,
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
                held_cards=tuple(cards[index] for index in range(len(cards)) if index not in indexes),
                deck_size=deck_size,
            )
            if best is None or _evaluation_sort_key(evaluation) > _evaluation_sort_key(best):
                best = evaluation

    if best is None:
        raise RuntimeError("No candidate plays were evaluated")
    return best


def _identify_hand_type(cards: tuple[Card, ...]) -> HandType:
    rank_counts = Counter(card.rank for card in cards)
    counts = sorted(rank_counts.values())
    has_five_cards = len(cards) == 5
    is_flush = has_five_cards and len({card.suit for card in cards}) == 1
    is_straight = has_five_cards and _is_straight(cards)
    max_count = max(counts)
    pair_count = counts.count(2)

    if is_flush and max_count == 5:
        return HandType.FLUSH_FIVE
    if is_flush and counts == [2, 3]:
        return HandType.FLUSH_HOUSE
    if max_count == 5:
        return HandType.FIVE_OF_A_KIND
    if is_flush and is_straight:
        return HandType.STRAIGHT_FLUSH
    if max_count == 4:
        return HandType.FOUR_OF_A_KIND
    if has_five_cards and counts == [2, 3]:
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


def _is_straight(cards: tuple[Card, ...]) -> bool:
    values = sorted({STRAIGHT_VALUES[card.rank] for card in cards})
    if len(values) != 5:
        return False
    if values == [2, 3, 4, 5, 14]:
        return True
    return values == list(range(values[0], values[0] + 5))


def _scoring_indices(cards: tuple[Card, ...], hand_type: HandType) -> tuple[int, ...]:
    if hand_type in {
        HandType.STRAIGHT,
        HandType.FLUSH,
        HandType.FULL_HOUSE,
        HandType.STRAIGHT_FLUSH,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }:
        return tuple(range(len(cards)))

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
    return RANK_VALUES[card.rank] + _enhancement_chips(card)


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


def _effect_adjustments(
    *,
    cards: tuple[Card, ...],
    hand_type: HandType,
    scoring_indices: tuple[int, ...],
    debuffed_suits: frozenset[str],
    jokers: tuple[Joker, ...],
    discards_remaining: int,
    held_cards: tuple[Card, ...],
    deck_size: int,
) -> tuple[int, int, float]:
    scored_cards = tuple(
        cards[index]
        for index in scoring_indices
        if not cards[index].debuffed and _normalize_suit(cards[index].suit) not in debuffed_suits
    )
    active_held_cards = tuple(card for card in held_cards if not card.debuffed)
    effect_chips = sum(_edition_chips(card.edition) for card in scored_cards)
    effect_mult = sum(_enhancement_mult(card) for card in scored_cards)
    effect_mult += sum(_edition_mult(card.edition) for card in scored_cards)
    effect_xmult = 1.0
    for card in scored_cards:
        effect_xmult *= _enhancement_xmult(card)
        effect_xmult *= _edition_xmult(card.edition)

    held_retrigger_count = 2 if any(joker.name == "Mime" for joker in jokers) else 1
    for card in active_held_cards:
        effect_xmult *= _held_card_xmult(card) ** held_retrigger_count

    for joker in jokers:
        name = joker.name
        effect_chips += _edition_chips(joker.edition)
        effect_mult += _edition_mult(joker.edition)
        effect_xmult *= _edition_xmult(joker.edition)
        if name == "Joker":
            effect_mult += 4
        elif name == "Stuntman":
            effect_chips += 250
        elif name == "Gros Michel":
            effect_mult += 15
        elif name == "Banner":
            effect_chips += 30 * max(0, discards_remaining)
        elif name == "Mystic Summit" and discards_remaining == 0:
            effect_mult += 15
        elif name == "Abstract Joker":
            effect_mult += 3 * len(jokers)
        elif name == "Swashbuckler":
            effect_mult += sum(other.sell_value or 0 for other in jokers if other is not joker)
        elif name == "Fibonacci":
            effect_mult += 8 * sum(1 for card in scored_cards if card.rank in {"A", "2", "3", "5", "8"})
        elif name == "Scholar":
            ace_count = sum(1 for card in scored_cards if card.rank == "A")
            effect_chips += 20 * ace_count
            effect_mult += 4 * ace_count
        elif name == "Scary Face":
            effect_chips += 30 * sum(1 for card in scored_cards if card.rank in {"J", "Q", "K"})
        elif name == "Arrowhead":
            effect_chips += 50 * sum(1 for card in scored_cards if _normalize_suit(card.suit) == "S")
        elif name == "Even Steven":
            effect_mult += 4 * sum(1 for card in scored_cards if card.rank in {"2", "4", "6", "8", "10", "T"})
        elif name == "Half Joker" and len(cards) <= 3:
            effect_mult += 20
        elif name == "Odd Todd":
            effect_chips += 31 * sum(1 for card in scored_cards if card.rank in {"A", "3", "5", "7", "9"})
        elif name == "Smiley Face":
            effect_mult += 5 * sum(1 for card in scored_cards if card.rank in {"J", "Q", "K"})
        elif name == "Walkie Talkie":
            walkie_count = sum(1 for card in scored_cards if card.rank in {"10", "T", "4"})
            effect_chips += 10 * walkie_count
            effect_mult += 4 * walkie_count
        elif name == "Onyx Agate":
            effect_mult += 7 * sum(1 for card in scored_cards if _normalize_suit(card.suit) == "C")
        elif name in SUIT_MULT_JOKERS:
            suit, mult = SUIT_MULT_JOKERS[name]
            effect_mult += mult * sum(1 for card in scored_cards if _normalize_suit(card.suit) == suit)
        elif name == "Jolly Joker" and _contains_pair(cards):
            effect_mult += 8
        elif name == "Zany Joker" and _contains_three_of_a_kind(cards):
            effect_mult += 12
        elif name == "Mad Joker" and hand_type == HandType.TWO_PAIR:
            effect_mult += 10
        elif name == "Crazy Joker" and _contains_straight(hand_type):
            effect_mult += 12
        elif name == "Droll Joker" and _contains_flush(hand_type):
            effect_mult += 10
        elif name == "Sly Joker" and _contains_pair(cards):
            effect_chips += 50
        elif name == "Wily Joker" and _contains_three_of_a_kind(cards):
            effect_chips += 100
        elif name == "Clever Joker" and hand_type == HandType.TWO_PAIR:
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
        elif name == "Shoot the Moon":
            effect_mult += 13 * held_retrigger_count * sum(1 for card in active_held_cards if card.rank == "Q")
        elif name == "Raised Fist" and active_held_cards:
            effect_mult += 2 * min(RANK_VALUES[card.rank] for card in active_held_cards)
        elif name == "Baron":
            effect_xmult *= 1.5 ** (held_retrigger_count * sum(1 for card in active_held_cards if card.rank == "K"))
        elif name == "Blackboard" and active_held_cards and all(
            _normalize_suit(card.suit) in {"S", "C"} for card in active_held_cards
        ):
            effect_xmult *= 3
        elif name == "Blue Joker" and deck_size > 0:
            effect_chips += 2 * deck_size
        elif name == "Wee Joker":
            effect_chips += _joker_current_plus(joker, suffix="chips")
            effect_chips += 8 * sum(1 for card in scored_cards if card.rank == "2")
        elif name == "Runner":
            effect_chips += _joker_current_plus(joker, suffix="chips")
            if _contains_straight(hand_type):
                effect_chips += 15
        elif name == "Green Joker":
            effect_mult += _joker_current_plus(joker, suffix="mult")
        elif name == "Ride the Bus":
            if not any(card.rank in {"J", "Q", "K"} for card in scored_cards):
                effect_mult += _joker_current_plus(joker, suffix="mult")
                effect_mult += 1
        elif name == "Spare Trousers":
            effect_mult += _joker_current_plus(joker, suffix="mult")
            if hand_type == HandType.TWO_PAIR:
                effect_mult += 2
        elif name in {"Fortune Teller", "Red Card", "Flash Card", "Popcorn", "Ceremonial Dagger"}:
            effect_mult += _joker_current_plus(joker, suffix="mult")
        elif name in {"Ice Cream", "Square Joker"}:
            effect_chips += _joker_current_plus(joker, suffix="chips")
        elif name in {"Constellation", "Madness", "Vampire", "Hologram", "Obelisk", "Lucky Cat"}:
            effect_xmult *= _joker_current_xmult(joker)
        elif name in {"Cavendish"}:
            effect_xmult *= 3

    return effect_chips, effect_mult, effect_xmult


def _enhancement_chips(card: Card) -> int:
    enhancement = _normalize_effect_name(card.enhancement)
    if enhancement in {"bonus", "bonus card"}:
        return 30
    if enhancement in {"stone", "stone card"}:
        return 50
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


def _contains_pair(cards: tuple[Card, ...]) -> bool:
    return max(Counter(card.rank for card in cards).values()) >= 2


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
