"""Monte Carlo odds for finding Balatro hand types.

The simulator intentionally models "can I play this hand type from my current
hand?" rather than the type of one random 5-card subset. That matches how a
Balatro player scans an 8-card hand and chooses the best playable subset.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from random import Random

from balatro_ai.rules.hand_evaluator import HandType

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("S", "H", "D", "C")
STRAIGHT_WINDOWS = tuple(tuple(range(start, start + 5)) for start in range(0, len(RANKS) - 4))
STRAIGHT_WINDOWS = (*STRAIGHT_WINDOWS, (12, 0, 1, 2, 3))
STRAIGHT_WINDOW_MASKS = tuple(sum(1 << rank for rank in window) for window in STRAIGHT_WINDOWS)

HAND_TYPES = (
    HandType.HIGH_CARD,
    HandType.PAIR,
    HandType.TWO_PAIR,
    HandType.THREE_OF_A_KIND,
    HandType.STRAIGHT,
    HandType.FLUSH,
    HandType.FULL_HOUSE,
    HandType.FOUR_OF_A_KIND,
    HandType.STRAIGHT_FLUSH,
    HandType.FIVE_OF_A_KIND,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
)

STANDARD_DECK_PRESET = "Standard 52"
ABANDONED_DECK_PRESET = "Abandoned"
CHECKERED_DECK_PRESET = "Checkered"
DECK_PRESETS = (STANDARD_DECK_PRESET, ABANDONED_DECK_PRESET, CHECKERED_DECK_PRESET)

DeckCounts = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True, order=True)
class CardKey:
    rank: int
    suit: int


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    deck_counts: DeckCounts
    hand_size: int = 8
    play_size: int = 5
    discard_size: int = 5
    discards: int = 4
    hands: int = 4
    trials: int = 2_000
    seed: int = 1


@dataclass(frozen=True, slots=True)
class HandTypeProbability:
    hand_type: HandType
    opening: float
    after_discards: float
    after_discards_and_hands: float


@dataclass(frozen=True, slots=True)
class SimulationResult:
    rows: tuple[HandTypeProbability, ...]
    trials: int
    deck_size: int
    hand_size: int
    play_size: int
    discards: int
    hands: int


def deck_preset(name: str) -> DeckCounts:
    if name == ABANDONED_DECK_PRESET:
        return tuple(
            tuple(0 if rank in (9, 10, 11) else 1 for rank in range(len(RANKS)))
            for _suit in SUITS
        )
    if name == CHECKERED_DECK_PRESET:
        return tuple(
            tuple(2 if suit in (0, 1) else 0 for _rank in RANKS)
            for suit in range(len(SUITS))
        )
    return standard_deck_counts()


def standard_deck_counts() -> DeckCounts:
    return tuple(tuple(1 for _rank in RANKS) for _suit in SUITS)


def deck_size(deck_counts: DeckCounts) -> int:
    return sum(sum(max(0, int(count)) for count in suit_counts) for suit_counts in deck_counts)


def build_deck(deck_counts: DeckCounts) -> list[CardKey]:
    deck: list[CardKey] = []
    for suit, suit_counts in enumerate(deck_counts):
        for rank, count in enumerate(suit_counts):
            deck.extend(CardKey(rank=rank, suit=suit) for _copy in range(max(0, int(count))))
    return deck


def available_hand_types(hand: Sequence[CardKey], *, play_size: int = 5) -> frozenset[HandType]:
    if not hand or play_size < 1:
        return frozenset()

    rank_counts, _suit_counts, suit_rank_counts = _hand_count_views(hand)
    return frozenset(
        hand_type
        for hand_type in HAND_TYPES
        if _has_hand_type_from_counts(
            hand_type,
            rank_counts=rank_counts,
            suit_rank_counts=suit_rank_counts,
            play_size=play_size,
        )
    )


def has_hand_type(hand_type: HandType, hand: Sequence[CardKey], *, play_size: int = 5) -> bool:
    if not hand or play_size < 1:
        return False
    rank_counts, _suit_counts, suit_rank_counts = _hand_count_views(hand)
    return _has_hand_type_from_counts(
        hand_type,
        rank_counts=rank_counts,
        suit_rank_counts=suit_rank_counts,
        play_size=play_size,
    )


def estimate_hand_type_probabilities(
    config: SimulationConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> SimulationResult:
    deck = build_deck(config.deck_counts)
    _validate_config(config, deck)

    rng = Random(config.seed)
    opening_hits = Counter[HandType]()
    discard_hits = Counter[HandType]()
    full_blind_hits = Counter[HandType]()
    hand_redraws = max(0, config.hands - 1)
    total_actions = max(0, config.discards) + hand_redraws
    report_every = max(1, config.trials // 100)

    completed = 0
    for trial in range(1, config.trials + 1):
        if should_stop is not None and should_stop():
            break

        shuffled = list(deck)
        rng.shuffle(shuffled)
        hand = shuffled[: config.hand_size]
        remaining = shuffled[config.hand_size :]
        opening = available_hand_types(hand, play_size=config.play_size)

        for hand_type in HAND_TYPES:
            if hand_type in opening:
                opening_hits[hand_type] += 1
                discard_hits[hand_type] += 1
                full_blind_hits[hand_type] += 1
                continue

            target_hand = list(hand)
            target_deck = list(remaining)
            found_after_discards = False
            found_after_all = False

            for action_index in range(total_actions):
                max_redraw = config.discard_size if action_index < config.discards else config.play_size
                discard_indices = choose_redraw_indices(
                    hand_type,
                    target_hand,
                    target_deck,
                    max_redraw=max_redraw,
                    play_size=config.play_size,
                )
                if discard_indices:
                    discard_set = set(discard_indices)
                    draw_count = min(len(discard_indices), len(target_deck))
                    target_hand = [card for index, card in enumerate(target_hand) if index not in discard_set]
                    target_hand.extend(target_deck[:draw_count])
                    target_deck = target_deck[draw_count:]

                if has_hand_type(hand_type, target_hand, play_size=config.play_size):
                    found_after_all = True
                    if action_index < config.discards:
                        found_after_discards = True
                    break

                if not target_deck:
                    break

            if found_after_discards:
                discard_hits[hand_type] += 1
                full_blind_hits[hand_type] += 1
            elif found_after_all:
                full_blind_hits[hand_type] += 1

        if progress is not None and (trial % report_every == 0 or trial == config.trials):
            progress(trial, config.trials)
        completed = trial

    denominator = max(1, completed)
    rows = tuple(
        HandTypeProbability(
            hand_type=hand_type,
            opening=opening_hits[hand_type] / denominator,
            after_discards=discard_hits[hand_type] / denominator,
            after_discards_and_hands=full_blind_hits[hand_type] / denominator,
        )
        for hand_type in HAND_TYPES
    )
    return SimulationResult(
        rows=rows,
        trials=denominator,
        deck_size=len(deck),
        hand_size=config.hand_size,
        play_size=config.play_size,
        discards=config.discards,
        hands=config.hands,
    )


def choose_redraw_indices(
    hand_type: HandType,
    hand: Sequence[CardKey],
    deck: Sequence[CardKey],
    *,
    max_redraw: int,
    play_size: int,
) -> tuple[int, ...]:
    if not hand or not deck or max_redraw <= 0:
        return ()
    rank_counts, suit_counts, suit_rank_counts = _hand_count_views(hand)
    if _has_hand_type_from_counts(
        hand_type,
        rank_counts=rank_counts,
        suit_rank_counts=suit_rank_counts,
        play_size=play_size,
    ):
        return ()

    max_action = min(max_redraw, len(hand), len(deck))
    if max_action <= 0:
        return ()

    deck_rank_counts, deck_suit_counts, deck_suit_rank_counts = _deck_count_views(deck)
    utility = _card_utility_scorer(
        hand_type,
        rank_counts,
        suit_counts,
        suit_rank_counts,
        deck_rank_counts,
        deck_suit_counts,
        deck_suit_rank_counts,
    )
    core = _core_keep_indices(
        hand_type,
        hand,
        rank_counts=rank_counts,
        suit_counts=suit_counts,
        suit_rank_counts=suit_rank_counts,
        deck_rank_counts=deck_rank_counts,
        deck_suit_counts=deck_suit_counts,
        deck_suit_rank_counts=deck_suit_rank_counts,
        play_size=play_size,
    )
    keep = set(core)
    min_keep = max(0, len(hand) - max_action)
    if len(keep) < min_keep:
        fillers = sorted(
            (index for index in range(len(hand)) if index not in keep),
            key=lambda index: utility(hand[index]),
            reverse=True,
        )
        keep.update(fillers[: min_keep - len(keep)])

    discardable = [index for index in range(len(hand)) if index not in keep]
    discardable.sort(key=lambda index: utility(hand[index]))
    return tuple(discardable[:max_action])


def _validate_config(config: SimulationConfig, deck: Sequence[CardKey]) -> None:
    if config.trials < 1:
        raise ValueError("Trials must be at least 1.")
    if config.hand_size < 1:
        raise ValueError("Hand size must be at least 1.")
    if config.play_size < 1:
        raise ValueError("Play size must be at least 1.")
    if config.discard_size < 1:
        raise ValueError("Discard size must be at least 1.")
    if config.discards < 0:
        raise ValueError("Discards cannot be negative.")
    if config.hands < 1:
        raise ValueError("Hands must be at least 1.")
    if len(deck) < config.hand_size:
        raise ValueError("Deck must contain at least as many cards as the hand size.")


def _core_keep_indices(
    hand_type: HandType,
    hand: Sequence[CardKey],
    *,
    rank_counts: tuple[int, ...],
    suit_counts: tuple[int, ...],
    suit_rank_counts: tuple[tuple[int, ...], ...],
    deck_rank_counts: tuple[int, ...],
    deck_suit_counts: tuple[int, ...],
    deck_suit_rank_counts: tuple[tuple[int, ...], ...],
    play_size: int,
) -> frozenset[int]:
    if hand_type == HandType.HIGH_CARD:
        return frozenset(range(min(1, len(hand))))

    if hand_type == HandType.PAIR:
        return _keep_for_rank_count(hand, rank_counts, deck_rank_counts, needed=2)
    if hand_type == HandType.THREE_OF_A_KIND:
        return _keep_for_rank_count(hand, rank_counts, deck_rank_counts, needed=3)
    if hand_type == HandType.FOUR_OF_A_KIND:
        return _keep_for_rank_count(hand, rank_counts, deck_rank_counts, needed=4)
    if hand_type == HandType.FIVE_OF_A_KIND:
        return _keep_for_five_of_a_kind(hand, rank_counts, suit_rank_counts, deck_rank_counts, deck_suit_rank_counts)
    if hand_type == HandType.TWO_PAIR:
        return _keep_for_rank_groups(hand, rank_counts, deck_rank_counts, groups=(2, 2))
    if hand_type == HandType.FULL_HOUSE:
        return _keep_for_rank_groups(hand, rank_counts, deck_rank_counts, groups=(3, 2))
    if hand_type == HandType.FLUSH:
        suit = _best_suit(suit_counts, deck_suit_counts, needed=5)
        return _first_indices(hand, lambda card: card.suit == suit, limit=min(5, play_size))
    if hand_type == HandType.STRAIGHT:
        ranks = _best_straight_window(rank_counts, deck_rank_counts)
        return _first_rank_indices(hand, ranks)
    if hand_type == HandType.STRAIGHT_FLUSH:
        suit, ranks = _best_straight_flush_window(suit_rank_counts, deck_suit_rank_counts)
        return _first_indices(hand, lambda card: card.suit == suit and card.rank in ranks, one_per_rank=True)
    if hand_type == HandType.FLUSH_HOUSE:
        return _keep_for_flush_rank_groups(hand, suit_rank_counts, deck_suit_rank_counts, groups=(3, 2))
    if hand_type == HandType.FLUSH_FIVE:
        return _keep_for_flush_five(hand, suit_rank_counts, deck_suit_rank_counts)
    return frozenset()


def _keep_for_rank_count(
    hand: Sequence[CardKey],
    rank_counts: tuple[int, ...],
    deck_rank_counts: tuple[int, ...],
    *,
    needed: int,
) -> frozenset[int]:
    rank = max(
        range(len(RANKS)),
        key=lambda candidate: (
            rank_counts[candidate] + deck_rank_counts[candidate] >= needed,
            min(rank_counts[candidate], needed),
            min(rank_counts[candidate] + deck_rank_counts[candidate], needed),
            deck_rank_counts[candidate],
        ),
    )
    return _first_indices(hand, lambda card: card.rank == rank, limit=needed)


def _keep_for_five_of_a_kind(
    hand: Sequence[CardKey],
    rank_counts: tuple[int, ...],
    suit_rank_counts: tuple[tuple[int, ...], ...],
    deck_rank_counts: tuple[int, ...],
    deck_suit_rank_counts: tuple[tuple[int, ...], ...],
) -> frozenset[int]:
    rank = max(
        range(len(RANKS)),
        key=lambda candidate: (
            _rank_can_make_non_flush_five(candidate, suit_rank_counts, deck_suit_rank_counts),
            min(rank_counts[candidate], 5),
            min(rank_counts[candidate] + deck_rank_counts[candidate], 5),
            deck_rank_counts[candidate],
        ),
    )
    matching = [index for index, card in enumerate(hand) if card.rank == rank]
    if len(matching) <= 5:
        return frozenset(matching)

    suit_groups: dict[int, list[int]] = {}
    for index in matching:
        suit_groups.setdefault(hand[index].suit, []).append(index)
    selected: list[int] = []
    for _suit, indexes in sorted(suit_groups.items(), key=lambda item: len(item[1])):
        if indexes:
            selected.append(indexes[0])
        if len(selected) >= 2:
            break
    for index in matching:
        if index not in selected:
            selected.append(index)
        if len(selected) >= 5:
            break
    return frozenset(selected)


def _keep_for_rank_groups(
    hand: Sequence[CardKey],
    rank_counts: tuple[int, ...],
    deck_rank_counts: tuple[int, ...],
    *,
    groups: tuple[int, int],
) -> frozenset[int]:
    best: tuple[int, int, int, int] | None = None
    best_ranks = (0, 1)
    for first_rank in range(len(RANKS)):
        for second_rank in range(len(RANKS)):
            if first_rank == second_rank:
                continue
            ranks = (first_rank, second_rank)
            first_needed, second_needed = groups
            current = min(rank_counts[first_rank], first_needed) + min(rank_counts[second_rank], second_needed)
            live = int(rank_counts[first_rank] + deck_rank_counts[first_rank] >= first_needed) + int(
                rank_counts[second_rank] + deck_rank_counts[second_rank] >= second_needed
            )
            future = deck_rank_counts[first_rank] + deck_rank_counts[second_rank]
            score = (live, current, future, -abs(first_rank - second_rank))
            if best is None or score > best:
                best = score
                best_ranks = ranks

    keep: set[int] = set()
    for rank, needed in zip(best_ranks, groups, strict=True):
        keep.update(_first_indices(hand, lambda card, target=rank: card.rank == target, limit=needed))
    return frozenset(keep)


def _keep_for_flush_rank_groups(
    hand: Sequence[CardKey],
    suit_rank_counts: tuple[tuple[int, ...], ...],
    deck_suit_rank_counts: tuple[tuple[int, ...], ...],
    *,
    groups: tuple[int, int],
) -> frozenset[int]:
    first_needed, second_needed = groups
    best: tuple[int, int, int, int, int] | None = None
    best_choice = (0, 0, 1)
    for suit in range(len(SUITS)):
        first_rank = max(
            range(len(RANKS)),
            key=lambda rank: _single_suit_group_score(
                suit_rank_counts,
                deck_suit_rank_counts,
                suit,
                rank,
                first_needed,
            ),
        )
        second_rank = max(
            (rank for rank in range(len(RANKS)) if rank != first_rank),
            key=lambda rank: _single_suit_group_score(
                suit_rank_counts,
                deck_suit_rank_counts,
                suit,
                rank,
                second_needed,
            ),
        )
        first_score = _single_suit_group_score(suit_rank_counts, deck_suit_rank_counts, suit, first_rank, first_needed)
        second_score = _single_suit_group_score(
            suit_rank_counts,
            deck_suit_rank_counts,
            suit,
            second_rank,
            second_needed,
        )
        score = (
            first_score[0] + second_score[0],
            first_score[1] + second_score[1],
            first_score[2] + second_score[2],
            first_score[3] + second_score[3],
            -abs(first_rank - second_rank),
        )
        if best is None or score > best:
            best = score
            best_choice = (suit, first_rank, second_rank)

    suit, triple_rank, pair_rank = best_choice
    keep: set[int] = set()
    keep.update(
        _first_indices(hand, lambda card: card.suit == suit and card.rank == triple_rank, limit=groups[0])
    )
    keep.update(_first_indices(hand, lambda card: card.suit == suit and card.rank == pair_rank, limit=groups[1]))
    return frozenset(keep)


def _single_suit_group_score(
    suit_rank_counts: tuple[tuple[int, ...], ...],
    deck_suit_rank_counts: tuple[tuple[int, ...], ...],
    suit: int,
    rank: int,
    needed: int,
) -> tuple[int, int, int, int]:
    current = suit_rank_counts[suit][rank]
    future = deck_suit_rank_counts[suit][rank]
    total = current + future
    return (
        int(total >= needed),
        min(current, needed),
        min(total, needed),
        future,
    )


def _keep_for_flush_five(
    hand: Sequence[CardKey],
    suit_rank_counts: tuple[tuple[int, ...], ...],
    deck_suit_rank_counts: tuple[tuple[int, ...], ...],
) -> frozenset[int]:
    suit, rank = max(
        ((suit, rank) for suit in range(len(SUITS)) for rank in range(len(RANKS))),
        key=lambda item: (
            suit_rank_counts[item[0]][item[1]],
            suit_rank_counts[item[0]][item[1]] + deck_suit_rank_counts[item[0]][item[1]] >= 5,
            deck_suit_rank_counts[item[0]][item[1]],
        ),
    )
    return _first_indices(hand, lambda card: card.suit == suit and card.rank == rank, limit=5)


def _best_suit(suit_counts: tuple[int, ...], deck_suit_counts: tuple[int, ...], *, needed: int) -> int:
    return max(
        range(len(SUITS)),
        key=lambda suit: (
            min(suit_counts[suit], needed),
            suit_counts[suit] + deck_suit_counts[suit] >= needed,
            deck_suit_counts[suit],
        ),
    )


def _best_straight_window(rank_counts: tuple[int, ...], deck_rank_counts: tuple[int, ...]) -> frozenset[int]:
    return frozenset(
        max(
            STRAIGHT_WINDOWS,
            key=lambda window: (
                sum(1 for rank in window if rank_counts[rank] > 0),
                sum(1 for rank in window if rank_counts[rank] + deck_rank_counts[rank] > 0),
                sum(deck_rank_counts[rank] for rank in window if rank_counts[rank] == 0),
            ),
        )
    )


def _best_straight_flush_window(
    suit_rank_counts: tuple[tuple[int, ...], ...],
    deck_suit_rank_counts: tuple[tuple[int, ...], ...],
) -> tuple[int, frozenset[int]]:
    best: tuple[int, int, int] | None = None
    best_choice = (0, frozenset(_straight_windows()[0]))
    for suit in range(len(SUITS)):
        for window in STRAIGHT_WINDOWS:
            score = (
                sum(1 for rank in window if suit_rank_counts[suit][rank] > 0),
                sum(1 for rank in window if suit_rank_counts[suit][rank] + deck_suit_rank_counts[suit][rank] > 0),
                sum(deck_suit_rank_counts[suit][rank] for rank in window if suit_rank_counts[suit][rank] == 0),
            )
            if best is None or score > best:
                best = score
                best_choice = (suit, frozenset(window))
    return best_choice


def _first_indices(
    hand: Sequence[CardKey],
    predicate: Callable[[CardKey], bool],
    *,
    limit: int | None = None,
    one_per_rank: bool = False,
) -> frozenset[int]:
    selected: set[int] = set()
    ranks_seen: set[int] = set()
    for index, card in enumerate(hand):
        if not predicate(card):
            continue
        if one_per_rank and card.rank in ranks_seen:
            continue
        selected.add(index)
        ranks_seen.add(card.rank)
        if limit is not None and len(selected) >= limit:
            break
    return frozenset(selected)


def _first_rank_indices(hand: Sequence[CardKey], ranks: frozenset[int]) -> frozenset[int]:
    selected: set[int] = set()
    ranks_seen: set[int] = set()
    for index, card in enumerate(hand):
        if card.rank in ranks and card.rank not in ranks_seen:
            selected.add(index)
            ranks_seen.add(card.rank)
    return frozenset(selected)


def _card_utility_scorer(
    hand_type: HandType,
    rank_counts: tuple[int, ...],
    suit_counts: tuple[int, ...],
    suit_rank_counts: tuple[tuple[int, ...], ...],
    deck_rank_counts: tuple[int, ...],
    deck_suit_counts: tuple[int, ...],
    deck_suit_rank_counts: tuple[tuple[int, ...], ...],
) -> Callable[[CardKey], int]:
    def utility(card: CardKey) -> int:
        if hand_type in {
            HandType.PAIR,
            HandType.THREE_OF_A_KIND,
            HandType.FOUR_OF_A_KIND,
            HandType.FIVE_OF_A_KIND,
            HandType.TWO_PAIR,
            HandType.FULL_HOUSE,
        }:
            return rank_counts[card.rank] + deck_rank_counts[card.rank]
        if hand_type == HandType.FLUSH:
            return suit_counts[card.suit] + deck_suit_counts[card.suit]
        if hand_type == HandType.STRAIGHT:
            return sum(
                1
                for window in STRAIGHT_WINDOWS
                if card.rank in window and all(rank_counts[rank] + deck_rank_counts[rank] > 0 for rank in window)
            )
        if hand_type == HandType.STRAIGHT_FLUSH:
            return sum(
                1
                for window in STRAIGHT_WINDOWS
                if card.rank in window
                and all(
                    suit_rank_counts[card.suit][rank] + deck_suit_rank_counts[card.suit][rank] > 0
                    for rank in window
                )
            )
        if hand_type in {HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
            return suit_rank_counts[card.suit][card.rank] + deck_suit_rank_counts[card.suit][card.rank]
        return 0

    return utility


def _hand_count_views(
    hand: Sequence[CardKey],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    rank_counts = [0] * len(RANKS)
    suit_counts = [0] * len(SUITS)
    suit_rank_counts = [[0] * len(RANKS) for _suit in SUITS]
    for card in hand:
        rank_counts[card.rank] += 1
        suit_counts[card.suit] += 1
        suit_rank_counts[card.suit][card.rank] += 1
    return tuple(rank_counts), tuple(suit_counts), tuple(tuple(counts) for counts in suit_rank_counts)


def _deck_count_views(
    deck: Sequence[CardKey],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    return _hand_count_views(deck)


def _rank_can_make_non_flush_five(
    rank: int,
    suit_rank_counts: tuple[tuple[int, ...], ...],
    deck_suit_rank_counts: tuple[tuple[int, ...], ...],
) -> bool:
    total = 0
    live_suits = 0
    for suit in range(len(SUITS)):
        suit_total = suit_rank_counts[suit][rank] + deck_suit_rank_counts[suit][rank]
        total += suit_total
        if suit_total > 0:
            live_suits += 1
    return total >= 5 and live_suits >= 2


def _has_straight(rank_counts: Sequence[int]) -> bool:
    mask = 0
    for rank, count in enumerate(rank_counts):
        if count > 0:
            mask |= 1 << rank
    return any(mask & window == window for window in STRAIGHT_WINDOW_MASKS)


def _has_hand_type_from_counts(
    hand_type: HandType,
    *,
    rank_counts: Sequence[int],
    suit_rank_counts: Sequence[Sequence[int]],
    play_size: int,
) -> bool:
    max_rank_count = max(rank_counts, default=0)
    if hand_type == HandType.HIGH_CARD:
        return play_size >= 1 and sum(rank_counts) >= 1
    if hand_type == HandType.PAIR:
        return play_size >= 2 and max_rank_count >= 2
    if hand_type == HandType.TWO_PAIR:
        return play_size >= 4 and sum(1 for count in rank_counts if count >= 2) >= 2
    if hand_type == HandType.THREE_OF_A_KIND:
        return play_size >= 3 and max_rank_count >= 3
    if hand_type == HandType.FOUR_OF_A_KIND:
        return play_size >= 4 and max_rank_count >= 4
    if hand_type == HandType.FIVE_OF_A_KIND:
        return play_size >= 5 and _has_exact_five_of_a_kind(suit_rank_counts)
    if play_size < 5:
        return False
    if hand_type == HandType.STRAIGHT:
        return _has_exact_straight(suit_rank_counts)
    if hand_type == HandType.FLUSH:
        return any(_suit_has_exact_flush(tuple(suit_ranks)) for suit_ranks in suit_rank_counts)
    if hand_type == HandType.FULL_HOUSE:
        return _has_exact_full_house(suit_rank_counts)
    if hand_type == HandType.STRAIGHT_FLUSH:
        return any(_has_straight(suit_ranks) for suit_ranks in suit_rank_counts)
    if hand_type == HandType.FLUSH_HOUSE:
        return any(_has_flush_house(suit_ranks) for suit_ranks in suit_rank_counts)
    if hand_type == HandType.FLUSH_FIVE:
        return any(max(suit_ranks, default=0) >= 5 for suit_ranks in suit_rank_counts)
    return False


def _has_exact_straight(suit_rank_counts: Sequence[Sequence[int]]) -> bool:
    for window in STRAIGHT_WINDOWS:
        union_suits: set[int] = set()
        for rank in window:
            rank_suits = {suit for suit in range(len(SUITS)) if suit_rank_counts[suit][rank] > 0}
            if not rank_suits:
                break
            union_suits.update(rank_suits)
        else:
            if len(union_suits) >= 2:
                return True
    return False


def _has_exact_five_of_a_kind(suit_rank_counts: Sequence[Sequence[int]]) -> bool:
    for rank in range(len(RANKS)):
        total = 0
        live_suits = 0
        for suit in range(len(SUITS)):
            count = suit_rank_counts[suit][rank]
            total += count
            if count > 0:
                live_suits += 1
        if total >= 5 and live_suits >= 2:
            return True
    return False


def _has_exact_full_house(suit_rank_counts: Sequence[Sequence[int]]) -> bool:
    rank_counts = tuple(sum(suit_rank_counts[suit][rank] for suit in range(len(SUITS))) for rank in range(len(RANKS)))
    for triple_rank in range(len(RANKS)):
        if rank_counts[triple_rank] < 3:
            continue
        for pair_rank in range(len(RANKS)):
            if pair_rank == triple_rank or rank_counts[pair_rank] < 2:
                continue
            triple_suits = tuple(suit_rank_counts[suit][triple_rank] for suit in range(len(SUITS)))
            pair_suits = tuple(suit_rank_counts[suit][pair_rank] for suit in range(len(SUITS)))
            if _rank_group_has_non_flush_choice(triple_suits, 3, pair_suits, 2):
                return True
    return False


@lru_cache(maxsize=4096)
def _suit_has_exact_flush(rank_counts: tuple[int, ...]) -> bool:
    selected = [0] * len(RANKS)

    def search(rank: int, total: int, mask: int) -> bool:
        if total == 5:
            counts = sorted(count for count in selected if count > 0)
            max_count = counts[-1]
            if max_count >= 4:
                return False
            if counts == [2, 3]:
                return False
            if counts == [1, 1, 1, 1, 1] and any(mask & window == window for window in STRAIGHT_WINDOW_MASKS):
                return False
            return True
        if rank >= len(RANKS):
            return False
        for take in range(0, min(rank_counts[rank], 5 - total) + 1):
            selected[rank] = take
            if search(rank + 1, total + take, mask | ((1 << rank) if take else 0)):
                selected[rank] = 0
                return True
        selected[rank] = 0
        return False

    return search(0, 0, 0)


def _has_flush_house(rank_counts: Sequence[int]) -> bool:
    pair_ranks = [rank for rank, count in enumerate(rank_counts) if count >= 2]
    if len(pair_ranks) < 2:
        return False
    return any(rank_counts[rank] >= 3 for rank in pair_ranks)


def _rank_group_has_non_flush_choice(
    first_suit_counts: tuple[int, ...],
    first_needed: int,
    second_suit_counts: tuple[int, ...],
    second_needed: int,
) -> bool:
    for first_choice in _suit_selection_vectors(first_suit_counts, first_needed):
        for second_choice in _suit_selection_vectors(second_suit_counts, second_needed):
            used_suits = sum(1 for suit in range(len(SUITS)) if first_choice[suit] + second_choice[suit] > 0)
            if used_suits >= 2:
                return True
    return False


@lru_cache(maxsize=4096)
def _suit_selection_vectors(suit_counts: tuple[int, ...], needed: int) -> tuple[tuple[int, ...], ...]:
    current = [0] * len(SUITS)
    vectors: list[tuple[int, ...]] = []

    def search(suit: int, remaining: int) -> None:
        if suit == len(SUITS):
            if remaining == 0:
                vectors.append(tuple(current))
            return
        for take in range(min(suit_counts[suit], remaining) + 1):
            current[suit] = take
            search(suit + 1, remaining - take)
        current[suit] = 0

    search(0, needed)
    return tuple(vectors)


def _straight_windows() -> tuple[tuple[int, ...], ...]:
    return STRAIGHT_WINDOWS
