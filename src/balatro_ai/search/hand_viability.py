"""Exact draw odds for playable poker hand patterns.

The shop search uses these probabilities as a reality check for narrow planets
and advanced hand plans. The implementation avoids Monte Carlo sampling: each
hand family is counted with compact rank/suit dynamic programs over the deck
multiset.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from math import comb

from balatro_ai.api.state import Card, GameState
from balatro_ai.rules.hand_evaluator import HandType
from balatro_ai.search.deck_model import DeckModel, _zone_cards, card_key

RANKS_LOW_TO_HIGH = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
RANK_INDEX = {rank: index for index, rank in enumerate(RANKS_LOW_TO_HIGH)}
SUITS = ("S", "H", "D", "C")
SMEARED_SUITS = ("B", "R")

ADVANCED_HAND_TYPES = frozenset(
    {
        HandType.STRAIGHT_FLUSH,
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }
)


@dataclass(frozen=True, slots=True)
class HandDrawOdds:
    """Probability that a random draw contains each playable hand pattern."""

    probabilities: dict[HandType, float]
    draw_size: int
    deck_size: int
    exact: bool

    def probability(self, hand_type: HandType) -> float:
        return float(self.probabilities.get(hand_type, 0.0))


# Hot path: hand_draw_odds is called ~tens of thousands of times per run (shop
# leaf eval + play viability), but the deck multiset is identity-stable across
# the thousands of evals between card acquisitions (measured ~99.9% repeat on the
# exact-deck path). Building the DeckModel (DeckModel.from_cards) and the rank/suit
# _deck_signature was ~8% of total bot CPU. Memoize the whole result keyed by the
# deck tuple's identity (O(1), no per-card scan) plus the flags/size that affect
# the answer. The deck tuple is PINNED in the cache value so its id cannot be
# reused by a freshly-allocated deck while cached, and an `is` guard on lookup
# makes a stale id-collision hit impossible. Falls back to the uncached path when
# there is no exact known_deck (standard-deck inference).
_HDO_CACHE: "OrderedDict[tuple, tuple[tuple, HandDrawOdds]]" = OrderedDict()
_HDO_CACHE_MAX = 256
# Content-keyed companion for the inferred-deck (no known_deck) branch, which
# the identity-pinned cache above cannot serve.
_HDO_INFERRED_CACHE: "OrderedDict[tuple, HandDrawOdds]" = OrderedDict()
# Identity front layer over the content cache: hand/modifiers identities are
# stable across the thousands of evals within a decision (modifiers is never
# mutated in place — states are rebuilt via replace()), so this converts
# per-call Counter+sort key construction into one content-key build per
# decision. Both objects are pinned with `is` guards (id-reuse safe).
_HDO_INFERRED_ID_CACHE: "OrderedDict[tuple, tuple[tuple, HandDrawOdds]]" = OrderedDict()


def _odds_from_model(model: DeckModel, size_in: int, joker_names: frozenset[str]) -> HandDrawOdds:
    size = max(0, min(size_in, model.total_cards))
    suit_keys = SMEARED_SUITS if "Smeared Joker" in joker_names else SUITS
    signature = _deck_signature(model, suit_keys=suit_keys)
    four_fingers = "Four Fingers" in joker_names
    probabilities = _cached_draw_probabilities(
        signature.rank_counts,
        signature.suit_rank_counts,
        signature.dead_count,
        size,
        4 if four_fingers else 5,
        4 if four_fingers else 5,
        "Shortcut" in joker_names,
    )
    return HandDrawOdds(
        probabilities={HandType(name): probability for name, probability in probabilities},
        draw_size=size,
        deck_size=model.total_cards,
        exact=model.exact,
    )


def hand_draw_odds(state: GameState, *, draw_size: int | None = None) -> HandDrawOdds:
    """Return exact hand-pattern probabilities for a fresh draw from ``state``."""

    size_in = _hand_size(state) if draw_size is None else int(draw_size)
    joker_names = frozenset(joker.name for joker in state.jokers)
    known = state.known_deck
    if known:
        smeared = "Smeared Joker" in joker_names
        size = max(0, min(size_in, len(known)))
        key = (
            id(known),
            size,
            smeared,
            "Four Fingers" in joker_names,
            "Shortcut" in joker_names,
        )
        entry = _HDO_CACHE.get(key)
        if entry is not None and entry[0] is known:
            _HDO_CACHE.move_to_end(key)
            return entry[1]
        model = DeckModel.from_cards(known, exact=len(known) == state.deck_size)
        result = _odds_from_model(model, size_in, joker_names)
        _HDO_CACHE[key] = (known, result)
        _HDO_CACHE.move_to_end(key)
        if len(_HDO_CACHE) > _HDO_CACHE_MAX:
            _HDO_CACHE.popitem(last=False)
        return result

    # Inferred-deck path (no known_deck: the bridge, BALATRO_NO_FORESIGHT=hide).
    # The model is a pure function of the visible-out-of-deck multiset, so a
    # content key dedupes the standard-deck rebuild + signature scan that
    # otherwise runs uncached on every call (~tens of thousands per run).
    smeared = "Smeared Joker" in joker_names
    four_fingers = "Four Fingers" in joker_names
    shortcut = "Shortcut" in joker_names
    id_key = (id(state.hand), id(state.modifiers), size_in, smeared, four_fingers, shortcut)
    id_entry = _HDO_INFERRED_ID_CACHE.get(id_key)
    if id_entry is not None and id_entry[0][0] is state.hand and id_entry[0][1] is state.modifiers:
        _HDO_INFERRED_ID_CACHE.move_to_end(id_key)
        return id_entry[1]
    visible = (
        state.hand
        + _zone_cards(state.modifiers, "played_pile")
        + _zone_cards(state.modifiers, "discard_pile")
        + _zone_cards(state.modifiers, "destroyed_cards")
    )
    # key=repr: CardKey holds str|None fields, which plain tuple ordering
    # cannot compare when an enhanced and plain copy of a card coexist.
    key = (
        tuple(sorted(Counter(card_key(card) for card in visible).items(), key=repr)),
        size_in,
        smeared,
        four_fingers,
        shortcut,
    )
    cached = _HDO_INFERRED_CACHE.get(key)
    if cached is None:
        model = DeckModel.from_state(state)
        cached = _odds_from_model(model, size_in, joker_names)
        _HDO_INFERRED_CACHE[key] = cached
        if len(_HDO_INFERRED_CACHE) > _HDO_CACHE_MAX:
            _HDO_INFERRED_CACHE.popitem(last=False)
    else:
        _HDO_INFERRED_CACHE.move_to_end(key)
    _HDO_INFERRED_ID_CACHE[id_key] = ((state.hand, state.modifiers), cached)
    if len(_HDO_INFERRED_ID_CACHE) > _HDO_CACHE_MAX:
        _HDO_INFERRED_ID_CACHE.popitem(last=False)
    return cached


def hand_type_draw_probability(state: GameState, hand_type: HandType, *, draw_size: int | None = None) -> float:
    return hand_draw_odds(state, draw_size=draw_size).probability(hand_type)


def hand_type_is_viable(
    state: GameState,
    hand_type: HandType,
    *,
    draw_size: int | None = None,
    minimum: float | None = None,
) -> bool:
    threshold = _default_minimum(hand_type) if minimum is None else minimum
    return hand_type_draw_probability(state, hand_type, draw_size=draw_size) >= threshold


def hand_type_viability_multiplier(
    state: GameState,
    hand_type: HandType,
    *,
    draw_size: int | None = None,
    minimum: float | None = None,
) -> float:
    threshold = max(0.001, _default_minimum(hand_type) if minimum is None else minimum)
    probability = hand_type_draw_probability(state, hand_type, draw_size=draw_size)
    return max(0.0, min(1.0, probability / threshold))


def _default_minimum(hand_type: HandType) -> float:
    if hand_type == HandType.STRAIGHT_FLUSH:
        return 0.10
    if hand_type == HandType.FOUR_OF_A_KIND:
        return 0.18
    if hand_type == HandType.FIVE_OF_A_KIND:
        return 0.08
    if hand_type == HandType.FLUSH_HOUSE:
        return 0.08
    if hand_type == HandType.FLUSH_FIVE:
        return 0.04
    return 0.0


@dataclass(frozen=True, slots=True)
class _DeckSignature:
    rank_counts: tuple[int, ...]
    suit_rank_counts: tuple[tuple[int, ...], ...]
    dead_count: int


def _deck_signature(model: DeckModel, *, suit_keys: tuple[str, ...]) -> _DeckSignature:
    rank_counts = [0] * len(RANKS_LOW_TO_HIGH)
    suit_rank_counts = [[0] * len(RANKS_LOW_TO_HIGH) for _ in suit_keys]
    suit_index = {suit: index for index, suit in enumerate(suit_keys)}
    dead_count = 0

    for key, count in model.counts.items():
        card = model.representatives[key]
        if _is_stone_card(card):
            dead_count += count
            continue
        rank_index = RANK_INDEX.get(_normalize_rank(card.rank))
        suit_key = _suit_key(card.suit, smeared=suit_keys == SMEARED_SUITS)
        if rank_index is None or suit_key not in suit_index:
            dead_count += count
            continue
        rank_counts[rank_index] += count
        suit_rank_counts[suit_index[suit_key]][rank_index] += count

    return _DeckSignature(
        rank_counts=tuple(rank_counts),
        suit_rank_counts=tuple(tuple(counts) for counts in suit_rank_counts),
        dead_count=dead_count,
    )


@lru_cache(maxsize=2048)
def _cached_draw_probabilities(
    rank_counts: tuple[int, ...],
    suit_rank_counts: tuple[tuple[int, ...], ...],
    dead_count: int,
    draw_size: int,
    straight_size: int,
    flush_size: int,
    shortcut: bool,
) -> tuple[tuple[str, float], ...]:
    total_cards = dead_count + sum(rank_counts)
    denominator = comb(total_cards, draw_size) if 0 <= draw_size <= total_cards else 0
    if denominator <= 0:
        return tuple((hand_type.value, 0.0) for hand_type in HandType)

    hits: dict[HandType, int] = {HandType.HIGH_CARD: denominator if draw_size > 0 else 0}
    hits.update(_rank_pattern_hit_counts(rank_counts, dead_count, draw_size))
    hits[HandType.STRAIGHT] = _straight_hit_count(rank_counts, dead_count, draw_size, straight_size, shortcut)
    hits[HandType.FLUSH] = _flush_hit_count(suit_rank_counts, dead_count, draw_size, flush_size)
    hits[HandType.STRAIGHT_FLUSH] = _straight_flush_hit_count(
        suit_rank_counts,
        dead_count,
        draw_size,
        straight_size,
        shortcut,
    )
    rare_flush = _rare_flush_hit_counts(suit_rank_counts, dead_count, draw_size)
    hits[HandType.FLUSH_HOUSE] = rare_flush[HandType.FLUSH_HOUSE]
    hits[HandType.FLUSH_FIVE] = rare_flush[HandType.FLUSH_FIVE]
    return tuple((hand_type.value, hits.get(hand_type, 0) / denominator) for hand_type in HandType)


def _rank_pattern_hit_counts(rank_counts: tuple[int, ...], dead_count: int, draw_size: int) -> dict[HandType, int]:
    states: dict[tuple[int, int, int, int, int], int] = {(0, 0, 0, 0, 0): 1}
    for count in rank_counts:
        next_states: dict[tuple[int, int, int, int, int], int] = {}
        for (total, pair_count, triple_count, max_rank, max_exact), ways in states.items():
            for take in range(0, min(count, draw_size - total) + 1):
                key = (
                    total + take,
                    min(2, pair_count + (1 if take >= 2 else 0)),
                    min(2, triple_count + (1 if take >= 3 else 0)),
                    min(5, max(max_rank, take)),
                    min(5, max(max_exact, take)),
                )
                next_states[key] = next_states.get(key, 0) + ways * comb(count, take)
        states = next_states
    states = _convolve_dead_rank_states(states, dead_count, draw_size)

    hits = {
        HandType.PAIR: 0,
        HandType.TWO_PAIR: 0,
        HandType.THREE_OF_A_KIND: 0,
        HandType.FULL_HOUSE: 0,
        HandType.FOUR_OF_A_KIND: 0,
        HandType.FIVE_OF_A_KIND: 0,
    }
    for (total, pair_count, triple_count, max_rank, _), ways in states.items():
        if total != draw_size:
            continue
        if max_rank >= 2:
            hits[HandType.PAIR] += ways
        if pair_count >= 2:
            hits[HandType.TWO_PAIR] += ways
        if max_rank >= 3:
            hits[HandType.THREE_OF_A_KIND] += ways
        if triple_count >= 1 and pair_count >= 2:
            hits[HandType.FULL_HOUSE] += ways
        if max_rank >= 4:
            hits[HandType.FOUR_OF_A_KIND] += ways
        if max_rank >= 5:
            hits[HandType.FIVE_OF_A_KIND] += ways
    return hits


def _convolve_dead_rank_states(
    states: dict[tuple[int, int, int, int, int], int],
    dead_count: int,
    draw_size: int,
) -> dict[tuple[int, int, int, int, int], int]:
    if dead_count <= 0:
        return states
    next_states: dict[tuple[int, int, int, int, int], int] = {}
    for (total, pair_count, triple_count, max_rank, max_exact), ways in states.items():
        for take in range(0, min(dead_count, draw_size - total) + 1):
            key = (total + take, pair_count, triple_count, max_rank, max_exact)
            next_states[key] = next_states.get(key, 0) + ways * comb(dead_count, take)
    return next_states


def _straight_hit_count(
    rank_counts: tuple[int, ...],
    dead_count: int,
    draw_size: int,
    straight_size: int,
    shortcut: bool,
) -> int:
    states: dict[tuple[int, int], int] = {(0, 0): 1}
    for rank_index, count in enumerate(rank_counts):
        next_states: dict[tuple[int, int], int] = {}
        for (total, mask), ways in states.items():
            for take in range(0, min(count, draw_size - total) + 1):
                next_mask = mask | ((1 << rank_index) if take > 0 else 0)
                key = (total + take, next_mask)
                next_states[key] = next_states.get(key, 0) + ways * comb(count, take)
        states = next_states
    states = _convolve_dead_mask_states(states, dead_count, draw_size)
    windows = _straight_windows(straight_size, shortcut=shortcut)
    return sum(ways for (total, mask), ways in states.items() if total == draw_size and any(mask & window == window for window in windows))


def _flush_hit_count(
    suit_rank_counts: tuple[tuple[int, ...], ...],
    dead_count: int,
    draw_size: int,
    flush_size: int,
) -> int:
    states: dict[tuple[int, bool], int] = {(0, False): 1}
    for ranks in suit_rank_counts:
        suit_total = sum(ranks)
        next_states: dict[tuple[int, bool], int] = {}
        for (total, has_flush), ways in states.items():
            for take in range(0, min(suit_total, draw_size - total) + 1):
                key = (total + take, has_flush or take >= flush_size)
                next_states[key] = next_states.get(key, 0) + ways * comb(suit_total, take)
        states = next_states
    states = _convolve_dead_bool_states(states, dead_count, draw_size)
    return sum(ways for (total, has_flush), ways in states.items() if total == draw_size and has_flush)


def _straight_flush_hit_count(
    suit_rank_counts: tuple[tuple[int, ...], ...],
    dead_count: int,
    draw_size: int,
    straight_size: int,
    shortcut: bool,
) -> int:
    states: dict[tuple[int, bool], int] = {(0, False): 1}
    for ranks in suit_rank_counts:
        suit_dist = _suit_straight_distribution(ranks, draw_size, straight_size, shortcut)
        next_states: dict[tuple[int, bool], int] = {}
        for (total, has_any), ways in states.items():
            for (take, has_suit_straight), suit_ways in suit_dist.items():
                if total + take > draw_size:
                    continue
                key = (total + take, has_any or has_suit_straight)
                next_states[key] = next_states.get(key, 0) + ways * suit_ways
        states = next_states
    states = _convolve_dead_bool_states(states, dead_count, draw_size)
    return sum(ways for (total, has_straight_flush), ways in states.items() if total == draw_size and has_straight_flush)


def _suit_straight_distribution(
    ranks: tuple[int, ...],
    draw_size: int,
    straight_size: int,
    shortcut: bool,
) -> dict[tuple[int, bool], int]:
    states: dict[tuple[int, int], int] = {(0, 0): 1}
    for rank_index, count in enumerate(ranks):
        next_states: dict[tuple[int, int], int] = {}
        for (total, mask), ways in states.items():
            for take in range(0, min(count, draw_size - total) + 1):
                next_mask = mask | ((1 << rank_index) if take > 0 else 0)
                key = (total + take, next_mask)
                next_states[key] = next_states.get(key, 0) + ways * comb(count, take)
        states = next_states
    windows = _straight_windows(straight_size, shortcut=shortcut)
    dist: dict[tuple[int, bool], int] = {}
    for (total, mask), ways in states.items():
        has_straight = any(mask & window == window for window in windows)
        key = (total, has_straight)
        dist[key] = dist.get(key, 0) + ways
    return dist


def _rare_flush_hit_counts(
    suit_rank_counts: tuple[tuple[int, ...], ...],
    dead_count: int,
    draw_size: int,
) -> dict[HandType, int]:
    states: dict[tuple[int, bool, bool], int] = {(0, False, False): 1}
    for ranks in suit_rank_counts:
        suit_dist = _suit_duplicate_distribution(ranks, draw_size)
        next_states: dict[tuple[int, bool, bool], int] = {}
        for (total, has_house, has_five), ways in states.items():
            for (take, suit_has_house, suit_has_five), suit_ways in suit_dist.items():
                if total + take > draw_size:
                    continue
                key = (total + take, has_house or suit_has_house, has_five or suit_has_five)
                next_states[key] = next_states.get(key, 0) + ways * suit_ways
        states = next_states
    if dead_count > 0:
        next_states: dict[tuple[int, bool, bool], int] = {}
        for (total, has_house, has_five), ways in states.items():
            for take in range(0, min(dead_count, draw_size - total) + 1):
                key = (total + take, has_house, has_five)
                next_states[key] = next_states.get(key, 0) + ways * comb(dead_count, take)
        states = next_states
    return {
        HandType.FLUSH_HOUSE: sum(ways for (total, has_house, _), ways in states.items() if total == draw_size and has_house),
        HandType.FLUSH_FIVE: sum(ways for (total, _, has_five), ways in states.items() if total == draw_size and has_five),
    }


def _suit_duplicate_distribution(ranks: tuple[int, ...], draw_size: int) -> dict[tuple[int, bool, bool], int]:
    states: dict[tuple[int, int, int, int], int] = {(0, 0, 0, 0): 1}
    for count in ranks:
        next_states: dict[tuple[int, int, int, int], int] = {}
        for (total, pair_count, triple_count, max_exact), ways in states.items():
            for take in range(0, min(count, draw_size - total) + 1):
                key = (
                    total + take,
                    min(2, pair_count + (1 if take >= 2 else 0)),
                    min(1, triple_count + (1 if take >= 3 else 0)),
                    min(5, max(max_exact, take)),
                )
                next_states[key] = next_states.get(key, 0) + ways * comb(count, take)
        states = next_states
    dist: dict[tuple[int, bool, bool], int] = {}
    for (total, pair_count, triple_count, max_exact), ways in states.items():
        key = (total, triple_count >= 1 and pair_count >= 2, max_exact >= 5)
        dist[key] = dist.get(key, 0) + ways
    return dist


def _convolve_dead_mask_states(states: dict[tuple[int, int], int], dead_count: int, draw_size: int) -> dict[tuple[int, int], int]:
    if dead_count <= 0:
        return states
    next_states: dict[tuple[int, int], int] = {}
    for (total, mask), ways in states.items():
        for take in range(0, min(dead_count, draw_size - total) + 1):
            key = (total + take, mask)
            next_states[key] = next_states.get(key, 0) + ways * comb(dead_count, take)
    return next_states


def _convolve_dead_bool_states(
    states: dict[tuple[int, bool], int],
    dead_count: int,
    draw_size: int,
) -> dict[tuple[int, bool], int]:
    if dead_count <= 0:
        return states
    next_states: dict[tuple[int, bool], int] = {}
    for (total, flag), ways in states.items():
        for take in range(0, min(dead_count, draw_size - total) + 1):
            key = (total + take, flag)
            next_states[key] = next_states.get(key, 0) + ways * comb(dead_count, take)
    return next_states


@lru_cache(maxsize=16)
def _straight_windows(size: int, *, shortcut: bool) -> tuple[int, ...]:
    if shortcut:
        # Exact Shortcut odds require gapped rank-window inclusion/exclusion.
        # Consecutive windows are conservative and avoid fantasy value.
        shortcut = False
    windows: set[int] = set()
    for start in range(0, len(RANKS_LOW_TO_HIGH) - size + 1):
        windows.add(sum(1 << index for index in range(start, start + size)))
    if size <= 5:
        wheel_ranks = ("A", "2", "3", "4", "5")[:size]
        windows.add(sum(1 << RANK_INDEX[rank] for rank in wheel_ranks))
    return tuple(windows)


def _hand_size(state: GameState) -> int:
    # Effective size = explicit base ("hand_size", 8 at run start) PLUS the
    # sim-accumulated "hand_size_delta" (Juggler/Stuntman/...): the two keys
    # coexist on sim states and mutating paths update exactly one of them.
    try:
        delta = int(state.modifiers.get("hand_size_delta", 0))
    except (TypeError, ValueError):
        delta = 0
    for key in ("hand_size", "hand_size_limit", "hand_size_max"):
        try:
            value = int(state.modifiers.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return max(1, value + delta)
    return max(1, 8 + delta)


def _normalize_rank(rank: str) -> str:
    return "10" if rank == "T" else str(rank)


def _suit_key(suit: str, *, smeared: bool) -> str:
    normalized = str(suit)
    if not smeared:
        return normalized
    if normalized in {"H", "D"}:
        return "R"
    if normalized in {"S", "C"}:
        return "B"
    return normalized


def _is_stone_card(card: Card) -> bool:
    return str(card.enhancement or "").lower() in {"stone", "stone card"}
