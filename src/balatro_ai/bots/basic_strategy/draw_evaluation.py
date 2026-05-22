"""Draw-shape evaluation for preferred-hand discard planning."""

from __future__ import annotations

from collections import Counter

from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy.cards import _normalize_suit, _rank_matches
from balatro_ai.bots.basic_strategy.data import (
    FLUSH_ARCHETYPE_HANDS,
    FOUR_KIND_CONTAINS_HANDS,
    PAIR_CONTAINS_HANDS,
    THREE_KIND_CONTAINS_HANDS,
    TWO_PAIR_CONTAINS_HANDS,
)
from balatro_ai.bots.basic_strategy.discard_state import _discard_scoring_state
from balatro_ai.bots.basic_strategy.draw_math import (
    _draw_all_groups_probability,
    _draw_at_least_k_probability,
    _draw_at_least_one_probability,
    _draw_group_min_counts_probability,
    _rank_matches_straight_value,
    _straight_rank_for_value,
    _straight_values_for_card,
    _straight_values_present,
    _straight_window_duplicate_penalty,
    _straight_window_is_open_ended,
    _strategy_rank_order,
)
from balatro_ai.bots.basic_strategy.hand_models import (
    _BlindContext,
    _StraightDrawEvaluation,
    _TargetDrawEvaluation,
)
from balatro_ai.bots.basic_strategy.hand_value import _straight_draw_potential
from balatro_ai.bots.basic_strategy.score_projection import (
    _best_score_from_cards,
    _dominant_suit_from_cards,
    _score_selected_cards,
)
from balatro_ai.rules.hand_evaluator import HandType, RANK_VALUES


def _straight_draw_evaluation(
    state: GameState,
    kept_cards: tuple[Card, ...],
    *,
    discarded_cards: tuple[Card, ...] = (),
    draw_count: int,
    context: _BlindContext | None = None,
) -> _StraightDrawEvaluation | None:
    context = context or _BlindContext()
    if not kept_cards or draw_count <= 0:
        return None

    value_to_cards: dict[int, list[Card]] = {}
    for card in kept_cards:
        for value in _straight_values_for_card(card):
            value_to_cards.setdefault(value, []).append(card)

    projected_state = _discard_scoring_state(state, discarded_cards, context)
    known_completion_score = (
        _best_score_from_cards(projected_state, (*kept_cards, *state.known_deck[:draw_count]), context)
        if state.known_deck and draw_count > 0
        else None
    )
    best: _StraightDrawEvaluation | None = None
    for start in range(1, 11):
        window_values = tuple(range(start, start + 5))
        present_values = tuple(value for value in window_values if value in value_to_cards)
        present_count = len(present_values)
        if present_count < 3:
            continue
        missing_values = tuple(value for value in window_values if value not in value_to_cards)
        missing_count = len(missing_values)
        if missing_count > draw_count:
            continue

        out_values = _straight_out_values_for_window(kept_cards, window_values, missing_values)
        out_counts = tuple(_straight_missing_value_out_count(state, value) for value in out_values)
        out_count = sum(out_counts)
        if missing_count > 0 and out_count <= 0:
            continue

        top_draw_out_count = _straight_top_draw_out_count(state, out_values, draw_count)
        completes_from_known_draw = _straight_known_draw_completes(state, missing_values, draw_count)
        completion_probability = _straight_completion_probability(
            state,
            missing_values=missing_values,
            out_values=out_values,
            out_counts=out_counts,
            draw_count=draw_count,
            completes_from_known_draw=completes_from_known_draw,
        )
        completion_score = known_completion_score
        if completion_score is None:
            completion_score = _straight_completion_score(
                projected_state,
                kept_cards,
                window_values=window_values,
                missing_values=missing_values,
                draw_count=draw_count,
                context=context,
            )
        duplicate_penalty = _straight_window_duplicate_penalty(value_to_cards, present_values)
        open_ended = _straight_window_is_open_ended(missing_values, window_values)
        gutshot = missing_count == 1 and not open_ended
        window_high = max(window_values)
        quality = _straight_draw_quality(
            present_count=present_count,
            missing_count=missing_count,
            out_count=out_count,
            top_draw_out_count=top_draw_out_count,
            completion_probability=completion_probability,
            completion_score=completion_score,
            window_high=window_high,
            open_ended=open_ended,
            gutshot=gutshot,
            duplicate_penalty=duplicate_penalty,
        )
        evaluation = _StraightDrawEvaluation(
            present_count=present_count,
            missing_count=missing_count,
            missing_values=missing_values,
            out_count=out_count,
            top_draw_out_count=top_draw_out_count,
            completion_probability=completion_probability,
            completion_score=completion_score,
            quality=quality,
            window_high=window_high,
            open_ended=open_ended,
            gutshot=gutshot,
            completes_from_known_draw=completes_from_known_draw,
        )
        if best is None or evaluation.quality > best.quality:
            best = evaluation
    return best


def _straight_draw_quality(
    *,
    present_count: int,
    missing_count: int,
    out_count: int,
    top_draw_out_count: int,
    completion_probability: float,
    completion_score: int,
    window_high: int,
    open_ended: bool,
    gutshot: bool,
    duplicate_penalty: int,
) -> float:
    quality = present_count * 110.0
    quality += out_count * 9.0
    quality += top_draw_out_count * 35.0
    quality += completion_probability * 260.0
    quality += min(120.0, completion_score * 0.012)
    quality += window_high * 2.5
    quality -= missing_count * 95.0
    quality -= duplicate_penalty * 22.0
    if open_ended:
        quality += 55.0
    if gutshot:
        quality -= 35.0
    if missing_count >= 2:
        quality -= 80.0
    return quality


def _straight_out_values_for_window(
    kept_cards: tuple[Card, ...],
    window_values: tuple[int, ...],
    missing_values: tuple[int, ...],
) -> tuple[int, ...]:
    if len(missing_values) != 1:
        return missing_values
    values = _straight_values_present(kept_cards)
    out_values: set[int] = set(missing_values)
    for start in range(1, 11):
        window = set(range(start, start + 5))
        missing = window - values
        if len(missing) == 1 and len(window & values) >= 4:
            out_values.update(missing)
    return tuple(sorted(out_values, key=lambda value: (value not in window_values, value)))


def _straight_missing_value_out_count(state: GameState, value: int) -> int:
    rank = _straight_rank_for_value(value)
    if state.known_deck:
        return sum(1 for card in state.known_deck if _rank_matches_straight_value(card, value))
    seen = sum(1 for card in state.hand if _rank_matches(card.rank, rank))
    return max(0, 4 - seen)


def _straight_top_draw_out_count(state: GameState, out_values: tuple[int, ...], draw_count: int) -> int:
    if not state.known_deck or draw_count <= 0:
        return 0
    top_draw = state.known_deck[:draw_count]
    return sum(1 for card in top_draw if any(_rank_matches_straight_value(card, value) for value in out_values))


def _straight_known_draw_completes(
    state: GameState,
    missing_values: tuple[int, ...],
    draw_count: int,
) -> bool:
    if not state.known_deck:
        return False
    top_values = _straight_values_present(tuple(state.known_deck[:draw_count]))
    return all(value in top_values for value in missing_values)


def _straight_completion_probability(
    state: GameState,
    *,
    missing_values: tuple[int, ...],
    out_values: tuple[int, ...],
    out_counts: tuple[int, ...],
    draw_count: int,
    completes_from_known_draw: bool,
) -> float:
    if not missing_values:
        return 1.0
    if state.known_deck:
        return 1.0 if completes_from_known_draw else 0.0
    if draw_count <= 0 or not out_counts:
        return 0.0
    deck_size = max(1, state.deck_size)
    draw_count = min(draw_count, deck_size)
    if len(missing_values) == 1:
        out_count = min(deck_size, sum(out_counts))
        return _draw_at_least_one_probability(deck_size, draw_count, out_count)
    per_missing_counts = tuple(
        max(0, _straight_missing_value_out_count(state, value))
        for value in missing_values
        if value in out_values
    )
    if len(per_missing_counts) != len(missing_values) or any(count <= 0 for count in per_missing_counts):
        return 0.0
    return _draw_all_groups_probability(deck_size, draw_count, per_missing_counts)


def _straight_completion_score(
    state: GameState,
    kept_cards: tuple[Card, ...],
    *,
    window_values: tuple[int, ...],
    missing_values: tuple[int, ...],
    draw_count: int,
    context: _BlindContext,
) -> int:
    if state.known_deck:
        return _best_score_from_cards(state, (*kept_cards, *state.known_deck[:draw_count]), context)
    selected_cards = _straight_completion_cards_for_values(state, kept_cards, window_values, missing_values, draw_count)
    if not selected_cards:
        return 0
    return _score_completion_selection(state, selected_cards, kept_cards, context)


def _straight_completion_cards_for_values(
    state: GameState,
    kept_cards: tuple[Card, ...],
    window_values: tuple[int, ...],
    missing_values: tuple[int, ...],
    draw_count: int,
) -> tuple[Card, ...]:
    if len(missing_values) > draw_count:
        return ()
    selected: list[Card] = []
    used_card_ids: set[int] = set()
    dominant_suit = _dominant_suit_from_cards(kept_cards)
    for value in window_values:
        kept = [
            (index, card)
            for index, card in enumerate(kept_cards)
            if id(card) not in used_card_ids and _rank_matches_straight_value(card, value)
        ]
        if kept:
            _, card = max(kept, key=lambda item: _completion_card_sort_key(item[1]))
            selected.append(card)
            used_card_ids.add(id(card))
            continue
        if value not in missing_values:
            return ()
        rank = _straight_rank_for_value(value)
        selected.append(_neutral_fill_card_for_rank(state, rank, avoid_suit=dominant_suit, selected=tuple(selected)))
    return tuple(selected[:5])


def _straight_draw_reason_detail(evaluation: _StraightDrawEvaluation) -> str:
    shape = "open" if evaluation.open_ended else "gutshot" if evaluation.gutshot else "made" if evaluation.missing_count == 0 else "two_gap"
    return (
        f"straight_draw={evaluation.present_count}/5 missing={evaluation.missing_count} "
        f"outs={evaluation.out_count} p={evaluation.completion_probability:.2f} "
        f"shape={shape} completion={evaluation.completion_score}"
    )


def _preferred_target_draw_evaluation(
    state: GameState,
    preferred: HandType,
    kept_cards: tuple[Card, ...],
    *,
    discarded_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext,
) -> _TargetDrawEvaluation | None:
    if not kept_cards or draw_count < 0:
        return None

    projected_state = _discard_scoring_state(state, discarded_cards, context)
    candidates: list[_TargetDrawEvaluation] = []
    if preferred == HandType.FLUSH:
        candidates.extend(_flush_target_draw_evaluations(projected_state, kept_cards, draw_count, context))
    elif preferred in {HandType.THREE_OF_A_KIND, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        target_count = {
            HandType.THREE_OF_A_KIND: 3,
            HandType.FOUR_OF_A_KIND: 4,
            HandType.FIVE_OF_A_KIND: 5,
        }[preferred]
        candidates.extend(
            _rank_target_draw_evaluations(
                projected_state,
                kept_cards,
                draw_count,
                context,
                hand_type=preferred,
                target_count=target_count,
            )
        )
    elif preferred == HandType.FULL_HOUSE:
        candidates.extend(_full_house_target_draw_evaluations(projected_state, kept_cards, draw_count, context))

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.completion_score, item.completion_probability, item.quality))


def _flush_target_draw_evaluations(
    state: GameState,
    kept_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext,
) -> list[_TargetDrawEvaluation]:
    evaluations: list[_TargetDrawEvaluation] = []
    for suit in ("S", "H", "D", "C"):
        present_count = sum(1 for card in kept_cards if _normalize_suit(card.suit) == suit)
        if present_count < 3:
            continue
        missing_count = max(0, 5 - present_count)
        if missing_count > draw_count:
            continue
        out_count = _flush_suit_out_count(state, suit)
        completion_probability = _flush_completion_probability(
            state,
            suit=suit,
            draw_count=draw_count,
            missing_count=missing_count,
            out_count=out_count,
        )
        if missing_count > 0 and completion_probability <= 0.0:
            continue
        completion_cards = _flush_completion_cards_for_suit(kept_cards, suit, draw_count)
        if not completion_cards:
            continue
        completion_score = _score_completion_selection(state, completion_cards, kept_cards, context)
        quality = _target_draw_quality(
            present_count=present_count,
            missing_count=missing_count,
            out_count=out_count,
            completion_probability=completion_probability,
            completion_score=completion_score,
            draw_count=draw_count,
        )
        evaluations.append(
            _TargetDrawEvaluation(
                hand_type=HandType.FLUSH,
                label=f"Flush:{suit}",
                present_count=present_count,
                missing_count=missing_count,
                out_count=out_count,
                completion_probability=completion_probability,
                completion_score=completion_score,
                quality=quality,
            )
        )
    return evaluations


def _rank_target_draw_evaluations(
    state: GameState,
    kept_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext,
    *,
    hand_type: HandType,
    target_count: int,
) -> list[_TargetDrawEvaluation]:
    rank_counts = Counter(card.rank for card in kept_cards)
    evaluations: list[_TargetDrawEvaluation] = []
    for rank in _strategy_rank_order():
        present_count = rank_counts.get(rank, 0)
        if present_count <= 0:
            continue
        missing_count = max(0, target_count - present_count)
        if missing_count > draw_count:
            continue
        out_count = _rank_out_count(state, rank)
        completion_probability = _rank_completion_probability(
            state,
            rank=rank,
            draw_count=draw_count,
            missing_count=missing_count,
            out_count=out_count,
        )
        if missing_count > 0 and completion_probability <= 0.0:
            continue
        completion_cards = _rank_completion_cards_for_rank(state, kept_cards, rank, target_count, draw_count)
        if not completion_cards:
            continue
        completion_score = _score_completion_selection(state, completion_cards, kept_cards, context)
        quality = _target_draw_quality(
            present_count=present_count,
            missing_count=missing_count,
            out_count=out_count,
            completion_probability=completion_probability,
            completion_score=completion_score,
            draw_count=draw_count,
        )
        evaluations.append(
            _TargetDrawEvaluation(
                hand_type=hand_type,
                label=f"{hand_type.value}:{rank}",
                present_count=present_count,
                missing_count=missing_count,
                out_count=out_count,
                completion_probability=completion_probability,
                completion_score=completion_score,
                quality=quality,
            )
        )
    return evaluations


def _full_house_target_draw_evaluations(
    state: GameState,
    kept_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext,
) -> list[_TargetDrawEvaluation]:
    rank_counts = Counter(card.rank for card in kept_cards)
    evaluations: list[_TargetDrawEvaluation] = []
    ranks = _strategy_rank_order()
    for trip_rank in ranks:
        trip_present = rank_counts.get(trip_rank, 0)
        if trip_present <= 0:
            continue
        for pair_rank in ranks:
            if pair_rank == trip_rank:
                continue
            pair_present = rank_counts.get(pair_rank, 0)
            if pair_present <= 0:
                continue
            trip_missing = max(0, 3 - trip_present)
            pair_missing = max(0, 2 - pair_present)
            missing_count = trip_missing + pair_missing
            if missing_count > draw_count:
                continue
            out_counts = (
                _rank_out_count(state, trip_rank),
                _rank_out_count(state, pair_rank),
            )
            completion_probability = _rank_group_completion_probability(
                state,
                draw_count=draw_count,
                ranks=(trip_rank, pair_rank),
                requirements=(trip_missing, pair_missing),
                out_counts=out_counts,
            )
            if missing_count > 0 and completion_probability <= 0.0:
                continue
            completion_cards = _full_house_completion_cards_for_ranks(
                state,
                kept_cards,
                trip_rank=trip_rank,
                pair_rank=pair_rank,
                draw_count=draw_count,
            )
            if not completion_cards:
                continue
            completion_score = _score_completion_selection(state, completion_cards, kept_cards, context)
            present_count = min(3, trip_present) + min(2, pair_present)
            quality = _target_draw_quality(
                present_count=present_count,
                missing_count=missing_count,
                out_count=sum(out_counts),
                completion_probability=completion_probability,
                completion_score=completion_score,
                draw_count=draw_count,
            )
            evaluations.append(
                _TargetDrawEvaluation(
                    hand_type=HandType.FULL_HOUSE,
                    label=f"Full House:{trip_rank}/{pair_rank}",
                    present_count=present_count,
                    missing_count=missing_count,
                    out_count=sum(out_counts),
                    completion_probability=completion_probability,
                    completion_score=completion_score,
                    quality=quality,
                )
            )
    return evaluations


def _target_draw_quality(
    *,
    present_count: int,
    missing_count: int,
    out_count: int,
    completion_probability: float,
    completion_score: int,
    draw_count: int,
) -> float:
    quality = present_count * 115.0
    quality += completion_probability * 520.0
    quality += min(180.0, completion_score * 0.012)
    quality += out_count * 6.0
    quality += draw_count * 8.0
    quality -= missing_count * 120.0
    return quality


def _target_draw_reason_detail(evaluation: _TargetDrawEvaluation) -> str:
    return (
        f"target={evaluation.label} present={evaluation.present_count} "
        f"missing={evaluation.missing_count} outs={evaluation.out_count} "
        f"p={evaluation.completion_probability:.2f} completion={evaluation.completion_score}"
    )


def _flush_suit_out_count(state: GameState, suit: str) -> int:
    if state.known_deck:
        return sum(1 for card in state.known_deck if _normalize_suit(card.suit) == suit)
    seen = sum(1 for card in state.hand if _normalize_suit(card.suit) == suit)
    return max(0, 13 - seen)


def _flush_completion_probability(
    state: GameState,
    *,
    suit: str,
    draw_count: int,
    missing_count: int,
    out_count: int,
) -> float:
    if missing_count <= 0:
        return 1.0
    if draw_count <= 0:
        return 0.0
    if state.known_deck:
        top_draw = state.known_deck[:draw_count]
        hits = sum(1 for card in top_draw if _normalize_suit(card.suit) == suit)
        return 1.0 if hits >= missing_count else 0.0
    return _draw_at_least_k_probability(max(1, state.deck_size), draw_count, out_count, missing_count)


def _rank_out_count(state: GameState, rank: str) -> int:
    if state.known_deck:
        return sum(1 for card in state.known_deck if _rank_matches(card.rank, rank))
    seen = sum(1 for card in state.hand if _rank_matches(card.rank, rank))
    return max(0, 4 - seen)


def _rank_completion_probability(
    state: GameState,
    *,
    rank: str,
    draw_count: int,
    missing_count: int,
    out_count: int,
) -> float:
    if missing_count <= 0:
        return 1.0
    if draw_count <= 0:
        return 0.0
    if state.known_deck:
        top_draw = state.known_deck[:draw_count]
        hits = sum(1 for card in top_draw if _rank_matches(card.rank, rank))
        return 1.0 if hits >= missing_count else 0.0
    return _draw_at_least_k_probability(max(1, state.deck_size), draw_count, out_count, missing_count)


def _rank_group_completion_probability(
    state: GameState,
    *,
    draw_count: int,
    ranks: tuple[str, ...],
    requirements: tuple[int, ...],
    out_counts: tuple[int, ...],
) -> float:
    needed = tuple(max(0, requirement) for requirement in requirements)
    if all(requirement <= 0 for requirement in needed):
        return 1.0
    if draw_count <= 0:
        return 0.0
    if state.known_deck:
        top_draw = state.known_deck[:draw_count]
        hits = []
        for index, requirement in enumerate(needed):
            if requirement <= 0:
                hits.append(requirement)
                continue
            rank = ranks[index] if index < len(ranks) else ""
            hits.append(sum(1 for card in top_draw if _rank_matches(card.rank, rank)))
        return 1.0 if all(hit >= requirement for hit, requirement in zip(hits, needed, strict=False)) else 0.0
    return _draw_group_min_counts_probability(max(1, state.deck_size), draw_count, needed, out_counts)


def _flush_completion_cards_for_suit(
    kept_cards: tuple[Card, ...],
    suit: str,
    draw_count: int,
) -> tuple[Card, ...]:
    present_count = sum(1 for card in kept_cards if _normalize_suit(card.suit) == suit)
    missing = max(0, 5 - present_count)
    if missing > draw_count:
        return ()
    existing_ranks = {card.rank for card in kept_cards if _normalize_suit(card.suit) == suit}
    fill = tuple(Card(rank, suit) for rank in _strategy_rank_order() if rank not in existing_ranks)
    suited_cards = tuple(card for card in kept_cards if _normalize_suit(card.suit) == suit)
    completion = (*suited_cards, *fill[:missing])
    return tuple(sorted(completion, key=_completion_card_sort_key, reverse=True)[:5])


def _rank_completion_cards_for_rank(
    state: GameState,
    kept_cards: tuple[Card, ...],
    rank: str,
    target_count: int,
    draw_count: int,
) -> tuple[Card, ...]:
    present_count = sum(1 for card in kept_cards if _rank_matches(card.rank, rank))
    missing = max(0, target_count - present_count)
    if missing > draw_count:
        return ()
    existing = tuple(card for card in kept_cards if _rank_matches(card.rank, rank))
    fill = _rank_fill_cards(state, kept_cards, rank, missing)
    kickers = tuple(
        sorted(
            (card for card in kept_cards if not _rank_matches(card.rank, rank)),
            key=_completion_card_sort_key,
            reverse=True,
        )[: max(0, 5 - target_count)]
    )
    return (*existing[:target_count], *fill, *kickers)[:5]


def _full_house_completion_cards_for_ranks(
    state: GameState,
    kept_cards: tuple[Card, ...],
    *,
    trip_rank: str,
    pair_rank: str,
    draw_count: int,
) -> tuple[Card, ...]:
    trip_missing = max(0, 3 - sum(1 for card in kept_cards if _rank_matches(card.rank, trip_rank)))
    pair_missing = max(0, 2 - sum(1 for card in kept_cards if _rank_matches(card.rank, pair_rank)))
    if trip_missing + pair_missing > draw_count:
        return ()
    trip_existing = tuple(card for card in kept_cards if _rank_matches(card.rank, trip_rank))[:3]
    trip_fill = _rank_fill_cards(state, kept_cards, trip_rank, trip_missing)
    pair_existing = tuple(card for card in kept_cards if _rank_matches(card.rank, pair_rank))[:2]
    pair_fill = _rank_fill_cards(state, (*kept_cards, *trip_fill), pair_rank, pair_missing)
    return (*trip_existing, *trip_fill, *pair_existing, *pair_fill)[:5]


def _rank_fill_cards(state: GameState, existing_cards: tuple[Card, ...], rank: str, count: int) -> tuple[Card, ...]:
    if count <= 0:
        return ()
    used_suits = {_normalize_suit(card.suit) for card in (*state.hand, *existing_cards) if _rank_matches(card.rank, rank)}
    fill_suits = tuple(suit for suit in ("S", "H", "D", "C") if suit not in used_suits)
    return tuple(Card(rank, suit) for suit in fill_suits[:count])


def _score_completion_selection(
    state: GameState,
    selected_cards: tuple[Card, ...],
    available_cards: tuple[Card, ...],
    context: _BlindContext,
) -> int:
    held_cards = _remaining_cards_after_selection(available_cards, selected_cards)
    return _score_selected_cards(state, selected_cards, context, held_cards=held_cards)


def _remaining_cards_after_selection(cards: tuple[Card, ...], selected_cards: tuple[Card, ...]) -> tuple[Card, ...]:
    remaining = list(cards)
    for selected in selected_cards:
        for index, card in enumerate(remaining):
            if card == selected:
                remaining.pop(index)
                break
    return tuple(remaining)


def _neutral_fill_card_for_rank(
    state: GameState,
    rank: str,
    *,
    avoid_suit: str | None,
    selected: tuple[Card, ...],
) -> Card:
    used_suits = {_normalize_suit(card.suit) for card in (*state.hand, *selected) if _rank_matches(card.rank, rank)}
    available_suits = tuple(suit for suit in ("S", "H", "D", "C") if suit not in used_suits)
    if avoid_suit is not None:
        for suit in available_suits:
            if _normalize_suit(suit) != _normalize_suit(avoid_suit):
                return Card(rank, suit)
    if available_suits:
        return Card(rank, available_suits[0])
    return Card(rank, "S")


def _completion_card_sort_key(card: Card) -> tuple[int, int]:
    suit_order = {"S": 0, "H": 1, "D": 2, "C": 3}
    return (RANK_VALUES.get(card.rank, 0), -suit_order.get(_normalize_suit(card.suit), 9))


def _hand_matches_preferred_family(hand_type: HandType, preferred: HandType) -> bool:
    return hand_type in _preferred_hand_family(preferred)


def _preferred_hand_family(preferred: HandType) -> set[HandType]:
    if preferred == HandType.PAIR:
        return set(PAIR_CONTAINS_HANDS)
    if preferred == HandType.TWO_PAIR:
        return set(TWO_PAIR_CONTAINS_HANDS)
    if preferred == HandType.THREE_OF_A_KIND:
        return set(THREE_KIND_CONTAINS_HANDS)
    if preferred == HandType.FULL_HOUSE:
        return {HandType.FULL_HOUSE, HandType.FLUSH_HOUSE}
    if preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        return set(FLUSH_ARCHETYPE_HANDS)
    if preferred == HandType.STRAIGHT:
        return {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}
    if preferred == HandType.FOUR_OF_A_KIND:
        return set(FOUR_KIND_CONTAINS_HANDS)
    if preferred == HandType.FIVE_OF_A_KIND:
        return {HandType.FIVE_OF_A_KIND, HandType.FLUSH_FIVE}
    return {preferred}
