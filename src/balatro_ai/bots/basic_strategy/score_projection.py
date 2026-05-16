"""Score projection helpers used by discard and draw planning."""

from __future__ import annotations

from collections import Counter
import sys

from balatro_ai.api.actions import Action
from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.bots.basic_strategy.blind_state import _is_boss_blind
from balatro_ai.bots.basic_strategy.cache import (
    _card_cache_key,
    _decision_scoped_cache,
    _freeze_for_cache,
    _joker_cache_key,
    _state_scoped_cache,
)
from balatro_ai.bots.basic_strategy.discard_state import (
    _discard_draw_count,
    _state_after_discard_for_projection,
)
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.hand_value import _straight_draw_potential
from balatro_ai.bots.basic_strategy.play_scoring import _boss_adjusted_score, _played_hand_counts
from balatro_ai.rules.hand_evaluator import (
    RANK_VALUES,
    STRAIGHT_VALUES,
    best_play_from_hand as _default_best_play_from_hand,
    debuffed_suits_for_blind,
)


def _projected_score_after_discard(
    state: GameState,
    action: Action,
    context: _BlindContext | None = None,
) -> int:
    context = context or _BlindContext()
    cache = _state_scoped_cache("projected_score_after_discard", state)
    cache_key = (
        tuple(action.card_indices),
        context.played_hand_types,
        context.discards_taken,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    discarded_cards = tuple(state.hand[index] for index in action.card_indices)
    draw_count = _discard_draw_count(state, action, len(kept_cards))
    known_draw = tuple(state.known_deck[: min(draw_count, len(state.known_deck))]) if state.known_deck else ()
    projected_state = _state_after_discard_for_projection(
        state,
        action,
        drawn_cards=known_draw,
        context=context,
        decrement_discard=False,
    )
    content_cache = _decision_scoped_cache("projected_score_after_discard_content")
    content_key = _projected_discard_cache_key(
        projected_state,
        kept_cards=kept_cards,
        discarded_cards=discarded_cards,
        draw_count=draw_count,
        drawn_cards=known_draw,
        context=context,
    )
    if content_cache is not None and content_key in content_cache:
        score = content_cache[content_key]
        if cache is not None:
            cache[cache_key] = score
        return int(score)

    if known_draw:
        score = _best_score_from_cards(projected_state, (*kept_cards, *known_draw), context)
        if cache is not None:
            cache[cache_key] = score
        if content_cache is not None:
            content_cache[content_key] = score
        return score

    kept_score = _best_score_from_cards(projected_state, kept_cards, context)
    optimistic_score = _optimistic_completion_score(projected_state, kept_cards, draw_count, context)
    realism = _discard_realism_factor(projected_state, kept_cards, draw_count)
    score = int(max(kept_score, (optimistic_score * realism) + (kept_score * (1.0 - realism))))
    if cache is not None:
        cache[cache_key] = score
    if content_cache is not None:
        content_cache[content_key] = score
    return score


def _projected_discard_cache_key(
    state: GameState,
    *,
    kept_cards: tuple[Card, ...],
    discarded_cards: tuple[Card, ...],
    draw_count: int,
    drawn_cards: tuple[Card, ...],
    context: _BlindContext,
) -> tuple[object, ...]:
    return (
        _hand_multiset_cache_key(kept_cards),
        _hand_multiset_cache_key(discarded_cards),
        draw_count,
        _jokers_cache_key(state.jokers),
        _scoring_state_cache_key(state, context),
        _freeze_for_cache(drawn_cards),
    )


def _best_score_from_cards(
    state: GameState,
    cards: tuple[Card, ...],
    context: _BlindContext | None = None,
) -> int:
    context = context or _BlindContext()
    if not cards:
        return 0
    content_cache = _decision_scoped_cache("best_score_from_cards_content")
    content_key = (
        _hand_multiset_cache_key(cards),
        _jokers_cache_key(state.jokers),
        _freeze_for_cache(state.hand_levels),
        _scoring_state_cache_key(state, context),
    )
    if content_cache is not None and content_key in content_cache:
        return int(content_cache[content_key])

    cache = _state_scoped_cache("best_score_from_cards", state)
    cache_key = (
        _freeze_for_cache(cards),
        context.played_hand_types,
        context.discards_taken,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    evaluation = _best_play_from_hand(
        cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=state.jokers,
        discards_remaining=max(0, state.discards_remaining - 1),
        hands_remaining=state.hands_remaining,
        deck_size=state.deck_size,
        money=state.money,
        played_hand_types_this_round=context.played_hand_types,
        played_hand_counts=_played_hand_counts(state),
    )
    score = _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)
    if cache is not None:
        cache[cache_key] = score
    if content_cache is not None:
        content_cache[content_key] = score
    return score


def _best_play_from_hand(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot.best_play_from_hand``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "best_play_from_hand", None) if strategy_module is not None else None
    if patched is not None and patched is not _default_best_play_from_hand:
        return patched(*args, **kwargs)

    cache = _decision_scoped_cache("best_play_from_hand_content")
    cache_key = _best_play_from_hand_cache_key(args, kwargs)
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]

    evaluation = _default_best_play_from_hand(*args, **kwargs)
    if cache is not None and cache_key is not None:
        cache[cache_key] = evaluation
    return evaluation


def _best_play_from_hand_cache_key(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[object, ...] | None:
    argument_names = (
        "hand",
        "hand_levels",
        "max_cards",
        "debuffed_suits",
        "blind_name",
        "jokers",
        "discards_remaining",
        "hands_remaining",
        "deck_size",
        "money",
        "played_hand_types_this_round",
        "played_hand_counts",
        "_joker_context",
    )
    if len(args) > len(argument_names):
        return None
    values: dict[str, object] = {
        "hand": (),
        "hand_levels": None,
        "max_cards": 5,
        "debuffed_suits": None,
        "blind_name": "",
        "jokers": (),
        "discards_remaining": 0,
        "hands_remaining": 0,
        "deck_size": 0,
        "money": 0,
        "played_hand_types_this_round": (),
        "played_hand_counts": None,
        "_joker_context": None,
    }
    for index, value in enumerate(args):
        values[argument_names[index]] = value
    for name, value in kwargs.items():
        if name not in values:
            return None
        values[name] = value
    if values["_joker_context"] is not None:
        return None

    return (
        tuple(_card_cache_key(card) for card in tuple(values["hand"])),
        _freeze_for_cache(values["hand_levels"] or {}),
        int(values["max_cards"]),
        _freeze_for_cache(values["debuffed_suits"] or frozenset()),
        str(values["blind_name"]),
        _jokers_cache_key(tuple(values["jokers"])),
        int(values["discards_remaining"]),
        int(values["hands_remaining"]),
        int(values["deck_size"]),
        int(values["money"]),
        _freeze_for_cache(values["played_hand_types_this_round"] or ()),
        _freeze_for_cache(values["played_hand_counts"] or {}),
    )


def _optimistic_completion_score(
    state: GameState,
    kept_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext | None = None,
) -> int:
    context = context or _BlindContext()
    if draw_count <= 0:
        return _best_score_from_cards(state, kept_cards, context)
    cache = _decision_scoped_cache("optimistic_completion_score_content")
    cache_key = (
        _hand_multiset_cache_key(kept_cards),
        draw_count,
        _jokers_cache_key(state.jokers),
        _freeze_for_cache(state.hand_levels),
        _scoring_state_cache_key(state, context),
    )
    if cache is not None and cache_key in cache:
        return int(cache[cache_key])

    candidates = [_fill_with_high_cards(kept_cards, draw_count)]
    flush_cards = _flush_completion_cards(kept_cards, draw_count)
    if flush_cards:
        candidates.append(flush_cards)
    straight_cards = _straight_completion_cards(kept_cards, draw_count)
    if straight_cards:
        candidates.append(straight_cards)
    rank_cards = _rank_completion_cards(kept_cards, draw_count)
    if rank_cards:
        candidates.append(rank_cards)

    score = max(_best_score_from_cards(state, candidate, context) for candidate in candidates)
    if cache is not None:
        cache[cache_key] = score
    return score


def _hand_multiset_cache_key(cards: tuple[Card, ...]) -> tuple[object, ...]:
    return tuple(sorted((_card_cache_key(card) for card in cards), key=repr))


def _jokers_cache_key(jokers: tuple[Joker, ...]) -> tuple[object, ...]:
    return tuple(_joker_cache_key(joker) for joker in jokers)


def _scoring_state_cache_key(state: GameState, context: _BlindContext) -> tuple[object, ...]:
    return (
        state.blind,
        state.discards_remaining,
        state.hands_remaining,
        state.deck_size,
        state.money,
        context.played_hand_types,
        context.discards_taken,
        _freeze_for_cache(state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))),
    )


def _fill_with_high_cards(kept_cards: tuple[Card, ...], draw_count: int) -> tuple[Card, ...]:
    existing_ranks = {card.rank for card in kept_cards}
    fill_pool = (
        Card("A", "S"),
        Card("K", "H"),
        Card("Q", "D"),
        Card("J", "C"),
        Card("10", "S"),
        Card("9", "H"),
        Card("8", "D"),
    )
    fill = tuple(card for card in fill_pool if card.rank not in existing_ranks)
    return (*kept_cards, *fill[:draw_count])[: len(kept_cards) + draw_count]


def _flush_completion_cards(kept_cards: tuple[Card, ...], draw_count: int) -> tuple[Card, ...]:
    suit_counts = Counter(card.suit for card in kept_cards)
    if not suit_counts:
        return ()
    suit, count = max(suit_counts.items(), key=lambda item: item[1])
    missing = max(0, 5 - count)
    if missing == 0 or missing > draw_count:
        return ()
    existing_ranks = {card.rank for card in kept_cards if card.suit == suit}
    fill = tuple(Card(rank, suit) for rank in ("A", "K", "Q", "J", "10") if rank not in existing_ranks)
    return (*kept_cards, *fill[:draw_count])[: len(kept_cards) + draw_count]


def _straight_completion_cards(kept_cards: tuple[Card, ...], draw_count: int) -> tuple[Card, ...]:
    if not kept_cards:
        return ()

    present_values = {STRAIGHT_VALUES[card.rank] for card in kept_cards}
    if 14 in present_values:
        present_values.add(1)

    best_missing: list[int] | None = None
    for start in range(1, 11):
        run = set(range(start, start + 5))
        missing = sorted(run - present_values)
        if len(missing) <= draw_count and (best_missing is None or len(missing) < len(best_missing)):
            best_missing = missing

    if not best_missing:
        return ()

    suit = _dominant_suit_from_cards(kept_cards) or "S"
    fill = tuple(Card(_straight_rank_for_value(value), suit) for value in best_missing)
    return (*kept_cards, *fill, *_fill_with_high_cards((), max(0, draw_count - len(fill))))[: len(kept_cards) + draw_count]


def _rank_completion_cards(kept_cards: tuple[Card, ...], draw_count: int) -> tuple[Card, ...]:
    rank_counts = Counter(card.rank for card in kept_cards)
    if not rank_counts:
        return ()

    rank, count = max(rank_counts.items(), key=lambda item: (item[1], RANK_VALUES[item[0]]))
    if count < 3:
        return ()

    target_count = 4 if count + draw_count >= 4 else 3
    missing = max(0, target_count - count)
    if missing == 0 or missing > draw_count:
        return ()

    used_suits = {card.suit for card in kept_cards if card.rank == rank}
    fill_suits = tuple(suit for suit in ("S", "H", "D", "C") if suit not in used_suits)
    fill = tuple(Card(rank, suit) for suit in fill_suits[:missing])
    remaining_draw = max(0, draw_count - len(fill))
    return (*kept_cards, *fill, *_fill_with_high_cards((), remaining_draw))[: len(kept_cards) + draw_count]


def _straight_rank_for_value(value: int) -> str:
    if value == 1 or value == 14:
        return "A"
    if value == 13:
        return "K"
    if value == 12:
        return "Q"
    if value == 11:
        return "J"
    return str(value)


def _dominant_suit_from_cards(cards: tuple[Card, ...]) -> str | None:
    suit_counts = Counter(card.suit for card in cards)
    if not suit_counts:
        return None
    return max(suit_counts.items(), key=lambda item: item[1])[0]


def _discard_realism_factor(state: GameState, kept_cards: tuple[Card, ...], draw_count: int) -> float:
    if draw_count <= 0:
        return 0.0
    draw_size = _strong_draw_size(kept_cards)
    if draw_size >= 4:
        factor = min(0.72, 0.30 + draw_size * 0.08 + draw_count * 0.04)
    elif draw_size == 3:
        factor = min(0.52, 0.24 + draw_count * 0.05)
    else:
        factor = min(0.45, 0.20 + draw_count * 0.05)

    if not state.known_deck:
        if state.ante >= 3:
            factor *= 0.65
        if _is_boss_blind(state):
            factor *= 0.85
        if state.discards_remaining <= 1:
            factor *= 0.75
    return factor


def _strong_draw_size(cards: tuple[Card, ...]) -> int:
    if not cards:
        return 0
    rank_counts = Counter(card.rank for card in cards)
    suit_counts = Counter(card.suit for card in cards)
    return max(
        max(rank_counts.values(), default=0),
        max(suit_counts.values(), default=0),
        _straight_draw_potential(cards),
    )
