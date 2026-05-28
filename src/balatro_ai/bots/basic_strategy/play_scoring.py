"""Play-action scoring and ordering helpers for the basic strategy bot."""

from __future__ import annotations

from math import ceil

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy.cache import _identity_cached_value
from balatro_ai.bots.basic_strategy.cards import _is_face_card_for_state
from balatro_ai.bots.basic_strategy.data import CARD_SHARP_REPEATABILITY_WEIGHTS
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext, _PlayCandidate
from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type
from balatro_ai.bots.basic_strategy.hand_value import _card_keep_scores
from balatro_ai.bots.basic_strategy.jokers import _current_plus_for_joker, _joker_names
from balatro_ai.rules.hand_evaluator import (
    HandType,
    evaluate_played_cards,
    debuffed_suits_for_blind,
    _prepare_joker_evaluation_context,
)


def _best_play_action(state: GameState, context: _BlindContext | None = None) -> Action | None:
    context = context or _BlindContext()
    candidates = _play_candidates(state, context)
    if not candidates:
        return None

    remaining_score = max(0, state.required_score - state.current_score)
    if remaining_score > 0:
        winning_candidates = [candidate for candidate in candidates if candidate.score >= remaining_score]
        if winning_candidates:
            if all(_ride_the_bus_reset_candidate(state, candidate) for candidate in winning_candidates):
                ride_safe_candidates = _ride_the_bus_non_reset_plausible_candidates(
                    state,
                    [candidate for candidate in candidates if candidate.score > 0],
                    remaining_score,
                )
                if ride_safe_candidates:
                    return min(
                        ride_safe_candidates,
                        key=lambda candidate: _hands_to_clear_key(state, candidate, remaining_score, context),
                    ).action
            return min(
                winning_candidates,
                key=lambda candidate: _minimum_sufficient_play_key(state, candidate, context),
            ).action

        feasible_candidates = [candidate for candidate in candidates if candidate.score > 0]
        if feasible_candidates:
            feasible_candidates = _ride_the_bus_survival_play_pool(
                state,
                feasible_candidates,
                remaining_score,
            )
            if _should_set_up_card_sharp(state, context):
                return min(
                    feasible_candidates,
                    key=lambda candidate: _card_sharp_setup_play_key(candidate, remaining_score),
                ).action
            return min(
                feasible_candidates,
                key=lambda candidate: _hands_to_clear_key(state, candidate, remaining_score, context),
            ).action

    return max(candidates, key=lambda candidate: _maximum_play_key(state, candidate, context)).action


def _play_candidates(state: GameState, context: _BlindContext | None = None) -> list[_PlayCandidate]:
    context = context or _BlindContext()
    play_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.PLAY_HAND and action.card_indices
    ]
    joker_context = _prepare_joker_evaluation_context(state.jokers)
    return [_play_candidate(state, action, context, joker_context=joker_context) for action in play_actions]


def _play_candidate(
    state: GameState,
    action: Action,
    context: _BlindContext | None = None,
    *,
    joker_context=None,
) -> _PlayCandidate:
    context = context or _BlindContext()
    evaluation = _evaluate_play_action(state, action, context, joker_context=joker_context)
    scoring_card_indices = tuple(action.card_indices[index] for index in evaluation.scoring_indices)
    cycle_value = _cycle_value_for_play(state, action, scoring_card_indices)
    cycle_count = sum(1 for index in action.card_indices if index not in scoring_card_indices)
    return _PlayCandidate(
        action=action,
        score=_boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context),
        hand_type=evaluation.hand_type,
        scoring_card_indices=scoring_card_indices,
        cycle_value=cycle_value,
        cycle_count=cycle_count,
    )


def _minimum_sufficient_play_key(
    state: GameState,
    candidate: _PlayCandidate,
    context: _BlindContext,
) -> tuple[int, int, float, float, int, int]:
    bonus = _playstyle_play_bonus(state, candidate, context)
    return (
        _winning_play_reset_penalty(state, candidate),
        candidate.score,
        -bonus,
        -candidate.cycle_value,
        -candidate.cycle_count,
        _action_index_sum(candidate.action),
    )


def _maximum_play_key(
    state: GameState,
    candidate: _PlayCandidate,
    context: _BlindContext,
) -> tuple[float, int, float, int]:
    return (
        candidate.score + _playstyle_play_bonus(state, candidate, context),
        candidate.score,
        candidate.cycle_value,
        -len(candidate.action.card_indices),
    )


def _hands_to_clear_key(
    state: GameState,
    candidate: _PlayCandidate,
    remaining_score: int,
    context: _BlindContext,
) -> tuple[int, float, int, float, int, int]:
    bonus = _playstyle_play_bonus(state, candidate, context)
    estimated_hands = ceil(remaining_score / max(1.0, candidate.score))
    return (
        estimated_hands,
        -bonus,
        -candidate.score,
        -candidate.cycle_value,
        -candidate.cycle_count,
        _action_index_sum(candidate.action),
    )


def _should_set_up_card_sharp(state: GameState, context: _BlindContext) -> bool:
    return (
        any(joker.name == "Card Sharp" for joker in state.jokers)
        and not context.played_hand_types
        and state.blind != "The Eye"
        and state.hands_remaining > 1
    )


def _card_sharp_setup_play_key(candidate: _PlayCandidate, remaining_score: int) -> tuple[int, float, int, float, int, int]:
    adjusted_score = _card_sharp_setup_score(candidate)
    estimated_hands = ceil(remaining_score / max(1.0, adjusted_score))
    return (
        estimated_hands,
        -adjusted_score,
        -candidate.score,
        -candidate.cycle_value,
        -candidate.cycle_count,
        _action_index_sum(candidate.action),
    )


def _card_sharp_setup_score(candidate: _PlayCandidate) -> float:
    repeatability = CARD_SHARP_REPEATABILITY_WEIGHTS.get(candidate.hand_type, 0.0)
    return candidate.score * (1.0 + repeatability)


def _playstyle_play_bonus(state: GameState, candidate: _PlayCandidate, context: _BlindContext) -> float:
    names = _joker_names(state)
    bonus = 0.0
    scoring_cards = _candidate_scoring_cards(state, candidate)

    if "Square Joker" in names:
        bonus += 90.0 if len(candidate.action.card_indices) == 4 else -20.0
    if "Green Joker" in names:
        bonus += 35.0
    if "Supernova" in names:
        if candidate.hand_type == _most_played_hand_type(state):
            bonus += 95.0
        if context.played_hand_types and candidate.hand_type == context.played_hand_types[-1]:
            bonus += 35.0
    if _ride_the_bus_reset_candidate(state, candidate):
        bonus -= _ride_the_bus_reset_score_penalty(state)
    if "Wee Joker" in names:
        bonus += 70.0 * sum(1 for card in scoring_cards if card.rank == "2")
    if "Hack" in names:
        bonus += 12.0 * sum(1 for card in scoring_cards if card.rank in {"2", "3", "4", "5"})
    if "Fibonacci" in names:
        bonus += 16.0 * sum(1 for card in scoring_cards if card.rank in {"A", "2", "3", "5", "8"})
    if "Vampire" in names:
        bonus += 75.0 * sum(1 for card in scoring_cards if card.enhancement)
    if "Midas Mask" in names:
        bonus += 30.0 * sum(1 for card in scoring_cards if _is_face_card_for_state(state, card))
    if "Hiker" in names:
        bonus += 18.0 * len(scoring_cards)
    return bonus


def _winning_play_reset_penalty(state: GameState, candidate: _PlayCandidate) -> int:
    return int(_ride_the_bus_reset_candidate(state, candidate))


def _ride_the_bus_survival_play_pool(
    state: GameState,
    candidates: list[_PlayCandidate],
    remaining_score: int,
) -> list[_PlayCandidate]:
    return _ride_the_bus_non_reset_plausible_candidates(state, candidates, remaining_score) or candidates


def _ride_the_bus_non_reset_plausible_candidates(
    state: GameState,
    candidates: list[_PlayCandidate],
    remaining_score: int,
) -> list[_PlayCandidate]:
    if not _ride_the_bus_policy_active(state):
        return []

    return [
        candidate
        for candidate in candidates
        if not _ride_the_bus_reset_candidate(state, candidate)
        and _candidate_estimated_hands_to_clear(candidate, remaining_score) <= state.hands_remaining
    ]


def _candidate_estimated_hands_to_clear(
    candidate: _PlayCandidate,
    remaining_score: int,
) -> int:
    return ceil(remaining_score / max(1.0, candidate.score))


def _ride_the_bus_reset_candidate(state: GameState, candidate: _PlayCandidate) -> bool:
    if not _ride_the_bus_policy_active(state):
        return False
    scoring_cards = _candidate_scoring_cards(state, candidate)
    return any(_is_face_card_for_state(state, card) for card in scoring_cards)


def _ride_the_bus_policy_active(state: GameState) -> bool:
    names = _joker_names(state)
    return "Ride the Bus" in names and "Pareidolia" not in names


def _ride_the_bus_reset_score_penalty(state: GameState) -> float:
    current_mult = _current_plus_for_joker(state, "Ride the Bus", suffix="mult")
    return 220.0 + max(0, current_mult) * 140.0


def _candidate_scoring_cards(state: GameState, candidate: _PlayCandidate) -> tuple[Card, ...]:
    return tuple(state.hand[index] for index in candidate.scoring_card_indices)


def _most_played_hand_type(state: GameState) -> HandType | None:
    hands = state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))
    if not isinstance(hands, dict):
        return None

    best: tuple[int, HandType] | None = None
    for name, value in hands.items():
        if not isinstance(value, dict):
            continue
        try:
            hand_type = HandType(str(name))
        except ValueError:
            continue
        played = _hand_play_count(value)
        if played <= 0:
            continue
        if best is None or played > best[0]:
            best = (played, hand_type)
    return best[1] if best is not None else None


def _hand_play_count(value: dict[str, object]) -> int:
    for key in ("played", "played_this_run", "played_count", "times_played", "count"):
        try:
            raw = value.get(key)
            if raw is not None:
                return int(raw)
        except (TypeError, ValueError):
            continue
    try:
        return int(value.get("played_this_round", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _action_index_sum(action: Action) -> int:
    return sum(action.card_indices)


def _score_play_action(
    state: GameState,
    action: Action,
    context: _BlindContext | None = None,
    *,
    joker_context=None,
) -> int:
    context = context or _BlindContext()
    # Rust fast path: returns (score, hand_type_str) — boss adjustment
    # still done Python-side. Bails on non-vanilla blinds + unsupported
    # jokers + Wild cards, falling through to the Python evaluator.
    from balatro_ai.search.rust_bridge import rust_evaluate_score_and_hand_type
    cards = tuple(state.hand[index] for index in action.card_indices)
    held_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    rust_result = rust_evaluate_score_and_hand_type(
        state, cards, held_cards, state.jokers,
        played_hand_types=context.played_hand_types,
    )
    if rust_result is not None:
        score, ht_str = rust_result
        try:
            hand_type = HandType(ht_str)
        except ValueError:
            hand_type = None
        if hand_type is not None:
            return _boss_adjusted_score(state, hand_type, score, context)
    evaluation = _evaluate_play_action(state, action, context, joker_context=joker_context)
    return _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)


def _evaluate_play_action(
    state: GameState,
    action: Action,
    context: _BlindContext | None = None,
    *,
    joker_context=None,
):
    context = context or _BlindContext()
    cards = tuple(state.hand[index] for index in action.card_indices)
    held_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    # Rust fast path: synthesize a minimal HandEvaluation with score
    # and hand_type populated. Most callers only need those two
    # fields (the `.score` property reads from score_override).
    # `scoring_indices` is left empty — the rare callers that need
    # it (blind_setup) get an empty tuple, which is correct for "no
    # cards score" but misleading otherwise. blind_setup uses it
    # only for FYI / annotation purposes; if a regression surfaces
    # there, narrow the wire-in via a `need_full` opt-in.
    from balatro_ai.search.rust_bridge import rust_evaluate_score_and_hand_type
    rust_result = rust_evaluate_score_and_hand_type(
        state, cards, held_cards, state.jokers,
        played_hand_types=context.played_hand_types,
    )
    if rust_result is not None:
        score, ht_str = rust_result
        try:
            hand_type = HandType(ht_str)
        except ValueError:
            hand_type = None
        if hand_type is not None:
            return _synthesize_partial_hand_evaluation(
                cards=cards, hand_type=hand_type, score=score,
            )
    return evaluate_played_cards(
        cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=state.jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=state.hands_remaining,
        held_cards=held_cards,
        deck_size=state.deck_size,
        money=state.money,
        played_hand_types_this_round=context.played_hand_types,
        played_hand_counts=_played_hand_counts(state),
        _joker_context=joker_context,
    )


def _synthesize_partial_hand_evaluation(
    cards: tuple,
    hand_type: HandType,
    score: int,
):
    """Build a HandEvaluation with just `cards`, `hand_type`, and
    `score_override` populated. `.score` reads score_override;
    `.hand_type` is direct; other numeric fields default to 0.
    `.scoring_indices` is empty — accept that limitation in
    exchange for the Rust fast path."""

    from balatro_ai.rules.hand_evaluator import HandEvaluation, BASE_HAND_VALUES
    base_chips, base_mult = BASE_HAND_VALUES[hand_type]
    return HandEvaluation(
        hand_type=hand_type,
        cards=cards,
        scoring_indices=(),
        base_chips=base_chips,
        base_mult=base_mult,
        card_chips=0,
        level=1,
        level_chips=0,
        level_mult=0,
        score_override=int(score),
    )


def _boss_adjusted_score(
    state: GameState,
    hand_type: HandType,
    score: int,
    context: _BlindContext,
) -> int:
    if score <= 0 or not context.played_hand_types:
        return score
    if state.blind == "The Eye" and hand_type in context.played_hand_types:
        return 0
    if state.blind == "The Mouth" and hand_type != context.played_hand_types[0]:
        return 0
    return score


def _played_hand_types_this_round(state: GameState) -> tuple[HandType, ...]:
    ordered = _round_played_hand_type_order(state.modifiers)
    if ordered:
        return ordered

    hands = state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))
    if not isinstance(hands, dict):
        return ()

    played: list[tuple[int, HandType]] = []
    for name, value in hands.items():
        if not isinstance(value, dict):
            continue
        try:
            played_count = int(value.get("played_this_round", 0) or 0)
        except (TypeError, ValueError):
            played_count = 0
        if played_count <= 0:
            continue
        try:
            hand_type = HandType(str(name))
        except ValueError:
            continue
        try:
            order = int(value.get("order", 0) or 0)
        except (TypeError, ValueError):
            order = 0
        played.extend((order, hand_type) for _ in range(played_count))

    return tuple(hand_type for _, hand_type in sorted(played, key=lambda item: item[0]))


def _round_played_hand_type_order(modifiers: dict[str, object]) -> tuple[HandType, ...]:
    raw = modifiers.get("round_played_hand_types", ())
    if not isinstance(raw, list | tuple):
        return ()

    ordered: list[HandType] = []
    for item in raw:
        try:
            ordered.append(item if isinstance(item, HandType) else HandType(str(item)))
        except ValueError:
            continue
    return tuple(ordered)


def _played_hand_counts(state: GameState) -> dict[str, int]:
    return _identity_cached_value("played_hand_counts", state, lambda: _played_hand_counts_uncached(state))


def _played_hand_counts_uncached(state: GameState) -> dict[str, int]:
    hands = state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))
    if not isinstance(hands, dict):
        return {}

    counts: dict[str, int] = {}
    for name, value in hands.items():
        if isinstance(value, dict):
            counts[str(name)] = _hand_play_count(value)
        else:
            try:
                counts[str(name)] = int(value)
            except (TypeError, ValueError):
                counts[str(name)] = 0
    return counts


def _cycle_value_for_play(
    state: GameState,
    action: Action,
    scoring_card_indices: tuple[int, ...],
) -> float:
    if len(action.card_indices) <= len(scoring_card_indices):
        return 0.0

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    scoring_set = set(scoring_card_indices)
    value = 0.0
    for index in action.card_indices:
        if index in scoring_set:
            continue
        value += 72.0 - keep_scores[index]
    return value
