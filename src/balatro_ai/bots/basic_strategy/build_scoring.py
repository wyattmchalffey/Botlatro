"""Sample-hand score projections for build and shop valuation."""

from __future__ import annotations

from dataclasses import replace

from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.bots.basic_strategy.cache import _decision_cached, _sample_build_score_cache_key
from balatro_ai.bots.basic_strategy.cards import (
    _is_joker_card,
    _joker_from_shop_card,
    _joker_would_overfill_slots,
    _normal_joker_slot_limit,
    _uses_normal_joker_slot,
)
from balatro_ai.bots.basic_strategy.data import WHITE_STAKE_SAMPLE_HANDS
from balatro_ai.bots.basic_strategy.hand_models import _SampleHand
from balatro_ai.bots.basic_strategy.hand_preferences import _dominant_suit, _preferred_hand_type
from balatro_ai.bots.basic_strategy.jokers import _joker_with_current_xmult
from balatro_ai.bots.basic_strategy.play_scoring import _played_hand_counts, _played_hand_types_this_round
from balatro_ai.bots.basic_strategy.rare_hands import _rare_hand_deck_manipulation_need
from balatro_ai.rules.hand_evaluator import (
    HandType,
    _prepare_joker_evaluation_context,
    best_play_from_hand,
    debuffed_suits_for_blind,
    evaluate_played_cards,
)
from balatro_ai.search.hand_viability import hand_type_is_viable


def _sample_score_gain_for_joker(state: GameState, joker: Joker) -> float:
    return max(0.0, _sample_score_delta_for_joker(state, joker))


def _sample_score_delta_for_joker(state: GameState, joker: Joker) -> float:
    current = _sample_build_score(state, state.jokers)
    with_candidate = _sample_build_score(state, _jokers_after_buy_for_scoring(state, joker))
    return with_candidate - current


def _jokers_after_buy_for_scoring(state: GameState, joker: Joker) -> tuple[Joker, ...]:
    jokers = (*state.jokers, joker)
    if not any(existing.name == "Joker Stencil" for existing in jokers):
        return jokers

    normal_slots_used = sum(1 for existing in jokers if _uses_normal_joker_slot(existing))
    stencil_xmult = max(1.0, float(_normal_joker_slot_limit(state) + 1 - normal_slots_used))
    return tuple(
        _joker_with_current_xmult(existing, stencil_xmult)
        if existing.name == "Joker Stencil"
        else existing
        for existing in jokers
    )


def _jokers_after_sell_for_scoring(state: GameState, *, remove_index: int) -> tuple[Joker, ...]:
    jokers = tuple(existing for index, existing in enumerate(state.jokers) if index != remove_index)
    if not any(existing.name == "Joker Stencil" for existing in jokers):
        return jokers

    normal_slots_used = sum(1 for existing in jokers if _uses_normal_joker_slot(existing))
    stencil_xmult = max(1.0, float(_normal_joker_slot_limit(state) + 1 - normal_slots_used))
    return tuple(
        _joker_with_current_xmult(existing, stencil_xmult)
        if existing.name == "Joker Stencil"
        else existing
        for existing in jokers
    )


def _buy_would_overfill_joker_slots(state: GameState, card: object) -> bool:
    return _is_joker_card(card) and _joker_would_overfill_slots(state, _joker_from_shop_card(card))


def _normal_slot_joker_card(card: object) -> bool:
    return _is_joker_card(card) and _uses_normal_joker_slot(_joker_from_shop_card(card))


def _sample_build_score(state: GameState, jokers: tuple[Joker, ...]) -> float:
    return _decision_cached(
        _sample_build_score_cache_key(state, jokers),
        lambda: _sample_build_score_uncached(state, jokers),
    )


def _repeatable_build_score(state: GameState, jokers: tuple[Joker, ...]) -> float:
    return _decision_cached(
        ("repeatable_build_score", _sample_build_score_cache_key(state, jokers)),
        lambda: _repeatable_build_score_uncached(state, jokers),
    )


def _sample_build_score_uncached(state: GameState, jokers: tuple[Joker, ...]) -> float:
    scoring_state = replace(state, jokers=jokers)
    joker_context = _prepare_joker_evaluation_context(jokers)
    weighted_total = 0.0
    total_weight = 0.0
    raw_scores: list[float] = []

    for sample in _score_samples_for_state(scoring_state):
        score = _sample_hand_build_score(scoring_state, jokers, sample, joker_context=joker_context)
        weighted_total += score * sample.weight
        total_weight += sample.weight
        raw_scores.append(score)

    visible_score = _visible_hand_sample_score(scoring_state, jokers, joker_context=joker_context)
    if visible_score > 0:
        weighted_total += visible_score * 1.15
        total_weight += 1.15
        raw_scores.append(float(visible_score))

    if total_weight <= 0 or not raw_scores:
        return 0.0
    expected = weighted_total / total_weight
    average_top = sum(sorted(raw_scores, reverse=True)[:3]) / min(3, len(raw_scores))
    return (expected * 0.78) + (average_top * 0.22)


def _repeatable_build_score_uncached(state: GameState, jokers: tuple[Joker, ...]) -> float:
    scoring_state = replace(state, jokers=jokers)
    joker_context = _prepare_joker_evaluation_context(jokers)
    weighted_total = 0.0
    total_weight = 0.0
    weighted_scores: list[tuple[float, float]] = []

    for sample in _score_samples_for_state(scoring_state):
        score = _sample_hand_build_score(scoring_state, jokers, sample, joker_context=joker_context)
        weighted_total += score * sample.weight
        total_weight += sample.weight
        weighted_scores.append((score, sample.weight))

    visible_score = _visible_hand_sample_score(scoring_state, jokers, joker_context=joker_context)
    if visible_score > 0:
        visible_weight = 0.35
        weighted_total += visible_score * visible_weight
        total_weight += visible_weight
        weighted_scores.append((float(visible_score), visible_weight))

    if total_weight <= 0 or not weighted_scores:
        return 0.0
    expected = weighted_total / total_weight
    lower_mid = _weighted_score_percentile(weighted_scores, 0.35)
    return (expected * 0.72) + (lower_mid * 0.28)


def _weighted_score_percentile(weighted_scores: list[tuple[float, float]], percentile: float) -> float:
    if not weighted_scores:
        return 0.0
    ordered = sorted(weighted_scores, key=lambda item: item[0])
    total_weight = sum(max(0.0, weight) for _, weight in ordered)
    if total_weight <= 0:
        return ordered[0][0]
    target = total_weight * max(0.0, min(1.0, percentile))
    cumulative = 0.0
    for score, weight in ordered:
        cumulative += max(0.0, weight)
        if cumulative >= target:
            return score
    return ordered[-1][0]


def _sample_hand_build_score(
    state: GameState,
    jokers: tuple[Joker, ...],
    sample: _SampleHand,
    *,
    joker_context,
) -> float:
    played_types = _played_hand_types_this_round(state)
    # Rust fast path: when the blind is safe + all jokers are supported,
    # call evaluate_simple_with_levels directly. This is the hottest
    # shop-search evaluation (~500K calls per trajectory).
    rust_score = _try_rust_sample_score(state, jokers, sample, played_types)
    if rust_score is not None:
        score = float(rust_score)
    else:
        evaluation = evaluate_played_cards(
            sample.cards,
            state.hand_levels,
            debuffed_suits=debuffed_suits_for_blind(state.blind),
            blind_name=state.blind,
            jokers=jokers,
            discards_remaining=state.discards_remaining,
            hands_remaining=max(1, state.hands_remaining),
            held_cards=sample.held_cards,
            deck_size=max(30, state.deck_size),
            money=state.money,
            played_hand_types_this_round=played_types,
            played_hand_counts=_played_hand_counts(state),
            _joker_context=joker_context,
        )
        score = float(evaluation.score)
    if not _should_project_card_sharp_repeat_value(state, joker_context, played_types):
        return score

    repeated_evaluation = evaluate_played_cards(
        sample.cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=max(1, state.hands_remaining),
        held_cards=sample.held_cards,
        deck_size=max(30, state.deck_size),
        money=state.money,
        played_hand_types_this_round=(evaluation.hand_type,),
        played_hand_counts=_played_hand_counts(state),
        _joker_context=joker_context,
    )
    active_weight = _card_sharp_repeat_projection_weight(state)
    return (score * (1.0 - active_weight)) + (float(repeated_evaluation.score) * active_weight)


def _should_project_card_sharp_repeat_value(
    state: GameState,
    joker_context,
    played_types: tuple[HandType, ...],
) -> bool:
    if "Card Sharp" not in joker_context.active_ability_names:
        return False
    if played_types:
        return False
    if state.blind == "The Eye":
        return False
    return _card_sharp_repeat_projection_weight(state) > 0.0


def _try_rust_sample_score(
    state: GameState,
    jokers: tuple[Joker, ...],
    sample: _SampleHand,
    played_types: tuple,
) -> int | None:
    """Rust fast path for shop sample-hand scoring. Delegates to the
    shared `search.rust_bridge` helper so the joker-data extraction
    is cached and re-used across the ~20-30 sample-hand calls per
    shop decision."""

    # The shop builder synthesizes states by overriding hand_levels,
    # hands_remaining, etc.; the bridge handles the boss-blind + joker
    # bail checks. The synthetic state passed here has state.blind set
    # to the live blind, so blind-safety filtering works correctly.
    from balatro_ai.search.rust_bridge import rust_evaluate_score
    return rust_evaluate_score(
        state, sample.cards, sample.held_cards, jokers,
        played_hand_types=played_types,
    )

    # played_hand_types is only needed for Card Sharp / Supernova /
    # Obelisk. Compute strs only when one of those is present.
    if any(n in {"Card Sharp", "Supernova", "Obelisk"} for n in joker_names):
        played_types_strs = [ht.value for ht in played_types]
    else:
        played_types_strs = []

    debuffed = [s for s in debuffed_suits_for_blind(state.blind) if len(s) == 1]
    slot_limit_raw = state.modifiers.get("joker_slot_limit", 5)
    try:
        joker_slot_limit = int(slot_limit_raw) if slot_limit_raw is not None else 5
    except (TypeError, ValueError):
        joker_slot_limit = 5
    try:
        result = balatro_core.evaluate_simple_with_levels(
            rust_cards, state.hand_levels, debuffed_suits=debuffed,
            joker_names=joker_names, joker_editions=joker_editions,
            held_cards=rust_held,
            joker_current_plus_mult=joker_plus_mult,
            joker_current_plus_chips=joker_plus_chips,
            joker_current_xmult=joker_xmult,
            joker_loyalty_ready=joker_loyalty_ready,
            joker_drivers_active=joker_drivers_active,
            joker_leading_plus_mult=joker_leading_plus_mult,
            joker_leading_plus_chips=joker_leading_plus_chips,
            joker_sell_value=joker_sell_value, joker_rarity=joker_rarity,
            joker_target_suit=joker_target_suit, joker_target_rank=joker_target_rank,
            joker_obelisk_gain=joker_obelisk_gain,
            money=int(state.money), joker_slot_limit=joker_slot_limit,
            discards_remaining=int(max(0, state.discards_remaining)),
            hands_remaining=int(max(1, state.hands_remaining)),
            played_hand_types=played_types_strs,
            deck_size=int(max(30, state.deck_size)),
        )
    except Exception:  # noqa: BLE001
        return None
    if result is None:
        return None
    _chips, _mult, score, _ht = result
    return int(score)


def _card_sharp_repeat_projection_weight(state: GameState) -> float:
    hands = max(1, int(state.hands_remaining or 4))
    if hands <= 1:
        return 0.0
    setup_discount = 0.85 if state.blind == "The Mouth" else 0.78
    return min(0.82, ((hands - 1) / hands) * setup_discount)


def _score_samples_for_state(state: GameState) -> tuple[_SampleHand, ...]:
    preferred = _preferred_hand_type(state)
    samples = list(WHITE_STAKE_SAMPLE_HANDS)
    samples.extend(_archetype_score_samples(state, preferred))
    return tuple(samples)


def _archetype_score_samples(state: GameState, preferred: HandType | None) -> tuple[_SampleHand, ...]:
    if preferred == HandType.PAIR:
        return (
            _SampleHand((Card("3", "S"), Card("3", "H")), (Card("9", "D"), Card("5", "C")), weight=1.2),
            _SampleHand((Card("8", "S"), Card("8", "D")), (Card("K", "C"), Card("4", "H")), weight=1.0),
            _SampleHand((Card("4", "S"), Card("4", "H"), Card("9", "D"), Card("9", "C")), weight=0.6),
        )
    if preferred == HandType.TWO_PAIR:
        return (
            _SampleHand((Card("4", "S"), Card("4", "H"), Card("9", "D"), Card("9", "C")), weight=1.4),
            _SampleHand((Card("6", "S"), Card("6", "D"), Card("J", "H"), Card("J", "C")), weight=1.0),
            _SampleHand((Card("7", "S"), Card("7", "H")), (Card("Q", "D"), Card("3", "C")), weight=0.6),
        )
    if preferred in {HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE}:
        return (
            _SampleHand((Card("7", "S"), Card("7", "H"), Card("7", "D")), weight=1.0),
            _SampleHand((Card("6", "S"), Card("6", "H"), Card("6", "D"), Card("J", "S"), Card("J", "C")), weight=0.8),
            _SampleHand((Card("5", "S"), Card("5", "H")), (Card("Q", "D"), Card("4", "C")), weight=0.8),
        )
    if preferred in {HandType.FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        dominant = _dominant_suit(state) or "H"
        samples = [
            _SampleHand(
                (
                    Card("A", dominant),
                    Card("Q", dominant),
                    Card("9", dominant),
                    Card("6", dominant),
                    Card("3", dominant),
                ),
                weight=1.4,
            ),
            _SampleHand(
                (
                    Card("K", dominant),
                    Card("J", dominant),
                    Card("8", dominant),
                    Card("5", dominant),
                    Card("2", dominant),
                ),
                weight=1.0,
            ),
        ]
        if preferred in {HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE} and _rare_hand_deck_manipulation_need(state, preferred) <= 0:
            samples.append(
                _SampleHand(
                    (
                        Card("7", dominant),
                        Card("7", dominant),
                        Card("7", dominant),
                        Card("4", dominant),
                        Card("4", dominant),
                    ),
                    weight=0.6,
                )
            )
        return tuple(samples)
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        samples = [
            _SampleHand((Card("9", "S"), Card("8", "H"), Card("7", "D"), Card("6", "C"), Card("5", "S")), weight=1.35),
            _SampleHand((Card("A", "S"), Card("K", "H"), Card("Q", "D"), Card("J", "C"), Card("10", "S")), weight=0.9),
            _SampleHand((Card("6", "S"), Card("6", "H")), (Card("Q", "D"), Card("4", "C")), weight=0.6),
        ]
        if preferred == HandType.STRAIGHT_FLUSH and hand_type_is_viable(state, HandType.STRAIGHT_FLUSH):
            dominant = _dominant_suit(state) or "H"
            samples.append(
                _SampleHand(
                    (
                        Card("9", dominant),
                        Card("8", dominant),
                        Card("7", dominant),
                        Card("6", dominant),
                        Card("5", dominant),
                    ),
                    weight=0.55,
                )
            )
        return tuple(samples)
    if preferred in {HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        if _rare_hand_deck_manipulation_need(state, preferred) > 0:
            return (
                _SampleHand((Card("8", "S"), Card("8", "H"), Card("8", "D")), weight=0.8),
                _SampleHand((Card("5", "S"), Card("5", "H")), (Card("Q", "D"), Card("4", "C")), weight=0.8),
            )
        return (
            _SampleHand((Card("8", "S"), Card("8", "H"), Card("8", "D"), Card("8", "C")), weight=1.0),
            _SampleHand((Card("6", "S"), Card("6", "H"), Card("6", "D"), Card("J", "S"), Card("J", "C")), weight=0.6),
        )
    return ()


def _visible_hand_sample_score(state: GameState, jokers: tuple[Joker, ...], *, joker_context=None) -> int:
    if not state.hand:
        return 0
    return best_play_from_hand(
        state.hand,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=max(1, state.hands_remaining),
        deck_size=max(30, state.deck_size),
        money=state.money,
        _joker_context=joker_context,
    ).score
