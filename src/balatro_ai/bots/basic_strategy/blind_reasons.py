"""Reason-string formatters for blind play and discard choices."""

from __future__ import annotations

from balatro_ai.api.actions import Action
from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy.cards import _is_face_card_for_state, _normalize_suit, _rank_matches
from balatro_ai.bots.basic_strategy.discard_policy import _estimated_hands_needed, _pace_safety_multiplier
from balatro_ai.bots.basic_strategy.discard_state import _known_draw_for_discard
from balatro_ai.bots.basic_strategy.draw_evaluation import _straight_draw_reason_detail
from balatro_ai.bots.basic_strategy.economy_hunt import _card_is_economy_hunt_target
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext, _PlayCandidate, _StraightDrawEvaluation
from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type
from balatro_ai.bots.basic_strategy.jokers import (
    _active_joker_names,
    _castle_target_suit,
    _mail_in_rebate_rank,
)
from balatro_ai.bots.basic_strategy.play_scoring import (
    _boss_adjusted_score,
    _evaluate_play_action,
    _score_play_action,
)
from balatro_ai.bots.basic_strategy.score_projection import _projected_score_after_discard
from balatro_ai.rules.hand_evaluator import HandType


def _play_reason(state: GameState, action: Action, context: _BlindContext | None = None) -> str:
    evaluation = _evaluate_play_action(state, action, context)
    context = context or _BlindContext()
    score = _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)
    remaining_score = max(0, state.required_score - state.current_score)
    hands_needed = _estimated_hands_needed(remaining_score, score)
    preferred = _preferred_hand_type(state)
    preferred_text = preferred.value if preferred is not None else "-"
    return (
        f"tactical_play hand={evaluation.hand_type.value} preferred={preferred_text} score={score} remaining={remaining_score} "
        f"hands_needed={hands_needed} hands_left={state.hands_remaining}"
    )


def _discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    current_needed = _estimated_hands_needed(remaining_score, best_score)
    projected_needed = _estimated_hands_needed(remaining_score, projected_score)
    return (
        f"tactical_discard current_score={best_score} projected_score={projected_score} "
        f"hands_needed={current_needed}->{projected_needed} hands_left={state.hands_remaining}"
    )


def _preferred_hand_hunt_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    projected_score: int,
    preferred: HandType,
    current_hand_type: HandType,
    context: _BlindContext | None = None,
    detail: str = "",
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    suffix = f" {detail}" if detail else ""
    return (
        f"preferred_hand_hunt preferred={preferred.value} current_hand={current_hand_type.value} "
        f"current_score={best_score} projected_score={projected_score} "
        f"remaining={remaining_score} discarding={len(discard.card_indices)}{suffix}"
    )


def _preferred_hand_hunt_redraw_play_reason(
    state: GameState,
    best_play: Action,
    candidate: _PlayCandidate,
    evaluation: _StraightDrawEvaluation,
    preferred: HandType,
    current_hand_type: HandType,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    return (
        f"preferred_hand_hunt_redraw preferred={preferred.value} current_hand={current_hand_type.value} "
        f"burn_hand={candidate.hand_type.value} burn_score={candidate.score} current_score={best_score} "
        f"remaining={remaining_score} playing={len(candidate.action.card_indices)} "
        f"{_straight_draw_reason_detail(evaluation)}"
    )


def _last_hand_hunt_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    context = context or _BlindContext()
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    return (
        f"last_hand_hunt current_score={best_score} projected_score={projected_score} "
        f"remaining={remaining_score} discards_left={state.discards_remaining}"
    )


def _winning_economy_hunt_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    gain: float,
    projected_score: int,
    context: _BlindContext | None = None,
) -> str:
    context = context or _BlindContext()
    best_score = _score_play_action(state, best_play, context)
    drawn_cards = _known_draw_for_discard(state, discard)
    drawn_label = ",".join(card.short_name for card in drawn_cards if _card_is_economy_hunt_target(state, card)) or "-"
    return (
        f"winning_economy_hunt current_score={best_score} projected_score={projected_score} "
        f"gain={gain:.1f} targets={drawn_label} discarding={len(discard.card_indices)}"
    )


def _joker_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    context = context or _BlindContext()
    best_score = _score_play_action(state, best_play, context)
    projected_score = _projected_score_after_discard(state, discard, context)
    discarded_cards = tuple(state.hand[index] for index in discard.card_indices)
    names = sorted(_joker_discard_triggers(state, discarded_cards))
    label = ",".join(names) if names else "joker"
    return (
        f"joker_discard triggers={label} current_score={best_score} "
        f"projected_score={projected_score} discarding={len(discard.card_indices)}"
    )


def _joker_discard_triggers(state: GameState, discarded_cards: tuple[Card, ...]) -> set[str]:
    names = _active_joker_names(state)
    triggers: set[str] = set()
    if "Trading Card" in names and len(discarded_cards) == 1:
        triggers.add("Trading Card")
    if "Burnt Joker" in names:
        triggers.add("Burnt Joker")
    if "Hit the Road" in names and any(not card.debuffed and card.rank == "J" for card in discarded_cards):
        triggers.add("Hit the Road")
    castle_suit = _castle_target_suit(state)
    if castle_suit is not None and any(
        not card.debuffed and _normalize_suit(card.suit) == castle_suit for card in discarded_cards
    ):
        triggers.add("Castle")
    mail_rank = _mail_in_rebate_rank(state)
    if mail_rank is not None and any(
        not card.debuffed and _rank_matches(card.rank, mail_rank) for card in discarded_cards
    ):
        triggers.add("Mail-In Rebate")
    if "Faceless Joker" in names and sum(1 for card in discarded_cards if _is_face_card_for_state(state, card)) >= 3:
        triggers.add("Faceless Joker")
    return triggers


def _safety_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    pace_score = remaining_score / max(1, state.hands_remaining)
    safety_score = pace_score * _pace_safety_multiplier(state)
    return (
        f"safety_discard current_score={best_score} projected_score={projected_score} "
        f"safety_score={safety_score:.1f} hands_left={state.hands_remaining} "
        f"discards_left={state.discards_remaining}"
    )


def _panic_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    pace_score = remaining_score / max(1, state.hands_remaining)
    return (
        f"panic_discard current_score={best_score} projected_score={projected_score} "
        f"pace_score={pace_score:.1f} hands_left={state.hands_remaining} "
        f"discards_left={state.discards_remaining}"
    )


def _mystic_summit_setup_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    score: int,
    active_score: int,
    context: _BlindContext | None = None,
) -> str:
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    return (
        f"mystic_summit_setup current_score={score} active_score={active_score} "
        f"projected_score={projected_score} remaining={remaining_score} "
        f"discards_left={state.discards_remaining}"
    )


def _first_blind_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    projected_score = _projected_score_after_discard(state, discard, context)
    return (
        f"first_blind_hunt current_score={best_score} projected_score={projected_score} "
        f"discarding={len(discard.card_indices)}"
    )


def _ante_one_upgrade_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    hand_type: HandType,
    projected_score: int,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    return (
        f"ante_one_upgrade hand={hand_type.value} current_score={best_score} "
        f"projected_score={projected_score} target_score={remaining_score} "
        f"discarding={len(discard.card_indices)}"
    )
