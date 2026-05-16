"""Discard action policy for blind play."""

from __future__ import annotations

from math import ceil
import sys

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy.blind_state import _is_boss_blind
from balatro_ai.bots.basic_strategy.cards import _is_face_card_for_state, _normalize_suit, _rank_matches
from balatro_ai.bots.basic_strategy.data import (
    BURNT_JOKER_DISCARD_HAND_VALUES,
    DISCARD_PENALTY_JOKERS,
)
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type
from balatro_ai.bots.basic_strategy.hand_value import (
    _card_keep_scores,
    _cheap_kept_hand_potential,
    _kept_hand_potential,
)
from balatro_ai.bots.basic_strategy.jokers import (
    _active_joker_names,
    _castle_target_suit,
    _current_plus_for_joker,
    _mail_in_rebate_rank,
)
from balatro_ai.bots.basic_strategy.play_scoring import _best_play_action, _score_play_action
from balatro_ai.bots.basic_strategy.score_projection import (
    _projected_score_after_discard as _default_projected_score_after_discard,
    _strong_draw_size,
)
from balatro_ai.bots.config import active_config
from balatro_ai.rules.hand_evaluator import HandType, RANK_VALUES, evaluate_played_cards


DISCARD_DETAIL_LIMIT = 28
LATE_DISCARD_DETAIL_LIMIT = 12


def _should_chase_discard(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    projected_score = _projected_score_after_discard(state, action, context)
    if state.known_deck:
        current_hands_needed = _estimated_hands_needed(remaining_score, current_score)
        projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
        return projected_score > current_score and projected_hands_needed <= current_hands_needed

    return _unknown_discard_projection_is_trustworthy(
        state,
        kept_cards=kept_cards,
        current_score=current_score,
        projected_score=projected_score,
        remaining_score=remaining_score,
        weak_chase=True,
    )


def _discard_penalty_jokers_active(state: GameState) -> bool:
    return bool(_active_joker_names(state) & DISCARD_PENALTY_JOKERS)


def _projected_score_after_discard(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._projected_score_after_discard``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_projected_score_after_discard", None) if strategy_module is not None else None
    return (patched or _default_projected_score_after_discard)(*args, **kwargs)


def _should_panic_discard(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    if remaining_score <= 0 or state.hands_remaining <= 1 or state.discards_remaining <= 0:
        return False

    pace_score = remaining_score / max(1, state.hands_remaining)
    if pace_score <= 0 or current_score >= pace_score * _panic_discard_ratio(state):
        return False

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    projected_score = _projected_score_after_discard(state, action, context)
    if projected_score > current_score:
        return True

    return _strong_draw_size(kept_cards) >= 3 and len(action.card_indices) >= 2


def _should_safety_discard(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    if remaining_score <= 0 or state.hands_remaining <= 1 or state.discards_remaining <= 0:
        return False

    pace_score = remaining_score / max(1, state.hands_remaining)
    safety_score = pace_score * _pace_safety_multiplier(state)
    if current_score >= safety_score:
        return False

    projected_score = _projected_score_after_discard(state, action, context)
    if state.discards_remaining <= 1 and projected_score < current_score:
        return False
    discard_penalty = _discard_penalty_jokers_active(state)
    if (
        state.ante >= 3
        and state.discards_remaining <= 1
        and not state.known_deck
        and projected_score < remaining_score
    ):
        projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
        if current_score >= pace_score or projected_hands_needed >= state.hands_remaining:
            return False
    if projected_score >= safety_score:
        if discard_penalty and projected_score <= current_score:
            return False
        return True

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    if state.discards_remaining <= 1 and not state.known_deck:
        return projected_score >= max(current_score * 1.35, pace_score)
    if discard_penalty:
        return projected_score >= max(current_score * 1.2, pace_score * 0.9)

    return projected_score >= max(current_score * 1.2, pace_score * 0.9) or _strong_draw_size(kept_cards) >= 4


def _should_last_hand_hunt_discard(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    if remaining_score <= 0 or state.hands_remaining > 1 or state.discards_remaining <= 0:
        return False

    projected_score = _projected_score_after_discard(state, action, context)
    if projected_score >= remaining_score and projected_score >= current_score:
        return True
    if current_score < remaining_score and state.discards_remaining > 1:
        return True
    if state.known_deck:
        return projected_score > current_score
    if projected_score > current_score:
        return True
    if _discard_penalty_jokers_active(state):
        return False

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    return _strong_draw_size(kept_cards) >= 4


def _panic_discard_ratio(state: GameState) -> float:
    cfg = active_config()
    ratio = cfg.panic_discard_base_ratio
    if state.ante >= 4:
        ratio += cfg.panic_discard_ante4_bonus
    if _is_boss_blind(state):
        ratio += cfg.panic_discard_boss_bonus
    if state.hands_remaining <= 2:
        ratio += cfg.panic_discard_low_hands_bonus
    if state.discards_remaining <= 1 and not state.known_deck:
        ratio -= cfg.panic_discard_desperate_penalty
    return max(cfg.panic_discard_floor, min(cfg.panic_discard_cap, ratio))


def _best_discard_action(
    state: GameState,
    *,
    current_best_score: int | None = None,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        return None

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    protected_count = max(2, min(4, len(state.hand) // 2))
    protected = {
        index
        for index, _ in sorted(
            enumerate(keep_scores),
            key=lambda item: (item[1], RANK_VALUES[state.hand[item[0]].rank]),
            reverse=True,
        )[:protected_count]
    }

    remaining_score = max(0, state.required_score - state.current_score)
    if current_best_score is None:
        current_best_play = _best_play_action(state, context)
        current_best_score = _score_play_action(state, current_best_play, context) if current_best_play is not None else 0
    current_hands_needed = _estimated_hands_needed(remaining_score, current_best_score or 0)
    detailed_actions = _prefilter_discard_actions(
        state,
        discard_actions,
        keep_scores,
        protected,
        preferred,
        limit=_discard_detail_limit(state),
    )

    def discard_score(action: Action) -> tuple[float, float, int, int]:
        protected_penalty = sum(1 for index in action.card_indices if index in protected) * 1000
        desirability = sum(100 - keep_scores[index] for index in action.card_indices)
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        kept_potential = _kept_hand_potential(state, kept_cards)
        projected_score = _projected_score_after_discard(state, action, context)
        projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
        speed_bonus = max(0, current_hands_needed - projected_hands_needed) * 900
        score_bonus = min(projected_score, remaining_score) * 0.08
        playstyle_bonus = _discard_action_playstyle_bonus(
            state,
            action,
            discarded_cards=discarded_cards,
            keep_scores=keep_scores,
            context=context,
        )
        return (
            desirability + kept_potential + speed_bonus + score_bonus + playstyle_bonus - protected_penalty,
            projected_score,
            kept_potential,
            len(action.card_indices),
        )

    best = max(detailed_actions, key=discard_score)
    return best if discard_score(best)[0] > -500 else None


def _discard_action_playstyle_bonus(
    state: GameState,
    action: Action,
    *,
    discarded_cards: tuple[Card, ...],
    keep_scores: tuple[float, ...],
    context: _BlindContext | None = None,
) -> float:
    context = context or _BlindContext()
    names = _active_joker_names(state)
    bonus = 0.0
    if not discarded_cards:
        return bonus

    if "Trading Card" in names and _is_first_discard_window(state, context) and len(discarded_cards) == 1:
        index = action.card_indices[0]
        bonus += 780.0 + max(0.0, 140.0 - keep_scores[index])
    if "Burnt Joker" in names and _is_first_discard_window(state, context):
        hand_type = evaluate_played_cards(discarded_cards, state.hand_levels).hand_type
        bonus += BURNT_JOKER_DISCARD_HAND_VALUES.get(hand_type, 0.0)
    if "Hit the Road" in names:
        bonus += 520.0 * sum(1 for card in discarded_cards if not card.debuffed and card.rank == "J")
    castle_suit = _castle_target_suit(state)
    if castle_suit is not None:
        bonus += 240.0 * sum(
            1 for card in discarded_cards if not card.debuffed and _normalize_suit(card.suit) == castle_suit
        )
    mail_rank = _mail_in_rebate_rank(state)
    if mail_rank is not None:
        bonus += 180.0 * sum(
            1 for card in discarded_cards if not card.debuffed and _rank_matches(card.rank, mail_rank)
        )
    if "Faceless Joker" in names:
        face_count = sum(1 for card in discarded_cards if _is_face_card_for_state(state, card))
        if face_count >= 3:
            bonus += 520.0 + face_count * 120.0
    if "Ride the Bus" in names and "Pareidolia" not in names:
        bonus += 90.0 * sum(1 for card in discarded_cards if _is_face_card_for_state(state, card))
    if "Green Joker" in names:
        bonus -= 280.0 + _current_plus_for_joker(state, "Green Joker", suffix="mult") * 30.0
    if "Delayed Gratification" in names:
        bonus -= 420.0
    if "Ramen" in names:
        bonus -= 80.0 * len(discarded_cards)
    return bonus


def _is_first_discard_window(state: GameState, context: _BlindContext) -> bool:
    if context.discards_taken > 0:
        return False
    if context.played_hand_types or state.current_score > 0:
        return False
    raw_discards = state.modifiers.get("round_discards_used", state.modifiers.get("discards_used"))
    try:
        if raw_discards is not None:
            return int(raw_discards) == 0
    except (TypeError, ValueError):
        pass
    return state.discards_remaining >= 3


def _prefilter_discard_actions(
    state: GameState,
    discard_actions: list[Action],
    keep_scores: tuple[float, ...],
    protected: set[int],
    preferred: HandType | None,
    *,
    limit: int = DISCARD_DETAIL_LIMIT,
) -> list[Action]:
    if len(discard_actions) <= limit:
        return discard_actions

    def cheap_score(action: Action) -> tuple[float, int]:
        protected_penalty = sum(1 for index in action.card_indices if index in protected) * 1000
        desirability = sum(100 - keep_scores[index] for index in action.card_indices)
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        return (
            desirability + _cheap_kept_hand_potential(kept_cards, preferred) - protected_penalty,
            len(action.card_indices),
        )

    return sorted(discard_actions, key=cheap_score, reverse=True)[:limit]


def _discard_detail_limit(state: GameState) -> int:
    if state.ante >= 3:
        return LATE_DISCARD_DETAIL_LIMIT
    return DISCARD_DETAIL_LIMIT


def _estimated_hands_needed(remaining_score: int, score: int | float) -> int:
    if remaining_score <= 0:
        return 0
    return ceil(remaining_score / max(1, score))


def _score_is_on_pace(state: GameState, score: int, remaining_score: int) -> bool:
    if state.hands_remaining <= 0:
        return True
    pace_score = remaining_score / max(1, state.hands_remaining)
    return score >= pace_score * _pace_safety_multiplier(state)


def _pace_safety_multiplier(state: GameState) -> float:
    multiplier = active_config().hand_pace_safety_base
    if state.ante >= 4:
        multiplier += 0.05
    if state.ante >= 5:
        multiplier += 0.06
    if _is_boss_blind(state):
        multiplier += 0.05
    if state.blind in {"The Wall", "The Needle"}:
        multiplier += 0.12
    elif state.blind in {"The Eye", "The Mouth"}:
        multiplier += 0.08
    elif state.blind in {"The Water", "The Arm"}:
        multiplier += 0.06
    if state.hands_remaining <= 2:
        multiplier += 0.04
    if state.discards_remaining >= 3 and state.hands_remaining >= 3:
        multiplier += 0.03
    return multiplier


def _discard_can_reduce_hands_needed(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    current_hands_needed = _estimated_hands_needed(remaining_score, current_score)
    if current_hands_needed <= 1:
        return False

    projected_score = _projected_score_after_discard(state, action, context)
    projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
    if projected_hands_needed >= current_hands_needed:
        return False
    if state.known_deck:
        return True

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    return _unknown_discard_projection_is_trustworthy(
        state,
        kept_cards=kept_cards,
        current_score=current_score,
        projected_score=projected_score,
        remaining_score=remaining_score,
    )


def _unknown_discard_projection_is_trustworthy(
    state: GameState,
    *,
    kept_cards: tuple[Card, ...],
    current_score: int,
    projected_score: int,
    remaining_score: int,
    weak_chase: bool = False,
) -> bool:
    if state.ante >= 3 and state.discards_remaining <= 1 and projected_score < remaining_score:
        return False

    draw_size = _strong_draw_size(kept_cards)
    if state.ante >= 3:
        if state.discards_remaining >= 3 and state.hands_remaining >= 3:
            if weak_chase:
                return projected_score >= current_score * 1.1 or draw_size >= 3
            return projected_score >= current_score * 1.25 or draw_size >= 3
        if draw_size < 4:
            return False
        return projected_score >= max(
            current_score * 1.6,
            (remaining_score / max(1, state.hands_remaining)) * _pace_safety_multiplier(state),
        )

    if weak_chase:
        return projected_score >= current_score * 1.1 or draw_size >= 2
    return draw_size >= 4 and projected_score >= current_score * 1.35
