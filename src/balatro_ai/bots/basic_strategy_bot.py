"""Basic rule bot with simple play/discard discipline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import ceil

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.random_bot import RandomBot
from balatro_ai.rules.hand_evaluator import RANK_VALUES, STRAIGHT_VALUES, debuffed_suits_for_blind, evaluate_played_cards


@dataclass(slots=True)
class BasicStrategyBot:
    """A small step above immediate-score greed.

    The bot avoids random shop spending, plays hands that are good enough for
    the current blind, and uses discards when the best immediate hand is behind
    the score pace.
    """

    seed: int | None = None
    name: str = "basic_strategy_bot"
    _fallback: RandomBot = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback = RandomBot(seed=self.seed)

    def choose_action(self, state: GameState) -> Action:
        for preferred_type in (
            ActionType.SELECT_BLIND,
            ActionType.CASH_OUT,
        ):
            action = _first_action_of_type(state, preferred_type)
            if action is not None:
                return action

        shop_buy = _shop_buy_action(state)
        if shop_buy is not None:
            return shop_buy

        end_shop = _first_action_of_type(state, ActionType.END_SHOP)
        if end_shop is not None:
            return end_shop

        pack_skip = _pack_skip_action(state)
        if pack_skip is not None:
            return pack_skip

        best_play = _best_play_action(state)
        if best_play is None:
            return self._fallback.choose_action(state)

        if _should_play_now(state, best_play):
            return best_play

        discard = _best_discard_action(state)
        return discard or best_play


def _first_action_of_type(state: GameState, action_type: ActionType) -> Action | None:
    for action in state.legal_actions:
        if action.action_type == action_type:
            return action
    return None


def _pack_skip_action(state: GameState) -> Action | None:
    for action in state.legal_actions:
        if action.action_type == ActionType.CHOOSE_PACK_CARD and action.target_id == "skip":
            return action
    return None


def _shop_buy_action(state: GameState) -> Action | None:
    if len(state.jokers) >= 2:
        return None

    shop_cards = state.modifiers.get("shop_cards", ())
    for action in state.legal_actions:
        if action.action_type != ActionType.BUY or action.metadata.get("kind") != "card":
            continue
        index = int(action.metadata.get("index", action.amount or 0))
        if index < len(shop_cards) and _is_joker_card(shop_cards[index]):
            return action
    return None


def _is_joker_card(card: object) -> bool:
    if not isinstance(card, dict):
        return False
    label = str(card.get("label", card.get("name", card.get("key", ""))))
    key = str(card.get("key", ""))
    card_set = str(card.get("set", "")).upper()
    return card_set == "JOKER" or "joker" in label.lower() or key.startswith("j_")


def _best_play_action(state: GameState) -> Action | None:
    play_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.PLAY_HAND and action.card_indices
    ]
    if not play_actions:
        return None
    return max(play_actions, key=lambda action: _score_play_action(state, action))


def _score_play_action(state: GameState, action: Action) -> int:
    cards = tuple(state.hand[index] for index in action.card_indices)
    held_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
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
    ).score


def _should_play_now(state: GameState, action: Action) -> bool:
    score = _score_play_action(state, action)
    remaining_score = max(0, state.required_score - state.current_score)
    if remaining_score == 0:
        return True
    if score >= remaining_score:
        return True
    if state.hands_remaining <= 1 or state.discards_remaining <= 0:
        return True

    needed_per_remaining_hand = ceil(remaining_score / max(1, state.hands_remaining))
    return score >= needed_per_remaining_hand


def _best_discard_action(state: GameState) -> Action | None:
    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        return None

    keep_scores = _card_keep_scores(state.hand)
    protected_count = max(2, min(4, len(state.hand) // 2))
    protected = {
        index
        for index, _ in sorted(
            enumerate(keep_scores),
            key=lambda item: (item[1], RANK_VALUES[state.hand[item[0]].rank]),
            reverse=True,
        )[:protected_count]
    }

    def discard_score(action: Action) -> tuple[float, int]:
        protected_penalty = sum(1 for index in action.card_indices if index in protected) * 1000
        desirability = sum(100 - keep_scores[index] for index in action.card_indices)
        return (desirability - protected_penalty, len(action.card_indices))

    best = max(discard_actions, key=discard_score)
    return best if discard_score(best)[0] > -500 else None


def _card_keep_scores(hand: tuple[Card, ...]) -> tuple[float, ...]:
    rank_counts = Counter(card.rank for card in hand)
    suit_counts = Counter(card.suit for card in hand)
    straight_values = [STRAIGHT_VALUES[card.rank] for card in hand]

    scores: list[float] = []
    for card in hand:
        straight_value = STRAIGHT_VALUES[card.rank]
        nearby = sum(
            1
            for value in straight_values
            if value != straight_value and abs(value - straight_value) <= 4
        )
        ace_low_nearby = 0
        if card.rank == "A":
            ace_low_nearby = sum(1 for value in straight_values if value in {2, 3, 4, 5})

        rank_count = rank_counts[card.rank]
        score = RANK_VALUES[card.rank]
        score += suit_counts[card.suit] * 8
        score += max(nearby, ace_low_nearby) * 6
        if rank_count >= 2:
            score += rank_count * 100
        scores.append(float(score))

    return tuple(scores)
