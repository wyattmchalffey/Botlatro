"""Greedy immediate-score bot."""

from __future__ import annotations

from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.random_bot import RandomBot
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards


@dataclass(slots=True)
class GreedyBot:
    """Pick the legal play-hand action with the highest immediate score."""

    seed: int | None = None
    name: str = "greedy_bot"
    _fallback: RandomBot = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback = RandomBot(seed=self.seed)

    def choose_action(self, state: GameState) -> Action:
        for preferred_type in (
            ActionType.SELECT_BLIND,
            ActionType.CASH_OUT,
            ActionType.END_SHOP,
        ):
            action = _first_action_of_type(state, preferred_type)
            if action is not None:
                return action

        pack_skip = _pack_skip_action(state)
        if pack_skip is not None:
            return pack_skip

        play_actions = [
            action
            for action in state.legal_actions
            if action.action_type == ActionType.PLAY_HAND and action.card_indices
        ]
        if not play_actions:
            return self._fallback.choose_action(state)

        return max(play_actions, key=lambda action: self._score_action(state, action))

    def _score_action(self, state: GameState, action: Action) -> int:
        cards = tuple(state.hand[index] for index in action.card_indices)
        held_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        return evaluate_played_cards(
            cards,
            state.hand_levels,
            debuffed_suits=debuffed_suits_for_blind(state.blind),
            blind_name=state.blind,
            jokers=state.jokers,
            discards_remaining=state.discards_remaining,
            held_cards=held_cards,
            deck_size=state.deck_size,
        ).score


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
