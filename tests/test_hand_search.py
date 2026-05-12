from __future__ import annotations

import unittest
from unittest.mock import patch

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, with_derived_legal_actions
from balatro_ai.search import hand_search
from balatro_ai.search.hand_search import HandSearchConfig, best_blind_beam_action, best_hand_action


class HandSearchTests(unittest.TestCase):
    def test_prefers_immediate_clear_to_first_blind_discard_value(self) -> None:
        play_pair = Action(ActionType.PLAY_HAND, card_indices=(0, 1))
        discard = Action(ActionType.DISCARD, card_indices=(2,))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=1,
            blind="Small Blind",
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=10,
            hand=(Card("A", "S"), Card("A", "H"), Card("2", "D")),
            known_deck=(Card("K", "C"),),
            legal_actions=(play_pair, discard),
        )

        action = best_hand_action(state, config=HandSearchConfig(draw_samples=2, leaf_samples=1, seed=1))

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))
        self.assertEqual(action.metadata["search"], "hand_expectimax")

    def test_hunts_for_one_hand_clear_when_first_blind_hand_is_weak(self) -> None:
        state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.SELECTING_HAND,
                ante=1,
                blind="Small Blind",
                required_score=300,
                current_score=0,
                hands_remaining=4,
                discards_remaining=3,
                deck_size=45,
                hand=(
                    Card("9", "H"),
                    Card("7", "D"),
                    Card("6", "H"),
                    Card("6", "D"),
                    Card("5", "C"),
                    Card("4", "D"),
                    Card("3", "H"),
                    Card("3", "C"),
                ),
            )
        )

        action = best_hand_action(state, config=HandSearchConfig(draw_samples=2, leaf_samples=1, seed=1))

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.metadata["search"], "hand_expectimax")

    def test_preserves_sixth_sense_setup_play(self) -> None:
        play_six = Action(ActionType.PLAY_HAND, card_indices=(0,))
        play_pair = Action(ActionType.PLAY_HAND, card_indices=(1, 2))
        discard = Action(ActionType.DISCARD, card_indices=(3,))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=1,
            blind="Small Blind",
            required_score=100,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=10,
            hand=(Card("6", "S"), Card("A", "S"), Card("A", "H"), Card("2", "D")),
            jokers=(Joker("Sixth Sense"),),
            legal_actions=(play_six, play_pair, discard),
        )

        action = best_hand_action(state, config=HandSearchConfig(draw_samples=2, leaf_samples=1, seed=1))

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0,))
        self.assertIn("joker_setup", action.metadata["reason"])

    def test_whole_blind_beam_can_discard_into_next_play_clear(self) -> None:
        state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.SELECTING_HAND,
                ante=1,
                blind="Small Blind",
                required_score=50,
                current_score=0,
                hands_remaining=1,
                discards_remaining=1,
                deck_size=1,
                hand=(Card("A", "S"), Card("K", "H"), Card("7", "D"), Card("4", "C")),
                known_deck=(Card("A", "H"),),
            )
        )

        action = best_blind_beam_action(
            state,
            config=HandSearchConfig(
                seed=1,
                max_play_actions=6,
                max_discard_actions=4,
                beam_depth=2,
                beam_width=4,
                beam_draw_samples=1,
                beam_leaf_samples=1,
            ),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.metadata["search"], "hand_beam")

    def test_beam_future_actions_do_not_call_basic_best_play(self) -> None:
        state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.SELECTING_HAND,
                ante=2,
                blind="Big Blind",
                required_score=800,
                current_score=0,
                hands_remaining=3,
                discards_remaining=2,
                deck_size=20,
                hand=(
                    Card("A", "S"),
                    Card("A", "H"),
                    Card("K", "D"),
                    Card("Q", "C"),
                    Card("J", "S"),
                    Card("10", "S"),
                ),
                jokers=(Joker("Joker"),),
            )
        )

        with patch.object(hand_search, "_best_play_action", side_effect=AssertionError("beam future used basic heuristic")):
            actions = hand_search._beam_future_actions(state, HandSearchConfig(beam_width=2))

        self.assertTrue(actions)
        self.assertLessEqual(len(actions), 3)

    def test_beam_leaf_value_memoizes_equivalent_states(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=2,
            discards_remaining=1,
            deck_size=10,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            jokers=(Joker("Joker"),),
        )
        equivalent_state = GameState(
            phase=state.phase,
            blind=state.blind,
            required_score=state.required_score,
            current_score=state.current_score,
            hands_remaining=state.hands_remaining,
            discards_remaining=state.discards_remaining,
            money=state.money,
            deck_size=state.deck_size,
            hand=tuple(reversed(state.hand)),
            jokers=state.jokers,
            hand_levels=dict(state.hand_levels),
            modifiers=dict(state.modifiers),
        )
        config = HandSearchConfig(seed=1, beam_leaf_samples=1)
        memo: dict[tuple[object, ...], float] = {}

        with patch.object(hand_search, "planning_value", side_effect=(0.42, 0.99)) as value:
            first = hand_search._beam_leaf_value(state, config=config, seed_offset=0, memo=memo)
            second = hand_search._beam_leaf_value(equivalent_state, config=config, seed_offset=10, memo=memo)

        self.assertEqual(first, 0.42)
        self.assertEqual(second, 0.42)
        self.assertEqual(value.call_count, 1)


if __name__ == "__main__":
    unittest.main()
