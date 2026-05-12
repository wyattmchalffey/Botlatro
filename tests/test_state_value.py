from __future__ import annotations

import unittest
from unittest.mock import patch

import context  # noqa: F401
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.search import state_value as state_values
from balatro_ai.search.state_value import clear_probability, future_value, headroom_value, planning_value, state_value


class StateValueTests(unittest.TestCase):
    def test_clear_probability_is_one_for_already_clear_state(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            required_score=100,
            current_score=100,
            hands_remaining=1,
            hand=(Card("2", "S"),),
        )

        self.assertEqual(clear_probability(state, samples=3), 1.0)

    def test_clear_probability_is_one_for_round_eval_state(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            required_score=100,
            current_score=40,
            hands_remaining=0,
            hand=(),
        )

        self.assertEqual(clear_probability(state, samples=3), 1.0)

    def test_clear_probability_is_zero_for_dead_state(self) -> None:
        state = GameState(
            phase=GamePhase.RUN_OVER,
            required_score=100,
            current_score=20,
            hands_remaining=0,
            run_over=True,
            hand=(Card("A", "S"), Card("A", "H")),
        )

        self.assertEqual(clear_probability(state, samples=3), 0.0)

    def test_clear_probability_rollout_clears_with_greedy_pair(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            current_score=0,
            hands_remaining=1,
            deck_size=0,
            hand=(Card("A", "S"), Card("A", "H"), Card("2", "D")),
        )

        self.assertEqual(clear_probability(state, samples=5, seed=7), 1.0)

    def test_clear_probability_rollout_fails_when_best_hand_is_short(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            hands_remaining=1,
            deck_size=0,
            hand=(Card("2", "S"), Card("7", "H")),
        )

        self.assertEqual(clear_probability(state, samples=5, seed=7), 0.0)

    def test_clear_probability_rollout_can_discard_dead_cards(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            current_score=0,
            hands_remaining=1,
            discards_remaining=1,
            deck_size=3,
            hand=(Card("A", "S"), Card("7", "H"), Card("4", "D"), Card("2", "C")),
            known_deck=(Card("A", "H"), Card("K", "S"), Card("Q", "D")),
        )

        self.assertEqual(clear_probability(state, samples=3, seed=7), 1.0)

    def test_future_value_is_monotonic_for_stronger_state(self) -> None:
        weak = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            money=0,
            hand=(Card("2", "S"), Card("7", "H")),
        )
        strong = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            money=25,
            hand=(Card("A", "S"), Card("A", "H"), Card("A", "D")),
            jokers=(Joker("Joker"), Joker("Sly Joker")),
        )

        self.assertGreater(future_value(strong), future_value(weak))

    def test_planning_value_rewards_headroom_above_bare_clear(self) -> None:
        bare_clear = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            current_score=0,
            hands_remaining=1,
            deck_size=0,
            hand=(Card("A", "S"), Card("A", "H")),
        )
        surplus_clear = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            current_score=0,
            hands_remaining=1,
            deck_size=0,
            hand=(Card("A", "S"), Card("A", "H"), Card("A", "D"), Card("A", "C")),
            jokers=(Joker("Joker"),),
        )

        self.assertGreater(headroom_value(surplus_clear), headroom_value(bare_clear))
        self.assertGreater(planning_value(surplus_clear, samples=1), planning_value(bare_clear, samples=1))

    def test_search_value_reuses_best_score_for_equivalent_hand_content(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            hands_remaining=2,
            discards_remaining=1,
            deck_size=20,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D"), Card("Q", "C")),
            jokers=(Joker("Joker"),),
        )
        equivalent_state = GameState(
            phase=state.phase,
            blind=state.blind,
            required_score=state.required_score,
            current_score=state.current_score,
            hands_remaining=state.hands_remaining,
            discards_remaining=state.discards_remaining,
            deck_size=state.deck_size,
            hand=tuple(reversed(state.hand)),
            jokers=state.jokers,
            hand_levels=dict(state.hand_levels),
            modifiers=dict(state.modifiers),
        )

        with state_values.state_value_cache_scope():
            with patch.object(state_values, "evaluate_played_cards", wraps=state_values.evaluate_played_cards) as evaluate:
                first_score = state_values._best_immediate_score(state)
                calls_after_first = evaluate.call_count
                second_score = state_values._best_immediate_score(equivalent_state)

        self.assertEqual(second_score, first_score)
        self.assertGreater(calls_after_first, 0)
        self.assertEqual(evaluate.call_count, calls_after_first)

    def test_search_value_reuses_greedy_action_score_for_equivalent_hand_content(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            hands_remaining=2,
            discards_remaining=1,
            deck_size=20,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D"), Card("Q", "C")),
            jokers=(Joker("Joker"),),
        )
        equivalent_state = GameState(
            phase=state.phase,
            blind=state.blind,
            required_score=state.required_score,
            current_score=state.current_score,
            hands_remaining=state.hands_remaining,
            discards_remaining=state.discards_remaining,
            deck_size=state.deck_size,
            hand=tuple(reversed(state.hand)),
            jokers=state.jokers,
            hand_levels=dict(state.hand_levels),
            modifiers=dict(state.modifiers),
        )

        with state_values.state_value_cache_scope():
            with patch.object(state_values, "evaluate_played_cards", wraps=state_values.evaluate_played_cards) as evaluate:
                first_action = state_values._best_greedy_play_action(state)
                calls_after_first = evaluate.call_count
                second_action = state_values._best_greedy_play_action(equivalent_state)

        self.assertIsNotNone(first_action)
        self.assertIsNotNone(second_action)
        self.assertEqual(
            sorted(state.hand[index].short_name for index in first_action.card_indices),
            sorted(equivalent_state.hand[index].short_name for index in second_action.card_indices),
        )
        self.assertGreater(calls_after_first, 0)
        self.assertEqual(evaluate.call_count, calls_after_first)

    def test_future_value_cashes_out_round_eval_leaf(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            required_score=100,
            current_score=120,
            money=0,
            hands_remaining=3,
            hand=(),
        )

        self.assertAlmostEqual(future_value(state), 0.024)

    def test_state_value_rewards_cash_out_economy_after_clear(self) -> None:
        lean_clear = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            required_score=100,
            current_score=120,
            money=0,
            hands_remaining=0,
            hand=(),
        )
        efficient_clear = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            required_score=100,
            current_score=120,
            money=0,
            hands_remaining=3,
            hand=(),
        )

        self.assertGreater(state_value(efficient_clear, samples=3), state_value(lean_clear, samples=3))

    def test_future_value_round_eval_to_do_list_fallback_does_not_raise(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Big Blind",
            required_score=1000,
            current_score=1200,
            money=10,
            hands_remaining=1,
            hand=(),
            jokers=(Joker("To Do List", metadata={"target_hand": "Pair"}),),
        )

        value = future_value(state)

        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_state_value_stays_in_unit_interval(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=100,
            current_score=0,
            hands_remaining=1,
            hand=(Card("A", "S"), Card("A", "H")),
        )

        value = state_value(state, samples=5, seed=1)

        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
