from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, with_derived_legal_actions
from balatro_ai.solver.search_v2.leaf_value import (
    ArchetypeAwareLeaf,
    ClearProbabilityLeaf,
    FastHeuristicLeaf,
    FutureBlindSurvivalLeaf,
)
from balatro_ai.solver.search_v2.play import SearchV2PlayPolicy, solver_beam_play_action


class _FixedFallback:
    def __init__(self, action: Action) -> None:
        self.action = action
        self.calls = 0

    def choose_action(self, _state: GameState) -> Action:
        self.calls += 1
        return self.action


class _FixedLeaf:
    def __init__(self, value: float) -> None:
        self.value = value

    def evaluate(self, _state: GameState) -> float:
        return self.value


class _FixedArchetype:
    def __init__(self, fit: float) -> None:
        self.fit = fit

    def archetype_fit_score(self, _state: GameState) -> float:
        return self.fit


class SolverSearchV2PlayTests(unittest.TestCase):
    def test_beam_prefers_immediate_clear(self) -> None:
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
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(2,)),
            ),
        )

        action = solver_beam_play_action(state, depth=2, width=2, leaf_evaluator=ClearProbabilityLeaf(samples=1))

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))

    def test_beam_can_discard_into_next_play_clear(self) -> None:
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

        action = solver_beam_play_action(state, depth=2, width=4, leaf_evaluator=ClearProbabilityLeaf(samples=1))

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.DISCARD)

    def test_policy_delegates_non_hand_phase(self) -> None:
        fallback = _FixedFallback(Action(ActionType.SELECT_BLIND))
        policy = SearchV2PlayPolicy(fallback=fallback)
        state = GameState(phase=GamePhase.BLIND_SELECT, legal_actions=(Action(ActionType.SELECT_BLIND),))

        action = policy.choose_action(state)

        self.assertEqual(action.action_type, ActionType.SELECT_BLIND)
        self.assertEqual(fallback.calls, 1)


class SolverSearchV2LeafTests(unittest.TestCase):
    def test_fast_leaf_rewards_cleared_blind_over_partial_progress(self) -> None:
        partial = GameState(
            phase=GamePhase.SELECTING_HAND,
            required_score=100,
            current_score=50,
            hands_remaining=2,
            discards_remaining=1,
            money=4,
        )
        cleared = GameState(
            phase=GamePhase.ROUND_EVAL,
            required_score=100,
            current_score=100,
            hands_remaining=2,
            discards_remaining=1,
            money=4,
        )

        leaf = FastHeuristicLeaf()

        self.assertGreater(leaf.evaluate(cleared), leaf.evaluate(partial))

    def test_future_leaf_scores_round_eval_as_survival(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            required_score=100,
            current_score=150,
            hands_remaining=2,
        )

        self.assertGreaterEqual(FutureBlindSurvivalLeaf(samples=1).evaluate(state), 1.0)

    def test_archetype_leaf_adds_fit_bonus(self) -> None:
        leaf = ArchetypeAwareLeaf(
            base=_FixedLeaf(1.0),
            archetype=_FixedArchetype(4.0),
            bonus_weight=0.1,
        )

        self.assertAlmostEqual(leaf.evaluate(GameState()), 1.4)

if __name__ == "__main__":
    unittest.main()
