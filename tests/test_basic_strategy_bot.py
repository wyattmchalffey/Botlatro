from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import unittest
from unittest.mock import patch

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.bots import basic_strategy_bot as strategy
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.rules.hand_evaluator import HandType


def _all_blind_actions(card_count: int) -> tuple[Action, ...]:
    actions: list[Action] = []
    for size in range(1, min(5, card_count) + 1):
        for indices in combinations(range(card_count), size):
            actions.append(Action(ActionType.PLAY_HAND, card_indices=indices))
            actions.append(Action(ActionType.DISCARD, card_indices=indices))
    return tuple(actions)


class BasicStrategyBotTests(unittest.TestCase):
    def test_plays_when_best_hand_beats_remaining_score(self) -> None:
        state = GameState(
            required_score=100,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("K", "C"),
                Card("2", "D"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)),
                Action(ActionType.DISCARD, card_indices=(3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1, 2))

    def test_rearranges_jokers_when_order_improves_score(self) -> None:
        state = GameState(
            phase=strategy.GamePhase.SELECTING_HAND,
            required_score=1200,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "D"),
                Card("2", "S"),
                Card("3", "C"),
            ),
            jokers=(Joker("Blackboard"), Joker("Joker", edition="HOLO")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REARRANGE)
        self.assertEqual(action.card_indices, (1, 0))

    def test_joker_rearrange_search_is_bounded_for_full_joker_lineups(self) -> None:
        jokers = (
            Joker("Blue Joker", edition="NEGATIVE"),
            Joker("Wrathful Joker"),
            Joker("Baseball Card"),
            Joker("Jolly Joker"),
            Joker("Green Joker"),
            Joker("The Idol"),
        )

        orders = strategy._joker_rearrange_candidate_orders(jokers)

        self.assertLessEqual(len(orders), 4)
        self.assertIn((0, 1, 3, 4, 2, 5), orders)

    def test_joker_rearrange_search_stays_exhaustive_for_small_lineups(self) -> None:
        jokers = (
            Joker("Joker"),
            Joker("Blackboard"),
            Joker("Jolly Joker"),
        )

        orders = strategy._joker_rearrange_candidate_orders(jokers)

        self.assertEqual(len(orders), 6)

    def test_joker_rearrange_search_uses_bounded_heuristics_for_five_jokers(self) -> None:
        jokers = (
            Joker("Joker"),
            Joker("Blackboard"),
            Joker("Jolly Joker"),
            Joker("The Duo"),
            Joker("Blue Joker"),
        )

        orders = strategy._joker_rearrange_candidate_orders(jokers)

        self.assertLessEqual(len(orders), 4)

    def test_prefers_smallest_hand_that_beats_remaining_score(self) -> None:
        state = GameState(
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("K", "C"),
                Card("2", "D"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))

    def test_adds_junk_kickers_to_scoring_play_to_cycle_deck(self) -> None:
        state = GameState(
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "C"),
                Card("9", "D"),
                Card("2", "C"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1, 2, 3, 4))
        self.assertIn("tactical_play", action.metadata["reason"])

    def test_does_not_cycle_cards_the_current_archetype_wants_to_keep(self) -> None:
        state = GameState(
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "D"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("T", "H"),
            ),
            jokers=(Joker("Droll Joker"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4, 5)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))

    def test_prefers_hand_that_reduces_hands_needed_over_pace_hand(self) -> None:
        state = GameState(
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("K", "C"),
                Card("2", "D"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)),
                Action(ActionType.DISCARD, card_indices=(3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1, 2))

    def test_discards_when_best_hand_is_behind_score_pace(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("9", "C"),
                Card("5", "D"),
                Card("4", "S"),
                Card("2", "C"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(3, 4, 5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertNotIn(0, action.card_indices)
        self.assertNotIn(1, action.card_indices)

    def test_plays_on_pace_when_discard_has_no_clear_speed_edge(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("9", "C"),
                Card("5", "D"),
                Card("4", "S"),
                Card("2", "C"),
            ),
            jokers=(Joker("Joker"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))
        self.assertIn("tactical_play", action.metadata["reason"])

    def test_big_blind_hunts_preferred_straight_draw_over_marginal_pair(self) -> None:
        state = GameState(
            ante=5,
            blind="Big Blind",
            required_score=700,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("9", "S"),
                Card("8", "H"),
                Card("7", "D"),
                Card("5", "C"),
                Card("A", "S"),
                Card("A", "H"),
                Card("2", "D"),
            ),
            jokers=(Joker("Devious Joker"), Joker("Joker")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(4, 5)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 2)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (4, 5, 6))
        self.assertIn("preferred_hand_hunt", action.metadata["reason"])

    def test_small_blind_hunts_clear_straight_draw_before_on_pace_play(self) -> None:
        state = GameState(
            ante=5,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=44,
            hand=(
                Card("9", "S"),
                Card("8", "H"),
                Card("7", "D"),
                Card("5", "C"),
                Card("A", "S"),
                Card("A", "H"),
                Card("2", "D"),
                Card("K", "C"),
            ),
            jokers=(Joker("Devious Joker"), Joker("Joker")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(4, 5)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6, 7)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 2)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (4, 5, 6, 7))
        self.assertIn("preferred_hand_hunt", action.metadata["reason"])
        self.assertIn("completion=1320", action.metadata["reason"])

    def test_blind_solver_hunts_nonpreferred_clear_flush_line(self) -> None:
        state = GameState(
            ante=3,
            blind="Big Blind",
            required_score=300,
            current_score=0,
            hands_remaining=2,
            discards_remaining=4,
            deck_size=44,
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("9", "H"),
                Card("2", "S"),
                Card("3", "C"),
                Card("4", "D"),
                Card("5", "S"),
            ),
            legal_actions=_all_blind_actions(8),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(set(action.card_indices), {4, 5, 6, 7})
        self.assertIn("blind_solver_clear_line", action.metadata["reason"])
        self.assertIn("target=Flush", action.metadata["reason"])

    def test_boss_blind_hunts_clear_straight_draw_under_pressure(self) -> None:
        state = GameState(
            ante=5,
            blind="The Wall",
            required_score=1200,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=44,
            hand=(
                Card("9", "S"),
                Card("8", "H"),
                Card("7", "D"),
                Card("5", "C"),
                Card("A", "S"),
                Card("A", "H"),
                Card("2", "D"),
                Card("K", "C"),
            ),
            jokers=(Joker("Devious Joker"), Joker("Joker")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(4, 5)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6, 7)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 2)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (4, 5, 6, 7))
        self.assertIn("preferred_hand_hunt", action.metadata["reason"])

    def test_no_discard_clear_straight_line_burns_non_core_hand(self) -> None:
        state = GameState(
            ante=5,
            blind="Small Blind",
            required_score=700,
            current_score=0,
            hands_remaining=2,
            discards_remaining=0,
            deck_size=44,
            hand=(
                Card("9", "S"),
                Card("8", "H"),
                Card("7", "D"),
                Card("5", "C"),
                Card("A", "S"),
                Card("A", "H"),
                Card("2", "D"),
                Card("K", "C"),
            ),
            jokers=(Joker("Devious Joker"), Joker("Joker")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(4, 5)),
                Action(ActionType.PLAY_HAND, card_indices=(4, 5, 6, 7)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (4, 5, 6, 7))
        self.assertIn("preferred_hand_hunt_redraw", action.metadata["reason"])

    def test_flush_plan_targets_required_scoring_suit_when_only_that_clears(self) -> None:
        state = GameState(
            ante=5,
            blind="Small Blind",
            required_score=3000,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=44,
            hand=(
                Card("A", "D"),
                Card("K", "D"),
                Card("Q", "D"),
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("2", "S"),
            ),
            jokers=(Joker("Droll Joker"), Joker("Greedy Joker")),
            legal_actions=_all_blind_actions(8),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertIn("target=Flush:D", action.metadata["reason"])
        self.assertIn("completion=4983", action.metadata["reason"])

    def test_flush_plan_prefers_more_likely_flush_when_multiple_suits_clear(self) -> None:
        state = GameState(
            ante=5,
            blind="Small Blind",
            required_score=2500,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=44,
            hand=(
                Card("A", "D"),
                Card("K", "D"),
                Card("Q", "D"),
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("2", "S"),
            ),
            jokers=(Joker("Droll Joker"), Joker("Greedy Joker")),
            legal_actions=_all_blind_actions(8),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertIn("target=Flush:H", action.metadata["reason"])
        self.assertIn("p=0.51", action.metadata["reason"])

    def test_straight_draw_eval_rewards_open_ended_outs_over_gutshot(self) -> None:
        open_state = GameState(
            ante=5,
            blind="Big Blind",
            deck_size=44,
            hand=(
                Card("5", "S"),
                Card("6", "H"),
                Card("7", "D"),
                Card("8", "C"),
                Card("A", "S"),
                Card("K", "H"),
                Card("Q", "D"),
                Card("2", "C"),
            ),
        )
        gutshot_state = replace(
            open_state,
            hand=(
                Card("5", "S"),
                Card("6", "H"),
                Card("8", "D"),
                Card("9", "C"),
                Card("A", "S"),
                Card("K", "H"),
                Card("Q", "D"),
                Card("2", "C"),
            ),
        )

        open_eval = strategy._straight_draw_evaluation(open_state, open_state.hand[:4], draw_count=3)
        gutshot_eval = strategy._straight_draw_evaluation(gutshot_state, gutshot_state.hand[:4], draw_count=3)

        self.assertIsNotNone(open_eval)
        self.assertIsNotNone(gutshot_eval)
        self.assertTrue(open_eval.open_ended)
        self.assertTrue(gutshot_eval.gutshot)
        self.assertGreater(open_eval.out_count, gutshot_eval.out_count)
        self.assertGreater(open_eval.completion_probability, gutshot_eval.completion_probability)

    def test_straight_hunt_uses_known_top_draw_outs(self) -> None:
        state = GameState(
            ante=5,
            blind="Big Blind",
            required_score=900,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=3,
            hand=(
                Card("9", "S"),
                Card("8", "H"),
                Card("7", "D"),
                Card("6", "C"),
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("Q", "C"),
            ),
            known_deck=(Card("10", "S"), Card("2", "H"), Card("3", "D")),
            jokers=(Joker("Devious Joker"), Joker("Joker")),
            legal_actions=_all_blind_actions(8),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertNotIn(0, action.card_indices)
        self.assertNotIn(1, action.card_indices)
        self.assertNotIn(2, action.card_indices)
        self.assertNotIn(3, action.card_indices)
        self.assertIn("straight_draw=4/5", action.metadata["reason"])

    def test_discards_when_known_draw_reduces_hands_needed(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("9", "C"),
                Card("5", "D"),
            ),
            known_deck=(Card("A", "D"), Card("Q", "C"), Card("J", "S")),
            jokers=(Joker("Joker"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (2, 3, 4))
        self.assertIn("hands_needed=4->2", action.metadata["reason"])

    def test_first_blind_uses_discards_to_hunt_one_hand_clear(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("Q", "H"),
                Card("Q", "D"),
                Card("8", "H"),
                Card("6", "C"),
                Card("4", "H"),
                Card("4", "D"),
                Card("2", "S"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(1, 2, 5, 6, 7)),
                Action(ActionType.DISCARD, card_indices=(0, 3, 4, 7)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6, 7)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertIn("first_blind_hunt", action.metadata["reason"])

    def test_first_blind_plays_when_current_hand_clears_in_one(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("2", "C"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(5,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1, 2, 3, 4))

    def test_last_hand_uses_discard_when_projection_can_clear(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="The Needle",
            required_score=800,
            current_score=0,
            hands_remaining=1,
            discards_remaining=1,
            jokers=(Joker("Joker"),),
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("2", "C"),
            ),
            known_deck=(Card("10", "H"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(4,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (4,))
        self.assertIn("last_hand_hunt", action.metadata["reason"])

    def test_winning_hand_discards_to_hold_drawn_gold_card(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("2", "C"),
            ),
            known_deck=(Card("3", "D", enhancement="GOLD"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(5,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (5,))
        self.assertIn("winning_economy_hunt", action.metadata["reason"])

    def test_winning_hand_discards_to_hold_drawn_blue_seal(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("2", "C"),
            ),
            known_deck=(Card("3", "D", seal="Blue"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(5,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (5,))
        self.assertIn("winning_economy_hunt", action.metadata["reason"])

    def test_winning_hand_does_not_hunt_blue_seal_without_consumable_room(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            consumables=("Mars", "Venus"),
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("2", "C"),
            ),
            known_deck=(Card("3", "D", seal="Blue"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(5,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)

    def test_winning_hand_value_discard_runs_before_opening_joker_setup(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            jokers=(Joker("DNA"),),
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("2", "C"),
            ),
            known_deck=(Card("3", "D", enhancement="GOLD"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.PLAY_HAND, card_indices=(5,)),
                Action(ActionType.DISCARD, card_indices=(5,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (5,))
        self.assertIn("winning_economy_hunt", action.metadata["reason"])

    def test_winning_hand_value_discard_accounts_for_delayed_gratification_cost(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            jokers=(Joker("Delayed Gratification"),),
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("2", "C"),
            ),
            known_deck=(Card("3", "D", enhancement="GOLD"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(5,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)

    def test_winning_hand_value_discard_can_outweigh_delayed_gratification(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            jokers=(Joker("Delayed Gratification"),),
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("2", "C"),
            ),
            known_deck=(Card("3", "D", seal="Blue"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(5,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (5,))
        self.assertIn("winning_economy_hunt", action.metadata["reason"])

    def test_winning_hand_value_discard_counts_trading_card_money_on_first_single_discard(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            jokers=(Joker("Delayed Gratification"), Joker("Trading Card")),
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("2", "C"),
            ),
            known_deck=(Card("3", "D", enhancement="GOLD"),),
            modifiers={"round_discards_used": 0},
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(5,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (5,))
        self.assertIn("winning_economy_hunt", action.metadata["reason"])

    def test_winning_hand_value_discard_does_not_count_trading_card_on_multi_discard(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            jokers=(Joker("Delayed Gratification"), Joker("Trading Card")),
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("2", "C"),
                Card("3", "C"),
            ),
            known_deck=(Card("4", "D", enhancement="GOLD"), Card("5", "C")),
            modifiers={"round_discards_used": 0},
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)

    def test_winning_hand_value_discard_projects_burnt_joker_first_discard_level(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=2,
            jokers=(Joker("Burnt Joker"),),
            hand_levels={"Pair": 1},
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D"), Card("7", "C"), Card("2", "S")),
            known_deck=(Card("3", "D"), Card("4", "C")),
            modifiers={"round_discards_used": 0},
        )

        projected = strategy._state_after_known_discard_for_economy_hunt(
            state,
            Action(ActionType.DISCARD, card_indices=(0, 1)),
            (Card("3", "D"), Card("4", "C")),
            strategy._BlindContext(),
        )

        self.assertEqual(projected.hand_levels["Pair"], 2)
        self.assertEqual(projected.modifiers["round_discards_used"], 1)

    def test_winning_hand_value_discard_does_not_project_burnt_joker_after_prior_discard(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            jokers=(Joker("Burnt Joker"),),
            hand_levels={"Pair": 1},
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D"), Card("7", "C"), Card("2", "S")),
            known_deck=(Card("3", "D"), Card("4", "C")),
            modifiers={"round_discards_used": 1},
        )

        projected = strategy._state_after_known_discard_for_economy_hunt(
            state,
            Action(ActionType.DISCARD, card_indices=(0, 1)),
            (Card("3", "D"), Card("4", "C")),
            strategy._BlindContext(discards_taken=1),
        )

        self.assertEqual(projected.hand_levels["Pair"], 1)
        self.assertEqual(projected.modifiers["round_discards_used"], 2)

    def test_ante_one_near_clear_flush_discards_to_upgrade_suit(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=3,
            hand=(
                Card("K", "H"),
                Card("8", "D"),
                Card("7", "H"),
                Card("7", "C"),
                Card("6", "H"),
                Card("4", "H"),
                Card("4", "C"),
                Card("2", "H"),
            ),
            known_deck=(Card("10", "D"), Card("A", "S"), Card("5", "S"), Card("J", "H")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 2, 4, 5, 7)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 4, 7)),
                Action(ActionType.DISCARD, card_indices=(1, 3, 6, 7)),
            ),
        )

        context = strategy._BlindContext(discards_taken=1)
        best_play = state.legal_actions[0]
        fallback_discard = state.legal_actions[1]
        action = strategy._ante_one_near_clear_upgrade_discard_action(
            state,
            best_play,
            fallback_discard,
            strategy._score_play_action(state, best_play, context),
            state.required_score,
            context,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (1, 3, 6, 7))
        self.assertIn("ante_one_upgrade", action.metadata["reason"])

    def test_ante_one_near_clear_straight_discards_to_upgrade_high_run(self) -> None:
        state = GameState(
            ante=1,
            blind="Small Blind",
            required_score=300,
            current_score=0,
            hands_remaining=4,
            discards_remaining=3,
            hand=(
                Card("Q", "H"),
                Card("Q", "D"),
                Card("9", "H"),
                Card("8", "C"),
                Card("7", "H"),
                Card("6", "S"),
                Card("6", "D"),
                Card("10", "S"),
            ),
            known_deck=(Card("J", "H"), Card("2", "C"), Card("3", "D"), Card("4", "S")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(2, 3, 4, 5, 7)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4, 7)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 5, 6)),
            ),
        )

        context = strategy._BlindContext(discards_taken=1)
        best_play = state.legal_actions[0]
        fallback_discard = state.legal_actions[1]
        action = strategy._ante_one_near_clear_upgrade_discard_action(
            state,
            best_play,
            fallback_discard,
            strategy._score_play_action(state, best_play, context),
            state.required_score,
            context,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (0, 1, 5, 6))
        self.assertIn("ante_one_upgrade", action.metadata["reason"])

    def test_discard_preserves_strong_flush_draw(self) -> None:
        state = GameState(
            required_score=900,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("K", "S"),
                Card("Q", "S"),
                Card("J", "S"),
                Card("2", "D"),
                Card("3", "C"),
                Card("4", "H"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0,)),
                Action(ActionType.DISCARD, card_indices=(3, 4, 5)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (4, 5, 6))

    def test_flush_build_discards_off_suit_cards_before_suited_core(self) -> None:
        state = GameState(
            required_score=1200,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("A", "S"),
                Card("A", "D"),
                Card("7", "C"),
                Card("8", "D"),
            ),
            jokers=(Joker("Droll Joker"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(4, 5)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 2)),
                Action(ActionType.DISCARD, card_indices=(4, 5, 6, 7)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (4, 5, 6, 7))

    def test_ante_three_preserves_last_discard_when_projection_is_unknown(self) -> None:
        state = GameState(
            ante=3,
            blind="Big Blind",
            required_score=3000,
            current_score=888,
            hands_remaining=3,
            discards_remaining=1,
            money=8,
            hand=(
                Card("A", "H"),
                Card("K", "C"),
                Card("J", "C"),
                Card("T", "C"),
                Card("9", "S"),
                Card("9", "H"),
                Card("9", "C"),
                Card("8", "H"),
            ),
            jokers=(
                Joker("Greedy Joker"),
                Joker("Crafty Joker"),
                Joker("Clever Joker"),
                Joker("Joker Stencil"),
                Joker("Wily Joker"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(4, 5, 6)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 3, 7)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (4, 5, 6))

    def test_panic_discards_when_badly_below_pace_with_discards_left(self) -> None:
        state = GameState(
            ante=4,
            blind="Big Blind",
            required_score=7500,
            current_score=0,
            hands_remaining=4,
            discards_remaining=2,
            money=28,
            hand=(
                Card("A", "D"),
                Card("Q", "H"),
                Card("9", "S"),
                Card("8", "H"),
                Card("7", "C"),
                Card("5", "H"),
                Card("3", "S"),
                Card("3", "H"),
            ),
            jokers=(
                Joker("Greedy Joker"),
                Joker("Crafty Joker"),
                Joker("Clever Joker"),
                Joker("Joker Stencil", metadata={"value": {"effect": "Currently X2"}}),
            ),
            legal_actions=_all_blind_actions(8),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertTrue(
            "panic_discard" in action.metadata["reason"]
            or "preferred_hand_hunt" in action.metadata["reason"]
        )

    def test_plays_on_last_hand_even_when_score_is_low(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=1,
            discards_remaining=4,
            hand=(Card("A", "S"), Card("A", "H")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(0,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)

    def test_buys_first_affordable_joker_before_leaving_shop(self) -> None:
        state = GameState(
            money=6,
            modifiers={
                "shop_cards": (
                    {"label": "Joker", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Mars", "set": "PLANET", "cost": {"buy": 3}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 0)

    def test_shop_prefers_stronger_joker_over_first_card(self) -> None:
        state = GameState(
            ante=1,
            money=10,
            modifiers={
                "shop_cards": (
                    {"label": "Credit Card", "set": "JOKER", "cost": {"buy": 1}},
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 1)
        self.assertIn("shop_audit", action.metadata)
        audit = action.metadata["shop_audit"]
        self.assertEqual(audit["decision"], "take")
        self.assertEqual(audit["chosen_item"]["name"], "Cavendish")
        self.assertGreaterEqual(len(audit["options"]), 2)

    def test_shop_can_buy_third_high_value_joker(self) -> None:
        state = GameState(
            ante=2,
            money=22,
            jokers=(
                Joker("Clever Joker"),
                Joker("Cartomancer"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Blueprint", "set": "JOKER", "cost": {"buy": 10}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 0)
        self.assertIn("reason", action.metadata)

    def test_shop_sells_weakest_joker_for_major_replacement(self) -> None:
        state = GameState(
            ante=3,
            money=22,
            jokers=(
                Joker("Credit Card", sell_value=1),
                Joker("Clever Joker", sell_value=2),
                Joker("Cartomancer", sell_value=3),
                Joker("Golden Joker", sell_value=3),
                Joker("Runner", sell_value=4),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Blueprint", "set": "JOKER", "cost": {"buy": 10}},
                )
            },
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.SELL, target_id="joker", amount=1, metadata={"kind": "joker", "index": 1}),
                Action(ActionType.SELL, target_id="joker", amount=2, metadata={"kind": "joker", "index": 2}),
                Action(ActionType.SELL, target_id="joker", amount=3, metadata={"kind": "joker", "index": 3}),
                Action(ActionType.SELL, target_id="joker", amount=4, metadata={"kind": "joker", "index": 4}),
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.SELL)
        self.assertEqual(action.amount, 0)
        self.assertIn("replace Credit Card", action.metadata["reason"])
        self.assertEqual(action.metadata["shop_audit"]["decision"], "replace")

    def test_shop_opens_power_pack_when_next_blind_pressure_is_high(self) -> None:
        state = GameState(
            ante=5,
            blind="Big Blind",
            required_score=16500,
            money=7,
            jokers=(Joker("Credit Card"), Joker("Cartomancer")),
            modifiers={
                "booster_packs": (
                    {"label": "Buffoon Pack", "set": "PACK", "cost": {"buy": 6}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertIn("pressure=", action.metadata["reason"])

    def test_shop_preserves_money_for_marginal_pack_when_build_is_safe(self) -> None:
        state = GameState(
            ante=1,
            blind="Small Blind",
            required_score=300,
            money=7,
            jokers=(Joker("Cavendish"), Joker("Banner")),
            modifiers={
                "booster_packs": (
                    {"label": "Buffoon Pack", "set": "PACK", "cost": {"buy": 6}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "skip")

    def test_early_shop_opens_visible_buffoon_pack_before_rerolling(self) -> None:
        state = GameState(
            ante=1,
            blind="Small Blind",
            required_score=300,
            money=9,
            modifiers={
                "shop_cards": (
                    {"label": "Venus", "set": "PLANET", "cost": {"buy": 3}},
                    {"label": "The Sun", "set": "TAROT", "cost": {"buy": 3}},
                ),
                "booster_packs": (
                    {"label": "Buffoon Pack", "set": "PACK", "cost": {"buy": 4}},
                    {"label": "Spectral Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=1, metadata={"kind": "pack", "index": 1}),
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertEqual(action.amount, 0)
        option_values = {
            option["type"]: option["value"]
            for option in action.metadata["shop_audit"]["options"]
            if option["type"] == "reroll"
        }
        self.assertEqual(option_values["reroll"], 0.0)

    def test_early_shop_rerolls_at_five_dollars_when_no_engine_is_visible(self) -> None:
        state = GameState(
            ante=1,
            blind="Small Blind",
            required_score=300,
            money=5,
            modifiers={
                "shop_cards": (
                    {"label": "Venus", "set": "PLANET", "cost": {"buy": 3}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        self.assertEqual(action.metadata["shop_audit"]["money_plan"]["reserve_money"], 25)

    def test_early_shop_does_not_sequence_pack_unless_it_remains_planned_after_buy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            blind="Small Blind",
            required_score=300,
            money=9,
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                ),
                "booster_packs": (
                    {"label": "Standard Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "take")

    def test_early_shop_does_not_sequence_pack_when_pair_is_unaffordable(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            blind="Small Blind",
            required_score=300,
            money=7,
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                ),
                "booster_packs": (
                    {"label": "Buffoon Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "take")

    def test_early_shop_opens_buffoon_before_planned_joker_plus_pack_line(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            blind="Small Blind",
            required_score=300,
            money=10,
            modifiers={
                "shop_cards": (
                    {"label": "Jupiter", "set": "PLANET", "cost": {"buy": 3}},
                    {"label": "Fortune Teller", "set": "JOKER", "cost": {"buy": 6}},
                ),
                "voucher_cards": (
                    {"label": "Seed Money", "set": "VOUCHER", "cost": {"buy": 10}},
                ),
                "booster_packs": (
                    {"label": "Buffoon Pack", "set": "PACK", "cost": {"buy": 4}},
                    {"label": "Standard Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=1, metadata={"kind": "pack", "index": 1}),
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertEqual(action.amount, 0)
        self.assertIn("planned_buy=Fortune Teller", action.metadata["reason"])

    def test_shop_never_buys_blocked_vouchers(self) -> None:
        blocked = (
            "Planet Merchant",
            "Magic Trick",
            "Director's Cut",
            "Crystal Ball",
            "Telescope",
        )
        for name in blocked:
            with self.subTest(name=name):
                state = GameState(
                    phase=GamePhase.SHOP,
                    ante=2,
                    blind="Small Blind",
                    required_score=800,
                    money=100,
                    modifiers={
                        "voucher_cards": (
                            {"label": name, "set": "VOUCHER", "cost": {"buy": 1}},
                        ),
                    },
                    legal_actions=(
                        Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}),
                        Action(ActionType.END_SHOP),
                    ),
                )

                action = BasicStrategyBot(seed=1).choose_action(state)

                self.assertEqual(action.action_type, ActionType.END_SHOP)
                self.assertEqual(strategy._voucher_value(state, {"label": name, "set": "VOUCHER"}), 0.0)

    def test_shop_can_buy_blank_before_ante_eight_but_not_in_ante_eight_shop(self) -> None:
        before_ante_eight = GameState(
            phase=GamePhase.SHOP,
            ante=7,
            blind="Small Blind",
            required_score=35000,
            money=100,
            modifiers={
                "voucher_cards": (
                    {"label": "Blank", "set": "VOUCHER", "cost": {"buy": 1}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        ante_eight = replace(before_ante_eight, ante=8)

        self.assertGreater(strategy._voucher_value(before_ante_eight, {"label": "Blank", "set": "VOUCHER"}), 0.0)
        self.assertEqual(strategy._voucher_value(ante_eight, {"label": "Blank", "set": "VOUCHER"}), 0.0)

        action = BasicStrategyBot(seed=1).choose_action(ante_eight)

        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_late_pressure_blocks_non_immediate_vouchers(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=5,
            blind="Small Blind",
            required_score=11000,
            money=80,
        )
        pressure = strategy._ShopPressure(
            target_score=24000,
            build_capacity=16000,
            ratio=1.5,
            raw_ratio=1.5,
            safety_multiplier=1.2,
            capacity_safety_factor=0.85,
        )

        for name in ("Seed Money", "Clearance Sale", "Reroll Surplus", "Overstock", "Tarot Merchant", "Hone"):
            with self.subTest(name=name):
                self.assertEqual(strategy._voucher_value(state, {"label": name, "set": "VOUCHER"}, pressure), 0.0)

        self.assertGreater(strategy._voucher_value(state, {"label": "Paint Brush", "set": "VOUCHER"}, pressure), 0.0)
        self.assertGreater(strategy._voucher_value(state, {"label": "Antimatter", "set": "VOUCHER"}, pressure), 0.0)
        self.assertGreater(strategy._voucher_value(state, {"label": "Blank", "set": "VOUCHER"}, pressure), 0.0)

    def test_pressure_allows_interest_voucher_with_money_scaling(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=5,
            blind="Small Blind",
            required_score=11000,
            money=80,
            jokers=(Joker("Bull"),),
        )
        pressure = strategy._ShopPressure(
            target_score=24000,
            build_capacity=16000,
            ratio=1.5,
            raw_ratio=1.5,
            safety_multiplier=1.2,
            capacity_safety_factor=0.85,
        )

        self.assertGreater(strategy._voucher_value(state, {"label": "Seed Money", "set": "VOUCHER"}, pressure), 0.0)

    def test_grabber_and_nacho_are_not_valued_for_needle_boss_shop(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=5,
            blind="Big Blind",
            required_score=16500,
            money=80,
            modifiers={
                "cleared_blind": {"kind": "BIG"},
                "upcoming_boss": {"name": "The Needle", "score": 22000},
            },
        )
        pressure = strategy._shop_pressure(state)

        self.assertEqual(pressure.boss_name, "The Needle")
        self.assertEqual(strategy._voucher_value(state, {"label": "Grabber", "set": "VOUCHER"}, pressure), 0.0)
        self.assertEqual(strategy._voucher_value(state, {"label": "Nacho Tong", "set": "VOUCHER"}, pressure), 0.0)
        self.assertGreater(strategy._voucher_value(state, {"label": "Paint Brush", "set": "VOUCHER"}, pressure), 0.0)

    def test_early_shop_opens_buffoon_even_when_second_joker_is_post_buy_plan(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            money=17,
            jokers=(Joker("Scary Face"), Joker("Abstract Joker")),
            modifiers={
                "shop_cards": (
                    {"label": "Hallucination", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Sly Joker", "set": "JOKER", "cost": {"buy": 3}},
                ),
                "booster_packs": (
                    {"label": "Celestial Pack", "set": "PACK", "cost": {"buy": 4}},
                    {"label": "Buffoon Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=1, metadata={"kind": "pack", "index": 1}),
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertEqual(action.amount, 1)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "sequence_info_first")
        self.assertIn("planned_buy=Hallucination", action.metadata["reason"])

    def test_early_shop_does_not_sequence_arcana_even_when_post_buy_planned(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=12,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                ),
                "booster_packs": (
                    {"label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "take")

    def test_early_shop_does_not_open_arcana_first_when_reroll_is_post_buy_plan(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=20,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                ),
                "booster_packs": (
                    {"label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "take")

    def test_early_shop_does_not_sequence_judgement_before_planned_joker_buy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=1,
            blind="Small Blind",
            required_score=300,
            money=9,
            consumables=("Judgement",),
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.USE_CONSUMABLE, target_id="consumable", amount=0, metadata={"kind": "consumable", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "take")

    def test_late_shop_does_not_sequence_judgement_before_planned_joker_buy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=9,
            consumables=("Judgement",),
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.USE_CONSUMABLE, target_id="consumable", amount=0, metadata={"kind": "consumable", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "take")

    def test_joker_stencil_is_worthless_when_it_fills_last_slot(self) -> None:
        state = GameState(
            ante=2,
            blind="Big Blind",
            required_score=1200,
            money=8,
            jokers=(
                Joker("Greedy Joker"),
                Joker("Crafty Joker"),
                Joker("Clever Joker"),
                Joker("Wily Joker"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Joker Stencil", "set": "JOKER", "cost": {"buy": 8}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        self.assertEqual(action.metadata["shop_audit"]["options"][0]["value"], 0.0)

    def test_rank_jokers_support_larger_matching_hands(self) -> None:
        self.assertIn(HandType.FOUR_OF_A_KIND, strategy.JOKER_HAND_SYNERGY["Wily Joker"])
        self.assertIn(HandType.FIVE_OF_A_KIND, strategy.JOKER_HAND_SYNERGY["Wily Joker"])
        self.assertIn(HandType.FULL_HOUSE, strategy.JOKER_HAND_SYNERGY["Clever Joker"])
        self.assertIn(HandType.FLUSH_FIVE, strategy.JOKER_HAND_SYNERGY["The Family"])

    def test_two_pair_support_joker_does_not_force_two_pair_plan(self) -> None:
        state = GameState(
            ante=3,
            jokers=(Joker("Clever Joker"),),
        )

        self.assertEqual(strategy._preferred_hand_type(state), HandType.FULL_HOUSE)

    def test_single_crafty_joker_does_not_force_flush_plan_early(self) -> None:
        state = GameState(
            ante=3,
            jokers=(Joker("Crafty Joker"),),
        )

        self.assertEqual(strategy._preferred_hand_type(state), HandType.PAIR)

    def test_crafty_joker_with_flush_investment_can_drive_flush_plan(self) -> None:
        state = GameState(
            ante=3,
            hand_levels={"Flush": 2},
            jokers=(Joker("Crafty Joker"),),
        )

        self.assertEqual(strategy._preferred_hand_type(state), HandType.FLUSH)

    def test_spare_trousers_can_still_drive_two_pair_plan(self) -> None:
        state = GameState(
            ante=3,
            jokers=(Joker("Spare Trousers"),),
        )

        self.assertEqual(strategy._preferred_hand_type(state), HandType.TWO_PAIR)

    def test_two_pair_planet_is_discounted_without_dedicated_scaling(self) -> None:
        uranus = {"label": "Uranus", "set": "PLANET", "cost": {"buy": 3}}
        support_only = GameState(
            ante=4,
            jokers=(Joker("Clever Joker"),),
        )
        dedicated = GameState(
            ante=4,
            jokers=(Joker("Spare Trousers"),),
        )

        self.assertGreater(
            strategy._planet_card_value(dedicated, uranus),
            strategy._planet_card_value(support_only, uranus) + 20,
        )

    def test_planet_value_uses_aligned_joker_capacity_gain(self) -> None:
        state = GameState(
            ante=4,
            jokers=(
                Joker("Droll Joker"),
                Joker("The Tribe"),
                Joker("Wrathful Joker"),
            ),
        )

        self.assertGreater(
            strategy._planet_card_value(state, {"label": "Jupiter", "set": "PLANET"}),
            strategy._planet_card_value(state, {"label": "Mercury", "set": "PLANET"}) + 50,
        )

    def test_unsupported_trip_joker_does_not_fill_last_slot(self) -> None:
        state = GameState(
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=26,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Greedy Joker"),
                Joker("Zany Joker"),
                Joker("Green Joker"),
                Joker("Sock and Buskin"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Wily Joker", "set": "JOKER", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        buy_option = next(option for option in action.metadata["shop_audit"]["options"] if option["type"] == "buy")
        self.assertLess(buy_option["value"], 0)

    def test_full_slots_block_direct_normal_joker_buy(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=80,
            jokers=(
                Joker("Throwback"),
                Joker("Blue Joker"),
                Joker("Half Joker"),
                Joker("Swashbuckler"),
                Joker("Scholar"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        buy_option = next(option for option in action.metadata["shop_audit"]["options"] if option["type"] == "buy")
        self.assertEqual(buy_option["value"], 0.0)

    def test_last_slot_buy_blocks_followup_joker_buy_if_bridge_state_is_stale(self) -> None:
        bot = BasicStrategyBot(seed=1)
        filled_last_slot = GameState(
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=27,
            jokers=(
                Joker("Half Joker"),
                Joker("Odd Todd"),
                Joker("Jolly Joker"),
                Joker("Gros Michel"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Castle", "set": "JOKER", "cost": {"buy": 6}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )
        stale_after_buy = GameState(
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=21,
            jokers=filled_last_slot.jokers,
            modifiers={
                "shop_cards": (
                    {"label": "Raised Fist", "set": "JOKER", "cost": {"buy": 5}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        first_action = bot.choose_action(filled_last_slot)
        second_action = bot.choose_action(stale_after_buy)

        self.assertEqual(first_action.action_type, ActionType.BUY)
        self.assertEqual(second_action.action_type, ActionType.END_SHOP)
        buy_option = next(
            option for option in second_action.metadata["shop_audit"]["options"] if option["type"] == "buy"
        )
        self.assertEqual(buy_option["value"], 0.0)

    def test_last_slot_pack_joker_blocks_stale_followup_pack_joker(self) -> None:
        bot = BasicStrategyBot(seed=1)
        filled_last_slot = GameState(
            ante=2,
            blind="Small Blind",
            required_score=5000,
            money=18,
            jokers=(
                Joker("Half Joker"),
                Joker("Odd Todd"),
                Joker("Jolly Joker"),
                Joker("Gros Michel"),
            ),
            modifiers={
                "pack_cards": (
                    {"label": "Devious Joker", "set": "JOKER", "cost": {"buy": 0}},
                ),
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )
        stale_after_pick = GameState(
            ante=2,
            blind="Small Blind",
            required_score=5000,
            money=18,
            jokers=filled_last_slot.jokers,
            modifiers={
                "pack_cards": (
                    {"label": "Raised Fist", "set": "JOKER", "cost": {"buy": 0}},
                ),
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        first_action = bot.choose_action(filled_last_slot)
        second_action = bot.choose_action(stale_after_pick)

        self.assertEqual(first_action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(first_action.target_id, "card")
        self.assertEqual(second_action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(second_action.target_id, "skip")

    def test_last_slot_pack_joker_blocks_stale_followup_shop_joker_buy(self) -> None:
        bot = BasicStrategyBot(seed=1)
        filled_last_slot = GameState(
            ante=2,
            blind="Small Blind",
            required_score=5000,
            money=18,
            jokers=(
                Joker("Half Joker"),
                Joker("Odd Todd"),
                Joker("Jolly Joker"),
                Joker("Gros Michel"),
            ),
            modifiers={
                "pack_cards": (
                    {"label": "Devious Joker", "set": "JOKER", "cost": {"buy": 0}},
                ),
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )
        stale_shop = GameState(
            ante=2,
            blind="Small Blind",
            required_score=5000,
            money=18,
            jokers=filled_last_slot.jokers,
            modifiers={
                "shop_cards": (
                    {"label": "Raised Fist", "set": "JOKER", "cost": {"buy": 5}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        first_action = bot.choose_action(filled_last_slot)
        second_action = bot.choose_action(stale_shop)

        self.assertEqual(first_action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(first_action.target_id, "card")
        self.assertEqual(second_action.action_type, ActionType.END_SHOP)
        buy_option = next(option for option in second_action.metadata["shop_audit"]["options"] if option["type"] == "buy")
        self.assertEqual(buy_option["value"], 0.0)

    def test_full_slots_can_still_buy_negative_joker(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=80,
            jokers=(
                Joker("Throwback"),
                Joker("Blue Joker"),
                Joker("Half Joker"),
                Joker("Swashbuckler"),
                Joker("Scholar"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "edition": "NEGATIVE", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 0)

    def test_compatible_rare_hand_does_not_become_primary_plan(self) -> None:
        state = GameState(
            jokers=(
                Joker("Crafty Joker"),
                Joker("Clever Joker"),
            ),
        )

        self.assertNotEqual(strategy._preferred_hand_type(state), HandType.FLUSH_HOUSE)

    def test_rare_hand_joker_is_discounted_without_deck_support(self) -> None:
        state = GameState(
            ante=2,
            blind="Big Blind",
            required_score=1200,
            money=12,
            modifiers={
                "shop_cards": (
                    {"label": "The Family", "set": "JOKER", "cost": {"buy": 8}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        self.assertLess(action.metadata["shop_audit"]["options"][0]["value"], 10)

    def test_rare_hand_joker_is_buyable_with_rank_support(self) -> None:
        state = GameState(
            ante=2,
            blind="Big Blind",
            required_score=1200,
            money=12,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("7", "C"),
                Card("3", "D"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "The Family", "set": "JOKER", "cost": {"buy": 8}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["chosen_item"]["name"], "The Family")

    def test_rare_hand_build_buys_rank_manipulation_tarot(self) -> None:
        state = GameState(
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=15,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("7", "C"),
                Card("3", "D"),
            ),
            jokers=(Joker("The Family"),),
            modifiers={
                "shop_cards": (
                    {"label": "Death", "set": "TAROT", "cost": {"buy": 3}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["chosen_item"]["name"], "Death")

    def test_five_of_a_kind_needs_manipulation_beyond_natural_four_kind(self) -> None:
        state = GameState(
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("A", "C"),
                Card("K", "H"),
            ),
        )
        with_death = GameState(
            hand=state.hand,
            consumables=("Death",),
        )

        self.assertGreater(strategy._rare_hand_deck_manipulation_need(state, HandType.FIVE_OF_A_KIND), 0)
        self.assertLess(
            strategy._rare_hand_deck_manipulation_need(with_death, HandType.FIVE_OF_A_KIND),
            strategy._rare_hand_deck_manipulation_need(state, HandType.FIVE_OF_A_KIND),
        )

    def test_flush_house_needs_manipulation_even_with_natural_trips_and_suit(self) -> None:
        state = GameState(
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("K", "H"),
                Card("Q", "H"),
            ),
        )
        with_death = GameState(
            hand=state.hand,
            consumables=("Death",),
        )

        self.assertGreater(strategy._rare_hand_deck_manipulation_need(state, HandType.FLUSH_HOUSE), 0)
        self.assertLess(
            strategy._rare_hand_deck_manipulation_need(with_death, HandType.FLUSH_HOUSE),
            strategy._rare_hand_deck_manipulation_need(state, HandType.FLUSH_HOUSE),
        )

    def test_owned_stencil_penalizes_filling_last_slot(self) -> None:
        state = GameState(
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=12,
            jokers=(
                Joker("Greedy Joker"),
                Joker("Crafty Joker"),
                Joker("Clever Joker"),
                Joker("Joker Stencil", metadata={"value": {"effect": "Currently X2"}}),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Wily Joker", "set": "JOKER", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        buy_option = next(option for option in action.metadata["shop_audit"]["options"] if option["type"] == "buy")
        self.assertLess(buy_option["value"], 0)

    def test_selling_joker_updates_owned_stencil_value(self) -> None:
        state = GameState(
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=12,
            jokers=(
                Joker("Greedy Joker"),
                Joker("Crafty Joker"),
                Joker("Clever Joker"),
                Joker("Wily Joker"),
                Joker("Joker Stencil", metadata={"value": {"effect": "Currently X1"}}),
            ),
        )

        jokers = strategy._jokers_after_sell_for_scoring(state, remove_index=3)
        stencil = next(joker for joker in jokers if joker.name == "Joker Stencil")

        self.assertIn("Currently X2", stencil.metadata["value"]["effect"])

    def test_early_shop_skips_economy_joker_before_scoring_is_safe(self) -> None:
        state = GameState(
            ante=1,
            money=7,
            modifiers={
                "shop_cards": (
                    {"label": "Mail-In Rebate", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_safe_economy_joker_buy_discounts_interest_floor_penalty(self) -> None:
        state = GameState(
            ante=3,
            money=25,
            jokers=(Joker("Gros Michel", metadata={"value": {"effect": "+15 Mult"}}),),
        )
        pressure = strategy._ShopPressure(
            target_score=1200,
            build_capacity=3000,
            ratio=0.6,
            raw_ratio=0.6,
            safety_multiplier=1.0,
            capacity_safety_factor=1.0,
        )
        golden = {"label": "Golden Joker", "set": "JOKER", "cost": {"buy": 6}}
        plain = {"label": "Joker", "set": "JOKER", "cost": {"buy": 6}}

        self.assertLess(
            strategy._cost_penalty(state, golden, pressure),
            strategy._cost_penalty(state, plain, pressure),
        )

    def test_early_shop_skips_target_required_tarot_until_targets_are_supported(self) -> None:
        state = GameState(
            ante=1,
            money=7,
            jokers=(Joker("Joker"),),
            modifiers={
                "shop_cards": (
                    {"label": "Death", "set": "TAROT", "cost": {"buy": 3}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_early_pack_skips_unsupported_narrow_joker(self) -> None:
        state = GameState(
            ante=1,
            money=6,
            modifiers={
                "pack_cards": (
                    {"label": "Devious Joker", "set": "JOKER", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")

    def test_late_full_build_stops_rerolling_before_bank_gets_too_low(self) -> None:
        state = GameState(
            ante=7,
            blind="Small Blind",
            required_score=35000,
            money=27,
            jokers=(
                Joker("Mad Joker"),
                Joker("Square Joker"),
                Joker("Gros Michel"),
                Joker("Scary Face"),
                Joker("Banner"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "skip")

    def test_under_pressure_buys_scaling_joker_below_interest_reserve(self) -> None:
        state = GameState(
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=24,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Greedy Joker"),
                Joker("Crafty Joker"),
                Joker("Clever Joker"),
                Joker("Joker Stencil", metadata={"value": {"effect": "Currently X2"}}),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Hologram", "set": "JOKER", "cost": {"buy": 5}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["chosen_item"]["name"], "Hologram")
        self.assertEqual(action.metadata["shop_audit"]["money_plan"]["reserve_money"], 25)

    def test_shop_safety_margin_opens_pack_before_score_wall(self) -> None:
        state = GameState(
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=55,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Runner"),
                Joker("Sock and Buskin"),
                Joker("Wrathful Joker"),
                Joker("Even Steven"),
                Joker("Hanging Chad"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Square Joker", "set": "JOKER", "cost": {"buy": 4}},
                ),
                "booster_packs": (
                    {"label": "Jumbo Celestial Pack", "set": "PACK", "cost": {"buy": 6}},
                    {"label": "Standard Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=1, metadata={"kind": "pack", "index": 1}),
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertEqual(action.amount, 0)
        pressure = action.metadata["shop_audit"]["pressure"]
        self.assertGreater(pressure["safety_multiplier"], 1.0)
        self.assertGreater(pressure["raw_ratio"], 1.0)
        self.assertLess(pressure["capacity_safety_factor"], 1.0)

    def test_shop_skips_early_arcana_pack_when_shop_hand_is_empty(self) -> None:
        state = GameState(
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=55,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Runner"),
                Joker("Sock and Buskin"),
                Joker("Wrathful Joker"),
                Joker("Even Steven"),
                Joker("Hanging Chad"),
            ),
            modifiers={
                "booster_packs": (
                    {"label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)

    def test_shop_safety_margin_spends_above_interest_on_buffer_pack(self) -> None:
        state = GameState(
            ante=4,
            blind="Big Blind",
            required_score=7500,
            money=43,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Runner"),
                Joker("Sock and Buskin"),
                Joker("Wrathful Joker"),
                Joker("Hanging Chad"),
                Joker("Even Steven"),
            ),
            modifiers={
                "booster_packs": (
                    {"label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4}},
                    {"label": "Jumbo Celestial Pack", "set": "PACK", "cost": {"buy": 6}},
                ),
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=1, metadata={"kind": "pack", "index": 1}),
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertGreaterEqual(state.money - 6, 25)
        pressure = action.metadata["shop_audit"]["pressure"]
        self.assertGreaterEqual(pressure["ratio"], 0.85)

    def test_shop_pressure_previews_upcoming_wall_boss(self) -> None:
        state = GameState(
            ante=4,
            blind="Big Blind",
            required_score=7500,
            money=40,
            jokers=(Joker("Joker"),),
            modifiers={
                "blinds": {
                    "boss": {"name": "The Wall", "type": "BOSS"},
                }
            },
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        pressure = strategy._shop_pressure(state)

        self.assertEqual(strategy._estimated_next_required_score(state), 20000)
        self.assertEqual(pressure.boss_name, "The Wall")
        self.assertEqual(pressure.boss_target_multiplier, 2.0)
        self.assertGreater(pressure.ratio, pressure.raw_ratio)

    def test_shop_pressure_uses_final_win_target_before_ante_eight(self) -> None:
        state = GameState(
            ante=7,
            blind="Big Blind",
            required_score=52500,
            money=55,
            jokers=(Joker("Joker"), Joker("Blue Joker"), Joker("Gros Michel")),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        next_required = strategy._estimated_next_required_score(state)
        planning_required = strategy._estimated_shop_planning_required_score(state)
        pressure = strategy._shop_pressure(state)

        self.assertEqual(next_required, 70000)
        self.assertEqual(planning_required, 100000)
        self.assertEqual(pressure.target_score, planning_required * pressure.safety_multiplier)

    def test_shop_pressure_discounts_final_boss_capacity(self) -> None:
        state = GameState(
            ante=8,
            blind="Big Blind",
            required_score=100000,
            money=90,
            jokers=(Joker("Joker"), Joker("Blue Joker"), Joker("Gros Michel")),
            modifiers={
                "blinds": {
                    "boss": {"name": "Crimson Heart", "type": "BOSS", "score": 100000},
                }
            },
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        pressure = strategy._shop_pressure(state)

        self.assertEqual(pressure.boss_name, "Crimson Heart")
        self.assertAlmostEqual(pressure.boss_capacity_factor, 0.60)
        self.assertGreater(pressure.ratio, pressure.raw_ratio)

    def test_crimson_heart_pressure_discounts_fragile_full_joker_lineup(self) -> None:
        stable_state = GameState(
            ante=8,
            blind="Big Blind",
            required_score=100000,
            money=90,
            jokers=(Joker("Joker"), Joker("Blue Joker"), Joker("Gros Michel")),
            modifiers={"blinds": {"boss": {"name": "Crimson Heart", "type": "BOSS", "score": 100000}}},
        )
        fragile_state = replace(
            stable_state,
            jokers=(
                Joker("Blue Joker"),
                Joker("Card Sharp"),
                Joker("Seeing Double"),
                Joker("Flower Pot"),
                Joker("Gros Michel"),
            ),
        )

        stable_pressure = strategy._shop_pressure(stable_state)
        fragile_pressure = strategy._shop_pressure(fragile_state)

        self.assertLess(fragile_pressure.boss_capacity_factor, stable_pressure.boss_capacity_factor)

    def test_shop_pressure_uses_cleared_big_blind_for_boss_shop(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=40,
            jokers=(Joker("Joker"),),
            modifiers={
                "cleared_blind": {
                    "ante": 4,
                    "blind": "Big Blind",
                    "required_score": 7500,
                    "kind": "BIG",
                },
                "blinds": {
                    "boss": {"name": "The Wall", "type": "BOSS", "score": 20000},
                },
            },
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        pressure = strategy._shop_pressure(state)

        self.assertEqual(strategy._estimated_next_required_score(state), 20000)
        self.assertEqual(pressure.boss_name, "The Wall")
        self.assertEqual(pressure.boss_target_multiplier, 1.0)
        self.assertGreater(pressure.ratio, 1.0)

    def test_shop_pressure_full_weights_needle_after_big_blind(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=3,
            blind="Small Blind",
            required_score=2000,
            money=25,
            jokers=(Joker("Jolly Joker"), Joker("Flower Pot"), Joker("Crafty Joker")),
            modifiers={
                "cleared_blind": {
                    "ante": 3,
                    "blind": "Big Blind",
                    "required_score": 3000,
                    "kind": "BIG",
                },
                "blinds": {
                    "boss": {"name": "The Needle", "type": "BOSS", "score": 2000},
                },
            },
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        pressure = strategy._shop_pressure(state)

        self.assertEqual(strategy._estimated_next_required_score(state), 2000)
        self.assertEqual(pressure.boss_name, "The Needle")
        self.assertEqual(pressure.boss_capacity_factor, 1.0)
        needle_state = replace(state, blind="The Needle", required_score=2000, hands_remaining=1)
        expected_capacity = strategy._sample_build_score(needle_state, needle_state.jokers) * 0.85
        self.assertAlmostEqual(pressure.build_capacity, expected_capacity, places=2)

    def test_post_big_shop_pressure_scores_flint_as_boss(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=35,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(Joker("Joker"), Joker("Banner")),
            modifiers={
                "cleared_blind": {
                    "ante": 4,
                    "blind": "Big Blind",
                    "required_score": 7500,
                    "kind": "BIG",
                },
                "blinds": {
                    "boss": {"name": "The Flint", "type": "BOSS", "score": 10000},
                },
            },
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        pressure = strategy._shop_pressure(state)

        flint_state = replace(state, blind="The Flint", required_score=10000)
        normal_score = strategy._sample_build_score(state, state.jokers)
        flint_score = strategy._sample_build_score(flint_state, flint_state.jokers)
        expected_capacity = flint_score * 3.5 * 0.85 * pressure.capacity_safety_factor
        self.assertEqual(pressure.boss_name, "The Flint")
        self.assertLess(flint_score, normal_score)
        self.assertAlmostEqual(pressure.build_capacity, expected_capacity, places=2)

    def test_post_big_shop_pressure_scores_water_with_no_discards(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=4,
            blind="Small Blind",
            required_score=5000,
            money=35,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(Joker("Banner"),),
            modifiers={
                "cleared_blind": {
                    "ante": 4,
                    "blind": "Big Blind",
                    "required_score": 7500,
                    "kind": "BIG",
                },
                "blinds": {
                    "boss": {"name": "The Water", "type": "BOSS", "score": 10000},
                },
            },
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        pressure = strategy._shop_pressure(state)

        water_state = replace(state, blind="The Water", required_score=10000, discards_remaining=0)
        normal_score = strategy._sample_build_score(state, state.jokers)
        water_score = strategy._sample_build_score(water_state, water_state.jokers)
        expected_capacity = water_score * 3.5 * 0.85 * pressure.capacity_safety_factor
        self.assertEqual(pressure.boss_name, "The Water")
        self.assertLess(water_score, normal_score)
        self.assertAlmostEqual(pressure.build_capacity, expected_capacity, places=2)

    def test_late_role_hunt_rerolls_money_above_interest_reserve(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=60,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        self.assertLess(action.metadata["shop_audit"]["money_plan"]["reserve_money"], 25)
        self.assertGreaterEqual(action.metadata["shop_audit"]["money_plan"]["spendable_money"], 35)

    def test_late_role_hunt_can_spend_above_raised_interest_cap_under_pressure(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=60,
            vouchers=("Seed Money",),
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        self.assertEqual(action.metadata["shop_audit"]["money_plan"]["interest_cap_money"], 50)
        self.assertGreaterEqual(state.money - 5, 50)

    def test_money_scaling_joker_holds_extra_bank_before_rerolling(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=79,
            jokers=(
                Joker("Bull"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        self.assertTrue(action.metadata["shop_audit"]["money_plan"]["has_money_scaling_joker"])
        self.assertEqual(action.metadata["shop_audit"]["money_plan"]["reserve_money"], 75)

    def test_severe_pressure_relaxes_money_scaling_reserve_before_ante_seven(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=60000,
            money=53,
            jokers=(
                Joker("Bull"),
                Joker("Half Joker"),
                Joker("Bloodstone"),
                Joker("Photograph"),
                Joker("Wrathful Joker"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        money_plan = action.metadata["shop_audit"]["money_plan"]
        self.assertTrue(money_plan["has_money_scaling_joker"])
        self.assertEqual(money_plan["reserve_money"], 35)
        self.assertEqual(money_plan["spendable_money"], 18)

    def test_severe_pressure_keeps_money_scaling_bank_when_roles_are_filled(self) -> None:
        state = GameState(
            ante=5,
            blind="Small Blind",
            required_score=31900,
            money=47,
            jokers=(
                Joker("Abstract Joker"),
                Joker("Bull"),
                Joker("Spare Trousers"),
                Joker("Blackboard"),
                Joker("Square Joker"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        money_plan = action.metadata["shop_audit"]["money_plan"]
        self.assertTrue(money_plan["has_money_scaling_joker"])
        self.assertEqual(money_plan["reserve_money"], 75)
        self.assertFalse(action.metadata["shop_audit"]["build_profile"]["missing_roles"])

    def test_money_scaling_joker_can_spend_above_extra_reserve(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=110,
            jokers=(
                Joker("Bull"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        self.assertEqual(action.metadata["shop_audit"]["money_plan"]["reserve_money"], 75)
        self.assertEqual(action.metadata["shop_audit"]["money_plan"]["spendable_money"], 35)

    def test_ante_eight_closing_reroll_spends_money_tree_bank_under_pressure(self) -> None:
        state = GameState(
            ante=8,
            blind="Big Blind",
            required_score=100000,
            money=80,
            vouchers=("Money Tree",),
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        money_plan = action.metadata["shop_audit"]["money_plan"]
        self.assertEqual(money_plan["interest_cap_money"], 100)
        self.assertEqual(money_plan["reserve_money"], 65)
        self.assertEqual(money_plan["spendable_money"], 15)

    def test_ante_seven_closing_pressure_relaxes_money_scaling_reserve(self) -> None:
        state = GameState(
            ante=7,
            blind="Big Blind",
            required_score=75000,
            money=80,
            jokers=(
                Joker("Bull"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        money_plan = action.metadata["shop_audit"]["money_plan"]
        self.assertTrue(money_plan["has_money_scaling_joker"])
        self.assertLess(money_plan["reserve_money"], 75)
        self.assertGreaterEqual(money_plan["spendable_money"], 40)

    def test_late_shop_prioritizes_missing_xmult_role(self) -> None:
        state = GameState(
            ante=5,
            blind="Big Blind",
            required_score=16500,
            money=80,
            jokers=(
                Joker("Banner"),
                Joker("Jolly Joker"),
                Joker("Green Joker"),
                Joker("Golden Joker"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Joker", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "The Duo", "set": "JOKER", "cost": {"buy": 8}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 1)
        audit = action.metadata["shop_audit"]
        self.assertIn("xmult", audit["build_profile"]["missing_roles"])
        self.assertEqual(audit["build_profile"]["archetype"], "rank")
        self.assertEqual(audit["chosen_item"]["name"], "The Duo")

    def test_build_profile_keeps_weak_flat_mult_as_deficit(self) -> None:
        profile = strategy._build_profile(GameState(ante=5, jokers=(Joker("Joker"),)))
        payload = strategy._build_profile_payload(profile)

        self.assertFalse(profile.has_mult)
        self.assertIn("mult", profile.missing_roles)
        self.assertEqual(payload["role_scores"]["mult"], 4.0)
        self.assertEqual(payload["role_requirements"]["mult"], 24.0)

    def test_inactive_loyalty_card_does_not_fill_xmult_role(self) -> None:
        profile = strategy._build_profile(GameState(ante=4, jokers=(Joker("Loyalty Card"),)))

        self.assertIn("xmult", profile.missing_roles)

    def test_late_build_profile_discounts_decaying_score_jokers(self) -> None:
        popcorn = Joker("Popcorn", metadata={"value": {"effect": "Currently +20 Mult"}})

        early_profile = strategy._build_profile(GameState(ante=3, jokers=(popcorn,)))
        late_profile = strategy._build_profile(GameState(ante=7, jokers=(popcorn,)))

        self.assertTrue(early_profile.has_mult)
        self.assertFalse(late_profile.has_mult)
        self.assertIn("mult", late_profile.missing_roles)

    def test_perishable_joker_does_not_satisfy_late_xmult_role(self) -> None:
        durable_profile = strategy._build_profile(GameState(ante=7, jokers=(Joker("Cavendish"),)))
        perishable_profile = strategy._build_profile(
            GameState(ante=7, jokers=(Joker("Cavendish", metadata={"perishable": True}),))
        )

        self.assertTrue(durable_profile.has_xmult)
        self.assertFalse(perishable_profile.has_xmult)
        self.assertIn("xmult", perishable_profile.missing_roles)

    def test_early_shop_prefers_real_chip_joker_over_inactive_loyalty_card(self) -> None:
        state = GameState(
            ante=2,
            blind="Small Blind",
            required_score=800,
            money=13,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Supernova"),
                Joker("Raised Fist"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Loyalty Card", "set": "JOKER", "cost": {"buy": 5}},
                    {"label": "Crafty Joker", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 1)

    def test_low_pressure_shop_values_visible_future_score_headroom(self) -> None:
        state = GameState(
            ante=2,
            blind="Small Blind",
            required_score=800,
            money=40,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Stuntman"),
                Joker("Joker"),
            ),
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
        )
        pressure = strategy._shop_pressure(state)
        joker = Joker("Cavendish")

        bonus = strategy._future_score_headroom_joker_bonus(state, joker, pressure)
        action_value = strategy._shop_action_value(state, state.legal_actions[0], pressure)

        self.assertLess(pressure.raw_ratio, 0.95)
        self.assertGreaterEqual(bonus, 20.0)
        self.assertGreaterEqual(action_value + strategy.SHOP_VALUE_TOLERANCE, strategy._shop_buy_threshold(state, pressure))

    def test_late_rich_build_rerolls_when_missing_xmult_and_scaling(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=90,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        missing_roles = action.metadata["shop_audit"]["build_profile"]["missing_roles"]
        self.assertIn("xmult", missing_roles)
        self.assertIn("scaling", missing_roles)

    def test_late_urgent_role_hunt_does_not_require_rich_profile(self) -> None:
        state = GameState(
            ante=8,
            blind="Big Blind",
            money=43,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
        )
        pressure = strategy._ShopPressure(
            target_score=100.0,
            build_capacity=80.0,
            ratio=2.6,
            raw_ratio=0.9,
            safety_multiplier=1.4,
            capacity_safety_factor=0.85,
        )
        profile = strategy._build_profile(state)

        self.assertEqual(profile.spendable_money, 18)
        self.assertEqual(strategy._spendable_money(state, pressure), 23)
        self.assertFalse(profile.rich)
        self.assertTrue(
            strategy._late_reroll_is_worth_it(
                state,
                pressure,
                profile,
                strategy._ShopContext(rerolls_in_shop=0),
            )
        )

    def test_late_non_rich_role_hunt_skips_soft_raw_pressure_reroll(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=58000,
            money=57,
            vouchers=("Seed Money", "Grabber", "Nacho Tong"),
            jokers=(
                Joker("Supernova"),
                Joker("Wily Joker"),
                Joker("Egg"),
                Joker("Odd Todd"),
                Joker("Swashbuckler"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Walkie Talkie", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Diet Cola", "set": "JOKER", "cost": {"buy": 6}},
                ),
                "booster_packs": (
                    {"label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4}},
                ),
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        pressure = strategy._ShopPressure(
            target_score=58000.0,
            build_capacity=25402.0,
            ratio=2.283,
            raw_ratio=0.968,
            safety_multiplier=2.36,
            capacity_safety_factor=0.85,
        )
        profile = strategy._build_profile(state)

        self.assertFalse(profile.rich)
        self.assertFalse(
            strategy._late_reroll_is_worth_it(
                state,
                pressure,
                profile,
                strategy._ShopContext(rerolls_in_shop=0),
            )
        )

    def test_late_shop_closer_allows_deeper_rerolls_when_rich_and_underpowered(self) -> None:
        bot = BasicStrategyBot(seed=1)
        state = GameState(
            seed=123,
            ante=6,
            blind="Big Blind",
            required_score=20000,
            money=90,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        actions = [bot.choose_action(state) for _ in range(9)]

        self.assertTrue(all(action.action_type == ActionType.REROLL for action in actions[:8]))
        self.assertEqual(actions[8].action_type, ActionType.END_SHOP)
        self.assertEqual(actions[8].metadata["shop_audit"]["shop_context"]["rerolls_in_shop"], 8)

    def test_late_shop_spreads_rerolls_after_small_blind(self) -> None:
        bot = BasicStrategyBot(seed=1)
        state = GameState(
            seed=123,
            ante=6,
            blind="Small Blind",
            required_score=20000,
            money=90,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        actions = [bot.choose_action(state) for _ in range(3)]

        self.assertTrue(all(action.action_type == ActionType.REROLL for action in actions[:2]))
        self.assertEqual(actions[2].action_type, ActionType.END_SHOP)
        self.assertEqual(actions[2].metadata["shop_audit"]["shop_context"]["rerolls_in_shop"], 2)

    def test_late_severe_pressure_can_extend_past_normal_reroll_cap(self) -> None:
        bot = BasicStrategyBot(seed=1)
        state = GameState(
            seed=123,
            ante=7,
            blind="Small Blind",
            required_score=52500,
            money=43,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        actions = [bot.choose_action(state) for _ in range(7)]

        self.assertTrue(all(action.action_type == ActionType.REROLL for action in actions[:6]))
        self.assertEqual(actions[6].action_type, ActionType.END_SHOP)
        self.assertEqual(actions[6].metadata["shop_audit"]["shop_context"]["rerolls_in_shop"], 6)

    def test_late_severe_pressure_extends_reroll_cap_when_money_is_available(self) -> None:
        bot = BasicStrategyBot(seed=1)
        state = GameState(
            seed=123,
            ante=7,
            blind="Small Blind",
            required_score=52500,
            money=70,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        actions = [bot.choose_action(state) for _ in range(8)]

        self.assertTrue(all(action.action_type == ActionType.REROLL for action in actions[:7]))
        self.assertEqual(actions[7].action_type, ActionType.END_SHOP)
        self.assertEqual(actions[7].metadata["shop_audit"]["shop_context"]["rerolls_in_shop"], 7)

    def test_late_shop_closer_rerolls_when_only_missing_chips_and_mult(self) -> None:
        state = GameState(
            ante=7,
            blind="Small Blind",
            required_score=52500,
            money=86,
            jokers=(
                Joker("Green Joker"),
                Joker("Hanging Chad"),
                Joker("Arrowhead"),
                Joker("Flower Pot"),
                Joker("Devious Joker"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        self.assertGreaterEqual(action.metadata["shop_audit"]["money_plan"]["spendable_money"], 60)
        self.assertNotIn("xmult", action.metadata["shop_audit"]["build_profile"]["missing_roles"])

    def test_late_shop_closer_spends_interest_bank_under_pressure(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=30000,
            money=93,
            vouchers=("Seed Money", "Money Tree"),
            jokers=(
                Joker("Erosion"),
                Joker("Ride the Bus"),
                Joker("Swashbuckler"),
                Joker("Seeing Double"),
                Joker("Odd Todd"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.REROLL)
        money_plan = action.metadata["shop_audit"]["money_plan"]
        self.assertEqual(money_plan["interest_cap_money"], 100)
        self.assertEqual(money_plan["reserve_money"], 65)
        self.assertEqual(money_plan["spendable_money"], 28)

    def test_ante_eight_bank_conversion_dips_below_high_interest_cap_for_missing_xmult(self) -> None:
        state = GameState(
            ante=8,
            blind="Big Blind",
            required_score=100000,
            money=90,
            vouchers=("Seed Money", "Money Tree"),
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
        )
        pressure = strategy._ShopPressure(
            target_score=100000.0,
            build_capacity=95000.0,
            ratio=1.05,
            raw_ratio=0.96,
            safety_multiplier=1.35,
            capacity_safety_factor=0.8,
            boss_name="Crimson Heart",
            boss_capacity_factor=0.72,
        )
        profile = strategy._build_profile(state)

        self.assertIn("xmult", profile.missing_roles)
        self.assertTrue(strategy._late_pressure_closer_mode(state, pressure, profile))
        self.assertEqual(strategy._desired_money_reserve(state, pressure), 75)
        self.assertEqual(strategy._spendable_money(state, pressure), 15)

    def test_bank_conversion_keeps_money_scaling_reserve_protected(self) -> None:
        state = GameState(
            ante=8,
            blind="Big Blind",
            required_score=100000,
            money=90,
            vouchers=("Seed Money", "Money Tree"),
            jokers=(
                Joker("Bull"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
        )
        pressure = strategy._ShopPressure(
            target_score=100000.0,
            build_capacity=95000.0,
            ratio=1.05,
            raw_ratio=0.96,
            safety_multiplier=1.35,
            capacity_safety_factor=0.8,
            boss_name="Crimson Heart",
            boss_capacity_factor=0.72,
        )

        profile = strategy._build_profile(state)

        self.assertFalse(strategy._late_bank_conversion_mode(state, pressure, profile))
        self.assertFalse(strategy._late_pressure_closer_mode(state, pressure, profile))
        self.assertEqual(strategy._desired_money_reserve(state, pressure), 50)

    def test_late_shop_does_not_expensive_reroll_below_pressure_floor(self) -> None:
        state = GameState(
            ante=7,
            blind="Small Blind",
            required_score=52500,
            money=31,
            modifiers={"reroll_cost": 7},
            jokers=(
                Joker("Green Joker"),
                Joker("Hanging Chad"),
                Joker("Arrowhead"),
                Joker("Flower Pot"),
                Joker("Devious Joker"),
            ),
            legal_actions=(
                Action(ActionType.REROLL),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.END_SHOP)
        self.assertEqual(action.metadata["shop_audit"]["money_plan"]["reserve_money"], 20)

    def test_late_shop_closer_opens_packs_after_normal_late_cap(self) -> None:
        state = GameState(
            ante=7,
            blind="Small Blind",
            required_score=52500,
            money=90,
            jokers=(
                Joker("Green Joker"),
                Joker("Hanging Chad"),
                Joker("Arrowhead"),
                Joker("Flower Pot"),
                Joker("Devious Joker"),
            ),
            modifiers={
                "booster_packs": (
                    {"label": "Jumbo Celestial Pack", "set": "PACK", "cost": {"buy": 6}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = strategy._shop_action(state, context=strategy._ShopContext(packs_opened_in_shop=2))

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.OPEN_PACK)

    def test_late_shop_opens_planet_pack_when_missing_scaling(self) -> None:
        state = GameState(
            ante=5,
            blind="Small Blind",
            required_score=11000,
            money=80,
            jokers=(
                Joker("Cavendish"),
                Joker("Banner"),
                Joker("Jolly Joker"),
                Joker("Golden Joker"),
            ),
            modifiers={
                "booster_packs": (
                    {"label": "Celestial Pack", "set": "PACK", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertIn("scaling", action.metadata["shop_audit"]["build_profile"]["missing_roles"])

    def test_high_pressure_forces_subthreshold_planet_spend_above_reserve(self) -> None:
        state = GameState(
            ante=7,
            blind="Small Blind",
            required_score=52500,
            money=45,
            jokers=(
                Joker("Stuntman"),
                Joker("Jolly Joker"),
                Joker("Blue Joker"),
                Joker("Golden Joker"),
                Joker("Gros Michel"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Jupiter", "set": "PLANET", "cost": {"buy": 3}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "forced_spend")
        self.assertLess(
            action.metadata["shop_audit"]["chosen_value"],
            action.metadata["shop_audit"]["threshold"],
        )

    def test_shop_buys_planet_for_direct_use_when_consumable_slots_are_full(self) -> None:
        state = GameState(
            ante=6,
            blind="Small Blind",
            required_score=43500,
            money=28,
            consumables=("Judgement", "Judgement"),
            jokers=(
                Joker("Constellation"),
                Joker("Photograph"),
                Joker("Swashbuckler"),
                Joker("Raised Fist"),
                Joker("Devious Joker"),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Saturn", "key": "c_saturn", "set": "PLANET", "cost": {"buy": 3}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.metadata["shop_audit"]["decision"], "take")

    def test_constellation_commits_to_planet_buys(self) -> None:
        base = GameState(
            ante=5,
            blind="Small Blind",
            required_score=11000,
            money=40,
            jokers=(Joker("Jolly Joker"), Joker("Blue Joker")),
            modifiers={
                "shop_cards": (
                    {"label": "Jupiter", "set": "PLANET", "cost": {"buy": 3}},
                )
            },
        )
        with_constellation = replace(base, jokers=(Joker("Constellation"), Joker("Jolly Joker"), Joker("Blue Joker")))
        card = base.modifiers["shop_cards"][0]

        self.assertGreater(
            strategy._planet_card_value(with_constellation, card),
            strategy._planet_card_value(base, card) + 12.0,
        )

    def test_hologram_commits_to_standard_pack_capacity(self) -> None:
        pack = {"label": "Jumbo Standard Pack", "set": "PACK", "cost": {"buy": 6}}
        base = GameState(
            ante=5,
            blind="Small Blind",
            required_score=11000,
            money=40,
            jokers=(Joker("Jolly Joker"), Joker("Blue Joker")),
        )
        with_hologram = replace(base, jokers=(Joker("Hologram"), Joker("Jolly Joker"), Joker("Blue Joker")))
        pressure = strategy._shop_pressure(base)

        self.assertGreater(
            strategy._pack_capacity_gain(with_hologram, pack, pressure),
            strategy._pack_capacity_gain(base, pack, pressure) + 20.0,
        )

    def test_late_pressured_build_opens_arcana_pack_without_visible_hand(self) -> None:
        state = GameState(
            ante=7,
            blind="Big Blind",
            required_score=52500,
            money=80,
            jokers=(
                Joker("Bull"),
                Joker("Bootstraps"),
                Joker("Seeing Double"),
                Joker("Green Joker"),
                Joker("Arrowhead"),
            ),
            modifiers={
                "booster_packs": (
                    {"label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)
        self.assertIn("pressure=", action.metadata["reason"])
        self.assertEqual(action.metadata["shop_audit"]["decision"], "take")

    def test_late_rich_shop_opens_celestial_pack_for_incremental_upgrades(self) -> None:
        state = GameState(
            ante=7,
            blind="Big Blind",
            required_score=52500,
            money=120,
            jokers=(
                Joker("Stuntman"),
                Joker("Gros Michel"),
                Joker("Sock and Buskin"),
                Joker("Smiley Face"),
                Joker("Photograph"),
            ),
            modifiers={
                "booster_packs": (
                    {"label": "Celestial Pack", "set": "PACK", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.OPEN_PACK)

    def test_high_ante_boss_discards_when_best_hand_is_barely_on_raw_pace(self) -> None:
        state = GameState(
            ante=7,
            blind="The Goad",
            required_score=3500,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("9", "C"),
                Card("5", "D"),
                Card("4", "S"),
                Card("2", "C"),
            ),
            jokers=(Joker("Joker"), Joker("Banner"), Joker("Blue Joker"), Joker("Green Joker")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4, 5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)

    def test_banner_preserve_play_blocks_projected_worse_discard_on_raw_pace(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=8,
            blind="Cerulean Bell",
            required_score=100000,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            jokers=(Joker("Banner"),),
        )
        play = Action(ActionType.PLAY_HAND, card_indices=(0,))
        discard = Action(ActionType.DISCARD, card_indices=(1,))

        with patch.object(strategy, "_score_play_action", return_value=27000):
            with patch.object(strategy, "_projected_score_after_discard", return_value=24000):
                reason = strategy._banner_preserve_play_reason(
                    state, play, discard, 30000, 100000, strategy._BlindContext()
                )

        self.assertIsNotNone(reason)
        self.assertIn("preserve_banner", reason or "")

    def test_banner_preserve_play_allows_projected_better_discard(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=8,
            blind="Cerulean Bell",
            required_score=100000,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            jokers=(Joker("Banner"),),
        )
        play = Action(ActionType.PLAY_HAND, card_indices=(0,))
        discard = Action(ActionType.DISCARD, card_indices=(1,))

        with patch.object(strategy, "_score_play_action", return_value=27000):
            with patch.object(strategy, "_projected_score_after_discard", return_value=40000):
                reason = strategy._banner_preserve_play_reason(
                    state, play, discard, 30000, 100000, strategy._BlindContext()
                )

        self.assertIsNone(reason)

    def test_banner_preserve_play_blocks_small_gain_when_hands_needed_same(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=8,
            blind="Cerulean Bell",
            required_score=100000,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            jokers=(Joker("Banner"),),
        )
        play = Action(ActionType.PLAY_HAND, card_indices=(0,))
        discard = Action(ActionType.DISCARD, card_indices=(1,))

        with patch.object(strategy, "_score_play_action", return_value=27000):
            with patch.object(strategy, "_projected_score_after_discard", return_value=31000):
                reason = strategy._banner_preserve_play_reason(
                    state, play, discard, 30000, 100000, strategy._BlindContext()
                )

        self.assertIsNotNone(reason)
        self.assertIn("hands_needed=4->4", reason or "")

    def test_last_safety_discard_does_not_take_known_worse_draw(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=3,
            blind="Big Blind",
            required_score=3000,
            current_score=0,
            hands_remaining=4,
            discards_remaining=1,
            deck_size=41,
            hand=(
                Card("K", "C"),
                Card("J", "C"),
                Card("8", "S"),
                Card("8", "C"),
                Card("5", "H"),
                Card("5", "D"),
                Card("3", "C"),
                Card("3", "D"),
            ),
            known_deck=(Card("6", "D"), Card("A", "H"), Card("10", "S")),
            jokers=(
                Joker("Crafty Joker", sell_value=2),
                Joker("Devious Joker", sell_value=2),
                Joker("Baron", sell_value=4),
                Joker("Swashbuckler", sell_value=2),
                Joker("Smiley Face", sell_value=2, metadata={"config": {"extra": 5}}),
            ),
        )
        discard = Action(ActionType.DISCARD, card_indices=(4,))
        context = strategy._BlindContext()

        projected_score = strategy._projected_score_after_discard(state, discard, context)

        self.assertLess(projected_score, 598)
        self.assertFalse(strategy._should_safety_discard(state, discard, 598, 3000, context))
        self.assertFalse(strategy._should_chase_discard(state, discard, 598, 3000, context))

    def test_safety_discard_requires_gain_with_discard_penalty_joker(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=7,
            blind="Small Blind",
            required_score=35000,
            current_score=11220,
            hands_remaining=3,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("K", "S"),
                Card("Q", "S"),
                Card("J", "S"),
                Card("9", "H"),
                Card("8", "D"),
                Card("5", "C"),
                Card("2", "H"),
            ),
            jokers=(Joker("Green Joker"),),
        )
        discard = Action(ActionType.DISCARD, card_indices=(4, 5, 6, 7))

        with patch.object(strategy, "_projected_score_after_discard", return_value=8000):
            with patch.object(strategy, "_strong_draw_size", return_value=4):
                self.assertFalse(
                    strategy._should_safety_discard(state, discard, 8000, 30000, strategy._BlindContext())
                )

    def test_safety_discard_allows_large_gain_with_discard_penalty_joker(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=7,
            blind="Small Blind",
            required_score=35000,
            current_score=11220,
            hands_remaining=3,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("K", "S"),
                Card("Q", "S"),
                Card("J", "S"),
                Card("9", "H"),
                Card("8", "D"),
                Card("5", "C"),
                Card("2", "H"),
            ),
            jokers=(Joker("Green Joker"),),
        )
        discard = Action(ActionType.DISCARD, card_indices=(4, 5, 6, 7))

        with patch.object(strategy, "_projected_score_after_discard", return_value=12000):
            self.assertTrue(strategy._should_safety_discard(state, discard, 8000, 30000, strategy._BlindContext()))

    def test_last_hand_hunt_does_not_spend_known_no_gain_discard(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=6,
            blind="The Wall",
            required_score=80000,
            current_score=37508,
            hands_remaining=1,
            discards_remaining=1,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("Q", "C"),
                Card("9", "S"),
            ),
            known_deck=(Card("2", "D"), Card("3", "C")),
        )
        discard = Action(ActionType.DISCARD, card_indices=(2, 3))

        with patch.object(strategy, "_projected_score_after_discard", return_value=9300):
            self.assertFalse(
                strategy._should_last_hand_hunt_discard(state, discard, 9300, 42492, strategy._BlindContext())
            )

    def test_last_hand_hunt_allows_desperation_dig_with_discards_left(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="The Needle",
            required_score=800,
            current_score=0,
            hands_remaining=1,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("K", "S"),
                Card("Q", "S"),
                Card("J", "S"),
                Card("9", "H"),
                Card("8", "D"),
                Card("5", "C"),
                Card("2", "H"),
            ),
            jokers=(Joker("Green Joker"),),
            known_deck=(Card("3", "D"), Card("4", "C")),
        )
        discard = Action(ActionType.DISCARD, card_indices=(4, 5, 6, 7))

        with patch.object(strategy, "_projected_score_after_discard", return_value=318):
            self.assertTrue(
                strategy._should_last_hand_hunt_discard(state, discard, 318, 800, strategy._BlindContext())
            )

    def test_last_hand_hunt_does_not_spend_final_penalty_discard_for_only_draw_shape(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=7,
            blind="Small Blind",
            required_score=35000,
            current_score=11220,
            hands_remaining=1,
            discards_remaining=1,
            hand=(
                Card("A", "S"),
                Card("K", "S"),
                Card("Q", "S"),
                Card("J", "S"),
                Card("9", "H"),
                Card("8", "D"),
                Card("5", "C"),
                Card("2", "H"),
            ),
            jokers=(Joker("Green Joker"),),
        )
        discard = Action(ActionType.DISCARD, card_indices=(4, 5, 6, 7))

        with patch.object(strategy, "_projected_score_after_discard", return_value=8000):
            with patch.object(strategy, "_strong_draw_size", return_value=4):
                self.assertFalse(
                    strategy._should_last_hand_hunt_discard(state, discard, 8000, 23780, strategy._BlindContext())
                )

    def test_projected_discard_score_reuses_equivalent_hand_content(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=800,
            current_score=0,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            hand=(
                Card("2", "S"),
                Card("2", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("Q", "C"),
                Card("J", "S"),
            ),
            jokers=(Joker("Joker"),),
        )
        context = strategy._BlindContext()

        with strategy.decision_cache_scope():
            with patch.object(strategy, "best_play_from_hand", wraps=strategy.best_play_from_hand) as best_play:
                first_score = strategy._projected_score_after_discard(state, Action(ActionType.DISCARD, card_indices=(0,)), context)
                calls_after_first = best_play.call_count
                second_score = strategy._projected_score_after_discard(state, Action(ActionType.DISCARD, card_indices=(1,)), context)

        self.assertEqual(second_score, first_score)
        self.assertGreater(calls_after_first, 0)
        self.assertEqual(best_play.call_count, calls_after_first)

    def test_projected_discard_score_decrements_banner_discard_count(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=5,
            blind="Big Blind",
            required_score=16500,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=40,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("Q", "C"),
                Card("J", "S"),
            ),
            known_deck=(Card("2", "D"),),
            jokers=(Joker("Banner"),),
        )
        action = Action(ActionType.DISCARD, card_indices=(2,))

        projected = strategy._projected_score_after_discard(
            state, action, strategy._BlindContext()
        )

        # Pair of Aces with Banner after spending one discard:
        # (10 base chips + 22 card chips + 3 * 30 Banner chips) * 2 mult.
        self.assertEqual(projected, 244)

    def test_optimistic_completion_score_reuses_equivalent_projected_states(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            blind="Small Blind",
            required_score=800,
            current_score=0,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            jokers=(Joker("Joker"),),
        )
        equivalent_state = GameState(
            phase=state.phase,
            ante=state.ante,
            blind=state.blind,
            required_score=state.required_score,
            current_score=state.current_score,
            hands_remaining=state.hands_remaining,
            discards_remaining=state.discards_remaining,
            money=state.money,
            deck_size=state.deck_size,
            hand=state.hand,
            jokers=state.jokers,
            hand_levels=dict(state.hand_levels),
            modifiers=dict(state.modifiers),
        )
        context = strategy._BlindContext()

        with strategy.decision_cache_scope():
            with patch.object(strategy, "best_play_from_hand", wraps=strategy.best_play_from_hand) as best_play:
                first_score = strategy._optimistic_completion_score(state, state.hand, 2, context)
                calls_after_first = best_play.call_count
                second_score = strategy._optimistic_completion_score(equivalent_state, equivalent_state.hand, 2, context)

        self.assertEqual(second_score, first_score)
        self.assertGreater(calls_after_first, 0)
        self.assertEqual(best_play.call_count, calls_after_first)

    def test_the_eye_avoids_repeating_played_hand_type(self) -> None:
        bot = BasicStrategyBot(seed=1)
        first_state = GameState(
            seed=123,
            ante=5,
            blind="The Eye",
            required_score=22000,
            current_score=0,
            hands_remaining=4,
            discards_remaining=0,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(2,)),
            ),
        )
        second_state = GameState(
            seed=123,
            ante=5,
            blind="The Eye",
            required_score=22000,
            current_score=100,
            hands_remaining=3,
            discards_remaining=0,
            hand=(Card("K", "S"), Card("K", "H"), Card("A", "D")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(2,)),
            ),
        )

        first_action = bot.choose_action(first_state)
        second_action = bot.choose_action(second_state)

        self.assertEqual(first_action.card_indices, (0, 1))
        self.assertEqual(second_action.card_indices, (2,))

    def test_the_eye_uses_bridge_played_this_round_as_hard_rule(self) -> None:
        state = GameState(
            seed=123,
            ante=5,
            blind="The Eye",
            required_score=22000,
            current_score=100,
            hands_remaining=3,
            discards_remaining=0,
            hand=(Card("K", "S"), Card("K", "H"), Card("A", "D")),
            modifiers={
                "hands": {
                    "Pair": {"played_this_round": 1, "order": 2},
                }
            },
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(2,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (2,))

    def test_the_mouth_sticks_to_first_played_hand_type(self) -> None:
        bot = BasicStrategyBot(seed=1)
        first_state = GameState(
            seed=456,
            ante=5,
            blind="The Mouth",
            required_score=22000,
            current_score=0,
            hands_remaining=4,
            discards_remaining=0,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(2,)),
            ),
        )
        second_state = GameState(
            seed=456,
            ante=5,
            blind="The Mouth",
            required_score=22000,
            current_score=100,
            hands_remaining=3,
            discards_remaining=0,
            hand=(
                Card("K", "S"),
                Card("K", "H"),
                Card("A", "S"),
                Card("Q", "S"),
                Card("J", "S"),
                Card("9", "S"),
                Card("7", "S"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(2, 3, 4, 5, 6)),
            ),
        )

        first_action = bot.choose_action(first_state)
        second_action = bot.choose_action(second_state)

        self.assertEqual(first_action.card_indices, (0, 1))
        self.assertEqual(second_action.card_indices, (0, 1))

    def test_card_sharp_repeats_prior_hand_type_for_xmult(self) -> None:
        bot = BasicStrategyBot(seed=1)
        first_state = GameState(
            seed=789,
            ante=2,
            blind="Big Blind",
            required_score=1200,
            current_score=0,
            hands_remaining=4,
            discards_remaining=0,
            hand=(Card("A", "S"), Card("A", "H"), Card("9", "D")),
            hand_levels={"Pair": 2},
            jokers=(Joker("Card Sharp"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
            ),
        )
        second_state = GameState(
            seed=789,
            ante=2,
            blind="Big Blind",
            required_score=1200,
            current_score=141,
            hands_remaining=3,
            discards_remaining=0,
            hand=(
                Card("K", "S"),
                Card("K", "H"),
                Card("2", "D"),
                Card("3", "C"),
                Card("4", "H"),
                Card("5", "S"),
                Card("6", "D"),
            ),
            hand_levels={"Pair": 2},
            jokers=(Joker("Card Sharp"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.PLAY_HAND, card_indices=(2, 3, 4, 5, 6)),
            ),
        )

        first_action = bot.choose_action(first_state)
        second_action = bot.choose_action(second_state)

        self.assertEqual(first_action.card_indices, (0, 1))
        self.assertEqual(second_action.card_indices, (0, 1))

    def test_sample_build_score_projects_card_sharp_repeat_xmult(self) -> None:
        state = GameState(
            ante=3,
            blind="Small Blind",
            hands_remaining=4,
            deck_size=40,
            hand_levels={"Pair": 1},
            hand=(Card("A", "S"), Card("A", "H")),
            jokers=(Joker("Joker"), Joker("Card Sharp")),
        )

        card_sharp_build = strategy._sample_build_score(state, state.jokers)
        flat_mult_build = strategy._sample_build_score(
            state,
            (Joker("Joker"), Joker("Joker")),
        )

        self.assertGreater(card_sharp_build, flat_mult_build)

    def test_owned_joker_value_counts_card_sharp_repeat_xmult(self) -> None:
        state = GameState(
            ante=4,
            blind="Small Blind",
            money=14,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=40,
            jokers=(
                Joker("Joker", sell_value=3),
                Joker("Mystic Summit", sell_value=3),
                Joker("Photograph", sell_value=3),
                Joker("Card Sharp", sell_value=3),
                Joker("Constellation", sell_value=3, metadata={"value": {"effect": "Currently X1.3 Mult"}}),
            ),
        )

        joker_value = strategy._owned_joker_value(state, state.jokers[0], remove_index=0)
        card_sharp_value = strategy._owned_joker_value(state, state.jokers[3], remove_index=3)

        self.assertGreater(card_sharp_value, joker_value)

    def test_mystic_summit_burns_discard_when_active_score_clears(self) -> None:
        state = GameState(
            ante=1,
            blind="The Pillar",
            required_score=600,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("9", "H"),
                Card("4", "C"),
                Card("3", "D"),
                Card("2", "S"),
            ),
            jokers=(Joker("Mystic Summit", sell_value=3),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(5, 6, 7)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (5, 6, 7))
        self.assertIn("mystic_summit_setup", action.metadata["reason"])

    def test_square_joker_prefers_four_card_clear_to_scale(self) -> None:
        state = GameState(
            ante=2,
            blind="Small Blind",
            required_score=100,
            current_score=0,
            hands_remaining=4,
            discards_remaining=2,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D"), Card("7", "C"), Card("2", "S")),
            jokers=(Joker("Square Joker", metadata={"value": {"effect": "Currently +0 Chips"}}),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1, 2, 3))

    def test_trading_card_takes_safe_first_single_card_discard(self) -> None:
        state = GameState(
            ante=2,
            blind="Small Blind",
            required_score=100,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D"), Card("7", "C"), Card("2", "S")),
            jokers=(Joker("Trading Card"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(4,)),
                Action(ActionType.DISCARD, card_indices=(2, 3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.card_indices, (4,))
        self.assertIn("Trading Card", action.metadata["reason"])

    def test_mail_in_rebate_rank_parser_ignores_payout_dollars(self) -> None:
        state = GameState(
            jokers=(
                Joker(
                    "Mail-In Rebate",
                    metadata={"value": {"effect": "Earn $5 for each discarded 8, rank changes every round"}},
                ),
            )
        )

        self.assertEqual(strategy._mail_in_rebate_rank(state), "8")

    def test_dna_opens_with_safe_single_card_copy(self) -> None:
        state = GameState(
            ante=2,
            blind="Small Blind",
            required_score=100,
            current_score=0,
            hands_remaining=4,
            discards_remaining=2,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D"), Card("7", "C"), Card("2", "S")),
            jokers=(Joker("DNA"),),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0,)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0,))
        self.assertIn("joker_setup", action.metadata["reason"])

    def test_basic_bot_selects_blind_even_with_safe_throwback_skip(self) -> None:
        state = GameState(
            ante=2,
            blind="Small Blind",
            required_score=800,
            hands_remaining=4,
            discards_remaining=4,
            money=20,
            jokers=(
                Joker("Throwback", metadata={"value": {"effect": "Currently X3"}}),
                Joker("Joker"),
                Joker("Stuntman"),
            ),
            legal_actions=(
                Action(ActionType.SELECT_BLIND),
                Action(ActionType.SKIP_BLIND),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.SELECT_BLIND)

    def test_full_shop_replacement_values_missing_xmult_role(self) -> None:
        state = GameState(
            ante=6,
            blind="Big Blind",
            required_score=30000,
            money=95,
            jokers=(
                Joker("Banner", sell_value=2),
                Joker("Jolly Joker", sell_value=2),
                Joker("Blue Joker", sell_value=3),
                Joker("Golden Joker", sell_value=3),
                Joker("Gros Michel", sell_value=2),
            ),
            modifiers={
                "shop_cards": (
                    {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
            legal_actions=(
                Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
                Action(ActionType.SELL, target_id="joker", amount=1, metadata={"kind": "joker", "index": 1}),
                Action(ActionType.SELL, target_id="joker", amount=2, metadata={"kind": "joker", "index": 2}),
                Action(ActionType.SELL, target_id="joker", amount=3, metadata={"kind": "joker", "index": 3}),
                Action(ActionType.SELL, target_id="joker", amount=4, metadata={"kind": "joker", "index": 4}),
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.SELL)
        audit = action.metadata["shop_audit"]
        self.assertEqual(audit["decision"], "replace")
        self.assertEqual(audit["chosen_replacement"], "Cavendish")
        self.assertGreater(audit["replacement_options"][0]["role_upgrade"], 0)

    def test_pack_choice_takes_useful_card_over_skip(self) -> None:
        state = GameState(
            money=6,
            modifiers={
                "pack_cards": (
                    {"label": "Judgement", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")

    def test_pack_choice_takes_positive_value_planet_under_old_threshold(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            ante=1,
            money=0,
            hands_remaining=4,
            discards_remaining=4,
            jokers=(Joker("Flash Card", sell_value=2),),
            modifiers={
                "pack_cards": (
                    {"label": "Pluto", "key": "c_pluto", "set": "PLANET", "config": {"hand_type": "High Card"}},
                    {"label": "Uranus", "key": "c_uranus", "set": "PLANET", "config": {"hand_type": "Two Pair"}},
                    {"label": "Mars", "key": "c_mars", "set": "PLANET", "config": {"hand_type": "Four of a Kind"}},
                    {"label": "Neptune", "key": "c_neptune", "set": "PLANET", "config": {"hand_type": "Straight Flush"}},
                    {"label": "Earth", "key": "c_earth", "set": "PLANET", "config": {"hand_type": "Full House"}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=2, metadata={"kind": "card", "index": 2}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=3, metadata={"kind": "card", "index": 3}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=4, metadata={"kind": "card", "index": 4}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")
        self.assertEqual(action.amount, 4)
        self.assertIn("pack_pick", action.metadata["reason"])

    def test_empty_booster_state_uses_no_op_instead_of_pack_skip(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            money=117,
            pack=(),
            modifiers={"pack_cards": ()},
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.NO_OP)

    def test_pack_choice_skips_medium_card_to_scale_red_card(self) -> None:
        state = GameState(
            ante=1,
            money=6,
            jokers=(Joker("Red Card", metadata={"value": {"effect": "Currently +0 Mult"}}),),
            modifiers={
                "pack_cards": (
                    {"label": "The Wheel of Fortune", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")
        self.assertIn("pack_skip", action.metadata["reason"])

    def test_unscaled_red_card_is_low_value_without_skip_plan(self) -> None:
        state = GameState(
            ante=1,
            money=10,
            modifiers={"joker_slots": 5},
        )

        red_card_value = strategy._joker_card_value(state, {"label": "Red Card", "set": "JOKER", "cost": {"buy": 4}})
        joker_value = strategy._joker_card_value(state, {"label": "Joker", "set": "JOKER", "cost": {"buy": 4}})

        self.assertLess(red_card_value, 35.0)
        self.assertLess(red_card_value, joker_value)

    def test_unscaled_red_card_gets_more_value_with_visible_skip_plan(self) -> None:
        without_pack = GameState(
            ante=1,
            money=10,
            modifiers={"joker_slots": 5},
        )
        with_pack = GameState(
            ante=1,
            money=10,
            modifiers={
                "joker_slots": 5,
                "booster_packs": ({"label": "Buffoon Pack", "set": "BOOSTER", "cost": {"buy": 4}},),
            },
        )

        without_value = strategy._joker_card_value(without_pack, {"label": "Red Card", "set": "JOKER", "cost": {"buy": 4}})
        with_value = strategy._joker_card_value(with_pack, {"label": "Red Card", "set": "JOKER", "cost": {"buy": 4}})

        self.assertGreater(with_value, without_value)

    def test_pack_choice_prioritizes_black_hole_over_matching_planet(self) -> None:
        state = GameState(
            money=46,
            ante=4,
            jokers=(Joker("Crafty Joker"), Joker("Constellation")),
            modifiers={
                "pack_cards": (
                    {"label": "Black Hole", "key": "c_black_hole", "set": "SPECTRAL", "cost": {"buy": 0}},
                    {"label": "Jupiter", "key": "c_jupiter", "set": "PLANET", "cost": {"buy": 0}},
                    {"label": "Mercury", "key": "c_mercury", "set": "PLANET", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=2, metadata={"kind": "card", "index": 2}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")
        self.assertEqual(action.amount, 0)
        self.assertTrue(action.metadata["reason"].startswith("pack_pick value="))

    def test_pack_choice_takes_black_hole_even_when_consumable_slots_are_full(self) -> None:
        state = GameState(
            money=46,
            consumables=("The Fool", "Death"),
            modifiers={
                "pack_cards": (
                    {"label": "Black Hole", "key": "c_black_hole", "set": "SPECTRAL", "cost": {"buy": 0}},
                    {"label": "Jupiter", "key": "c_jupiter", "set": "PLANET", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")
        self.assertEqual(action.amount, 0)

    def test_pack_choice_takes_immolate_for_money(self) -> None:
        state = GameState(
            money=6,
            hand=(
                Card("A", "S"),
                Card("K", "H"),
                Card("Q", "D"),
                Card("J", "C"),
                Card("9", "S"),
                Card("8", "H"),
                Card("4", "D"),
                Card("2", "C"),
            ),
            modifiers={
                "pack_cards": (
                    {"label": "Immolate", "key": "c_immolate", "set": "SPECTRAL", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")
        self.assertEqual(action.amount, 0)
        self.assertIn("pack_pick", action.metadata["reason"])

    def test_pack_choice_takes_targeted_spectral_upgrades(self) -> None:
        for name in ("Trance", "Aura", "Medium", "Talisman", "Deja Vu"):
            with self.subTest(name=name):
                state = GameState(
                    money=6,
                    hand=(Card("A", "S"), Card("K", "H")),
                    modifiers={
                        "pack_cards": (
                            {"label": name, "key": f"c_{name.lower().replace(' ', '_')}", "set": "SPECTRAL", "cost": {"buy": 0}},
                        )
                    },
                    legal_actions=(
                        Action(
                            ActionType.CHOOSE_PACK_CARD,
                            card_indices=(0,),
                            target_id="card",
                            amount=0,
                            metadata={"kind": "card", "index": 0},
                        ),
                        Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
                    ),
                )

                action = BasicStrategyBot(seed=1).choose_action(state)

                self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
                self.assertEqual(action.target_id, "card")
                self.assertEqual(action.card_indices, (0,))

    def test_pack_choice_can_skip_low_value_spectral_card(self) -> None:
        state = GameState(
            money=6,
            modifiers={
                "pack_cards": (
                    {"label": "Ouija", "key": "c_ouija", "set": "SPECTRAL", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")

    def test_pack_choice_skips_target_required_tarot_until_targets_are_supported(self) -> None:
        state = GameState(
            money=6,
            modifiers={
                "pack_cards": (
                    {"label": "Strength", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")

    def test_pack_choice_targets_supported_suit_tarot(self) -> None:
        state = GameState(
            money=6,
            hand=(
                Card("A", "S"),
                Card("K", "H"),
                Card("Q", "D"),
                Card("J", "C"),
                Card("9", "S"),
            ),
            jokers=(Joker("Crafty Joker"),),
            modifiers={
                "pack_cards": (
                    {"label": "The World", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")
        self.assertTrue(action.card_indices)
        self.assertLessEqual(len(action.card_indices), 3)

    def test_pack_choice_synthesizes_skip_when_only_unsafe_target_tarot_is_legal(self) -> None:
        state = GameState(
            money=6,
            modifiers={
                "pack_cards": (
                    {"label": "The World", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")

    def test_pack_choice_skips_joker_when_slots_are_full(self) -> None:
        state = GameState(
            money=6,
            jokers=(
                Joker("Joker"),
                Joker("Jolly Joker"),
                Joker("Sly Joker"),
                Joker("Banner"),
                Joker("Scholar"),
            ),
            modifiers={
                "pack_cards": (
                    {"label": "Green Joker", "set": "JOKER", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "skip")

    def test_pack_choice_takes_planet_when_consumable_slots_are_full(self) -> None:
        state = GameState(
            money=6,
            consumables=("The Fool", "Death"),
            modifiers={
                "pack_cards": (
                    {"label": "Jupiter", "key": "c_jupiter", "set": "PLANET", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")

    def test_shop_uses_held_planet_before_ending_shop(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            ante=2,
            blind="Small Blind",
            required_score=800,
            consumables=("Jupiter",),
            legal_actions=(
                Action(ActionType.USE_CONSUMABLE, target_id="consumable", amount=0, metadata={"kind": "consumable", "index": 0}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.USE_CONSUMABLE)
        self.assertEqual(action.amount, 0)

    def test_held_tarot_prefers_max_legal_targets(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=4,
            blind="Big Blind",
            required_score=7500,
            consumables=("The Moon",),
            hand=(
                Card("A", "S"),
                Card("K", "D"),
                Card("Q", "H"),
                Card("J", "C"),
                Card("10", "C"),
            ),
            legal_actions=(
                Action(
                    ActionType.USE_CONSUMABLE,
                    card_indices=(0,),
                    target_id="consumable",
                    amount=0,
                    metadata={"kind": "consumable", "index": 0},
                ),
                Action(
                    ActionType.USE_CONSUMABLE,
                    card_indices=(0, 1, 2),
                    target_id="consumable",
                    amount=0,
                    metadata={"kind": "consumable", "index": 0},
                ),
                Action(ActionType.PLAY_HAND, card_indices=(3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.USE_CONSUMABLE)
        self.assertEqual(action.card_indices, (0, 1, 2))

    def test_pack_choice_preserves_legal_target_indices_for_non_suit_tarot(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            money=6,
            hand=(Card("2", "S"), Card("K", "H")),
            modifiers={
                "pack_cards": (
                    {"label": "Strength", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, card_indices=(0,), target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")
        self.assertEqual(action.card_indices, (0,))

    def test_pack_choice_prefers_max_legal_targets_for_tarot(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            money=6,
            hand=(Card("2", "S"), Card("K", "H")),
            modifiers={
                "pack_cards": (
                    {"label": "Strength", "set": "TAROT", "cost": {"buy": 0}},
                )
            },
            legal_actions=(
                Action(ActionType.CHOOSE_PACK_CARD, card_indices=(0,), target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, card_indices=(0, 1), target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.CHOOSE_PACK_CARD)
        self.assertEqual(action.target_id, "card")
        self.assertEqual(action.card_indices, (0, 1))

    def test_discard_draw_count_uses_modified_hand_size_and_serpent(self) -> None:
        large_hand_state = GameState(
            hand=tuple(Card(str(rank), "S") for rank in ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5")),
            known_deck=tuple(Card(str(rank), "H") for rank in ("2", "3", "4", "5", "6")),
            deck_size=30,
            modifiers={"hand_size": 10},
        )
        discard_five = Action(ActionType.DISCARD, card_indices=(0, 1, 2, 3, 4))

        self.assertEqual(len(strategy._known_draw_for_discard(large_hand_state, discard_five)), 5)

        serpent_state = GameState(
            blind="The Serpent",
            hand=tuple(Card(str(rank), "S") for rank in ("A", "K", "Q", "J", "10", "9", "8", "7")),
            deck_size=30,
        )
        discard_one = Action(ActionType.DISCARD, card_indices=(0,))

        self.assertEqual(strategy._discard_draw_count(serpent_state, discard_one, kept_count=7), 3)

    def test_projected_discard_score_applies_burnt_joker_level(self) -> None:
        hand = (
            Card("2", "S"),
            Card("2", "H"),
            Card("A", "S"),
            Card("A", "H"),
            Card("K", "D"),
            Card("Q", "C"),
            Card("J", "D"),
            Card("8", "C"),
        )
        known = (Card("3", "S"), Card("4", "H"))
        action = Action(ActionType.DISCARD, card_indices=(0, 1))
        hand_levels = {hand_type.value: 1 for hand_type in HandType}
        enabled = GameState(
            hand=hand,
            known_deck=known,
            deck_size=40,
            jokers=(Joker("Burnt Joker"),),
            hand_levels=hand_levels,
            discards_remaining=4,
            hands_remaining=4,
        )
        disabled = replace(
            enabled,
            jokers=(Joker("Burnt Joker", metadata={"effect": "All abilities are disabled"}),),
        )

        self.assertGreater(
            strategy._projected_score_after_discard(enabled, action),
            strategy._projected_score_after_discard(disabled, action),
        )

    def test_discard_heuristics_ignore_disabled_jokers_and_debuffed_cards(self) -> None:
        action = Action(ActionType.DISCARD, card_indices=(0,))
        disabled_trading = GameState(
            hand=(Card("A", "S"),),
            jokers=(Joker("Trading Card", metadata={"effect": "All abilities are disabled"}),),
            discards_remaining=4,
        )
        debuffed_hit_road = GameState(
            hand=(Card("J", "S", debuffed=True),),
            jokers=(Joker("Hit the Road"),),
            discards_remaining=4,
        )

        self.assertEqual(
            strategy._discard_action_playstyle_bonus(
                disabled_trading,
                action,
                discarded_cards=disabled_trading.hand,
                keep_scores=(1.0,),
            ),
            0.0,
        )
        self.assertEqual(
            strategy._discard_action_playstyle_bonus(
                debuffed_hit_road,
                action,
                discarded_cards=debuffed_hit_road.hand,
                keep_scores=(1.0,),
            ),
            0.0,
        )
        self.assertEqual(
            strategy._jokers_after_discard_for_scoring(debuffed_hit_road, debuffed_hit_road.hand)[0].metadata,
            {},
        )

    def test_consumable_room_uses_slot_limit_modifier(self) -> None:
        state = GameState(
            consumables=("The Fool", "Death"),
            modifiers={"consumable_slot_limit": 3},
        )

        self.assertTrue(strategy._has_consumable_room(state))
        self.assertEqual(strategy._basic_consumable_open_slots(state), 1)


if __name__ == "__main__":
    unittest.main()
