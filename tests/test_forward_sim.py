from __future__ import annotations

import unittest
from dataclasses import replace

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.search.forward_sim import (
    StochasticPlayOutcomes,
    simulate_cash_out,
    simulate_buy,
    simulate_choose_pack_card,
    simulate_discard,
    simulate_end_shop,
    simulate_open_pack,
    simulate_play,
    simulate_reroll,
    simulate_select_blind,
    simulate_sell,
    simulate_use_consumable,
)


class ForwardSimTests(unittest.TestCase):
    def test_simulate_play_scores_selected_cards_and_draws_replacements_when_blind_continues(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=10,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D"), Card("2", "C")),
            modifiers={"hands": {}},
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
            drawn_cards=(Card("Q", "S"), Card("Q", "C")),
        )

        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(next_state.current_score, 64)
        self.assertEqual(next_state.hands_remaining, 3)
        self.assertEqual(next_state.discards_remaining, 3)
        self.assertEqual(next_state.deck_size, 8)
        self.assertEqual(tuple(card.short_name for card in next_state.hand), ("KD", "QS", "QC", "2C"))
        self.assertEqual(next_state.modifiers["hands"]["Pair"]["played_this_round"], 1)
        self.assertEqual(next_state.legal_actions, ())

    def test_simulate_play_returns_round_cards_to_deck_when_blind_clears(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            current_score=0,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=1,
            known_deck=(Card("2", "S"),),
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D")),
            modifiers={
                "hands": {},
                "played_pile": (Card("Q", "C"),),
                "discard_pile": (Card("7", "D"),),
            },
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1)))

        self.assertEqual(next_state.phase, GamePhase.ROUND_EVAL)
        self.assertEqual(next_state.hand, ())
        self.assertEqual(next_state.deck_size, 6)
        self.assertEqual(tuple(card.short_name for card in next_state.known_deck), ("2S", "KD", "AS", "AH", "QC", "7D"))
        self.assertNotIn("played_pile", next_state.modifiers)
        self.assertNotIn("discard_pile", next_state.modifiers)
        self.assertEqual(next_state.modifiers["hands"]["Pair"]["played_this_round"], 1)

    def test_simulate_play_applies_injected_hook_discards_to_held_cards(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Hook",
            required_score=500,
            current_score=0,
            hands_remaining=4,
            deck_size=2,
            known_deck=(Card("9", "S"), Card("8", "D")),
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            drawn_cards=(Card("9", "S"), Card("8", "D")),
            stochastic_outcomes=StochasticPlayOutcomes(hook_discarded_cards=(Card("Q", "D"),)),
        )

        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(tuple(card.short_name for card in next_state.hand), ("KH", "9S", "8D"))
        self.assertEqual(tuple(card.short_name for card in next_state.modifiers["discard_pile"]), ("QD",))
        self.assertEqual(next_state.deck_size, 0)

    def test_simulate_play_applies_hook_discard_joker_context_before_scoring(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Hook",
            required_score=10_000,
            current_score=0,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            jokers=(Joker("Green Joker", metadata={"current_mult": 5}),),
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            stochastic_outcomes=StochasticPlayOutcomes(hook_discarded_cards=(Card("Q", "D"),)),
        )

        self.assertEqual(next_state.current_score, 96)
        self.assertEqual(next_state.jokers[0].metadata["current_mult"], 5)

    def test_simulate_play_does_not_advance_disabled_green_joker(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Crimson Heart",
            required_score=10_000,
            current_score=0,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"),),
            jokers=(Joker("Green Joker", metadata={"state": {"debuff": True}, "current_mult": 5}),),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.current_score, 16)
        self.assertEqual(next_state.jokers[0].metadata["current_mult"], 5)

    def test_simulate_play_applies_crimson_heart_next_disabled_after_scoring(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Crimson Heart",
            required_score=10_000,
            current_score=0,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"),),
            jokers=(
                Joker("Stuntman", metadata={"state": {"debuff": True}, "value": {"effect": "All abilities are disabled"}}),
                Joker("Joker", metadata={"value": {"effect": "+4 Mult"}}),
            ),
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            stochastic_outcomes=StochasticPlayOutcomes(crimson_heart_disabled_joker_index=1),
        )

        self.assertEqual(next_state.current_score, 80)
        self.assertFalse(next_state.jokers[0].metadata.get("state"))
        self.assertEqual(next_state.jokers[1].metadata["state"]["debuff"], True)
        self.assertEqual(next_state.jokers[1].metadata["value"]["effect"], "All abilities are disabled")

    def test_simulate_play_returns_hook_discards_to_deck_when_blind_clears(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Hook",
            required_score=1,
            current_score=0,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            stochastic_outcomes=StochasticPlayOutcomes(hook_discarded_cards=(Card("Q", "D"),)),
        )

        self.assertEqual(next_state.phase, GamePhase.ROUND_EVAL)
        self.assertEqual(next_state.hand, ())
        self.assertEqual(next_state.deck_size, 3)
        self.assertNotIn("discard_pile", next_state.modifiers)

    def test_simulate_play_grows_egg_sell_value_when_blind_clears(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            hands_remaining=4,
            hand=(Card("A", "S"), Card("A", "H")),
            jokers=(Joker("Egg", sell_value=2, metadata={"extra_value": 0, "extra": 3}),),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1)))

        self.assertEqual(next_state.phase, GamePhase.ROUND_EVAL)
        self.assertEqual(next_state.jokers[0].sell_value, 5)
        self.assertEqual(next_state.jokers[0].metadata["extra_value"], 3)

    def test_simulate_play_grows_egg_sell_value_when_final_hand_loses(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Big Blind",
            required_score=1200,
            current_score=700,
            hands_remaining=1,
            hand=(Card("A", "S"),),
            jokers=(Joker("Egg", sell_value=8, metadata={"extra_value": 6, "extra": 3}),),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.phase, GamePhase.RUN_OVER)
        self.assertEqual(next_state.jokers[0].sell_value, 11)
        self.assertEqual(next_state.jokers[0].metadata["extra_value"], 9)

    def test_simulate_play_removes_expired_popcorn_when_blind_ends(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            hands_remaining=4,
            hand=(Card("A", "S"), Card("A", "H")),
            jokers=(Joker("Popcorn", sell_value=2, metadata={"current_mult": 4, "extra": 4}),),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1)))

        self.assertEqual(next_state.phase, GamePhase.ROUND_EVAL)
        self.assertEqual(next_state.jokers, ())

    def test_simulate_play_removes_injected_extinct_joker_when_blind_clears(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            hands_remaining=4,
            hand=(Card("A", "S"), Card("A", "H")),
            jokers=(Joker("Gros Michel"), Joker("Cavendish")),
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
            stochastic_outcomes=StochasticPlayOutcomes(removed_jokers=("Gros Michel",)),
        )

        self.assertEqual(next_state.phase, GamePhase.ROUND_EVAL)
        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("Cavendish",))

    def test_simulate_play_applies_deterministic_scoring_money_delta(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Tooth",
            required_score=500,
            current_score=0,
            money=10,
            hands_remaining=4,
            hand=(
                Card("K", "S", seal="Gold"),
                Card("K", "D", enhancement="GOLD"),
                Card("Q", "C"),
                Card("J", "H"),
            ),
            jokers=(Joker("Golden Ticket"),),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1)))

        self.assertEqual(next_state.money, 15)

    def test_simulate_play_accepts_8_ball_created_tarot_hook(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            hands_remaining=4,
            hand=(Card("8", "S"), Card("A", "H")),
            jokers=(Joker("8 Ball"),),
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            created_consumables=("The Fool",),
        )

        self.assertEqual(next_state.consumables, ("The Fool",))
        self.assertEqual(next_state.modifiers["created_tarot_count"], 1)

    def test_simulate_play_updates_lucky_cat_from_injected_lucky_triggers(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            hands_remaining=4,
            hand=(Card("A", "S"),),
            jokers=(Joker("Lucky Cat", metadata={"current_xmult": 1.0}),),
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            stochastic_outcomes=StochasticPlayOutcomes(lucky_card_triggers=2),
        )

        self.assertEqual(next_state.jokers[0].metadata["current_xmult"], 1.5)

    def test_simulate_play_removes_injected_shattered_glass_cards(self) -> None:
        glass_king = Card("K", "H", enhancement="GLASS")
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            current_score=0,
            hands_remaining=4,
            hand=(glass_king,),
            jokers=(
                Joker("Glass Joker", metadata={"current_xmult": 1.0}),
                Joker("Caino", metadata={"current_xmult": 1.0}),
            ),
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            stochastic_outcomes=StochasticPlayOutcomes(shattered_glass_cards=(glass_king,)),
        )
        by_name = {joker.name: joker.metadata for joker in next_state.jokers}

        self.assertEqual(next_state.hand, ())
        self.assertEqual(next_state.modifiers["destroyed_cards"], (glass_king,))
        self.assertEqual(next_state.modifiers["played_pile"], ())
        self.assertEqual(by_name["Glass Joker"]["current_xmult"], 1.75)
        self.assertEqual(by_name["Caino"]["current_xmult"], 2.0)

    def test_simulate_play_updates_space_joker_hand_level(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            hands_remaining=4,
            hand=(Card("A", "S"),),
            jokers=(Joker("Space Joker"),),
            modifiers={"hands": {}},
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            stochastic_outcomes=StochasticPlayOutcomes(space_joker_triggers=1),
        )

        self.assertEqual(next_state.current_score, 52)
        self.assertEqual(next_state.hand_levels["High Card"], 2)
        self.assertEqual(next_state.modifiers["hands"]["High Card"]["level"], 2)

    def test_simulate_play_pays_held_gold_cards_when_blind_clears(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=50,
            current_score=0,
            money=10,
            hands_remaining=4,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D", enhancement="GOLD")),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1)))

        self.assertEqual(next_state.phase, GamePhase.ROUND_EVAL)
        self.assertEqual(next_state.money, 13)

    def test_simulate_play_keeps_blind_phase_when_score_is_short(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=500,
            current_score=0,
            hands_remaining=4,
            deck_size=10,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D"), Card("2", "C")),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1)))

        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(next_state.current_score, 64)
        self.assertEqual(next_state.hand, (Card("K", "D"), Card("2", "C")))

    def test_simulate_play_uses_chicot_disabled_boss_blind(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Flint",
            required_score=500,
            current_score=0,
            hands_remaining=4,
            hand=(Card("A", "S"), Card("A", "H")),
            modifiers={"boss_disabled": True},
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1)))

        self.assertEqual(next_state.current_score, 64)

    def test_simulate_play_mr_bones_saves_losing_round_at_threshold(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=200,
            current_score=0,
            hands_remaining=1,
            hand=(Card("A", "S"), Card("A", "H")),
            jokers=(Joker("Mr. Bones"), Joker("Joker")),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1)))

        self.assertEqual(next_state.phase, GamePhase.ROUND_EVAL)
        self.assertFalse(next_state.run_over)
        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("Joker",))

    def test_simulate_discard_removes_cards_and_draws_without_changing_score(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            current_score=120,
            hands_remaining=4,
            discards_remaining=2,
            deck_size=9,
            hand=(Card("A", "S"), Card("7", "H"), Card("K", "D"), Card("2", "C")),
        )

        next_state = simulate_discard(
            state,
            Action(ActionType.DISCARD, card_indices=(1, 3)),
            drawn_cards=(Card("Q", "S"), Card("Q", "C")),
        )

        self.assertEqual(next_state.current_score, 120)
        self.assertEqual(next_state.hands_remaining, 4)
        self.assertEqual(next_state.discards_remaining, 1)
        self.assertEqual(next_state.deck_size, 7)
        self.assertEqual(tuple(card.short_name for card in next_state.hand), ("AS", "KD", "QS", "QC"))
        self.assertEqual(next_state.legal_actions, ())

    def test_simulate_discard_tracks_discard_pile_and_known_deck_draws(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=3,
            deck_size=2,
            hand=(Card("A", "S"), Card("7", "H")),
            known_deck=(Card("Q", "S"), Card("Q", "C")),
        )

        next_state = simulate_discard(
            state,
            Action(ActionType.DISCARD, card_indices=(1,)),
            drawn_cards=(Card("Q", "S"),),
        )

        self.assertEqual(tuple(card.short_name for card in next_state.hand), ("AS", "QS"))
        self.assertEqual(tuple(card.short_name for card in next_state.known_deck), ("QC",))
        self.assertEqual(next_state.deck_size, 1)
        self.assertEqual(tuple(card.short_name for card in next_state.modifiers["discard_pile"]), ("7H",))

    def test_simulate_play_rejects_non_play_actions(self) -> None:
        state = GameState(hand=(Card("A", "S"),))

        with self.assertRaises(ValueError):
            simulate_play(state, Action(ActionType.DISCARD, card_indices=(0,)))

    def test_simulate_play_updates_deterministic_scaling_jokers(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            hands_remaining=4,
            deck_size=20,
            hand=(Card("2", "S"), Card("2", "H"), Card("9", "D"), Card("9", "C"), Card("K", "S")),
            jokers=(
                Joker("Square Joker", metadata={"current_chips": 8}),
                Joker("Spare Trousers", metadata={"current_mult": 4}),
                Joker("Green Joker", metadata={"current_mult": 2}),
                Joker("Ride the Bus", metadata={"current_mult": 5}),
                Joker("Wee Joker", metadata={"current_chips": 16}),
                Joker("Ice Cream", metadata={"current_chips": 20}),
                Joker("Seltzer", metadata={"current_remaining": 2}),
            ),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3)))
        values = {joker.name: joker.metadata for joker in next_state.jokers}

        self.assertEqual(values["Square Joker"]["current_chips"], 12)
        self.assertEqual(values["Spare Trousers"]["current_mult"], 6)
        self.assertEqual(values["Green Joker"]["current_mult"], 3)
        self.assertEqual(values["Ride the Bus"]["current_mult"], 6)
        self.assertEqual(values["Wee Joker"]["current_chips"], 48)
        self.assertEqual(values["Ice Cream"]["current_chips"], 15)
        self.assertEqual(values["Seltzer"]["current_remaining"], 1)

    def test_simulate_play_removes_consumed_ice_cream_and_seltzer(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            hands_remaining=4,
            hand=(Card("A", "S"),),
            jokers=(
                Joker("Ice Cream", metadata={"current_chips": 5}),
                Joker("Seltzer", metadata={"current_remaining": 1}),
            ),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.jokers, ())

    def test_simulate_play_reads_visible_seltzer_countdown_text(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            hands_remaining=4,
            hand=(Card("A", "S"),),
            jokers=(
                Joker(
                    "Seltzer",
                    metadata={"value": {"effect": "Retrigger all cards played for the next 1 hands"}},
                ),
            ),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.jokers, ())

    def test_simulate_play_records_persistent_card_mutations(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            hands_remaining=4,
            hand=(Card("K", "S", enhancement="BONUS"), Card("K", "H"), Card("3", "D")),
            jokers=(
                Joker("Hiker"),
                Joker("Midas Mask"),
                Joker("Vampire", metadata={"current_xmult": 1.0}),
            ),
        )

        next_state = simulate_play(state, Action(ActionType.PLAY_HAND, card_indices=(0, 1)))
        played_pile = next_state.modifiers["played_pile"]
        values = {joker.name: joker.metadata for joker in next_state.jokers}

        self.assertEqual(tuple(card.short_name for card in played_pile), ("KS", "KH"))
        self.assertIsNone(played_pile[0].enhancement)
        self.assertIsNone(played_pile[1].enhancement)
        self.assertEqual(played_pile[0].metadata["modifier"]["perma_bonus"], 5)
        self.assertEqual(played_pile[1].metadata["modifier"]["perma_bonus"], 5)
        self.assertAlmostEqual(values["Vampire"]["current_xmult"], 1.2)

    def test_simulate_discard_updates_deterministic_discard_jokers(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=3,
            deck_size=20,
            hand=(Card("J", "S"), Card("7", "H"), Card("4", "D")),
            jokers=(
                Joker("Ramen", metadata={"current_xmult": 1.5}),
                Joker("Green Joker", metadata={"current_mult": 3}),
                Joker("Hit the Road", metadata={"current_xmult": 2.0}),
            ),
        )

        next_state = simulate_discard(state, Action(ActionType.DISCARD, card_indices=(0, 1)))
        values = {joker.name: joker.metadata for joker in next_state.jokers}

        self.assertAlmostEqual(values["Ramen"]["current_xmult"], 1.48)
        self.assertEqual(values["Green Joker"]["current_mult"], 2)
        self.assertEqual(values["Hit the Road"]["current_xmult"], 2.5)

    def test_simulate_discard_removes_eaten_ramen(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=3,
            hand=(Card("J", "S"),),
            jokers=(Joker("Ramen", metadata={"current_xmult": 1.01}),),
        )

        next_state = simulate_discard(state, Action(ActionType.DISCARD, card_indices=(0,)))

        self.assertEqual(next_state.jokers, ())

    def test_simulate_discard_levels_burnt_joker_hand_once(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=3,
            deck_size=20,
            hand=(Card("A", "S"), Card("A", "H"), Card("K", "D")),
            jokers=(Joker("Burnt Joker"),),
            hand_levels={"Pair": 1},
        )

        after_first = simulate_discard(
            state,
            Action(ActionType.DISCARD, card_indices=(0, 1)),
            drawn_cards=(Card("Q", "S"), Card("Q", "C")),
        )
        after_second = simulate_discard(after_first, Action(ActionType.DISCARD, card_indices=(1, 2)))

        self.assertEqual(after_first.hand_levels["Pair"], 2)
        self.assertEqual(after_first.modifiers["round_discards_used"], 1)
        self.assertEqual(after_second.hand_levels["Pair"], 2)
        self.assertEqual(after_second.modifiers["round_discards_used"], 2)

    def test_simulate_discard_skips_burnt_joker_after_known_prior_discard(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=3,
            hand=(Card("A", "S"), Card("A", "H")),
            jokers=(Joker("Burnt Joker"),),
            hand_levels={"Pair": 1},
            modifiers={"round_discards_used": 1},
        )

        next_state = simulate_discard(state, Action(ActionType.DISCARD, card_indices=(0, 1)))

        self.assertEqual(next_state.hand_levels["Pair"], 1)

    def test_simulate_discard_updates_visible_economy_and_countdown_jokers(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=3,
            money=2,
            hand=(Card("K", "H"), Card("Q", "H"), Card("J", "S"), Card("7", "D")),
            jokers=(
                Joker("Faceless Joker"),
                Joker("Mail-In Rebate", metadata={"target_rank": "King"}),
                Joker("Castle", metadata={"current_chips": 6, "target_suit": "Heart"}),
                Joker("Yorick", metadata={"current_remaining": 2, "current_xmult": 1.0}),
            ),
        )

        next_state = simulate_discard(state, Action(ActionType.DISCARD, card_indices=(0, 1, 2)))
        values = {joker.name: joker.metadata for joker in next_state.jokers}

        self.assertEqual(next_state.money, 12)
        self.assertEqual(values["Castle"]["current_chips"], 12)
        self.assertEqual(values["Yorick"]["current_remaining"], 22)
        self.assertEqual(values["Yorick"]["current_xmult"], 2.0)

    def test_simulate_discard_parses_mail_in_rebate_rank_after_discarded_text(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=3,
            money=4,
            hand=(Card("5", "S"), Card("5", "D"), Card("8", "H")),
            jokers=(
                Joker(
                    "Mail-In Rebate",
                    metadata={"value": {"effect": "Earn $5 for each discarded 8, rank changes every round"}},
                ),
            ),
        )

        no_payout = simulate_discard(state, Action(ActionType.DISCARD, card_indices=(0, 1)))
        payout = simulate_discard(state, Action(ActionType.DISCARD, card_indices=(2,)))

        self.assertEqual(no_payout.money, 4)
        self.assertEqual(payout.money, 9)

    def test_simulate_discard_adds_trading_card_money_only_on_first_single_discard(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=3,
            money=4,
            hand=(Card("7", "D"), Card("Q", "H")),
            jokers=(Joker("Trading Card"),),
        )

        after_first = simulate_discard(
            state,
            Action(ActionType.DISCARD, card_indices=(0,)),
            drawn_cards=(Card("3", "S"),),
        )
        after_second = simulate_discard(after_first, Action(ActionType.DISCARD, card_indices=(1,)))

        self.assertEqual(after_first.money, 7)
        self.assertEqual(after_second.money, 7)
        self.assertEqual(after_first.modifiers["discard_pile"], ())
        self.assertEqual(tuple(card.short_name for card in after_first.modifiers["destroyed_cards"]), ("7D",))

    def test_simulate_cash_out_returns_hand_played_and_discard_piles_to_deck(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            current_score=500,
            deck_size=1,
            hand=(Card("A", "H"),),
            known_deck=(Card("2", "S"),),
            modifiers={
                "played_pile": (Card("K", "S", metadata={"modifier": {"perma_bonus": 5}}),),
                "discard_pile": (Card("7", "D"),),
                "destroyed_cards": (Card("6", "C"),),
            },
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(next_state.phase, GamePhase.SHOP)
        self.assertEqual(next_state.current_score, 0)
        self.assertEqual(next_state.hand, ())
        self.assertEqual(tuple(card.short_name for card in next_state.known_deck), ("2S", "AH", "KS", "7D"))
        self.assertEqual(next_state.deck_size, 4)
        self.assertNotIn("played_pile", next_state.modifiers)
        self.assertNotIn("discard_pile", next_state.modifiers)
        self.assertEqual(tuple(card.short_name for card in next_state.modifiers["destroyed_cards"]), ("6C",))

    def test_simulate_cash_out_updates_visible_end_of_round_jokers(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="The Hook",
            current_score=500,
            deck_size=0,
            jokers=(
                Joker("Campfire", metadata={"current_xmult": 2.5}),
                Joker("Rocket", metadata={"current_dollars": 3, "increase": 2}),
                Joker("Popcorn", metadata={"current_mult": 12, "extra": 4}),
                Joker("Turtle Bean", metadata={"current_h_size": 2, "h_mod": 1}),
                Joker("Invisible Joker", metadata={"current_rounds": 1}),
                Joker("Egg", sell_value=2, metadata={"extra_value": 0, "extra": 3}),
                Joker("Hit the Road", metadata={"current_xmult": 2.5}),
                Joker("To Do List", metadata={"target_hand": "Pair"}),
            ),
        )

        next_state = simulate_cash_out(state, next_to_do_targets=("Flush",))
        values = {joker.name: joker for joker in next_state.jokers}

        self.assertEqual(values["Campfire"].metadata["current_xmult"], 1.0)
        self.assertEqual(values["Rocket"].metadata["current_dollars"], 5)
        self.assertEqual(values["Popcorn"].metadata["current_mult"], 8)
        self.assertEqual(values["Turtle Bean"].metadata["current_h_size"], 1)
        self.assertEqual(values["Invisible Joker"].metadata["current_rounds"], 2)
        self.assertEqual(values["Egg"].metadata["extra_value"], 0)
        self.assertEqual(values["Egg"].sell_value, 2)
        self.assertEqual(values["Hit the Road"].metadata["current_xmult"], 1.0)
        self.assertEqual(values["To Do List"].metadata["target_hand"], "Flush")

    def test_simulate_cash_out_adds_deterministic_money_rewards(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="The Hook",
            money=17,
            hands_remaining=2,
            discards_remaining=3,
            hand=(Card("9", "S"), Card("A", "H")),
            known_deck=(Card("9", "H"), Card("2", "C")),
            jokers=(
                Joker("Golden Joker"),
                Joker("Rocket", metadata={"current_dollars": 3}),
                Joker("Delayed Gratification"),
                Joker("Cloud 9", metadata={"current_nines": 2}),
            ),
            modifiers={"current_blind": {"type": "BOSS", "dollars": 5}, "money_per_discard": 1},
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(next_state.money, 45)

    def test_simulate_cash_out_honors_no_blind_reward_and_no_extra_hand_money(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Big Blind",
            money=17,
            hands_remaining=2,
            modifiers={
                "current_blind": {"type": "BIG", "dollars": 4},
                "no_blind_reward": {"Big": True},
                "no_extra_hand_money": True,
            },
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(next_state.money, 20)

    def test_simulate_cash_out_uses_logged_round_eval_dollars_when_available(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Big Blind",
            money=17,
            hands_remaining=2,
            modifiers={"cash_out_money_delta": 8},
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(next_state.money, 25)

    def test_simulate_cash_out_adds_satellite_money_when_planets_are_visible(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            money=10,
            jokers=(Joker("Satellite", metadata={"current_planets": 3}),),
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(next_state.money, 18)

    def test_planet_pick_updates_constellation_and_satellite_usage(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            blind="Small Blind",
            money=0,
            hand_levels={"Flush": 1},
            jokers=(Joker("Constellation", metadata={"current_xmult": 1.0}), Joker("Satellite")),
            modifiers={"pack_cards": ({"label": "Jupiter", "set": "PLANET"},)},
        )

        chosen = simulate_choose_pack_card(
            state,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"index": 0}),
        )
        paid = simulate_cash_out(replace(chosen, phase=GamePhase.ROUND_EVAL, hands_remaining=0))

        self.assertEqual(chosen.hand_levels["Flush"], 2)
        self.assertEqual(chosen.jokers[0].metadata["current_xmult"], 1.1)
        self.assertEqual(chosen.modifiers["planet_cards_used"], {"Jupiter": 1})
        self.assertEqual(paid.money, 4)

    def test_simulate_cash_out_reads_rocket_payout_from_visible_effect_text(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            money=0,
            jokers=(
                Joker(
                    "Rocket",
                    metadata={
                        "value": {
                            "effect": "Earn $5 at end of round. Payout increases by $2 when Boss Blind is defeated."
                        }
                    },
                ),
            ),
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(next_state.money, 8)

    def test_simulate_cash_out_increments_rocket_from_visible_effect_text_on_boss(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="The Hook",
            money=0,
            jokers=(
                Joker(
                    "Rocket",
                    metadata={
                        "value": {
                            "effect": "Earn $5 at end of round. Payout increases by $2 when Boss Blind is defeated."
                        }
                    },
                ),
            ),
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(next_state.jokers[0].metadata["current_dollars"], 7)

    def test_simulate_cash_out_infers_to_the_moon_interest_from_visible_joker(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            money=25,
            jokers=(Joker("To the Moon"),),
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(next_state.money, 38)

    def test_simulate_cash_out_resets_burglar_temporary_hands_after_round(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            hands_remaining=4,
            jokers=(Joker("Burglar"),),
            modifiers={"hands": {"Pair": {"played_this_round": 3, "played": 3, "order": 1}}},
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(next_state.hands_remaining, 4)

    def test_simulate_play_gift_card_grows_sell_values_at_end_of_round(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=1,
            current_score=0,
            hands_remaining=4,
            deck_size=47,
            hand=(Card("A", "H"), Card("K", "H"), Card("Q", "H"), Card("J", "H"), Card("10", "H")),
            jokers=(Joker("Gift Card", sell_value=2), Joker("Joker", sell_value=3)),
            consumables=("Mars", "The Fool"),
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
        )

        self.assertEqual(next_state.phase, GamePhase.ROUND_EVAL)
        self.assertEqual(tuple(joker.sell_value for joker in next_state.jokers), (3, 4))
        self.assertEqual(next_state.modifiers["consumable_sell_value_bonus"], 2)

    def test_simulate_cash_out_does_not_apply_gift_card_again(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            jokers=(Joker("Gift Card", sell_value=2), Joker("Joker", sell_value=3)),
            consumables=("Mars", "The Fool"),
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(tuple(joker.sell_value for joker in next_state.jokers), (2, 3))
        self.assertNotIn("consumable_sell_value_bonus", next_state.modifiers)

    def test_simulate_cash_out_removes_expired_popcorn_and_turtle_bean(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            jokers=(
                Joker("Popcorn", metadata={"current_mult": 4, "extra": 4}),
                Joker("Turtle Bean", metadata={"current_h_size": 1, "h_mod": 1}),
                Joker("Joker"),
            ),
        )

        next_state = simulate_cash_out(state)

        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("Joker",))

    def test_simulate_cash_out_removes_injected_extinct_jokers(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            jokers=(Joker("Gros Michel"), Joker("Cavendish")),
        )

        next_state = simulate_cash_out(state, removed_jokers=("Gros Michel",))

        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("Cavendish",))

    def test_simulate_cash_out_requires_injected_to_do_list_target(self) -> None:
        state = GameState(
            phase=GamePhase.ROUND_EVAL,
            blind="Small Blind",
            jokers=(Joker("To Do List", metadata={"target_hand": "Pair"}),),
        )

        with self.assertRaisesRegex(ValueError, "To Do List needs 1 injected"):
            simulate_cash_out(state)

    def test_simulate_buy_adds_joker_and_removes_shop_card(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            shop=("Joker", "Mars"),
            modifiers={
                "shop_cards": (
                    {"label": "Joker", "set": "JOKER", "cost": {"buy": 4, "sell": 2}},
                    {"label": "Mars", "set": "PLANET", "cost": {"buy": 3}},
                )
            },
        )

        next_state = simulate_buy(state, Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}))

        self.assertEqual(next_state.money, 6)
        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("Joker",))
        self.assertEqual(next_state.jokers[0].sell_value, 2)
        self.assertEqual(next_state.shop, ("Mars",))
        self.assertEqual(tuple(item["label"] for item in next_state.modifiers["shop_cards"]), ("Mars",))

    def test_simulate_buy_rejects_normal_joker_but_allows_negative_at_full_slots(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            jokers=(Joker("One"),),
            modifiers={
                "joker_slots": 1,
                "shop_cards": (
                    {"label": "Joker", "set": "JOKER", "cost": {"buy": 4, "sell": 2}},
                    {"label": "Cavendish", "set": "JOKER", "edition": "NEGATIVE", "cost": {"buy": 4, "sell": 2}},
                ),
            },
        )

        with self.assertRaisesRegex(ValueError, "full joker slots"):
            simulate_buy(state, Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}))

        next_state = simulate_buy(
            state,
            Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
        )

        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("One", "Cavendish"))
        self.assertEqual(next_state.jokers[1].edition, "NEGATIVE")

    def test_simulate_buy_adds_voucher_consumable_and_playing_card(self) -> None:
        voucher_state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            modifiers={"voucher_cards": ({"label": "Blank", "set": "VOUCHER", "cost": {"buy": 6}},)},
        )

        after_voucher = simulate_buy(
            voucher_state,
            Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}),
        )

        self.assertEqual(after_voucher.money, 4)
        self.assertEqual(after_voucher.vouchers, ("Blank",))
        self.assertEqual(after_voucher.modifiers["voucher_cards"], ())

    def test_simulate_buy_clearance_sale_reprices_existing_cards_like_game_set_cost(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=20,
            jokers=(
                Joker(
                    "Green Joker",
                    sell_value=2,
                    metadata={"cost": {"buy": 4, "sell": 2}, "value": {"effect": "+1 Mult per hand played (Currently +24 Mult)"}},
                ),
                Joker("Driver's License", edition="POLYCHROME", sell_value=6, metadata={"cost": {"buy": 12, "sell": 6}}),
            ),
            modifiers={
                "shop_cards": ({"name": "Wily Joker", "set": "JOKER", "cost": {"buy": 4, "sell": 2}},),
                "voucher_cards": (
                    {"name": "Clearance Sale", "set": "VOUCHER", "cost": {"buy": 10, "sell": 5}},
                ),
                "booster_packs": ({"name": "Celestial Pack", "set": "BOOSTER", "cost": {"buy": 4, "sell": 2}},),
            },
        )

        next_state = simulate_buy(
            state,
            Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}),
        )

        self.assertEqual(next_state.money, 10)
        self.assertEqual(next_state.vouchers, ("Clearance Sale",))
        self.assertEqual(next_state.modifiers["discount_percent"], 25)
        self.assertEqual(tuple(joker.sell_value for joker in next_state.jokers), (1, 4))
        self.assertEqual(next_state.modifiers["shop_cards"][0]["cost"], {"buy": 3, "sell": 1})
        self.assertEqual(next_state.modifiers["booster_packs"][0]["cost"], {"buy": 3, "sell": 1})

        driver_cost = next_state.jokers[1].metadata["cost"]
        self.assertEqual(driver_cost, {"buy": 9, "sell": 4})

    def test_simulate_buy_wasteful_applies_discard_delta_immediately_without_double_counting(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=20,
            hands_remaining=4,
            discards_remaining=4,
            modifiers={
                "voucher_cards": (
                    {"name": "Wasteful", "set": "VOUCHER", "cost": {"buy": 10, "sell": 5}},
                )
            },
        )

        next_state = simulate_buy(
            state,
            Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}),
        )
        after_shop = simulate_end_shop(next_state)

        self.assertEqual(next_state.money, 10)
        self.assertEqual(next_state.vouchers, ("Wasteful",))
        self.assertEqual(next_state.discards_remaining, 5)
        self.assertEqual(next_state.modifiers["round_discards_delta"], 1)
        self.assertEqual(next_state.modifiers["applied_round_discards_delta"], 1)
        self.assertEqual(after_shop.discards_remaining, 5)

    def test_simulate_buy_can_inject_overstock_shop_surface(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=20,
            shop=("Pluto",),
            modifiers={
                "shop_slots": 2,
                "shop_cards": (
                    {"name": "Pluto", "set": "PLANET", "cost": {"buy": 3}},
                ),
                "voucher_cards": (
                    {"name": "Overstock", "set": "VOUCHER", "cost": {"buy": 10, "sell": 5}},
                ),
            },
        )

        next_state = simulate_buy(
            state,
            Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}),
            next_shop_cards=(
                {"name": "Pluto", "set": "PLANET", "cost": {"buy": 3}},
                {"name": "Earth", "set": "PLANET", "cost": {"buy": 3}},
                {"name": "The Chariot", "set": "TAROT", "cost": {"buy": 3}},
            ),
        )

        self.assertEqual(next_state.money, 10)
        self.assertEqual(next_state.vouchers, ("Overstock",))
        self.assertEqual(next_state.modifiers["shop_slots"], 3)
        self.assertEqual(next_state.shop, ("Pluto", "Earth", "The Chariot"))
        self.assertEqual(tuple(item["name"] for item in next_state.modifiers["shop_cards"]), ("Pluto", "Earth", "The Chariot"))

        consumable_state = GameState(
            phase=GamePhase.SHOP,
            money=5,
            modifiers={"shop_cards": ({"label": "Mars", "set": "PLANET", "cost": {"buy": 3}},)},
        )

        after_consumable = simulate_buy(
            consumable_state,
            Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
        )

        self.assertEqual(after_consumable.money, 2)
        self.assertEqual(after_consumable.consumables, ("Mars",))

        full_consumable_state = GameState(
            phase=GamePhase.SHOP,
            money=5,
            consumables=("Jupiter", "Saturn"),
            modifiers={"shop_cards": ({"label": "Temperance", "set": "TAROT", "cost": {"buy": 3}},)},
        )

        with self.assertRaisesRegex(ValueError, "full consumable slots"):
            simulate_buy(
                full_consumable_state,
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            )

        card_state = GameState(
            phase=GamePhase.SHOP,
            money=5,
            deck_size=2,
            known_deck=(Card("A", "S"), Card("K", "H")),
            modifiers={"shop_cards": ({"rank": "Q", "suit": "D", "set": "DEFAULT", "cost": {"buy": 2}},)},
        )

        after_card = simulate_buy(
            card_state,
            Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
        )

        self.assertEqual(after_card.money, 3)
        self.assertEqual(after_card.deck_size, 3)
        self.assertEqual(tuple(card.short_name for card in after_card.known_deck), ("AS", "KH", "QD"))

    def test_simulate_buy_applies_passive_joker_modifiers(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=30,
            modifiers={
                "joker_slots": 6,
                "shop_cards": (
                    {"label": "Juggler", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Drunkard", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Credit Card", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "To the Moon", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Oops! All 6s", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Showman", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
        )

        next_state = state
        for _ in range(6):
            next_state = simulate_buy(
                next_state,
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            )

        self.assertEqual(next_state.modifiers["hand_size_delta"], 1)
        self.assertEqual(next_state.modifiers["round_discards_delta"], 1)
        self.assertEqual(next_state.modifiers["bankrupt_at"], -20)
        self.assertEqual(next_state.modifiers["interest_amount"], 2)
        self.assertEqual(next_state.modifiers["probability_multiplier"], 2.0)
        self.assertTrue(next_state.modifiers["allow_duplicate_jokers"])

    def test_multiple_real_oops_stack_and_unstack_probability_multiplier(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=30,
            modifiers={
                "shop_cards": (
                    {"label": "Showman", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Oops! All 6s", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Oops! All 6s", "set": "JOKER", "cost": {"buy": 4}},
                )
            },
        )

        next_state = state
        for _ in range(3):
            next_state = simulate_buy(
                next_state,
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            )

        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("Showman", "Oops! All 6s", "Oops! All 6s"))
        self.assertEqual(next_state.modifiers["probability_multiplier"], 4.0)

        next_state = simulate_sell(
            next_state,
            Action(ActionType.SELL, target_id="joker", amount=1, metadata={"kind": "joker", "index": 1}),
        )
        self.assertEqual(next_state.modifiers["probability_multiplier"], 2.0)

        next_state = simulate_sell(
            next_state,
            Action(ActionType.SELL, target_id="joker", amount=1, metadata={"kind": "joker", "index": 1}),
        )
        self.assertEqual(next_state.modifiers["probability_multiplier"], 1.0)

    def test_simulate_buy_initializes_visible_dynamic_joker_counters(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=20,
            modifiers={
                "shop_cards": (
                    {"label": "Ice Cream", "set": "JOKER", "cost": {"buy": 5, "sell": 2}},
                    {"label": "Popcorn", "set": "JOKER", "cost": {"buy": 5, "sell": 2}},
                    {"label": "Ramen", "set": "JOKER", "cost": {"buy": 6, "sell": 3}},
                )
            },
        )

        next_state = state
        for _ in range(3):
            next_state = simulate_buy(
                next_state,
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            )

        self.assertEqual(next_state.jokers[0].metadata["current_chips"], 100)
        self.assertEqual(next_state.jokers[1].metadata["current_mult"], 20)
        self.assertEqual(next_state.jokers[2].metadata["current_xmult"], 2.0)

    def test_simulate_buy_astronomer_makes_planets_free(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=5,
            jokers=(Joker("Astronomer"),),
            modifiers={"shop_cards": ({"label": "Mars", "set": "PLANET", "cost": {"buy": 3}},)},
        )

        next_state = simulate_buy(
            state,
            Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
        )

        self.assertEqual(next_state.money, 5)
        self.assertEqual(next_state.consumables, ("Mars",))

    def test_simulate_sell_removes_joker_adds_money_and_updates_campfire(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=2,
            jokers=(
                Joker("Campfire", metadata={"current_xmult": 1.0, "extra": 0.25}),
                Joker("Spare Joker", sell_value=3),
            ),
        )

        next_state = simulate_sell(state, Action(ActionType.SELL, target_id="joker", amount=1, metadata={"kind": "joker", "index": 1}))

        self.assertEqual(next_state.money, 5)
        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("Campfire",))
        self.assertEqual(next_state.jokers[0].metadata["current_xmult"], 1.25)

    def test_simulate_sell_removes_passive_joker_modifiers(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=2,
            jokers=(Joker("Credit Card", sell_value=3), Joker("To the Moon", sell_value=3)),
            modifiers={"bankrupt_at": -20, "interest_amount": 2},
        )

        after_credit_card = simulate_sell(
            state,
            Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
        )
        after_to_the_moon = simulate_sell(
            after_credit_card,
            Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
        )

        self.assertEqual(after_credit_card.modifiers["bankrupt_at"], 0)
        self.assertEqual(after_to_the_moon.modifiers["interest_amount"], 1)

    def test_simulate_sell_handles_diet_cola_and_luchador_triggers(self) -> None:
        diet_cola_state = GameState(
            phase=GamePhase.SHOP,
            money=2,
            jokers=(Joker("Diet Cola", sell_value=1),),
        )

        after_diet_cola = simulate_sell(
            diet_cola_state,
            Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
        )

        self.assertEqual(after_diet_cola.modifiers["tags"], ("Double Tag",))

        luchador_state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Hook",
            jokers=(Joker("Luchador", sell_value=1),),
            modifiers={"current_blind": {"type": "BOSS"}},
        )

        after_luchador = simulate_sell(
            luchador_state,
            Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}),
        )

        self.assertTrue(after_luchador.modifiers["boss_disabled"])

    def test_simulate_sell_can_sell_consumable_and_respect_gift_card_bonus(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            money=2,
            consumables=("Mars", "Aura"),
            modifiers={"consumable_sell_value_bonus": 4},
        )

        next_state = simulate_sell(
            state,
            Action(ActionType.SELL, target_id="consumable", amount=1, metadata={"kind": "consumable", "index": 1}),
        )

        self.assertEqual(next_state.money, 6)
        self.assertEqual(next_state.consumables, ("Mars",))
        self.assertEqual(next_state.modifiers["consumable_sell_value_bonus"], 2)

    def test_simulate_use_consumable_uses_stored_tarot_in_blind(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            money=2,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            jokers=(Joker("Fortune Teller", metadata={"current_mult": 0}),),
            consumables=("The Empress",),
        )

        next_state = simulate_use_consumable(
            state,
            Action(ActionType.USE_CONSUMABLE, card_indices=(0, 1), target_id="consumable", amount=0, metadata={"kind": "consumable", "index": 0}),
        )

        self.assertEqual(next_state.consumables, ())
        self.assertEqual(tuple(card.enhancement for card in next_state.hand), ("MULT", "MULT", None))
        self.assertEqual(next_state.jokers[0].metadata["current_mult"], 1)
        self.assertEqual(next_state.modifiers["tarot_cards_used"]["The Empress"], 1)

    def test_simulate_use_consumable_keeps_pack_open_while_using_stored_tarot(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("8S",),
            hand=(Card("A", "S"), Card("K", "H")),
            consumables=("Death",),
            modifiers={"pack_cards": ({"rank": "8", "suit": "S", "set": "Base"},)},
        )

        next_state = simulate_use_consumable(
            state,
            Action(ActionType.USE_CONSUMABLE, card_indices=(0, 1), target_id="consumable", amount=0, metadata={"kind": "consumable", "index": 0}),
        )

        self.assertEqual(next_state.phase, GamePhase.BOOSTER_OPENED)
        self.assertEqual(next_state.pack, ("8S",))
        self.assertEqual(tuple(card.short_name for card in next_state.hand), ("KH", "KH"))
        self.assertEqual(next_state.consumables, ())

    def test_simulate_reroll_spends_cost_replaces_shop_and_updates_flash_card(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            shop=("Joker",),
            jokers=(Joker("Flash Card", metadata={"current_mult": 2, "extra": 2}),),
            modifiers={
                "reroll_cost": 5,
                "reroll_cost_increase": 0,
                "shop_cards": ({"label": "Joker", "set": "JOKER", "cost": {"buy": 4}},),
            },
        )

        next_state = simulate_reroll(
            state,
            Action(ActionType.REROLL),
            new_shop_cards=(
                {"label": "Cavendish", "set": "JOKER", "cost": {"buy": 4}},
                {"label": "Saturn", "set": "PLANET", "cost": {"buy": 3}},
            ),
        )

        self.assertEqual(next_state.money, 5)
        self.assertEqual(next_state.shop, ("Cavendish", "Saturn"))
        self.assertEqual(next_state.modifiers["reroll_cost"], 6)
        self.assertEqual(next_state.modifiers["reroll_cost_increase"], 1)
        self.assertEqual(next_state.jokers[0].metadata["current_mult"], 4)

    def test_simulate_reroll_consumes_free_reroll_before_money(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=2,
            modifiers={"free_rerolls": 1, "reroll_cost": 5, "shop_cards": ({"label": "Joker"},)},
        )

        next_state = simulate_reroll(state, Action(ActionType.REROLL), new_shop_cards=({"label": "Mars", "set": "PLANET"},))

        self.assertEqual(next_state.money, 2)
        self.assertEqual(next_state.modifiers["free_rerolls"], 0)
        self.assertEqual(next_state.modifiers["reroll_cost"], 5)
        self.assertEqual(next_state.shop, ("Mars",))

    def test_simulate_open_pack_astronomer_and_hallucination_hooks(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=4,
            jokers=(Joker("Astronomer"), Joker("Hallucination")),
            modifiers={"booster_packs": ({"label": "Celestial Pack", "set": "BOOSTER", "cost": {"buy": 4}},)},
        )

        opened = simulate_open_pack(
            state,
            Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
            pack_contents=({"label": "Mars", "set": "PLANET"},),
            created_consumables=("The Fool",),
        )

        self.assertEqual(opened.money, 4)
        self.assertEqual(opened.consumables, ("The Fool",))
        self.assertEqual(opened.modifiers["created_tarot_count"], 1)

    def test_simulate_open_pack_and_choose_pack_card(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=9,
            modifiers={
                "booster_packs": (
                    {"label": "Arcana Pack", "set": "PACK", "cost": {"buy": 4}},
                    {"label": "Buffoon Pack", "set": "PACK", "cost": {"buy": 6}},
                )
            },
        )

        opened = simulate_open_pack(
            state,
            Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
            pack_contents=(
                {"label": "Green Joker", "set": "JOKER", "cost": {"sell": 2}},
                {"label": "Mars", "set": "PLANET"},
            ),
        )

        self.assertEqual(opened.phase, GamePhase.BOOSTER_OPENED)
        self.assertEqual(opened.money, 5)
        self.assertEqual(opened.pack, ("Green Joker", "Mars"))
        self.assertEqual(tuple(item["label"] for item in opened.modifiers["booster_packs"]), ("Buffoon Pack",))

        chosen = simulate_choose_pack_card(
            opened,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
        )

        self.assertEqual(chosen.phase, GamePhase.SHOP)
        self.assertEqual(chosen.pack, ())
        self.assertEqual(chosen.modifiers["pack_cards"], ())
        self.assertEqual(tuple(joker.name for joker in chosen.jokers), ("Green Joker",))
        self.assertEqual(chosen.jokers[0].sell_value, 2)

    def test_simulate_open_pack_can_draw_temporary_targeting_hand_and_return_it_on_skip(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            money=10,
            deck_size=10,
            modifiers={
                "booster_packs": (
                    {"label": "Spectral Pack", "set": "BOOSTER", "cost": {"buy": 4}},
                )
            },
        )

        opened = simulate_open_pack(
            state,
            Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0}),
            pack_contents=("Aura", "Hex"),
            drawn_cards=(Card("A", "S"), Card("K", "D")),
        )
        skipped = simulate_choose_pack_card(
            opened,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip"}),
        )

        self.assertEqual(opened.phase, GamePhase.BOOSTER_OPENED)
        self.assertEqual(opened.deck_size, 8)
        self.assertEqual(tuple(card.short_name for card in opened.hand), ("AS", "KD"))
        self.assertEqual(skipped.phase, GamePhase.SHOP)
        self.assertEqual(skipped.deck_size, 10)
        self.assertEqual(skipped.hand, ())

    def test_simulate_choose_pack_card_skip_returns_to_shop(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("Mars",),
            modifiers={"pack_cards": ({"label": "Mars", "set": "PLANET"},)},
        )

        next_state = simulate_choose_pack_card(
            state,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
        )

        self.assertEqual(next_state.phase, GamePhase.SHOP)
        self.assertEqual(next_state.pack, ())
        self.assertEqual(next_state.modifiers["pack_cards"], ())
        self.assertEqual(next_state.consumables, ())

    def test_simulate_choose_pack_card_skip_updates_red_card(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("Joker",),
            jokers=(Joker("Red Card", metadata={"current_mult": 0}),),
            modifiers={"pack_cards": ({"label": "Joker", "set": "JOKER"},)},
        )

        next_state = simulate_choose_pack_card(
            state,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}),
        )

        self.assertEqual(next_state.phase, GamePhase.SHOP)
        self.assertEqual(next_state.jokers[0].metadata["current_mult"], 3)

    def test_simulate_choose_pack_card_uses_planet_from_pack_immediately(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("Mars", "Saturn"),
            hand_levels={"Four of a Kind": 1},
            modifiers={
                "pack_cards": (
                    {"label": "Mars", "set": "PLANET"},
                    {"label": "Saturn", "set": "PLANET"},
                ),
                "hands": {"Four of a Kind": {"level": 1, "played": 0, "played_this_round": 0}},
            },
        )

        next_state = simulate_choose_pack_card(
            state,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
        )

        self.assertEqual(next_state.phase, GamePhase.SHOP)
        self.assertEqual(next_state.consumables, ())
        self.assertEqual(next_state.hand_levels["Four of a Kind"], 2)
        self.assertEqual(next_state.modifiers["hands"]["Four of a Kind"]["level"], 2)

    def test_simulate_choose_pack_card_uses_hermit_from_pack_immediately(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            money=20,
            pack=("The Hermit",),
            jokers=(Joker("Fortune Teller", metadata={"current_mult": 0}),),
            modifiers={"pack_cards": ({"label": "The Hermit", "set": "TAROT"},)},
        )

        next_state = simulate_choose_pack_card(
            state,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
        )

        self.assertEqual(next_state.phase, GamePhase.SHOP)
        self.assertEqual(next_state.money, 40)
        self.assertEqual(next_state.consumables, ())
        self.assertEqual(next_state.modifiers["tarot_cards_used"]["The Hermit"], 1)
        self.assertEqual(next_state.jokers[0].metadata["current_mult"], 1)

    def test_simulate_choose_pack_card_uses_targeted_tarot_from_pack_immediately(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("The Empress",),
            hand=(Card("A", "S"), Card("K", "H")),
            modifiers={"pack_cards": ({"label": "The Empress", "set": "TAROT"},)},
        )

        next_state = simulate_choose_pack_card(
            state,
            Action(
                ActionType.CHOOSE_PACK_CARD,
                card_indices=(0, 1),
                target_id="card",
                amount=0,
                metadata={"kind": "card", "index": 0},
            ),
            return_hand_to_deck=False,
        )

        self.assertEqual(next_state.phase, GamePhase.SHOP)
        self.assertEqual(tuple(card.enhancement for card in next_state.hand), ("MULT", "MULT"))
        self.assertEqual(next_state.modifiers["tarot_cards_used"]["The Empress"], 1)

    def test_simulate_choose_pack_card_wheel_of_fortune_applies_injected_edition(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("The Wheel of Fortune",),
            jokers=(Joker("Joker", sell_value=1), Joker("Fortune Teller", metadata={"current_mult": 0})),
            modifiers={"pack_cards": ({"label": "The Wheel of Fortune", "set": "TAROT"},)},
        )

        next_state = simulate_choose_pack_card(
            state,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            wheel_of_fortune_joker_index=0,
            wheel_of_fortune_edition="holo",
        )

        self.assertEqual(next_state.jokers[0].edition, "HOLOGRAPHIC")
        self.assertEqual(next_state.jokers[0].sell_value, 2)
        self.assertEqual(next_state.jokers[1].metadata["current_mult"], 1)
        self.assertEqual(next_state.modifiers["tarot_cards_used"]["The Wheel of Fortune"], 1)

    def test_simulate_choose_pack_card_wheel_of_fortune_miss_only_records_tarot(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("The Wheel of Fortune",),
            jokers=(Joker("Joker", sell_value=1), Joker("Fortune Teller", metadata={"current_mult": 0})),
            modifiers={"pack_cards": ({"label": "The Wheel of Fortune", "set": "TAROT"},)},
        )

        next_state = simulate_choose_pack_card(
            state,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
        )

        self.assertIsNone(next_state.jokers[0].edition)
        self.assertEqual(next_state.jokers[1].metadata["current_mult"], 1)
        self.assertEqual(next_state.modifiers["tarot_cards_used"]["The Wheel of Fortune"], 1)

    def test_simulate_choose_pack_card_rejects_normal_joker_when_slots_are_full(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            jokers=(Joker("One"),),
            pack=("Riff-raff",),
            modifiers={
                "joker_slots": 1,
                "pack_cards": ({"label": "Riff-raff", "set": "JOKER", "cost": {"sell": 3}},),
            },
        )

        with self.assertRaisesRegex(ValueError, "full joker slots"):
            simulate_choose_pack_card(
                state,
                Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            )

    def test_simulate_choose_pack_card_can_keep_mega_pack_open(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("Saturn", "Neptune", "Uranus"),
            modifiers={
                "pack_cards": (
                    {"label": "Saturn", "set": "PLANET"},
                    {"label": "Neptune", "set": "PLANET"},
                    {"label": "Uranus", "set": "PLANET"},
                )
            },
        )

        next_state = simulate_choose_pack_card(
            state,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
            remaining_pack_cards=(
                {"label": "Neptune", "set": "PLANET"},
                {"label": "Uranus", "set": "PLANET"},
            ),
        )

        self.assertEqual(next_state.phase, GamePhase.BOOSTER_OPENED)
        self.assertEqual(next_state.pack, ("Neptune", "Uranus"))
        self.assertEqual(next_state.hand_levels["Straight"], 2)

    def test_simulate_choose_pack_card_adds_standard_pack_card_to_deck(self) -> None:
        state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            deck_size=52,
            known_deck=(Card("A", "S"),),
            pack=("Wild Card",),
            modifiers={"pack_cards": ({"name": "Wild Card", "key": "C_A", "set": "ENHANCED"},)},
        )

        next_state = simulate_choose_pack_card(
            state,
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
        )

        self.assertEqual(next_state.deck_size, 53)
        self.assertEqual(tuple(card.short_name for card in next_state.known_deck), ("AS", "AC"))
        self.assertEqual(next_state.known_deck[-1].enhancement, "WILD")

    def test_simulate_end_shop_moves_to_blind_select_and_clears_shop_surface(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            shop=("Joker",),
            pack=("The Fool",),
            modifiers={
                "shop_cards": ({"name": "Joker"},),
                "voucher_cards": ({"name": "Blank"},),
                "booster_packs": ({"name": "Arcana Pack"},),
                "pack_cards": ({"name": "The Fool"},),
            },
        )

        next_state = simulate_end_shop(state)

        self.assertEqual(next_state.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(next_state.shop, ())
        self.assertEqual(next_state.pack, ())
        self.assertEqual(next_state.modifiers["shop_cards"], ())
        self.assertEqual(next_state.modifiers["voucher_cards"], ())
        self.assertEqual(next_state.modifiers["booster_packs"], ())
        self.assertEqual(next_state.modifiers["pack_cards"], ())

    def test_simulate_end_shop_applies_new_shop_passive_round_count_delta_once(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            hands_remaining=4,
            discards_remaining=4,
            modifiers={
                "round_hands_delta": -1,
                "round_discards_delta": 1,
                "applied_round_hands_delta": 0,
                "applied_round_discards_delta": 0,
            },
        )

        next_state = simulate_end_shop(state)
        blind_state = simulate_select_blind(next_state)

        self.assertEqual(next_state.hands_remaining, 3)
        self.assertEqual(next_state.discards_remaining, 5)
        self.assertEqual(next_state.modifiers["applied_round_hands_delta"], -1)
        self.assertEqual(next_state.modifiers["applied_round_discards_delta"], 1)
        self.assertEqual(blind_state.hands_remaining, 3)
        self.assertEqual(blind_state.discards_remaining, 5)

    def test_simulate_end_shop_accepts_injected_perkeo_copy(self) -> None:
        state = GameState(
            phase=GamePhase.SHOP,
            consumables=("Mars",),
            jokers=(Joker("Perkeo"),),
        )

        next_state = simulate_end_shop(state, created_consumables=("Mars",))

        self.assertEqual(next_state.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(next_state.consumables, ("Mars", "Mars"))
        self.assertEqual(next_state.modifiers["created_negative_consumable_count"], 1)

    def test_simulate_select_blind_draws_and_applies_injected_card_creation(self) -> None:
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="Small Blind",
            required_score=300,
            current_score=120,
            hands_remaining=4,
            discards_remaining=3,
            deck_size=4,
            known_deck=(Card("A", "S"), Card("K", "H"), Card("Q", "D"), Card("J", "C")),
            jokers=(Joker("Certificate"), Joker("Marble Joker")),
            modifiers={
                "played_pile": (Card("2", "C"),),
                "discard_pile": (Card("3", "C"),),
                "round_hands_played": 2,
                "round_discards_used": 1,
                "hands": {"Pair": {"played": 4, "played_this_round": 1, "order": 1}},
            },
        )

        next_state = simulate_select_blind(
            state,
            drawn_cards=(Card("A", "S"), Card("K", "H")),
            created_hand_cards=(Card("7", "D", seal="Blue"),),
            created_deck_cards=(Card("9", "C"),),
        )

        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(next_state.current_score, 0)
        self.assertEqual(tuple(card.short_name for card in next_state.hand), ("AS", "KH", "7D"))
        self.assertEqual(next_state.hand[2].seal, "Blue")
        self.assertEqual(tuple(card.short_name for card in next_state.known_deck), ("9C", "QD", "JC"))
        self.assertEqual(next_state.known_deck[0].enhancement, "STONE")
        self.assertEqual(next_state.deck_size, 3)
        self.assertNotIn("played_pile", next_state.modifiers)
        self.assertNotIn("discard_pile", next_state.modifiers)
        self.assertEqual(next_state.modifiers["round_hands_played"], 0)
        self.assertEqual(next_state.modifiers["round_discards_used"], 0)
        self.assertEqual(next_state.modifiers["hands"]["Pair"]["played_this_round"], 0)

    def test_simulate_select_blind_accepts_cartomancer_and_riff_raff_injections(self) -> None:
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="Small Blind",
            hands_remaining=4,
            discards_remaining=3,
            jokers=(Joker("Cartomancer"), Joker("Riff-raff")),
            modifiers={"consumable_slot_limit": 3},
        )

        next_state = simulate_select_blind(
            state,
            created_consumables=("The Fool",),
            created_jokers=(
                {"label": "Joker", "set": "JOKER", "cost": {"buy": 4, "sell": 2}},
                {"label": "Greedy Joker", "set": "JOKER", "cost": {"buy": 5, "sell": 2}},
            ),
        )

        self.assertEqual(next_state.consumables, ("The Fool",))
        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("Cartomancer", "Riff-raff", "Joker", "Greedy Joker"))
        self.assertEqual(next_state.modifiers["created_tarot_count"], 1)
        self.assertEqual(next_state.modifiers["created_common_joker_count"], 2)

    def test_simulate_select_blind_applies_round_hand_and_discard_deltas(self) -> None:
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="Small Blind",
            hands_remaining=4,
            discards_remaining=3,
            modifiers={"round_hands_delta": -1, "round_discards_delta": 3},
        )

        next_state = simulate_select_blind(state)

        self.assertEqual(next_state.hands_remaining, 3)
        self.assertEqual(next_state.discards_remaining, 6)

    def test_simulate_select_blind_applies_burglar_and_chicot(self) -> None:
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The Flint",
            hands_remaining=4,
            discards_remaining=3,
            jokers=(Joker("Burglar"), Joker("Chicot")),
            modifiers={"current_blind": {"type": "BOSS"}},
        )

        next_state = simulate_select_blind(state)

        self.assertEqual(next_state.hands_remaining, 7)
        self.assertEqual(next_state.discards_remaining, 0)
        self.assertTrue(next_state.modifiers["boss_disabled"])

    def test_simulate_select_blind_applies_round_target_jokers(self) -> None:
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="Small Blind",
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Ancient Joker"),
                Joker("The Idol"),
                Joker("Castle"),
                Joker("Mail-In Rebate"),
            ),
        )

        next_state = simulate_select_blind(
            state,
            next_ancient_suit="Heart",
            next_idol_card=Card("Q", "D"),
            next_castle_suit="Club",
            next_mail_rank="10",
        )
        by_name = {joker.name: joker.metadata for joker in next_state.jokers}

        self.assertEqual(by_name["Ancient Joker"]["target_suit"], "H")
        self.assertEqual(by_name["The Idol"]["target_rank"], "Q")
        self.assertEqual(by_name["The Idol"]["target_suit"], "D")
        self.assertEqual(by_name["Castle"]["target_suit"], "C")
        self.assertEqual(by_name["Mail-In Rebate"]["target_rank"], "10")

    def test_simulate_select_blind_applies_dagger_and_madness(self) -> None:
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="Small Blind",
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Ceremonial Dagger", sell_value=3, metadata={"current_mult": 0}),
                Joker("Drunkard", sell_value=3),
                Joker("Madness", sell_value=4, metadata={"current_xmult": 1.0}),
                Joker("Joker", sell_value=2),
            ),
            modifiers={"round_discards_delta": 1},
        )

        next_state = simulate_select_blind(
            state,
            ceremonial_destroyed_joker_index=1,
            madness_destroyed_joker_index=2,
        )

        self.assertEqual(tuple(joker.name for joker in next_state.jokers), ("Ceremonial Dagger", "Madness"))
        self.assertEqual(next_state.jokers[0].metadata["current_mult"], 6)
        self.assertEqual(next_state.jokers[1].metadata["current_xmult"], 1.5)
        self.assertEqual(next_state.modifiers["round_discards_delta"], 0)

    def test_simulate_select_blind_refreshes_deck_tally_jokers(self) -> None:
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="Small Blind",
            deck_size=3,
            known_deck=(
                Card("A", "S", enhancement="STONE"),
                Card("K", "H", enhancement="STEEL"),
                Card("Q", "D", enhancement="BONUS"),
            ),
            hands_remaining=4,
            discards_remaining=4,
            jokers=(
                Joker("Joker Stencil"),
                Joker("Stone Joker"),
                Joker("Steel Joker"),
                Joker("Driver's License"),
                Joker("Erosion"),
            ),
            modifiers={"starting_deck_size": 4},
        )

        next_state = simulate_select_blind(state)
        by_name = {joker.name: joker.metadata for joker in next_state.jokers}

        self.assertEqual(by_name["Joker Stencil"]["current_xmult"], 1.0)
        self.assertEqual(by_name["Stone Joker"]["stone_tally"], 1)
        self.assertEqual(by_name["Stone Joker"]["current_chips"], 25)
        self.assertEqual(by_name["Steel Joker"]["steel_tally"], 1)
        self.assertEqual(by_name["Steel Joker"]["current_xmult"], 1.2)
        self.assertEqual(by_name["Driver's License"]["driver_tally"], 3)
        self.assertEqual(by_name["Erosion"]["current_mult"], 4)

    def test_dynamic_erosion_counts_hidden_draw_deck_not_just_visible_hand(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=999999,
            current_score=0,
            hands_remaining=4,
            deck_size=44,
            hand=(
                Card("A", "H"),
                Card("K", "H"),
                Card("Q", "H"),
                Card("J", "H"),
                Card("10", "H"),
                Card("2", "S"),
                Card("3", "C"),
                Card("4", "S"),
            ),
            jokers=(Joker("Erosion"),),
            modifiers={"starting_deck_size": 52},
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
        )

        self.assertEqual(next_state.current_score, 1208)
        self.assertEqual(next_state.jokers[0].metadata["current_mult"], 0)

    def test_simulate_select_blind_applies_boss_starting_hand_and_discard_rules(self) -> None:
        water_state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The Water",
            hands_remaining=4,
            discards_remaining=4,
        )
        needle_state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The Needle",
            hands_remaining=4,
            discards_remaining=4,
        )

        after_water = simulate_select_blind(water_state)
        after_needle = simulate_select_blind(needle_state)

        self.assertEqual(after_water.hands_remaining, 4)
        self.assertEqual(after_water.discards_remaining, 0)
        self.assertEqual(after_needle.hands_remaining, 1)
        self.assertEqual(after_needle.discards_remaining, 4)

    def test_simulate_select_blind_requires_injected_random_created_cards(self) -> None:
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            deck_size=1,
            known_deck=(Card("A", "S"),),
            jokers=(Joker("Certificate"),),
        )

        with self.assertRaisesRegex(ValueError, "Certificate needs 1 injected"):
            simulate_select_blind(state, drawn_cards=(Card("A", "S"),))

    def test_simulate_play_dna_adds_copy_to_hand_on_first_single_card_hand(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            current_score=0,
            hands_remaining=4,
            deck_size=1,
            hand=(Card("A", "S", enhancement="BONUS"), Card("K", "H")),
            known_deck=(Card("Q", "S"),),
            jokers=(Joker("DNA"),),
            modifiers={"round_hands_played": 0},
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            drawn_cards=(Card("Q", "S"),),
        )

        self.assertEqual(tuple(card.short_name for card in next_state.hand), ("AS", "KH", "QS"))
        self.assertEqual(next_state.hand[0].enhancement, "BONUS")
        self.assertEqual(next_state.known_deck, ())
        self.assertEqual(next_state.deck_size, 0)

    def test_simulate_play_sixth_sense_destroys_single_six_and_accepts_injected_spectral(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            current_score=0,
            hands_remaining=4,
            deck_size=1,
            hand=(Card("6", "S"), Card("K", "H")),
            known_deck=(Card("Q", "S"),),
            jokers=(Joker("Sixth Sense"),),
            modifiers={"round_hands_played": 0},
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0,)),
            drawn_cards=(Card("Q", "S"),),
            created_consumables=("Aura",),
        )

        self.assertEqual(tuple(card.short_name for card in next_state.hand), ("KH", "QS"))
        self.assertEqual(next_state.modifiers["played_pile"], ())
        self.assertEqual(tuple(card.short_name for card in next_state.modifiers["destroyed_cards"]), ("6S",))
        self.assertEqual(next_state.modifiers["created_spectral_count"], 1)
        self.assertEqual(next_state.consumables, ("Aura",))

    def test_copy_jokers_do_not_copy_incompatible_sixth_sense(self) -> None:
        blueprint_state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            hands_remaining=4,
            hand=(Card("6", "S"), Card("K", "H")),
            jokers=(Joker("Blueprint"), Joker("Sixth Sense")),
            modifiers={"round_hands_played": 0, "consumable_slot_limit": 2},
        )
        brainstorm_state = replace(blueprint_state, jokers=(Joker("Sixth Sense"), Joker("Brainstorm")))

        for state in (blueprint_state, brainstorm_state):
            next_state = simulate_play(
                state,
                Action(ActionType.PLAY_HAND, card_indices=(0,)),
                created_consumables=("Aura",),
            )
            self.assertEqual(next_state.modifiers["created_spectral_count"], 1)
            self.assertEqual(next_state.consumables, ("Aura",))

    def test_simulate_play_accepts_injected_tarots_from_hand_triggers(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            current_score=0,
            money=4,
            hands_remaining=4,
            hand=(Card("A", "S"), Card("K", "S"), Card("Q", "S"), Card("J", "S"), Card("10", "S")),
            jokers=(Joker("Seance"), Joker("Superposition"), Joker("Vagabond")),
            modifiers={"consumable_slot_limit": 5},
        )

        next_state = simulate_play(
            state,
            Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
            created_consumables=("Death", "The Fool", "Strength"),
        )

        self.assertEqual(next_state.consumables, ("Death", "The Fool", "Strength"))
        self.assertEqual(next_state.modifiers["created_spectral_count"], 1)
        self.assertEqual(next_state.modifiers["created_tarot_count"], 2)
        self.assertEqual(next_state.modifiers["created_consumable_count"], 3)

    def test_simulate_play_rejects_too_many_injected_created_consumables(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            money=4,
            hands_remaining=4,
            hand=(Card("A", "S"), Card("K", "D"), Card("Q", "C"), Card("J", "H"), Card("10", "S")),
            jokers=(Joker("Superposition"),),
        )

        with self.assertRaisesRegex(ValueError, "play-created consumables"):
            simulate_play(
                state,
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)),
                created_consumables=("Death", "The Fool"),
            )

    def test_state_from_mapping_preserves_round_discard_counter(self) -> None:
        state = GameState.from_mapping(
            {
                "state": "SELECTING_HAND",
                "round": {"discards_left": 2, "discards_used": 1, "hands_left": 4, "hands_played": 0},
                "hand": {"cards": []},
            }
        )

        self.assertEqual(state.discards_remaining, 2)
        self.assertEqual(state.modifiers["round_discards_used"], 1)
        self.assertEqual(state.modifiers["discards_used"], 1)


if __name__ == "__main__":
    unittest.main()
