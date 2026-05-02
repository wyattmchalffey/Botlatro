from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.state import Card, Joker
from balatro_ai.rules.hand_evaluator import (
    HandType,
    best_play_from_hand,
    debuffed_suits_for_blind,
    evaluate_played_cards,
)


def cards(names: list[str]) -> tuple[Card, ...]:
    return tuple(Card.from_mapping(name) for name in names)


def joker_with_effect(name: str, effect: str) -> Joker:
    return Joker(name, metadata={"value": {"effect": effect}})


def uncommon_joker(name: str) -> Joker:
    return Joker(name, metadata={"rarity": "Uncommon"})


class HandEvaluatorTests(unittest.TestCase):
    def test_identifies_core_hand_types(self) -> None:
        examples = [
            (["AS", "KD", "7H", "4C", "2D"], HandType.HIGH_CARD),
            (["AS", "AD", "7H", "4C", "2D"], HandType.PAIR),
            (["AS", "AD", "7H", "7C", "2D"], HandType.TWO_PAIR),
            (["AS", "AD", "AH", "4C", "2D"], HandType.THREE_OF_A_KIND),
            (["AS", "KD", "QH", "JC", "10D"], HandType.STRAIGHT),
            (["AS", "KS", "7S", "4S", "2S"], HandType.FLUSH),
            (["AS", "AD", "AH", "2C", "2D"], HandType.FULL_HOUSE),
            (["AS", "AD", "AH", "AC", "2D"], HandType.FOUR_OF_A_KIND),
            (["AS", "KS", "QS", "JS", "10S"], HandType.STRAIGHT_FLUSH),
            (["AS", "AS", "AS", "AS", "AS"], HandType.FLUSH_FIVE),
        ]

        for hand, expected_type in examples:
            with self.subTest(hand=hand):
                self.assertEqual(evaluate_played_cards(cards(hand)).hand_type, expected_type)

    def test_score_includes_base_level_and_scored_cards(self) -> None:
        evaluation = evaluate_played_cards(cards(["AS", "AD"]), {"Pair": 2})

        self.assertEqual(evaluation.hand_type, HandType.PAIR)
        self.assertEqual(evaluation.card_chips, 22)
        self.assertEqual(evaluation.chips, 47)
        self.assertEqual(evaluation.mult, 3)
        self.assertEqual(evaluation.score, 141)

    def test_best_play_from_hand_finds_strongest_immediate_score(self) -> None:
        evaluation = best_play_from_hand(cards(["AS", "AD", "AH", "7C", "2D"]))

        self.assertEqual(evaluation.hand_type, HandType.THREE_OF_A_KIND)

    def test_suit_debuff_zeroes_card_chips_but_keeps_hand_type(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS", "AD", "QS", "QC"]),
            debuffed_suits=debuffed_suits_for_blind("The Club"),
        )

        self.assertEqual(evaluation.hand_type, HandType.TWO_PAIR)
        self.assertEqual(evaluation.card_chips, 32)
        self.assertEqual(evaluation.score, 104)

    def test_simple_joker_effects_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["TC", "9S", "8S", "7S", "6D"]),
            jokers=(Joker("Wrathful Joker"), Joker("Stuntman")),
        )

        self.assertEqual(evaluation.hand_type, HandType.STRAIGHT)
        self.assertEqual(evaluation.chips, 320)
        self.assertEqual(evaluation.mult, 13)
        self.assertEqual(evaluation.score, 4160)

    def test_basic_card_enhancements_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="A", suit="S", enhancement="m_bonus"),
                Card(rank="A", suit="D", enhancement="m_mult"),
            )
        )

        self.assertEqual(evaluation.card_chips, 52)
        self.assertEqual(evaluation.mult, 6)
        self.assertEqual(evaluation.score, 372)

    def test_wild_card_can_complete_flush(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="K", suit="D", enhancement="WILD"),
                Card(rank="Q", suit="S"),
                Card(rank="9", suit="S"),
                Card(rank="7", suit="S"),
                Card(rank="4", suit="S"),
            )
        )

        self.assertEqual(evaluation.hand_type, HandType.FLUSH)

    def test_wild_card_counts_for_suit_debuffs(self) -> None:
        evaluation = evaluate_played_cards(
            (Card(rank="K", suit="D", enhancement="WILD"),),
            debuffed_suits=debuffed_suits_for_blind("The Goad"),
        )

        self.assertEqual(evaluation.card_chips, 0)
        self.assertEqual(evaluation.score, 5)

    def test_debuffed_wild_card_does_not_complete_flush(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="K", suit="H", enhancement="WILD", debuffed=True),
                Card(rank="Q", suit="S"),
                Card(rank="9", suit="S"),
                Card(rank="7", suit="S"),
                Card(rank="4", suit="S"),
            )
        )

        self.assertEqual(evaluation.hand_type, HandType.HIGH_CARD)

    def test_smeared_joker_extends_suit_based_card_effects(self) -> None:
        wrathful = evaluate_played_cards(
            cards(["AC"]),
            jokers=(Joker("Smeared Joker"), Joker("Wrathful Joker")),
        )
        greedy = evaluate_played_cards(
            cards(["JH"]),
            jokers=(Joker("Smeared Joker"), Joker("Greedy Joker")),
        )
        onyx = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Smeared Joker"), Joker("Onyx Agate")),
        )

        self.assertEqual(wrathful.score, 64)
        self.assertEqual(greedy.score, 60)
        self.assertEqual(onyx.score, 128)

    def test_smeared_joker_extends_seeing_double_like_source(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Smeared Joker"), Joker("Seeing Double")),
        )

        self.assertEqual(evaluation.effect_xmult, 2)
        self.assertEqual(evaluation.score, 32)

    def test_wild_cards_fill_missing_flower_pot_suits_like_source(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="A", suit="H"),
                Card(rank="K", suit="D"),
                Card(rank="Q", suit="S"),
                Card(rank="2", suit="H", enhancement="WILD"),
            ),
            jokers=(Joker("Splash"), Joker("Flower Pot")),
        )

        self.assertEqual(evaluation.effect_xmult, 3)
        self.assertEqual(evaluation.score, 114)

    def test_blackboard_counts_held_wild_cards_as_black_like_source(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Blackboard"),),
            held_cards=(Card(rank="7", suit="H", enhancement="WILD"),),
        )

        self.assertEqual(evaluation.effect_xmult, 3)
        self.assertEqual(evaluation.score, 48)

    def test_stone_card_scores_exactly_50_chips(self) -> None:
        evaluation = evaluate_played_cards(
            (Card(rank="A", suit="H", enhancement="STONE"),),
            debuffed_suits=debuffed_suits_for_blind("The Head"),
        )

        self.assertEqual(evaluation.card_chips, 50)
        self.assertEqual(evaluation.chips, 55)
        self.assertEqual(evaluation.score, 55)

    def test_stone_card_uses_displayed_chip_total(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card.from_mapping(
                    {
                        "label": "Stone Card",
                        "modifier": {"enhancement": "STONE"},
                        "value": {"rank": "3", "suit": "H", "effect": "+55 chips no rank or suit"},
                    }
                ),
            )
        )

        self.assertEqual(evaluation.card_chips, 55)
        self.assertEqual(evaluation.score, 60)

    def test_stone_card_has_no_rank_or_suit_for_scoring_effects(self) -> None:
        evaluation = evaluate_played_cards(
            (Card(rank="K", suit="S", enhancement="STONE"),),
            jokers=(Joker("Scholar"), Joker("Scary Face"), Joker("Photograph"), Joker("Wrathful Joker")),
        )

        self.assertEqual(evaluation.card_chips, 50)
        self.assertEqual(evaluation.effect_chips, 0)
        self.assertEqual(evaluation.effect_mult, 0)
        self.assertEqual(evaluation.score, 55)

    def test_stone_card_scores_alongside_made_hand_without_changing_hand_type(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="A", suit="S"),
                Card(rank="A", suit="D"),
                Card(rank="A", suit="H", enhancement="STONE"),
            )
        )

        self.assertEqual(evaluation.hand_type, HandType.PAIR)
        self.assertEqual(evaluation.scoring_indices, (0, 1, 2))
        self.assertEqual(evaluation.card_chips, 72)
        self.assertEqual(evaluation.score, 164)

    def test_midas_mask_removes_face_card_enhancement_before_scoring(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="Q", suit="H", enhancement="BONUS"),
                Card(rank="Q", suit="D"),
            ),
            jokers=(Joker("Midas Mask"),),
        )

        self.assertEqual(evaluation.card_chips, 20)
        self.assertEqual(evaluation.score, 60)

    def test_vampire_strips_scoring_enhancements_before_same_hand_xmult(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="A", suit="S", enhancement="BONUS"),
                Card(rank="A", suit="D", enhancement="MULT"),
            ),
            jokers=(Joker("Vampire", metadata={"current_xmult": 1.5}),),
        )

        self.assertEqual(evaluation.hand_type, HandType.PAIR)
        self.assertEqual(evaluation.card_chips, 22)
        self.assertEqual(evaluation.mult, 2)
        self.assertAlmostEqual(evaluation.effect_xmult, 1.7)
        self.assertEqual(evaluation.score, 108)

    def test_permanent_card_chip_metadata_is_applied(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="A", suit="S", metadata={"modifier": {"perma_bonus": 10}}),
                Card(rank="A", suit="D", metadata={"bonus_chips": 5}),
            )
        )

        self.assertEqual(evaluation.card_chips, 37)
        self.assertEqual(evaluation.score, 94)

    def test_displayed_card_chip_total_supplies_missing_permanent_chips(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card.from_mapping(
                    {
                        "label": "Base Card",
                        "value": {"rank": "7", "suit": "H", "effect": "+47 chips"},
                    }
                ),
            )
        )

        self.assertEqual(evaluation.card_chips, 47)
        self.assertEqual(evaluation.score, 52)

    def test_displayed_extra_chip_text_supplies_missing_permanent_chips(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card.from_mapping(
                    {
                        "label": "Base Card",
                        "value": {"rank": "K", "suit": "S", "effect": "+10 chips +15 extra chips"},
                    }
                ),
            )
        )

        self.assertEqual(evaluation.card_chips, 25)
        self.assertEqual(evaluation.score, 30)

    def test_bonus_card_extra_chip_text_does_not_double_count_bonus_enhancement(self) -> None:
        plain_bonus = evaluate_played_cards(
            (
                Card.from_mapping(
                    {
                        "label": "Bonus Card",
                        "value": {"rank": "K", "suit": "S", "effect": "+10 chips +30 extra chips"},
                        "set": "ENHANCED",
                    }
                ),
            )
        )
        hiked_bonus = evaluate_played_cards(
            (
                Card.from_mapping(
                    {
                        "label": "Bonus Card",
                        "value": {"rank": "8", "suit": "S", "effect": "+8 chips +40 extra chips"},
                        "set": "ENHANCED",
                    }
                ),
            )
        )

        self.assertEqual(plain_bonus.card_chips, 40)
        self.assertEqual(hiked_bonus.card_chips, 48)

    def test_glass_and_joker_edition_xmult_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="A", suit="S", enhancement="m_glass"),
                Card(rank="A", suit="D"),
            ),
            jokers=(Joker("Joker", edition="POLY"),),
        )

        self.assertEqual(evaluation.score, 384)

    def test_holographic_short_edition_is_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["QS", "QC", "QD", "8S", "8H"]),
            jokers=(Joker("Wrathful Joker"), Joker("Stuntman", edition="HOLO")),
        )

        self.assertEqual(evaluation.score, 6720)

    def test_fibonacci_and_family_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["8S", "8H", "8C", "8D"]),
            jokers=(Joker("Fibonacci"), Joker("The Family")),
        )

        self.assertEqual(evaluation.score, 14352)

    def test_mystic_summit_and_wily_joker_stack(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["5S", "5H", "5C", "5D"]),
            jokers=(Joker("Mystic Summit"), Joker("Wily Joker")),
            discards_remaining=0,
        )

        self.assertEqual(evaluation.score, 3960)

    def test_rank_and_suit_card_jokers_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["TC", "4C", "AC", "9C", "8C"]),
            jokers=(Joker("Walkie Talkie"), Joker("Onyx Agate"), Joker("Scholar")),
        )

        self.assertEqual(evaluation.hand_type, HandType.FLUSH)
        self.assertEqual(evaluation.chips, 117)
        self.assertEqual(evaluation.mult, 51)
        self.assertEqual(evaluation.score, 5967)

    def test_xmult_family_jokers_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS", "KS", "QS", "JS", "TS"]),
            jokers=(Joker("The Order"), Joker("The Tribe")),
        )

        self.assertEqual(evaluation.hand_type, HandType.STRAIGHT_FLUSH)
        self.assertEqual(evaluation.effect_xmult, 6)
        self.assertEqual(evaluation.score, 7248)

    def test_abstract_joker_counts_owned_jokers(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Abstract Joker"), Joker("Gros Michel")),
        )

        self.assertEqual(evaluation.mult, 22)
        self.assertEqual(evaluation.score, 352)

    def test_swashbuckler_and_scary_face_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS", "KD"]),
            jokers=(
                Joker("Swashbuckler", sell_value=2),
                Joker("Scary Face", sell_value=3),
                Joker("Joker", sell_value=4),
            ),
        )

        self.assertEqual(evaluation.chips, 90)
        self.assertEqual(evaluation.mult, 13)
        self.assertEqual(evaluation.score, 1170)

    def test_the_flint_halves_base_chips_and_mult(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["QS", "JC", "TD", "9S", "8D"]),
            blind_name="The Flint",
        )

        self.assertEqual(evaluation.base_chips, 15)
        self.assertEqual(evaluation.base_mult, 2)
        self.assertEqual(evaluation.score, 124)

    def test_the_flint_rounds_halved_base_values_up(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            blind_name="The Flint",
        )

        self.assertEqual(evaluation.base_chips, 3)
        self.assertEqual(evaluation.base_mult, 1)
        self.assertEqual(evaluation.score, 14)

    def test_the_flint_halves_leveled_hand_values(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["QC", "QD", "JS", "JD", "5C"]),
            {"Two Pair": 4},
            blind_name="The Flint",
        )

        self.assertEqual(evaluation.base_chips, 10)
        self.assertEqual(evaluation.level_chips, 30)
        self.assertEqual(evaluation.base_mult, 1)
        self.assertEqual(evaluation.level_mult, 2)
        self.assertEqual(evaluation.score, 240)

    def test_the_psychic_zeroes_non_five_card_plays(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS", "KC", "8S", "8H"]),
            blind_name="The Psychic",
            jokers=(Joker("Shoot the Moon"), Joker("Abstract Joker")),
        )

        self.assertEqual(evaluation.score, 0)

    def test_the_eye_zeroes_repeated_hand_type(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS", "KC"]),
            blind_name="The Eye",
            played_hand_types_this_round=(HandType.PAIR,),
        )

        self.assertEqual(evaluation.hand_type, HandType.PAIR)
        self.assertEqual(evaluation.score, 0)

    def test_the_mouth_zeroes_different_hand_type_after_first_play(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS", "KS", "QS", "JS", "TS"]),
            blind_name="The Mouth",
            played_hand_types_this_round=(HandType.PAIR,),
        )

        self.assertEqual(evaluation.hand_type, HandType.STRAIGHT_FLUSH)
        self.assertEqual(evaluation.score, 0)

    def test_card_sharp_applies_xmult_to_repeated_hand_type(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS", "KC"]),
            jokers=(Joker("Card Sharp"),),
            played_hand_types_this_round=(HandType.PAIR,),
        )

        self.assertEqual(evaluation.hand_type, HandType.PAIR)
        self.assertEqual(evaluation.effect_xmult, 3)
        self.assertEqual(evaluation.score, 180)

    def test_best_play_respects_the_mouth_existing_hand_type(self) -> None:
        evaluation = best_play_from_hand(
            cards(["KS", "KC", "AS", "QS", "JS", "TS", "9S"]),
            blind_name="The Mouth",
            played_hand_types_this_round=(HandType.PAIR,),
        )

        self.assertEqual(evaluation.hand_type, HandType.PAIR)
        self.assertGreater(evaluation.score, 0)

    def test_arrowhead_is_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["QD", "JS", "TS", "9S", "8H"]),
            jokers=(Joker("Arrowhead"),),
        )

        self.assertEqual(evaluation.hand_type, HandType.STRAIGHT)
        self.assertEqual(evaluation.chips, 227)
        self.assertEqual(evaluation.mult, 4)
        self.assertEqual(evaluation.score, 908)

    def test_even_steven_and_half_joker_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["QS", "QH", "QD"]),
            jokers=(Joker("Even Steven"), Joker("Half Joker")),
        )

        self.assertEqual(evaluation.hand_type, HandType.THREE_OF_A_KIND)
        self.assertEqual(evaluation.mult, 23)
        self.assertEqual(evaluation.score, 1380)

    def test_held_card_effects_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Shoot the Moon"), Joker("Raised Fist"), Joker("Baron")),
            held_cards=(
                Card(rank="Q", suit="S"),
                Card(rank="K", suit="C"),
                Card(rank="2", suit="D", enhancement="STEEL"),
            ),
        )

        self.assertEqual(evaluation.mult, 18)
        self.assertEqual(evaluation.effect_xmult, 2.25)
        self.assertEqual(evaluation.score, 568)

    def test_blackboard_and_blue_joker_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS", "AD"]),
            jokers=(Joker("Blackboard"), Joker("Blue Joker")),
            held_cards=cards(["2S", "3C"]),
            deck_size=40,
        )

        self.assertEqual(evaluation.chips, 112)
        self.assertEqual(evaluation.effect_xmult, 3)
        self.assertEqual(evaluation.score, 672)

    def test_card_debuff_state_zeroes_scored_card_effects(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="A", suit="S", enhancement="BONUS", debuffed=True),
                Card(rank="A", suit="D"),
            ),
            jokers=(Joker("Scholar"),),
        )

        self.assertEqual(evaluation.card_chips, 11)
        self.assertEqual(evaluation.chips, 41)
        self.assertEqual(evaluation.mult, 6)
        self.assertEqual(evaluation.score, 246)

    def test_dynamic_current_plus_jokers_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["2H", "2C", "2D"]),
            jokers=(
                Joker("Zany Joker"),
                joker_with_effect(
                    "Wee Joker",
                    "This Joker gains +8 Chips when each played 2 is scored (Currently +0 Chips)",
                ),
            ),
        )

        self.assertEqual(evaluation.chips, 60)
        self.assertEqual(evaluation.mult, 15)
        self.assertEqual(evaluation.score, 900)

    def test_green_joker_uses_current_value(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS", "AD"]),
            jokers=(
                joker_with_effect(
                    "Green Joker",
                    "+1 Mult per hand played -1 Mult per discard (Currently +3 Mult)",
                ),
            ),
        )

        self.assertEqual(evaluation.mult, 6)
        self.assertEqual(evaluation.score, 192)

    def test_ride_the_bus_and_runner_use_current_values(self) -> None:
        bus = evaluate_played_cards(
            cards(["9S"]),
            jokers=(
                joker_with_effect(
                    "Ride the Bus",
                    "This Joker gains +1 Mult per consecutive hand played without a scoring face card (Currently +2 Mult)",
                ),
            ),
        )
        runner = evaluate_played_cards(
            cards(["TS", "9C", "8D", "7H", "6S"]),
            jokers=(
                joker_with_effect(
                    "Runner",
                    "Gains +15 Chips if played hand contains a Straight (Currently +45 Chips)",
                ),
            ),
        )

        self.assertEqual(bus.score, 56)
        self.assertEqual(runner.score, 520)

    def test_ride_the_bus_does_not_score_on_scoring_face_cards(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS", "QS", "JS", "TS", "9H"]),
            jokers=(
                joker_with_effect(
                    "Ride the Bus",
                    "This Joker gains +1 Mult per consecutive hand played without a scoring face card (Currently +2 Mult)",
                ),
            ),
        )

        self.assertEqual(evaluation.score, 316)

    def test_four_fingers_and_shortcut_change_straight_detection(self) -> None:
        four_fingers = evaluate_played_cards(
            cards(["AS", "KD", "QH", "JC", "2D"]),
            jokers=(Joker("Four Fingers"),),
        )
        shortcut = evaluate_played_cards(
            cards(["TS", "8D", "6C", "5H", "3S"]),
            jokers=(Joker("Shortcut"),),
        )

        self.assertEqual(four_fingers.hand_type, HandType.STRAIGHT)
        self.assertEqual(four_fingers.scoring_indices, (0, 1, 2, 3))
        self.assertEqual(four_fingers.score, 284)
        self.assertEqual(shortcut.hand_type, HandType.STRAIGHT)
        self.assertEqual(shortcut.score, 248)

    def test_splash_scores_all_played_cards(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS", "AD", "7H"]),
            jokers=(Joker("Splash"),),
        )

        self.assertEqual(evaluation.hand_type, HandType.PAIR)
        self.assertEqual(evaluation.scoring_indices, (0, 1, 2))
        self.assertEqual(evaluation.score, 78)

    def test_disabled_splash_does_not_change_scoring_indices(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["TS", "TD", "8S", "8H", "2S"]),
            jokers=(Joker("Splash", metadata={"state": {"debuff": True}}),),
        )

        self.assertEqual(evaluation.hand_type, HandType.TWO_PAIR)
        self.assertEqual(evaluation.scoring_indices, (0, 1, 2, 3))
        self.assertEqual(evaluation.score, 112)

    def test_pareidolia_makes_all_cards_face_cards(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS", "AD"]),
            jokers=(Joker("Pareidolia"), Joker("Scary Face")),
        )

        self.assertEqual(evaluation.chips, 92)
        self.assertEqual(evaluation.score, 184)

    def test_copy_jokers_copy_adjacent_abilities(self) -> None:
        blueprint = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Blueprint"), Joker("Joker")),
        )
        brainstorm = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Joker"), Joker("Brainstorm")),
        )

        self.assertEqual(blueprint.mult, 9)
        self.assertEqual(blueprint.score, 144)
        self.assertEqual(brainstorm.mult, 9)
        self.assertEqual(brainstorm.score, 144)

    def test_retrigger_jokers_repeat_card_and_when_scored_effects(self) -> None:
        hack = evaluate_played_cards(
            cards(["2S"]),
            jokers=(Joker("Hack"), Joker("Fibonacci")),
        )
        sock = evaluate_played_cards(
            cards(["KS"]),
            jokers=(Joker("Sock and Buskin"), Joker("Smiley Face")),
        )

        self.assertEqual(hack.chips, 9)
        self.assertEqual(hack.mult, 17)
        self.assertEqual(hack.score, 153)
        self.assertEqual(sock.chips, 25)
        self.assertEqual(sock.mult, 11)
        self.assertEqual(sock.score, 275)

    def test_hiker_permanent_chips_affect_same_card_retriggers(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Hiker"), Joker("Hanging Chad")),
        )

        self.assertEqual(evaluation.chips, 53)
        self.assertEqual(evaluation.score, 53)

    def test_hanging_chad_and_photograph_stack_on_first_face_card(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS"]),
            jokers=(Joker("Hanging Chad"), Joker("Photograph")),
        )

        self.assertEqual(evaluation.chips, 35)
        self.assertEqual(evaluation.effect_xmult, 8)
        self.assertEqual(evaluation.score, 280)

    def test_photograph_triggers_before_flat_joker_mult(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS", "AC", "QC", "QD"]),
            jokers=(Joker("Abstract Joker"), Joker("Photograph")),
        )

        self.assertEqual(evaluation.chips, 62)
        self.assertEqual(evaluation.mult, 8)
        self.assertEqual(evaluation.effect_xmult, 2)
        self.assertEqual(evaluation.score, 620)

    def test_photograph_triggers_before_earlier_xmult_jokers(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS"]),
            jokers=(Joker("Cavendish"), Joker("Photograph")),
        )

        self.assertEqual(evaluation.effect_xmult, 6)
        self.assertEqual(evaluation.score, 90)

    def test_photograph_triggers_before_earlier_polychrome_joker(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS"]),
            jokers=(Joker("Joker", edition="Polychrome"), Joker("Photograph")),
        )

        self.assertEqual(evaluation.mult, 5)
        self.assertEqual(evaluation.effect_xmult, 3)
        self.assertEqual(evaluation.score, 135)

    def test_disabled_joker_does_not_apply_ability_or_edition(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["8S", "8D"]),
            jokers=(
                Joker("The Order", edition="HOLO", metadata={"value": {"effect": "All abilities are disabled"}}),
                Joker("The Tribe", edition="POLYCHROME", metadata={"state": {"debuff": True}}),
                Joker("Bull", metadata={"value": {"effect": "+2 Chips for each $1 you have (Currently +330 Chips)"}}),
                Joker("Even Steven"),
                Joker("Bootstraps", metadata={"value": {"effect": "+2 Mult for every $5 you have (Currently +66 Mult)"}}),
            ),
            money=165,
        )

        self.assertEqual(evaluation.chips, 356)
        self.assertEqual(evaluation.mult, 76)
        self.assertEqual(evaluation.score, 27056)

    def test_raised_fist_does_not_skip_lower_debuffed_held_card(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["QS", "QC"]),
            jokers=(Joker("Raised Fist"), Joker("Photograph")),
            held_cards=(
                Card(rank="A", suit="H", debuffed=True),
                Card(rank="7", suit="H", debuffed=True),
                Card(rank="10", suit="S"),
            ),
        )

        self.assertEqual(evaluation.score, 120)

    def test_raised_fist_respects_blind_debuffed_held_suit(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["QH", "QC"]),
            jokers=(Joker("Raised Fist"),),
            held_cards=(Card(rank="7", suit="S"), Card(rank="10", suit="H")),
            debuffed_suits=debuffed_suits_for_blind("The Goad"),
        )

        self.assertEqual(evaluation.score, 60)

    def test_raised_fist_uses_rank_order_for_ten_face_ties(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AH", "AC"]),
            jokers=(Joker("Raised Fist"),),
            held_cards=(Card(rank="Q", suit="H"), Card(rank="10", suit="S")),
            debuffed_suits=debuffed_suits_for_blind("The Goad"),
        )

        self.assertEqual(evaluation.score, 64)

    def test_raised_fist_uses_last_card_when_lowest_rank_ties(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["JH"]),
            jokers=(Joker("Raised Fist"),),
            held_cards=(
                Card(rank="J", suit="D"),
                Card(rank="6", suit="S"),
                Card(rank="6", suit="H"),
                Card(rank="6", suit="C"),
            ),
            debuffed_suits=debuffed_suits_for_blind("The Club"),
        )

        self.assertEqual(evaluation.score, 15)

    def test_hanging_chad_does_not_move_to_next_card_when_first_scoring_card_is_debuffed(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["7S", "7H", "2H", "2C", "2D"]),
            jokers=(Joker("Hanging Chad"), Joker("Odd Todd")),
            debuffed_suits=debuffed_suits_for_blind("The Goad"),
        )

        self.assertEqual(evaluation.score, 336)

    def test_money_scaled_jokers_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Bull"), Joker("Bootstraps")),
            money=11,
        )

        self.assertEqual(evaluation.chips, 38)
        self.assertEqual(evaluation.mult, 5)
        self.assertEqual(evaluation.score, 190)

    def test_scoring_time_dollars_feed_money_scaled_jokers(self) -> None:
        evaluation = evaluate_played_cards(
            (Card(rank="A", suit="D", enhancement="GOLD"),),
            jokers=(Joker("Golden Ticket"), Joker("Rough Gem"), Joker("Bull"), Joker("Bootstraps")),
        )

        self.assertEqual(evaluation.chips, 26)
        self.assertEqual(evaluation.mult, 3)
        self.assertEqual(evaluation.score, 78)

    def test_the_tooth_reduces_money_before_money_scaled_jokers(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS", "KD", "QC", "JH", "TC"]),
            jokers=(Joker("Bull"), Joker("Bootstraps")),
            blind_name="The Tooth",
            money=11,
        )

        self.assertEqual(evaluation.chips, 93)
        self.assertEqual(evaluation.mult, 6)
        self.assertEqual(evaluation.score, 558)

    def test_acrobat_and_suit_xmult_jokers_are_applied(self) -> None:
        acrobat = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Acrobat"),),
            hands_remaining=1,
        )
        seeing_double = evaluate_played_cards(
            cards(["KC", "KS"]),
            jokers=(Joker("Seeing Double"),),
        )
        flower_pot = evaluate_played_cards(
            cards(["TD", "9C", "8H", "7S", "6D"]),
            jokers=(Joker("Flower Pot"),),
        )
        flower_pot_with_debuffed_suit = evaluate_played_cards(
            cards(["TD", "9C", "8H", "7S", "6D"]),
            jokers=(Joker("Flower Pot"),),
            debuffed_suits=debuffed_suits_for_blind("The Goad"),
        )

        self.assertEqual(acrobat.score, 48)
        self.assertEqual(seeing_double.score, 120)
        self.assertEqual(flower_pot.score, 840)
        self.assertEqual(flower_pot_with_debuffed_suit.score, 756)

    def test_flower_pot_needs_five_played_cards(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["8H", "8C", "8D", "2S"]),
            jokers=(Joker("Flower Pot"),),
        )

        self.assertEqual(evaluation.score, 162)

    def test_flower_pot_ignores_kicker_suits_outside_scored_hand(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["QH", "QC", "7S", "3H", "3D"]),
            jokers=(Joker("Flower Pot"),),
        )

        self.assertEqual(evaluation.score, 92)

    def test_contains_two_pair_jokers_apply_to_full_house(self) -> None:
        mad = evaluate_played_cards(
            cards(["9S", "9D", "5S", "5C", "5D"]),
            jokers=(Joker("Mad Joker"),),
        )
        clever = evaluate_played_cards(
            cards(["9S", "9D", "5S", "5C", "5D"]),
            jokers=(Joker("Clever Joker"),),
        )

        self.assertEqual(mad.score, 1022)
        self.assertEqual(clever.score, 612)

    def test_spare_trousers_applies_to_full_house(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["JS", "JC", "JD", "5C", "5D"]),
            jokers=(
                joker_with_effect(
                    "Spare Trousers",
                    "This Joker gains +2 Mult if played hand contains a Two Pair (Currently +6 Mult)",
                ),
            ),
        )

        self.assertEqual(evaluation.chips, 80)
        self.assertEqual(evaluation.mult, 12)
        self.assertEqual(evaluation.score, 960)

    def test_specific_scored_card_xmult_jokers_are_applied(self) -> None:
        ancient = evaluate_played_cards(
            cards(["AS", "AH"]),
            jokers=(
                joker_with_effect(
                    "Ancient Joker",
                    "Each played card with Heart suit gives X1.5 Mult when scored",
                ),
            ),
        )
        idol = evaluate_played_cards(
            cards(["KS"]),
            jokers=(
                joker_with_effect(
                    "The Idol",
                    "Each played King of Spades gives X2 Mult when scored",
                ),
            ),
        )
        triboulet = evaluate_played_cards(
            cards(["KS", "KH", "QS", "QH", "QD"]),
            jokers=(Joker("Triboulet"),),
        )

        self.assertEqual(ancient.score, 96)
        self.assertEqual(idol.score, 30)
        self.assertEqual(triboulet.score, 11520)

    def test_current_value_and_rarity_xmult_jokers_are_applied(self) -> None:
        supernova = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Supernova", "Adds times played to Mult (Currently +4 Mult)"),),
        )
        supernova_from_hand_counts = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Supernova"),),
            played_hand_counts={"High Card": 3},
        )
        ramen = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Ramen", "X2 Mult, loses X0.01 per card discarded (Currently X1.6 Mult)"),),
        )
        ramen_live_text = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Ramen", "X1.92 Mult, loses X0.01 Mult per card discarded"),),
        )
        ramen_visible_rounding = evaluate_played_cards(
            cards(["AS", "AD", "KH", "9S", "9D"]),
            {"Two Pair": 7},
            jokers=(
                Joker("Bootstraps"),
                joker_with_effect("Ramen", "X1.58 Mult, loses X0.01 Mult per card discarded"),
            ),
            money=105,
        )
        baseball = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Baseball Card"), uncommon_joker("Uncommon One"), uncommon_joker("Uncommon Two")),
        )
        baseball_with_fallback_rarity = evaluate_played_cards(
            cards(["AS"]),
            jokers=(
                Joker("Baseball Card"),
                joker_with_effect("Erosion", "+4 Mult for each card below starting deck size (Currently +4 Mult)"),
            ),
        )

        self.assertEqual(supernova.score, 80)
        self.assertEqual(supernova_from_hand_counts.score, 80)
        self.assertEqual(ramen.score, 25)
        self.assertEqual(ramen_live_text.score, 30)
        self.assertEqual(ramen_visible_rounding.score, 14219)
        self.assertEqual(baseball.score, 36)
        self.assertEqual(baseball_with_fallback_rarity.score, 120)

    def test_baseball_card_uses_local_rarity_fallbacks(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(
                Joker("Baseball Card"),
                Joker("Blackboard"),
                Joker("Acrobat"),
                Joker("Baron"),
                Joker("Credit Card"),
            ),
        )

        self.assertEqual(evaluation.score, 36)

    def test_baseball_card_accepts_numeric_rarity_metadata(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(
                Joker("Baseball Card"),
                Joker("Numeric Uncommon", metadata={"rarity": 2}),
                Joker("Nested Numeric Uncommon", metadata={"value": {"rarity": "2"}}),
                Joker("Numeric Rare", metadata={"rarity": 3}),
            ),
        )

        self.assertEqual(evaluation.score, 36)

    def test_current_value_chip_and_mult_jokers_are_applied(self) -> None:
        stone = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Stone Joker", "Gives +25 Chips for each Stone Card in your full deck (Currently +50 Chips)"),),
        )
        ice_cream_from_metadata = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Ice Cream", metadata={"ability": {"extra": {"chips": 85}}}),),
        )
        ice_cream_live_text = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Ice Cream", "+95 Chips -5 Chips for every hand played"),),
        )
        blue_joker_live_text = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Blue Joker", "+2 Chips for each remaining card in deck (Currently +64 Chips)"),),
            deck_size=10,
        )
        hidden_blue_joker = evaluate_played_cards(
            cards(["AS"]),
            jokers=(
                Joker(
                    "Blue Joker",
                    metadata={
                        "state": {"hidden": True},
                        "value": {"effect": "+2 Chips for each remaining card in deck (Currently +64 Chips)"},
                    },
                ),
            ),
            deck_size=10,
        )
        castle = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Castle", "This Joker gains +3 Chips per discarded Spade card (Currently +9 Chips)"),),
        )
        square_four_card_play = evaluate_played_cards(
            cards(["AS", "KH", "QD", "JC"]),
            jokers=(Joker("Square Joker", metadata={"current_chips": 8}),),
        )
        erosion = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Erosion", "+4 Mult for each card below starting deck size (Currently +8 Mult)"),),
        )

        self.assertEqual(stone.score, 66)
        self.assertEqual(ice_cream_from_metadata.score, 101)
        self.assertEqual(ice_cream_live_text.score, 111)
        self.assertEqual(blue_joker_live_text.score, 80)
        self.assertEqual(hidden_blue_joker.score, 36)
        self.assertEqual(castle.score, 25)
        self.assertEqual(square_four_card_play.score, 28)
        self.assertEqual(erosion.score, 144)

    def test_played_gold_seal_money_is_available_to_bootstraps(self) -> None:
        evaluation = evaluate_played_cards(
            (Card(rank="A", suit="S", seal="GOLD"),),
            jokers=(Joker("Bootstraps"),),
            money=59,
        )

        self.assertEqual(evaluation.mult, 25)
        self.assertEqual(evaluation.score, 400)

    def test_current_value_mult_jokers_can_use_metadata(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Popcorn", metadata={"config": {"extra": {"mult": 12}}}),),
        )
        live_text = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Popcorn", "+16 Mult -4 Mult per round played"),),
        )

        self.assertEqual(evaluation.mult, 13)
        self.assertEqual(evaluation.score, 208)
        self.assertEqual(live_text.mult, 17)
        self.assertEqual(live_text.score, 272)

    def test_current_value_xmult_jokers_are_applied(self) -> None:
        steel = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Steel Joker", "Gives X0.2 Mult for each Steel Card in your full deck (Currently X1.4 Mult)"),),
        )
        glass = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Glass Joker", "Gains X0.75 Mult when Glass Cards are destroyed (Currently X1.75 Mult)"),),
        )
        stencil = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Joker Stencil", "X1 Mult for each empty Joker slot. Joker Stencil included (Currently X3)"),),
        )
        hit_the_road = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Hit the Road", "Gains X0.5 Mult for every Jack discarded this round (Currently X2.5 Mult)"),),
        )
        ramen_from_metadata = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Ramen", metadata={"ability": {"extra": {"xmult": 1.5}}}),),
        )

        self.assertEqual(steel.score, 22)
        self.assertEqual(glass.score, 28)
        self.assertEqual(stencil.score, 48)
        self.assertEqual(hit_the_road.score, 40)
        self.assertEqual(ramen_from_metadata.score, 24)

    def test_license_and_loyalty_card_are_applied_from_effect_text(self) -> None:
        license_active = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Driver's License", "X3 Mult if you have at least 16 Enhanced cards in your full deck (Currently 16)"),),
        )
        license_inactive = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Driver's License", "X3 Mult if you have at least 16 Enhanced cards in your full deck (Currently 15)"),),
        )
        loyalty_ready = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Loyalty Card", "X4 Mult every 6 hands played 0 remaining"),),
        )
        loyalty_active = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Loyalty Card", "X4 Mult every 6 hands played Active!"),),
        )
        loyalty_waiting = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Loyalty Card", "X4 Mult every 6 hands played 2 remaining"),),
        )
        loyalty_active_from_metadata = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Loyalty Card", metadata={"config": {"extra": {"remaining": 0}}}),),
        )
        loyalty_waiting_from_metadata = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Loyalty Card", metadata={"config": {"extra": {"remaining": 1}}}),),
        )

        self.assertEqual(license_active.score, 48)
        self.assertEqual(license_inactive.score, 16)
        self.assertEqual(loyalty_ready.score, 64)
        self.assertEqual(loyalty_active.score, 64)
        self.assertEqual(loyalty_waiting.score, 16)
        self.assertEqual(loyalty_active_from_metadata.score, 64)
        self.assertEqual(loyalty_waiting_from_metadata.score, 16)

    def test_stochastic_play_outcomes_can_be_injected(self) -> None:
        misprint = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Misprint"),),
            stochastic_outcomes={"misprint_mult": 10},
        )
        bloodstone = evaluate_played_cards(
            cards(["AH"]),
            jokers=(Joker("Bloodstone"),),
            stochastic_outcomes={"bloodstone_triggers": 1},
        )
        space = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Space Joker"),),
            stochastic_outcomes={"space_joker_triggers": 1},
        )

        self.assertEqual(misprint.score, 176)
        self.assertEqual(bloodstone.score, 32)
        self.assertEqual(space.score, 52)

    def test_scoring_time_money_outcomes_feed_money_scaled_jokers(self) -> None:
        business_card = evaluate_played_cards(
            cards(["KS"]),
            money=4,
            jokers=(Joker("Business Card"), Joker("Bootstraps")),
            stochastic_outcomes={"business_card_triggers": 1},
        )
        reserved_parking = evaluate_played_cards(
            cards(["AS"]),
            money=4,
            held_cards=cards(["KS"]),
            jokers=(Joker("Reserved Parking"), Joker("Bootstraps")),
            stochastic_outcomes={"reserved_parking_triggers": 1},
        )
        matador = evaluate_played_cards(
            cards(["AS"]),
            money=0,
            jokers=(Joker("Matador"), Joker("Bootstraps")),
            stochastic_outcomes={"matador_triggered": True},
        )

        self.assertEqual(business_card.money_delta, 2)
        self.assertEqual(business_card.score, 45)
        self.assertEqual(reserved_parking.money_delta, 1)
        self.assertEqual(reserved_parking.score, 48)
        self.assertEqual(matador.money_delta, 8)
        self.assertEqual(matador.score, 48)


if __name__ == "__main__":
    unittest.main()
