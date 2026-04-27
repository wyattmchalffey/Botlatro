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

    def test_permanent_card_chip_metadata_is_applied(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="A", suit="S", metadata={"modifier": {"perma_bonus": 10}}),
                Card(rank="A", suit="D", metadata={"bonus_chips": 5}),
            )
        )

        self.assertEqual(evaluation.card_chips, 37)
        self.assertEqual(evaluation.score, 94)

    def test_glass_and_joker_edition_xmult_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            (
                Card(rank="A", suit="S", enhancement="m_glass"),
                Card(rank="A", suit="D"),
            ),
            jokers=(Joker("Joker", edition="POLY"),),
        )

        self.assertEqual(evaluation.score, 576)

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

    def test_the_psychic_zeroes_non_five_card_plays(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS", "KC", "8S", "8H"]),
            blind_name="The Psychic",
            jokers=(Joker("Shoot the Moon"), Joker("Abstract Joker")),
        )

        self.assertEqual(evaluation.score, 0)

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
        self.assertEqual(evaluation.score, 648)

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

    def test_hanging_chad_and_photograph_stack_on_first_face_card(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["KS"]),
            jokers=(Joker("Hanging Chad"), Joker("Photograph")),
        )

        self.assertEqual(evaluation.chips, 35)
        self.assertEqual(evaluation.effect_xmult, 8)
        self.assertEqual(evaluation.score, 280)

    def test_money_scaled_jokers_are_applied(self) -> None:
        evaluation = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Bull"), Joker("Bootstraps")),
            money=11,
        )

        self.assertEqual(evaluation.chips, 38)
        self.assertEqual(evaluation.mult, 5)
        self.assertEqual(evaluation.score, 190)

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

        self.assertEqual(acrobat.score, 48)
        self.assertEqual(seeing_double.score, 120)
        self.assertEqual(flower_pot.score, 840)

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
        ramen = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Ramen", "X2 Mult, loses X0.01 per card discarded (Currently X1.6 Mult)"),),
        )
        baseball = evaluate_played_cards(
            cards(["AS"]),
            jokers=(Joker("Baseball Card"), uncommon_joker("Uncommon One"), uncommon_joker("Uncommon Two")),
        )

        self.assertEqual(supernova.score, 80)
        self.assertEqual(ramen.score, 25)
        self.assertEqual(baseball.score, 36)

    def test_current_value_chip_and_mult_jokers_are_applied(self) -> None:
        stone = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Stone Joker", "Gives +25 Chips for each Stone Card in your full deck (Currently +50 Chips)"),),
        )
        castle = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Castle", "This Joker gains +3 Chips per discarded Spade card (Currently +9 Chips)"),),
        )
        erosion = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Erosion", "+4 Mult for each card below starting deck size (Currently +8 Mult)"),),
        )

        self.assertEqual(stone.score, 66)
        self.assertEqual(castle.score, 25)
        self.assertEqual(erosion.score, 144)

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

        self.assertEqual(steel.score, 22)
        self.assertEqual(glass.score, 28)
        self.assertEqual(stencil.score, 48)
        self.assertEqual(hit_the_road.score, 40)

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
        loyalty_waiting = evaluate_played_cards(
            cards(["AS"]),
            jokers=(joker_with_effect("Loyalty Card", "X4 Mult every 6 hands played 2 remaining"),),
        )

        self.assertEqual(license_active.score, 48)
        self.assertEqual(license_inactive.score, 16)
        self.assertEqual(loyalty_ready.score, 64)
        self.assertEqual(loyalty_waiting.score, 16)


if __name__ == "__main__":
    unittest.main()
