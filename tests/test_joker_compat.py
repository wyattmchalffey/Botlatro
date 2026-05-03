from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.rules.joker_compat import BLUEPRINT_INCOMPATIBLE_JOKERS


class JokerCompatTests(unittest.TestCase):
    def test_blueprint_incompatible_base_joker_list_matches_source_dump(self) -> None:
        self.assertEqual(
            BLUEPRINT_INCOMPATIBLE_JOKERS,
            {
                "Four Fingers",
                "Credit Card",
                "Chaos the Clown",
                "Delayed Gratification",
                "Pareidolia",
                "Egg",
                "Splash",
                "Sixth Sense",
                "Shortcut",
                "Cloud 9",
                "Rocket",
                "Midas Mask",
                "Gift Card",
                "Turtle Bean",
                "To the Moon",
                "Juggler",
                "Drunkard",
                "Golden Joker",
                "Trading Card",
                "Mr. Bones",
                "Troubadour",
                "Smeared Joker",
                "Showman",
                "Merry Andy",
                "Oops! All 6s",
                "Invisible Joker",
                "Satellite",
                "Astronomer",
                "Chicot",
            },
        )


if __name__ == "__main__":
    unittest.main()
