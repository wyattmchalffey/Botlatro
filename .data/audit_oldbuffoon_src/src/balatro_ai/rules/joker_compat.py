"""Base-game joker compatibility flags from Balatro's joker centers."""

from __future__ import annotations

from balatro_ai.api.state import Joker


BLUEPRINT_INCOMPATIBLE_JOKERS = frozenset(
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
    }
)


def is_blueprint_compatible(joker: Joker) -> bool:
    """Return whether Blueprint/Brainstorm can copy this base-game joker."""

    return joker.name not in BLUEPRINT_INCOMPATIBLE_JOKERS
