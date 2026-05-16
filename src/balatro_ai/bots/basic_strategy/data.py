"""Static strategy data tables for the basic strategy bot."""

from __future__ import annotations

from balatro_ai.api.state import Card
from balatro_ai.bots.basic_strategy.hand_models import _SampleHand
from balatro_ai.rules.hand_evaluator import HandType


DISCARD_PENALTY_JOKERS = {"Banner", "Delayed Gratification", "Green Joker", "Ramen"}

VOUCHER_INTEREST_CAP_MONEY = {
    "Seed Money": 50,
    "Money Tree": 100,
}

MONEY_SCALING_RESERVE_TARGETS = {
    "Bull": 75,
    "Bootstraps": 75,
}

RARE_HAND_TYPES = frozenset(
    {
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }
)

IMPOSSIBLE_STARTER_DECK_HANDS = frozenset(
    {
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }
)

IMPOSSIBLE_HAND_MANIPULATION_GAP = {
    HandType.FIVE_OF_A_KIND: 1.5,
    HandType.FLUSH_HOUSE: 1.5,
    HandType.FLUSH_FIVE: 2.0,
}

RARE_RANK_HAND_TYPES = frozenset(
    {
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }
)

RARE_FLUSH_HAND_TYPES = frozenset({HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE})

RARE_HAND_JOKER_TARGETS = {
    "The Family": HandType.FOUR_OF_A_KIND,
}

RANK_DECK_MANIPULATION_TAROTS = {
    "Strength",
    "Death",
    "The Hanged Man",
}

FLUSH_DECK_MANIPULATION_TAROTS = {
    "Death",
    "The Hanged Man",
    "The Star",
    "The Moon",
    "The Sun",
    "The World",
}

RARE_HAND_SUPPORT_TAROTS = RANK_DECK_MANIPULATION_TAROTS | FLUSH_DECK_MANIPULATION_TAROTS

DEDICATED_TWO_PAIR_BUILD_JOKERS = {
    "Spare Trousers",
}

DEDICATED_PAIR_BUILD_JOKERS = {
    "The Duo",
    "Jolly Joker",
    "Sly Joker",
}

TWO_PAIR_SUPPORT_JOKERS = {
    "Mad Joker",
    "Clever Joker",
}


WHITE_STAKE_SAMPLE_HANDS = (
    _SampleHand((Card("A", "S"),), (Card("K", "H"), Card("Q", "C"), Card("7", "D")), weight=0.55),
    _SampleHand((Card("2", "S"), Card("2", "H")), (Card("9", "D"), Card("5", "C")), weight=1.35),
    _SampleHand((Card("7", "S"), Card("7", "H")), (Card("K", "D"), Card("4", "C")), weight=1.2),
    _SampleHand((Card("A", "S"), Card("A", "H")), (Card("8", "D"), Card("4", "C")), weight=0.75),
    _SampleHand((Card("4", "S"), Card("4", "H"), Card("9", "D"), Card("9", "C")), weight=0.95),
    _SampleHand((Card("Q", "S"), Card("Q", "H"), Card("Q", "D")), weight=0.45),
    _SampleHand((Card("9", "S"), Card("8", "H"), Card("7", "D"), Card("6", "C"), Card("5", "S")), weight=0.3),
    _SampleHand((Card("A", "H"), Card("K", "H"), Card("Q", "H"), Card("7", "H"), Card("2", "H")), weight=0.3),
    _SampleHand((Card("J", "S"), Card("J", "H"), Card("J", "D"), Card("4", "S"), Card("4", "C")), weight=0.25),
)


PLANET_TO_HAND = {
    "Pluto": HandType.HIGH_CARD,
    "Mercury": HandType.PAIR,
    "Uranus": HandType.TWO_PAIR,
    "Venus": HandType.THREE_OF_A_KIND,
    "Earth": HandType.FULL_HOUSE,
    "Mars": HandType.FOUR_OF_A_KIND,
    "Jupiter": HandType.FLUSH,
    "Saturn": HandType.STRAIGHT,
    "Neptune": HandType.STRAIGHT_FLUSH,
    "Planet X": HandType.FIVE_OF_A_KIND,
    "Ceres": HandType.FLUSH_HOUSE,
    "Eris": HandType.FLUSH_FIVE,
}

PAIR_CONTAINS_HANDS = (
    HandType.PAIR,
    HandType.TWO_PAIR,
    HandType.THREE_OF_A_KIND,
    HandType.FULL_HOUSE,
    HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
)

TWO_PAIR_CONTAINS_HANDS = (
    HandType.TWO_PAIR,
    HandType.FULL_HOUSE,
    HandType.FLUSH_HOUSE,
)

THREE_KIND_CONTAINS_HANDS = (
    HandType.THREE_OF_A_KIND,
    HandType.FULL_HOUSE,
    HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
)

FOUR_KIND_CONTAINS_HANDS = (
    HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND,
    HandType.FLUSH_FIVE,
)

FLUSH_ARCHETYPE_HANDS = (
    HandType.FLUSH,
    HandType.STRAIGHT_FLUSH,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
)

RANK_ARCHETYPE_HANDS = tuple(
    dict.fromkeys((*PAIR_CONTAINS_HANDS, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND))
)

CARD_SHARP_REPEATABILITY_WEIGHTS = {
    HandType.HIGH_CARD: 1.9,
    HandType.PAIR: 2.2,
    HandType.TWO_PAIR: 1.4,
    HandType.THREE_OF_A_KIND: 1.0,
    HandType.STRAIGHT: 0.7,
    HandType.FLUSH: 0.7,
    HandType.FULL_HOUSE: 0.45,
    HandType.FOUR_OF_A_KIND: 0.15,
    HandType.STRAIGHT_FLUSH: 0.1,
    HandType.FIVE_OF_A_KIND: 0.0,
    HandType.FLUSH_HOUSE: 0.0,
    HandType.FLUSH_FIVE: 0.0,
}

PREFERRED_HAND_HUNT_TYPES = {
    HandType.THREE_OF_A_KIND,
    HandType.STRAIGHT,
    HandType.FLUSH,
    HandType.FULL_HOUSE,
    HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND,
    HandType.STRAIGHT_FLUSH,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
}

BURNT_JOKER_DISCARD_HAND_VALUES = {
    HandType.HIGH_CARD: 180.0,
    HandType.PAIR: 420.0,
    HandType.TWO_PAIR: 500.0,
    HandType.THREE_OF_A_KIND: 540.0,
    HandType.STRAIGHT: 580.0,
    HandType.FLUSH: 580.0,
    HandType.FULL_HOUSE: 620.0,
    HandType.FOUR_OF_A_KIND: 650.0,
    HandType.STRAIGHT_FLUSH: 700.0,
    HandType.FIVE_OF_A_KIND: 720.0,
    HandType.FLUSH_HOUSE: 720.0,
    HandType.FLUSH_FIVE: 740.0,
}


JOKER_HAND_SYNERGY = {
    "Jolly Joker": PAIR_CONTAINS_HANDS,
    "Sly Joker": PAIR_CONTAINS_HANDS,
    "Zany Joker": THREE_KIND_CONTAINS_HANDS,
    "Wily Joker": THREE_KIND_CONTAINS_HANDS,
    "Mad Joker": TWO_PAIR_CONTAINS_HANDS,
    "Clever Joker": TWO_PAIR_CONTAINS_HANDS,
    "Crazy Joker": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "Devious Joker": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "Droll Joker": FLUSH_ARCHETYPE_HANDS,
    "Crafty Joker": FLUSH_ARCHETYPE_HANDS,
    "Spare Trousers": TWO_PAIR_CONTAINS_HANDS,
    "Runner": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "The Duo": PAIR_CONTAINS_HANDS,
    "The Trio": THREE_KIND_CONTAINS_HANDS,
    "The Family": FOUR_KIND_CONTAINS_HANDS,
    "The Order": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "The Tribe": FLUSH_ARCHETYPE_HANDS,
}


JOKER_PRIMARY_HAND = {
    "Jolly Joker": HandType.PAIR,
    "Sly Joker": HandType.PAIR,
    "Zany Joker": HandType.THREE_OF_A_KIND,
    "Wily Joker": HandType.THREE_OF_A_KIND,
    "Mad Joker": HandType.TWO_PAIR,
    "Clever Joker": HandType.TWO_PAIR,
    "Crazy Joker": HandType.STRAIGHT,
    "Devious Joker": HandType.STRAIGHT,
    "Droll Joker": HandType.FLUSH,
    "Crafty Joker": HandType.FLUSH,
    "Spare Trousers": HandType.TWO_PAIR,
    "Runner": HandType.STRAIGHT,
    "The Duo": HandType.PAIR,
    "The Trio": HandType.THREE_OF_A_KIND,
    "The Family": HandType.FOUR_OF_A_KIND,
    "The Order": HandType.STRAIGHT,
    "The Tribe": HandType.FLUSH,
}


EARLY_POWER_JOKERS = {
    "Joker",
    "Jolly Joker",
    "Zany Joker",
    "Mad Joker",
    "Crazy Joker",
    "Droll Joker",
    "Sly Joker",
    "Wily Joker",
    "Clever Joker",
    "Devious Joker",
    "Crafty Joker",
    "Greedy Joker",
    "Lusty Joker",
    "Wrathful Joker",
    "Gluttonous Joker",
    "Half Joker",
    "Mystic Summit",
    "Misprint",
    "Gros Michel",
    "Popcorn",
    "Ice Cream",
    "Even Steven",
    "Odd Todd",
    "Scary Face",
    "Abstract Joker",
}


NARROW_EARLY_JOKERS = {
    "Crazy Joker",
    "Devious Joker",
    "Droll Joker",
    "Crafty Joker",
    "Runner",
    "The Order",
    "The Tribe",
}


NARROW_CHIP_PRIMARY_JOKERS = {
    "Crafty Joker",
    "Devious Joker",
    "Runner",
}


JOKER_BASE_VALUES = {
    "Blueprint": 75,
    "Brainstorm": 72,
    "Cavendish": 58,
    "Gros Michel": 42,
    "Popcorn": 34,
    "Ice Cream": 38,
    "Misprint": 32,
    "Abstract Joker": 28,
    "Half Joker": 24,
    "Mystic Summit": 22,
    "Photograph": 32,
    "Hanging Chad": 36,
    "Sock and Buskin": 34,
    "Hack": 28,
    "Dusk": 30,
    "Seltzer": 26,
    "Seeing Double": 34,
    "Flower Pot": 26,
    "Ancient Joker": 24,
    "The Idol": 24,
    "Baseball Card": 30,
    "Steel Joker": 28,
    "Stone Joker": 22,
    "Driver's License": 32,
    "Joker Stencil": 35,
    "Blackboard": 30,
    "Baron": 36,
}


JOKER_SCALING_VALUES = {
    "Green Joker": 34,
    "Ride the Bus": 34,
    "Supernova": 30,
    "Square Joker": 28,
    "Runner": 28,
    "Spare Trousers": 38,
    "Hologram": 40,
    "Constellation": 38,
    "Flash Card": 30,
    "Red Card": 28,
    "Castle": 30,
    "Erosion": 24,
    "Wee Joker": 34,
    "Lucky Cat": 34,
    "Glass Joker": 34,
    "Campfire": 32,
    "Throwback": 24,
    "Obelisk": 18,
}


DECAYING_SCORE_JOKERS = frozenset(
    {
        "Ice Cream",
        "Popcorn",
        "Ramen",
        "Seltzer",
        "Turtle Bean",
    }
)


FINITE_SCORE_JOKERS = frozenset(
    {
        "Seltzer",
        "Turtle Bean",
    }
)


ROUND_RESET_SCORE_JOKERS = frozenset(
    {
        "Campfire",
    }
)


TEMPORARY_SCORE_JOKERS = DECAYING_SCORE_JOKERS | ROUND_RESET_SCORE_JOKERS | frozenset({"Gros Michel"})


JOKER_ECONOMY_VALUES = {
    "Golden Joker": 28,
    "Rocket": 32,
    "Cloud 9": 18,
    "Business Card": 18,
    "Reserved Parking": 16,
    "Delayed Gratification": 14,
    "To the Moon": 22,
    "Mail-In Rebate": 18,
    "Golden Ticket": 20,
    "Faceless Joker": 12,
    "Egg": 14,
    "Gift Card": 18,
    "Trading Card": 20,
    "Satellite": 16,
    "Hallucination": 24,
}


JOKER_ORDER_XMULT = {
    "The Duo",
    "The Trio",
    "The Family",
    "The Order",
    "The Tribe",
    "Acrobat",
    "Seeing Double",
    "Flower Pot",
    "Blackboard",
    "Baron",
    "Constellation",
    "Madness",
    "Vampire",
    "Hologram",
    "Obelisk",
    "Lucky Cat",
    "Canio",
    "Caino",
    "Yorick",
    "Ramen",
    "Campfire",
    "Throwback",
    "Steel Joker",
    "Glass Joker",
    "Joker Stencil",
    "Hit the Road",
    "Cavendish",
    "Loyalty Card",
    "Driver's License",
    "Card Sharp",
    "Ancient Joker",
    "The Idol",
    "Triboulet",
}


JOKER_ORDER_MULT = {
    "Joker",
    "Gros Michel",
    "Mystic Summit",
    "Abstract Joker",
    "Swashbuckler",
    "Supernova",
    "Bootstraps",
    "Fibonacci",
    "Scholar",
    "Even Steven",
    "Half Joker",
    "Odd Todd",
    "Smiley Face",
    "Walkie Talkie",
    "Onyx Agate",
    "Greedy Joker",
    "Lusty Joker",
    "Wrathful Joker",
    "Gluttonous Joker",
    "Jolly Joker",
    "Zany Joker",
    "Mad Joker",
    "Crazy Joker",
    "Droll Joker",
    "Shoot the Moon",
    "Raised Fist",
    "Green Joker",
    "Ride the Bus",
    "Spare Trousers",
    "Fortune Teller",
    "Red Card",
    "Flash Card",
    "Popcorn",
    "Ceremonial Dagger",
    "Erosion",
}


JOKER_ORDER_CHIPS = {
    "Stuntman",
    "Bull",
    "Banner",
    "Scary Face",
    "Arrowhead",
    "Walkie Talkie",
    "Sly Joker",
    "Wily Joker",
    "Clever Joker",
    "Devious Joker",
    "Crafty Joker",
    "Blue Joker",
    "Wee Joker",
    "Runner",
    "Ice Cream",
    "Square Joker",
    "Stone Joker",
    "Castle",
}


LOW_PRIORITY_JOKERS = {
    "Credit Card",
    "Loyalty Card",
    "Seance",
    "Superposition",
    "To Do List",
    "Matador",
}


GLASS_CANNON_JOKERS = {
    "Popcorn",
    "Ice Cream",
    "Seltzer",
    "Gros Michel",
    "Ramen",
}


CHIP_JOKERS = {
    "Sly Joker",
    "Wily Joker",
    "Clever Joker",
    "Devious Joker",
    "Crafty Joker",
    "Scary Face",
    "Odd Todd",
    "Scholar",
    "Arrowhead",
    "Banner",
    "Stuntman",
    "Bull",
    "Blue Joker",
    "Square Joker",
    "Wee Joker",
    "Runner",
    "Ice Cream",
    "Stone Joker",
    "Castle",
}


MULT_JOKERS = {
    "Joker",
    "Jolly Joker",
    "Zany Joker",
    "Mad Joker",
    "Crazy Joker",
    "Droll Joker",
    "Greedy Joker",
    "Lusty Joker",
    "Wrathful Joker",
    "Gluttonous Joker",
    "Half Joker",
    "Mystic Summit",
    "Misprint",
    "Gros Michel",
    "Popcorn",
    "Abstract Joker",
    "Swashbuckler",
    "Supernova",
    "Bootstraps",
    "Fibonacci",
    "Scholar",
    "Even Steven",
    "Smiley Face",
    "Walkie Talkie",
    "Onyx Agate",
    "Green Joker",
    "Ride the Bus",
    "Spare Trousers",
    "Fortune Teller",
    "Flash Card",
    "Red Card",
    "Erosion",
    "Shoot the Moon",
    "Raised Fist",
}


XMULT_JOKERS = {
    "Blueprint",
    "Brainstorm",
    "Cavendish",
    "Photograph",
    "Ramen",
    "Acrobat",
    "Blackboard",
    "The Duo",
    "The Trio",
    "The Family",
    "The Order",
    "The Tribe",
    "Seeing Double",
    "Flower Pot",
    "Ancient Joker",
    "The Idol",
    "Baseball Card",
    "Steel Joker",
    "Glass Joker",
    "Driver's License",
    "Joker Stencil",
    "Baron",
    "Hologram",
    "Constellation",
    "Lucky Cat",
    "Campfire",
    "Throwback",
    "Obelisk",
    "Canio",
    "Caino",
    "Yorick",
    "Madness",
    "Vampire",
    "Card Sharp",
    "Loyalty Card",
}


SCALING_JOKERS = {
    "Green Joker",
    "Ride the Bus",
    "Supernova",
    "Square Joker",
    "Runner",
    "Spare Trousers",
    "Hologram",
    "Constellation",
    "Flash Card",
    "Red Card",
    "Castle",
    "Erosion",
    "Wee Joker",
    "Lucky Cat",
    "Glass Joker",
    "Campfire",
    "Throwback",
    "Obelisk",
    "Fortune Teller",
    "Steel Joker",
    "Vampire",
    "Madness",
}


FLEX_SCALING_JOKERS = {
    "Abstract Joker",
    "Banner",
    "Blue Joker",
    "Bull",
    "Green Joker",
    "Hologram",
    "Ride the Bus",
    "Square Joker",
    "Supernova",
}


ROLE_MISSING_BONUSES = {
    "chips": 18.0,
    "mult": 18.0,
    "xmult": 34.0,
    "scaling": 26.0,
    "economy": 14.0,
}


ROLE_UNIQUE_VALUES = {
    "chips": 18.0,
    "mult": 18.0,
    "xmult": 34.0,
    "scaling": 24.0,
    "economy": 10.0,
}


TAROT_VALUES = {
    "The Fool": 18,
    "The Magician": 24,
    "The High Priestess": 22,
    "The Empress": 22,
    "The Emperor": 16,
    "The Hierophant": 18,
    "The Lovers": 20,
    "The Chariot": 24,
    "Justice": 32,
    "The Hermit": 34,
    "The Wheel of Fortune": 16,
    "Strength": 28,
    "The Hanged Man": 28,
    "Death": 34,
    "Temperance": 26,
    "The Devil": 20,
    "The Tower": 18,
    "The Star": 22,
    "The Moon": 22,
    "The Sun": 22,
    "Judgement": 32,
    "The World": 22,
}


TARGET_REQUIRED_TAROTS = {
    "The Magician",
    "The Empress",
    "The Hierophant",
    "The Lovers",
    "The Chariot",
    "Justice",
    "Strength",
    "The Hanged Man",
    "Death",
    "The Devil",
    "The Tower",
    "The Star",
    "The Moon",
    "The Sun",
    "The World",
}


SPECTRAL_CARD_NAMES = {
    "Familiar",
    "Grim",
    "Incantation",
    "Talisman",
    "Aura",
    "Wraith",
    "Sigil",
    "Ouija",
    "Ectoplasm",
    "Immolate",
    "Ankh",
    "Deja Vu",
    "Hex",
    "Trance",
    "Medium",
    "Cryptid",
    "The Soul",
    "Black Hole",
}


SPECTRAL_SEAL_VALUES = {
    "Deja Vu": 58.0,
    "Trance": 42.0,
    "Medium": 40.0,
    "Talisman": 38.0,
}


SUIT_TAROT_TARGET_SUITS = {
    "The Star": "D",
    "The Moon": "C",
    "The Sun": "H",
    "The World": "S",
}

VOUCHER_BUY_DENYLIST = frozenset(
    {
        "planet merchant",
        "magic trick",
        "directors cut",
        "crystal ball",
        "telescope",
    }
)

VOUCHER_IMMEDIATE_SCORE_NAMES = frozenset(
    {
        "Grabber",
        "Nacho Tong",
        "Wasteful",
        "Recyclomancy",
        "Paint Brush",
        "Palette",
        "Hone",
        "Glow Up",
        "Retcon",
        "Antimatter",
    }
)

VOUCHER_PRESSURE_ALLOWED_NAMES = frozenset(
    {
        "Grabber",
        "Nacho Tong",
        "Wasteful",
        "Recyclomancy",
        "Paint Brush",
        "Palette",
        "Antimatter",
    }
)

DANGEROUS_BOSS_BLINDS = frozenset(
    {
        "The Wall",
        "The Needle",
        "Violet Vessel",
        "Verdant Leaf",
        "Crimson Heart",
        "Amber Acorn",
        "Cerulean Bell",
        "The Eye",
        "The Mouth",
        "The Pillar",
        "The Psychic",
        "The Flint",
        "The Water",
        "The Arm",
        "The Manacle",
        "The Club",
        "The Goad",
        "The Head",
        "The Window",
    }
)

FINAL_BOSS_BLINDS = frozenset(
    {
        "Amber Acorn",
        "Cerulean Bell",
        "Crimson Heart",
        "Verdant Leaf",
        "Violet Vessel",
    }
)

FINAL_BOSS_FRAGILE_JOKERS = frozenset(
    {
        "Ancient Joker",
        "Blackboard",
        "Card Sharp",
        "Driver's License",
        "Flower Pot",
        "Photograph",
        "Seeing Double",
        "The Idol",
    }
)

ORDER_SENSITIVE_JOKERS = frozenset(
    {
        "Ancient Joker",
        "Baseball Card",
        "Baron",
        "Blackboard",
        "Blueprint",
        "Brainstorm",
        "Card Sharp",
        "Flower Pot",
        "Hanging Chad",
        "Photograph",
        "Seeing Double",
        "The Idol",
    }
)


VOUCHER_VALUES = {
    "Overstock": 34,
    "Overstock Plus": 42,
    "Clearance Sale": 38,
    "Liquidation": 46,
    "Hone": 30,
    "Glow Up": 40,
    "Reroll Surplus": 32,
    "Reroll Glut": 40,
    "Tarot Merchant": 24,
    "Tarot Tycoon": 30,
    "Planet Merchant": 0,
    "Planet Tycoon": 24,
    "Crystal Ball": 26,
    "Omen Globe": 34,
    "Telescope": 26,
    "Observatory": 36,
    "Grabber": 28,
    "Nacho Tong": 36,
    "Wasteful": 22,
    "Recyclomancy": 30,
    "Paint Brush": 34,
    "Palette": 44,
    "Seed Money": 38,
    "Money Tree": 46,
    "Blank": 22,
    "Magic Trick": 0,
    "Illusion": 18,
    "Antimatter": 58,
    "Hieroglyph": 22,
    "Petroglyph": 28,
    "Director's Cut": 28,
    "Retcon": 36,
}


ANTE_SMALL_BLIND_SCORES = {
    1: 300,
    2: 800,
    3: 2000,
    4: 5000,
    5: 11000,
    6: 20000,
    7: 35000,
    8: 50000,
}
