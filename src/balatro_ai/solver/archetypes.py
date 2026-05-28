"""Build archetypes for the offline solver (Milestone M6a).

An `Archetype` declares a build plan the solver commits to at the run's
root. The plan biases shop and pack decisions toward items that
accelerate the plan, even when those items aren't the highest scoring
in isolation. This directly addresses the "full slots, no xmult" loss
pattern that the live bot keeps hitting.

M6a ships the scaffold plus one archetype (Flush). M6b will add the
remaining archetypes and the multi-archetype branching that picks the
best trajectory across archetype attempts per seed.

Status: M6a of `SOLVER_PLAN.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

from balatro_ai.api.state import GameState
from balatro_ai.rules.hand_evaluator import HandType


@dataclass(frozen=True, slots=True)
class Archetype:
    """A build plan the solver commits to at the run's root.

    Attributes
    ----------
    name: human-readable label for the archetype (e.g. "flush").
    target_hand_types: hand types this archetype wants to play often.
                       Used for matching against the played-hand counter.
    key_joker_keys: joker keys (e.g. ``"j_smeared"``) that accelerate
                    this archetype. Shop scoring adds a bonus per match.
    key_consumable_keys: tarot/spectral/planet keys this archetype wants.
                         Same bonus mechanism.
    shop_bonus_per_match: how much extra leaf value to award per matched
                          owned key. Tuned to nudge decisions rather than
                          force them — soft bias, not hard bias.
    """

    name: str
    target_hand_types: frozenset[HandType]
    key_joker_keys: frozenset[str]
    key_consumable_keys: frozenset[str] = frozenset()
    shop_bonus_per_match: float = 1.5

    def archetype_fit_score(self, state: GameState) -> float:
        """Return the bonus to add to a shop leaf's value for this archetype.

        Counts owned items (jokers + consumables) that match this
        archetype's key lists and multiplies by `shop_bonus_per_match`.
        Stateless beyond the input — safe to call repeatedly during
        beam expansion.
        """

        if not self.key_joker_keys and not self.key_consumable_keys:
            return 0.0

        bonus = 0.0
        if self.key_joker_keys:
            for joker in state.jokers:
                joker_key = _joker_key(joker)
                if joker_key in self.key_joker_keys:
                    bonus += self.shop_bonus_per_match

        if self.key_consumable_keys:
            for consumable in state.consumables:
                if str(consumable) in self.key_consumable_keys:
                    bonus += self.shop_bonus_per_match

        return bonus


def _joker_key(joker) -> str:
    """Best-effort recovery of the joker's `j_*` key.

    `Joker` parsed from bridge state keeps the original key in metadata.
    Locally-constructed jokers (some test fixtures) may only have the
    display name, which we map back through a small reverse lookup.
    """

    meta = joker.metadata if hasattr(joker, "metadata") else {}
    key = meta.get("key") if isinstance(meta, dict) else None
    if isinstance(key, str) and key.startswith("j_"):
        return key
    # Fallback: try the name. Live games always have the key in metadata,
    # so this path is mostly for tests.
    name = getattr(joker, "name", None)
    if isinstance(name, str):
        return _NAME_TO_KEY.get(name, "")
    return ""


# Reverse map for jokers commonly referenced by archetypes. Extend as
# more archetypes ship. Source of truth is `pools.py` joker key lists
# (which use the `j_*` keys directly) so we don't need to mirror the
# whole 150-joker map here.
_NAME_TO_KEY: dict[str, str] = {
    "Smeared Joker": "j_smeared",
    "Four Fingers": "j_four_fingers",
    "Shortcut": "j_shortcut",
    "Driver's License": "j_drivers_license",
    "Throwback": "j_throwback",
    "Flower Pot": "j_flower_pot",
    "Vampire": "j_vampire",
    "Hologram": "j_hologram",
    "Steel Joker": "j_steel_joker",
    "Stone Joker": "j_stone",
    "Lucky Cat": "j_lucky_cat",
    "Golden Ticket": "j_ticket",
    "Glass Joker": "j_glass",
}


# --- Built-in archetypes -----------------------------------------------

FLUSH_ARCHETYPE = Archetype(
    name="flush",
    target_hand_types=frozenset({HandType.FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE, HandType.STRAIGHT_FLUSH}),
    key_joker_keys=frozenset({
        # Suit-flexibility helpers — make flushes easier to build.
        "j_smeared",          # Hearts == Diamonds, Spades == Clubs
        "j_four_fingers",     # 4-card flushes count
        "j_shortcut",          # related: skips can fill straights, helps Straight Flush
        # Flush-payoff jokers — extra mult/chips on flush.
        "j_flower_pot",       # bonus when each scored card has a unique suit (NOT flush per se, but suit-aware)
        "j_droll",             # +10 mult on flush
        "j_sly",               # +50 chips on pair, weaker but archetype-adjacent
        "j_wily",              # +100 chips on 3-of-a-kind
        # Scaling jokers that benefit from consistent flush hands.
        "j_hologram",          # +X mult on every card played
        "j_vampire",           # consumes enhancements for XMult
    }),
    key_consumable_keys=frozenset({
        # Tarots that convert suits — strengthen the flush plan.
        "c_star",              # Diamonds
        "c_moon",              # Clubs
        "c_sun",               # Hearts
        "c_world",             # Spades
    }),
    shop_bonus_per_match=2.0,
)


SCALING_JOKER_ARCHETYPE = Archetype(
    name="scaling_joker",
    target_hand_types=frozenset({HandType.HIGH_CARD, HandType.PAIR, HandType.TWO_PAIR}),
    key_joker_keys=frozenset({
        # Per-hand-played scaling.
        "j_green_joker",       # +1 mult per hand
        "j_ride_the_bus",       # +1 mult per consecutive non-face
        "j_square",             # +4 chips per 4-card hand played
        "j_wee",                # +8 chips per 2 played
        "j_runner",             # +15 chips per straight
        # Round-counter scaling.
        "j_supernova",          # +mult per hand-type played
        "j_constellation",      # +0.1 xmult per planet used
        "j_madness",            # +0.5 xmult per blind selected w/ small/big
        "j_red_card",           # +3 mult per pack skipped
        "j_castle",             # +3 chips per suit card discarded
        "j_flash",              # +2 mult per reroll
        "j_trousers",           # +2 mult per two pair
        # Money-scaled jokers also count as "scaling" in spirit.
        "j_bull",               # +2 chips per $1
        "j_bootstraps",         # +2 mult per $5
    }),
    shop_bonus_per_match=2.0,
)


HIGH_CARD_MULT_ARCHETYPE = Archetype(
    name="high_card_mult",
    target_hand_types=frozenset({HandType.HIGH_CARD, HandType.PAIR}),
    key_joker_keys=frozenset({
        # Held-in-hand mult jokers — high-card play with banked aces.
        "j_mime",               # retrigger held cards
        "j_raised_fist",        # +2 mult per held lowest rank
        "j_baron",              # +1 mult per held King (Xmult x1.5)
        "j_shoot_the_moon",     # +13 mult per held Queen
        # Face/Ace-specific.
        "j_scary_face",         # +30 chips per face played
        "j_smiley",             # +5 mult per face played
        "j_business",           # 1-in-2 face card gives $2
        "j_photograph",          # first scoring face xmult
        "j_pareidolia",         # all cards count as face
    }),
    shop_bonus_per_match=2.0,
)


PAIR_RETRIGGER_ARCHETYPE = Archetype(
    name="pair_retrigger",
    target_hand_types=frozenset({HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND, HandType.FOUR_OF_A_KIND}),
    key_joker_keys=frozenset({
        # Retriggers — extra scoring per scored card.
        "j_sock_and_buskin",    # retrigger face cards
        "j_hanging_chad",        # retrigger first scoring card 2x
        "j_dusk",                # retrigger scoring cards on last hand
        "j_seltzer",            # retrigger all scoring cards (limited)
        # Pair/face-pair payoff jokers.
        "j_jolly",              # +8 mult on pair
        "j_zany",               # +12 mult on 3 of a kind
        "j_mad",                # +10 mult on 2 pair
        "j_clever",             # +80 chips on 2 pair
    }),
    shop_bonus_per_match=2.0,
)


BUILT_IN_ARCHETYPES: tuple[Archetype, ...] = (
    FLUSH_ARCHETYPE,
    SCALING_JOKER_ARCHETYPE,
    HIGH_CARD_MULT_ARCHETYPE,
    PAIR_RETRIGGER_ARCHETYPE,
)
