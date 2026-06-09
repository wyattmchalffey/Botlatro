"""Item pools for shop prediction, extracted from Balatro game.lua.

Each pool is sorted by the source's ``order`` field, matching how Balatro
builds ``G.P_CENTER_POOLS`` (sorted ascending by order before
``pseudorandom_element`` iterates). Lists are tuples to make accidental
mutation obvious.

To regenerate after a game update, run the helper at the bottom of this
module against a fresh `.data/balatro-source/game.lua` extraction.
"""

from __future__ import annotations


# --- Tarot (Major Arcana) pool -------------------------------------------
TAROT_POOL: tuple[str, ...] = (
    "c_fool",            # order  1
    "c_magician",        # order  2
    "c_high_priestess",  # order  3
    "c_empress",         # order  4
    "c_emperor",         # order  5
    "c_heirophant",      # order  6
    "c_lovers",          # order  7
    "c_chariot",         # order  8
    "c_justice",         # order  9
    "c_hermit",          # order 10
    "c_wheel_of_fortune",# order 11
    "c_strength",        # order 12
    "c_hanged_man",      # order 13
    "c_death",           # order 14
    "c_temperance",      # order 15
    "c_devil",           # order 16
    "c_tower",           # order 17
    "c_star",            # order 18
    "c_moon",            # order 19
    "c_sun",             # order 20
    "c_judgement",       # order 21
    "c_world",           # order 22
)


# --- Planet pool ----------------------------------------------------------
# Last three (Planet X, Ceres, Eris) are softlocked: only available if the
# player has played the corresponding hand type (Five of a Kind, Flush House,
# Flush Five). At a fresh ante-1 first shop, these are typically UNAVAILABLE
# unless a special hand was already played.
PLANET_POOL: tuple[str, ...] = (
    "c_mercury",   # order  1, Pair
    "c_venus",     # order  2, Three of a Kind
    "c_earth",     # order  3, Full House
    "c_mars",      # order  4, Four of a Kind
    "c_jupiter",   # order  5, Flush
    "c_saturn",    # order  6, Straight
    "c_uranus",    # order  7, Two Pair
    "c_neptune",   # order  8, Straight Flush
    "c_pluto",     # order  9, High Card
    "c_planet_x",  # order 10, Five of a Kind (softlock)
    "c_ceres",     # order 11, Flush House    (softlock)
    "c_eris",      # order 12, Flush Five     (softlock)
)

PLANET_SOFTLOCKED: frozenset[str] = frozenset({"c_planet_x", "c_ceres", "c_eris"})


# --- Joker pools by rarity (1=common, 2=uncommon, 3=rare, 4=legendary) ----
# Auto-extracted from game.lua via re.findall on `j_<key>= {... order = N,
# rarity = M ...}`. Sorted by order ascending. 150 total: 61 + 64 + 20 + 5.

JOKER_POOL_COMMON: tuple[str, ...] = (
    "j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker",
    "j_gluttenous_joker", "j_jolly", "j_zany", "j_mad", "j_crazy", "j_droll",
    "j_sly", "j_wily", "j_clever", "j_devious", "j_crafty", "j_half",
    "j_credit_card", "j_banner", "j_mystic_summit", "j_8_ball", "j_misprint",
    "j_raised_fist", "j_chaos", "j_scary_face", "j_abstract", "j_delayed_grat",
    "j_gros_michel", "j_even_steven", "j_odd_todd", "j_scholar", "j_business",
    "j_supernova", "j_ride_the_bus", "j_egg", "j_runner", "j_ice_cream",
    "j_splash", "j_blue_joker", "j_faceless", "j_green_joker",
    "j_superposition", "j_todo_list", "j_cavendish", "j_red_card", "j_square",
    "j_riff_raff", "j_photograph", "j_reserved_parking", "j_mail",
    "j_hallucination", "j_fortune_teller", "j_juggler", "j_drunkard",
    "j_golden", "j_popcorn", "j_walkie_talkie", "j_smiley", "j_ticket",
    "j_swashbuckler", "j_hanging_chad", "j_shoot_the_moon",
)

JOKER_POOL_UNCOMMON: tuple[str, ...] = (
    "j_stencil", "j_four_fingers", "j_mime", "j_ceremonial", "j_marble",
    "j_loyalty_card", "j_dusk", "j_fibonacci", "j_steel_joker", "j_hack",
    "j_pareidolia", "j_space", "j_burglar", "j_blackboard", "j_sixth_sense",
    "j_constellation", "j_hiker", "j_card_sharp", "j_madness", "j_seance",
    "j_vampire", "j_shortcut", "j_hologram", "j_cloud_9", "j_rocket",
    "j_midas_mask", "j_luchador", "j_gift", "j_turtle_bean", "j_erosion",
    "j_to_the_moon", "j_stone", "j_lucky_cat", "j_bull", "j_diet_cola",
    "j_trading", "j_flash", "j_trousers", "j_ramen", "j_selzer", "j_castle",
    "j_mr_bones", "j_acrobat", "j_sock_and_buskin", "j_troubadour",
    "j_certificate", "j_smeared", "j_throwback", "j_rough_gem", "j_bloodstone",
    "j_arrowhead", "j_onyx_agate", "j_glass", "j_ring_master", "j_flower_pot",
    "j_merry_andy", "j_oops", "j_idol", "j_seeing_double", "j_matador",
    "j_satellite", "j_cartomancer", "j_astronomer", "j_bootstraps",
)

JOKER_POOL_RARE: tuple[str, ...] = (
    "j_dna", "j_vagabond", "j_baron", "j_obelisk", "j_baseball", "j_ancient",
    "j_campfire", "j_blueprint", "j_wee", "j_hit_the_road", "j_duo", "j_trio",
    "j_family", "j_order", "j_tribe", "j_stuntman", "j_invisible",
    "j_brainstorm", "j_drivers_license", "j_burnt",
)

JOKER_POOL_LEGENDARY: tuple[str, ...] = (
    "j_caino", "j_triboulet", "j_yorick", "j_chicot", "j_perkeo",
)


JOKER_POOL_BY_RARITY: dict[int, tuple[str, ...]] = {
    1: JOKER_POOL_COMMON,
    2: JOKER_POOL_UNCOMMON,
    3: JOKER_POOL_RARE,
    4: JOKER_POOL_LEGENDARY,
}

ENHANCEMENT_GATED_JOKERS: dict[str, str] = {
    "j_steel_joker": "m_steel",
    "j_stone": "m_stone",
    "j_lucky_cat": "m_lucky",
    "j_ticket": "m_gold",
    "j_glass": "m_glass",
}

PERISHABLE_INCOMPATIBLE_JOKERS: frozenset[str] = frozenset({
    "j_ceremonial",
    "j_ride_the_bus",
    "j_runner",
    "j_constellation",
    "j_green_joker",
    "j_red_card",
    "j_madness",
    "j_square",
    "j_vampire",
    "j_hologram",
    "j_rocket",
    "j_obelisk",
    "j_lucky_cat",
    "j_flash",
    "j_trousers",
    "j_castle",
    "j_glass",
    "j_wee",
})

ETERNAL_INCOMPATIBLE_JOKERS: frozenset[str] = frozenset({
    "j_gros_michel",
    "j_ice_cream",
    "j_cavendish",
    "j_luchador",
    "j_turtle_bean",
    "j_diet_cola",
    "j_popcorn",
    "j_ramen",
    "j_selzer",
    "j_mr_bones",
    "j_invisible",
})


# Jokers that ship with `unlocked = false` in game.lua and would be filtered
# out as UNAVAILABLE on a fresh profile. The user's actual profile may have
# unlocked some of these; predictions for a profile-specific session should
# pass the unlocked set into the pool builder.
LOCKED_BY_DEFAULT: frozenset[str] = frozenset({
    "j_ticket",          # common
    "j_swashbuckler",    # common
    "j_hanging_chad",    # common
    "j_shoot_the_moon",  # common
    # Note: more locked jokers exist at higher rarities; this set captures
    # the ones the empirical 4-seed fixture run did not encounter as picks.
    # Extend as cross-validation surfaces more.
})


def planet_pool_for_ante(
    played_hand_types: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Return the Planet pool with softlocked entries marked UNAVAILABLE.

    ``played_hand_types`` should contain the *names* of hand types the
    player has played this run (e.g. ``{"Five of a Kind"}``). Any softlock
    whose required hand isn't in the set becomes UNAVAILABLE.
    """

    softlock_required = {
        "c_planet_x": "Five of a Kind",
        "c_ceres": "Flush House",
        "c_eris": "Flush Five",
    }
    return tuple(
        key if (key not in softlock_required or softlock_required[key] in played_hand_types)
        else "UNAVAILABLE"
        for key in PLANET_POOL
    )


# Pool-flag-gated jokers (Balatro common_events.lua:2027-2028). The only two: Gros
# Michel disappears once it has gone extinct; Cavendish only appears AFTER it does.
# Keyed on the run pool flag ``gros_michel_extinct``. Default (no flags set = the
# common pre-extinction state) correctly makes Cavendish UNAVAILABLE and Gros Michel
# available -- which is exactly the case the predictor previously got wrong (it
# predicted Cavendish, which `_record_available` then excluded -> reroll divergence).
_JOKER_NO_POOL_FLAG: dict[str, str] = {"j_gros_michel": "gros_michel_extinct"}
_JOKER_YES_POOL_FLAG: dict[str, str] = {"j_cavendish": "gros_michel_extinct"}


def joker_pool_for_rarity(
    rarity: int,
    *,
    unlocked: frozenset[str] | None = None,
    used: frozenset[str] = frozenset(),
    available_enhancements: frozenset[str] = frozenset(),
    pool_flags: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Return the rarity pool with locked/used/flag-gated entries marked UNAVAILABLE.

    ``unlocked`` is the set of joker keys the player's profile has unlocked
    (None = treat everything as unlocked, matching most active profiles).
    ``used`` is the set of keys already drawn this run (Showman bypasses
    this; not modeled here yet). ``pool_flags`` is the set of SET run pool flags
    (G.GAME.pool_flags), used to gate Gros Michel / Cavendish.
    """

    pool = JOKER_POOL_BY_RARITY[rarity]
    out = []
    for key in pool:
        is_locked = key in LOCKED_BY_DEFAULT and unlocked is not None and key not in unlocked
        is_used = key in used
        gate = ENHANCEMENT_GATED_JOKERS.get(key)
        is_gated = gate is not None and gate not in available_enhancements
        no_flag = _JOKER_NO_POOL_FLAG.get(key)
        yes_flag = _JOKER_YES_POOL_FLAG.get(key)
        is_flag_gated = (no_flag is not None and no_flag in pool_flags) or (
            yes_flag is not None and yes_flag not in pool_flags
        )
        out.append("UNAVAILABLE" if (is_locked or is_used or is_gated or is_flag_gated) else key)
    return tuple(out)
