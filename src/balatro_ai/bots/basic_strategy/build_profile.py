"""Build profile and joker role scoring policy."""

from __future__ import annotations

import os

from balatro_ai.api.state import GameState, Joker
from balatro_ai.bots.basic_strategy.cache import _identity_cached_value
from balatro_ai.bots.basic_strategy.cards import (
    _edition_chips_value,
    _edition_mult_value,
    _edition_xmult_value,
    _normal_joker_open_slots,
)
from balatro_ai.bots.basic_strategy.data import (
    DECAYING_SCORE_JOKERS,
    FINITE_SCORE_JOKERS,
    FLEX_SCALING_JOKERS,
    JOKER_ECONOMY_VALUES,
    JOKER_SCALING_VALUES,
    ROLE_UNIQUE_VALUES,
    ROUND_RESET_SCORE_JOKERS,
    SCALING_JOKERS,
)
from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type
from balatro_ai.bots.basic_strategy.jokers import (
    _driver_license_readiness_factor,
    _joker_current_plus_value,
    _joker_current_xmult_value,
    _joker_has_sticker,
    _joker_is_disabled_for_build,
    _joker_remaining_count,
    _joker_roles,
    _static_chip_role_score,
    _static_mult_role_score,
    _static_xmult_role_score,
)
from balatro_ai.bots.basic_strategy.profile import _BuildProfile, _ShopPressure, _role_requirement
from balatro_ai.bots.basic_strategy.shop_money import _interest_cap_money, _spendable_money
from balatro_ai.bots.basic_strategy.shop_safety import _has_money_scaling_joker
from balatro_ai.rules.hand_evaluator import HandType


def _build_profile(state: GameState) -> _BuildProfile:
    return _identity_cached_value("build_profile", state, lambda: _build_profile_uncached(state))


def _build_profile_uncached(state: GameState) -> _BuildProfile:
    joker_roles = [_joker_roles(joker) for joker in state.jokers if not _joker_is_disabled_for_build(joker)]
    preferred = _preferred_hand_type(state)
    role_scores = _build_role_scores(state)
    return _BuildProfile(
        preferred_hand=preferred,
        archetype=_build_archetype(state, preferred),
        chip_score=role_scores["chips"],
        mult_score=role_scores["mult"],
        xmult_score=role_scores["xmult"],
        scaling_score=role_scores["scaling"],
        economy_score=role_scores["economy"],
        has_chips=role_scores["chips"] >= _role_requirement("chips", state.ante),
        has_mult=role_scores["mult"] >= _role_requirement("mult", state.ante),
        has_xmult=role_scores["xmult"] >= _role_requirement("xmult", state.ante),
        has_scaling=role_scores["scaling"] >= _role_requirement("scaling", state.ante),
        has_economy=(
            state.money >= _interest_cap_money(state)
            or any("economy" in roles for roles in joker_roles)
            or role_scores["economy"] >= _role_requirement("economy", state.ante)
        ),
        open_joker_slots=_normal_joker_open_slots(state),
        money=state.money,
        spendable_money=_spendable_money(state),
        ante=state.ante,
    )


def _hand_level_scaling_score(state: GameState) -> float:
    """Planet / hand-leveling IS scaling investment — permanent per-round growth on
    the hand you play. The joker-only scaling score ignored it, so a leveled build read
    as 'no scaling': the bot hunted scaling jokers it didn't need and its capacity
    penalties inflated pressure. Credit levels above 1 (full weight on the preferred
    hand, partial on others), calibrated so a level ~4-5 preferred hand registers as
    scaling (requirement is 22-30). A/B-measured winrate-negative (12->9/100), so OFF
    by default — env-gated for reuse."""
    if os.environ.get("BALATRO_PLANET_SCALING", "0") != "1":
        return 0.0
    levels = state.hand_levels or {}
    preferred = _preferred_hand_type(state)
    preferred_key = preferred.value if preferred is not None else None
    total = 0.0
    for hand_key, level in levels.items():
        above = max(0, int(level) - 1)
        if above <= 0:
            continue
        total += above * (1.0 if hand_key == preferred_key else 0.35)
    return min(30.0, total * 7.0)


def _build_role_scores(state: GameState) -> dict[str, float]:
    scores = {"chips": 0.0, "mult": 0.0, "xmult": 0.0, "scaling": 0.0, "economy": 0.0}
    for joker in state.jokers:
        joker_scores = _joker_role_scores(state, joker)
        for role, value in joker_scores.items():
            scores[role] += value
    scores["scaling"] += _hand_level_scaling_score(state)
    scores["economy"] += min(18.0, max(0.0, state.money - 5) * 0.6)
    if state.money >= _interest_cap_money(state):
        scores["economy"] += 18.0
    return scores


def _joker_role_scores(state: GameState, joker: Joker) -> dict[str, float]:
    if _joker_is_disabled_for_build(joker):
        return {"chips": 0.0, "mult": 0.0, "xmult": 0.0, "scaling": 0.0, "economy": 0.0}

    name = joker.name
    scores = {
        "chips": float(_edition_chips_value(joker.edition) + _joker_current_plus_value(joker, suffix="chips")),
        "mult": float(_edition_mult_value(joker.edition) + _joker_current_plus_value(joker, suffix="mult")),
        "xmult": max(0.0, (_edition_xmult_value(joker.edition) - 1.0) * 42.0),
        "scaling": 0.0,
        "economy": float(JOKER_ECONOMY_VALUES.get(name, 0)),
    }

    current_xmult = _joker_current_xmult_value(joker)
    if current_xmult > 1.0:
        scores["xmult"] += (current_xmult - 1.0) * 42.0

    if name == "Red Card":
        current_mult = _joker_current_plus_value(joker, suffix="mult")
        if current_mult > 0:
            scores["scaling"] += min(30.0, current_mult * 1.5)
        return _durable_joker_role_scores(state, joker, scores)

    scores["chips"] += _static_chip_role_score(name)
    scores["mult"] += _static_mult_role_score(name)
    static_xmult = _static_xmult_role_score(name)
    if name == "Driver's License":
        static_xmult *= _driver_license_readiness_factor(state, joker)
    scores["xmult"] += static_xmult
    if name in SCALING_JOKERS or name in JOKER_SCALING_VALUES:
        scores["scaling"] += float(JOKER_SCALING_VALUES.get(name, 24))
        if name in {"Green Joker", "Ride the Bus", "Square Joker", "Runner", "Spare Trousers"}:
            scores["scaling"] += max(0.0, _joker_current_plus_value(joker, suffix="mult") * 0.8)
            scores["scaling"] += max(0.0, _joker_current_plus_value(joker, suffix="chips") * 0.25)
        if current_xmult > 1.0 and name in {"Hologram", "Constellation", "Lucky Cat", "Glass Joker", "Campfire"}:
            scores["scaling"] += min(24.0, (current_xmult - 1.0) * 22.0)

    if (os.environ.get("BALATRO_RESET_JOKER_DISCOUNT", "0") == "1"
            and name in ROUND_RESET_SCORE_JOKERS and state.ante < 6):
        # Campfire-class jokers RESET their gain every boss and need a sell-economy to
        # ramp, so they are NOT real scalers/xmult until late game — don't let them fill
        # the scaling/xmult role early (the over-valuation that bought a 0-scoring
        # Campfire under pressure). Env-gated for A/B.
        scores["scaling"] *= 0.15
        scores["xmult"] *= 0.15

    return _durable_joker_role_scores(state, joker, scores)


def _durable_joker_role_scores(
    state: GameState,
    joker: Joker,
    scores: dict[str, float],
) -> dict[str, float]:
    durability = _joker_late_durability_factor(state, joker)
    if durability < 1.0:
        for role in ("chips", "mult", "xmult", "scaling"):
            scores[role] *= durability
        if _joker_has_sticker(joker, "perishable"):
            scores["economy"] *= durability
    return scores


def _joker_late_durability_factor(state: GameState, joker: Joker) -> float:
    if _joker_is_disabled_for_build(joker):
        return 0.0

    factor = 1.0
    if _joker_has_sticker(joker, "perishable"):
        if state.ante >= 7:
            factor *= 0.35
        elif state.ante >= 5:
            factor *= 0.55
        else:
            factor *= 0.75

    if joker.name in DECAYING_SCORE_JOKERS:
        if state.ante >= 7:
            factor *= 0.45
        elif state.ante >= 5:
            factor *= 0.68
        else:
            factor *= 0.90

    if joker.name in FINITE_SCORE_JOKERS:
        remaining = _joker_remaining_count(joker)
        if remaining is not None:
            factor *= min(1.0, max(0.35, remaining / 8.0))

    if joker.name in ROUND_RESET_SCORE_JOKERS:
        if state.ante >= 7:
            factor *= 0.62
        elif state.ante >= 5:
            factor *= 0.78

    if joker.name == "Gros Michel":
        if state.ante >= 7:
            factor *= 0.72
        elif state.ante >= 5:
            factor *= 0.88

    return max(0.0, min(1.0, factor))


def _build_archetype(state: GameState, preferred: HandType | None) -> str:
    if preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        return "flush"
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return "straight"
    if preferred in {
        HandType.PAIR,
        HandType.TWO_PAIR,
        HandType.THREE_OF_A_KIND,
        HandType.FULL_HOUSE,
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
    }:
        return "rank"
    if any(joker.name in FLEX_SCALING_JOKERS for joker in state.jokers):
        return "flex_scaling"
    if not state.jokers:
        return "open"
    return "flex"


def _owned_role_value(state: GameState, joker: Joker, *, remove_index: int) -> float:
    roles = _joker_roles(joker)
    if not roles:
        return 0.0

    remaining_roles = [
        _joker_roles(existing)
        for index, existing in enumerate(state.jokers)
        if index != remove_index
    ]
    value = 0.0
    for role in ("chips", "mult", "xmult", "scaling", "economy"):
        if role in roles and not any(role in existing_roles for existing_roles in remaining_roles):
            value += ROLE_UNIQUE_VALUES[role]
    return value


def _reroll_role_hunt_bonus(profile: _BuildProfile, pressure: _ShopPressure) -> float:
    if not profile.rich and pressure.ratio < 1.05:
        return 0.0
    bonus = 0.0
    if not profile.has_xmult and (profile.late or pressure.ratio >= 1.0):
        bonus += 16 * max(0.35, profile.role_deficit_ratio("xmult"))
    if not profile.has_scaling and (profile.late or pressure.ratio >= 1.0):
        bonus += 10 * max(0.35, profile.role_deficit_ratio("scaling"))
    if not profile.has_scaling and pressure.ratio >= 1.15:
        bonus += 12 * max(0.35, profile.role_deficit_ratio("scaling"))
    if profile.open_joker_slots >= 2 and ("chips" in profile.missing_roles or "mult" in profile.missing_roles):
        bonus += 8
    return bonus


def _urgent_late_role_hunt(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> bool:
    if not profile.late or _has_money_scaling_joker(state):
        return False
    if not ({"xmult", "scaling"} & set(profile.missing_roles)):
        return False
    return state.money >= 75 or pressure.ratio >= 0.85 or pressure.raw_ratio >= 0.95
