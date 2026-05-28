"""Run-level strategy plan derived from the current state.

The build profile describes what the bot has right now. The run plan describes
what the bot is trying to become, and how aggressively the current shop should
turn money into that plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.data import (
    DANGEROUS_BOSS_BLINDS,
    FINAL_BOSS_BLINDS,
    FLEX_SCALING_JOKERS,
    MONEY_SCALING_RESERVE_TARGETS,
    RARE_HAND_TYPES,
)
from balatro_ai.bots.basic_strategy.jokers import _active_joker_names
from balatro_ai.bots.basic_strategy.profile import _BuildProfile, _ShopPressure
from balatro_ai.bots.basic_strategy.shop_forecast import _has_planet_investment, _shop_has_followup_big_blind_shop
from balatro_ai.rules.hand_evaluator import HandType


_FLUSH_HANDS = frozenset({HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE})
_STRAIGHT_HANDS = frozenset({HandType.STRAIGHT, HandType.STRAIGHT_FLUSH})
_RANK_HANDS = frozenset(
    {
        HandType.PAIR,
        HandType.TWO_PAIR,
        HandType.THREE_OF_A_KIND,
        HandType.FULL_HOUSE,
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }
)
@dataclass(frozen=True, slots=True)
class _RunPlan:
    archetype: str
    commitment: str
    primary_hands: tuple[HandType, ...]
    missing_roles: tuple[str, ...]
    shop_posture: str
    reroll_budget: int
    desired_score_buffer: float
    pack_priorities: tuple[str, ...]
    voucher_priorities: tuple[str, ...]
    spectral_permissions: tuple[str, ...]
    reasons: tuple[str, ...] = ()

    def prioritizes_pack(self, pack_kind: str) -> bool:
        return pack_kind.lower() in self.pack_priorities

    def prioritizes_voucher(self, voucher_name: str) -> bool:
        return voucher_name in self.voucher_priorities

    def permits_spectral(self, spectral_name: str) -> bool:
        return spectral_name in self.spectral_permissions


def _run_plan(
    state: GameState,
    profile: _BuildProfile,
    pressure: _ShopPressure,
) -> _RunPlan:
    primary_hands = _primary_hands(state, profile)
    archetype = _plan_archetype(state, profile, primary_hands)
    missing_roles = profile.missing_roles
    commitment, commitment_reasons = _plan_commitment(state, profile, pressure, primary_hands)
    posture, posture_reasons = _shop_posture(state, profile, pressure, commitment)
    reroll_budget = _plan_reroll_budget(state, profile, pressure, posture, commitment)
    desired_score_buffer = _desired_score_buffer(state, pressure, commitment, posture)
    pack_priorities = _pack_priorities(state, profile, pressure, archetype, primary_hands, posture)
    spectral_permissions = _spectral_permissions(state, profile, archetype, primary_hands)
    voucher_priorities = _voucher_priorities(state, profile, pressure, archetype, primary_hands, posture)
    return _RunPlan(
        archetype=archetype,
        commitment=commitment,
        primary_hands=primary_hands,
        missing_roles=missing_roles,
        shop_posture=posture,
        reroll_budget=reroll_budget,
        desired_score_buffer=desired_score_buffer,
        pack_priorities=pack_priorities,
        voucher_priorities=voucher_priorities,
        spectral_permissions=spectral_permissions,
        reasons=tuple((*commitment_reasons, *posture_reasons)),
    )


def _run_plan_payload(plan: _RunPlan) -> dict[str, object]:
    return {
        "archetype": plan.archetype,
        "commitment": plan.commitment,
        "primary_hands": [hand_type.value for hand_type in plan.primary_hands],
        "missing_roles": list(plan.missing_roles),
        "shop_posture": plan.shop_posture,
        "reroll_budget": plan.reroll_budget,
        "desired_score_buffer": round(plan.desired_score_buffer, 3),
        "pack_priorities": list(plan.pack_priorities),
        "voucher_priorities": list(plan.voucher_priorities),
        "spectral_permissions": list(plan.spectral_permissions),
        "reasons": list(plan.reasons),
    }


def _primary_hands(state: GameState, profile: _BuildProfile) -> tuple[HandType, ...]:
    ordered: list[HandType] = []
    if profile.preferred_hand is not None:
        ordered.append(profile.preferred_hand)

    level_votes: list[tuple[int, HandType]] = []
    for name, level in state.hand_levels.items():
        try:
            hand_type = HandType(str(name))
        except ValueError:
            continue
        try:
            numeric_level = int(level)
        except (TypeError, ValueError):
            continue
        if numeric_level > 1:
            level_votes.append((numeric_level, hand_type))

    for _level, hand_type in sorted(level_votes, key=lambda item: item[0], reverse=True):
        if hand_type not in ordered:
            ordered.append(hand_type)

    if not ordered:
        if profile.archetype == "flush":
            ordered.append(HandType.FLUSH)
        elif profile.archetype == "straight":
            ordered.append(HandType.STRAIGHT)
        elif profile.archetype == "rank":
            ordered.append(HandType.FULL_HOUSE)

    return tuple(ordered[:3])


def _plan_archetype(state: GameState, profile: _BuildProfile, primary_hands: tuple[HandType, ...]) -> str:
    names = _active_joker_names(state)
    if names & set(MONEY_SCALING_RESERVE_TARGETS):
        return "money_scaling"
    if names & FLEX_SCALING_JOKERS:
        return "joker_scaling"
    if any(hand_type in _FLUSH_HANDS for hand_type in primary_hands):
        return "flush"
    if any(hand_type in _STRAIGHT_HANDS for hand_type in primary_hands):
        return "straight"
    if any(hand_type in _RANK_HANDS for hand_type in primary_hands):
        return "rank"
    return profile.archetype


def _plan_commitment(
    state: GameState,
    profile: _BuildProfile,
    pressure: _ShopPressure,
    primary_hands: tuple[HandType, ...],
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    missing_critical = _missing_critical(profile)
    has_hand_investment = _has_hand_investment(state, primary_hands)
    has_score_engine = profile.has_chips and profile.has_mult

    if (pressure.ratio >= 2.0 or pressure.raw_ratio >= 1.35) and missing_critical:
        reasons.append("severe_pressure_missing_critical")
        return "desperate", tuple(reasons)
    if state.ante >= 5 and primary_hands and (has_hand_investment or profile.has_xmult or profile.has_scaling):
        reasons.append("late_primary_plan")
        return "committed", tuple(reasons)
    if primary_hands and (has_hand_investment or profile.has_scaling or profile.has_xmult):
        reasons.append("supported_primary_hand")
        return "committed", tuple(reasons)
    if primary_hands or has_score_engine or state.ante >= 3:
        reasons.append("early_direction")
        return "leaning", tuple(reasons)
    reasons.append("no_stable_direction")
    return "exploratory", tuple(reasons)


def _shop_posture(
    state: GameState,
    profile: _BuildProfile,
    pressure: _ShopPressure,
    commitment: str,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    missing_critical = _missing_critical(profile)

    if (pressure.ratio >= 2.0 or pressure.raw_ratio >= 1.35) and missing_critical:
        reasons.append("panic_pressure")
        return "panic", tuple(reasons)
    if pressure.ratio >= 1.05 or pressure.raw_ratio >= 1.0:
        reasons.append("pressure_spend")
        return "spend", tuple(reasons)
    if state.ante >= 5 and missing_critical and profile.spendable_money >= 15:
        reasons.append("late_missing_critical")
        return "spend", tuple(reasons)
    if state.ante <= 4 and (not profile.has_economy or not profile.has_chips or not profile.has_mult):
        reasons.append("early_engine_build")
        return "build", tuple(reasons)
    if commitment in {"exploratory", "leaning"} and state.ante <= 4:
        reasons.append("forming_plan")
        return "build", tuple(reasons)
    if pressure.ratio < 0.75 and profile.has_economy and not missing_critical:
        reasons.append("safe_complete_build")
        return "conserve", tuple(reasons)
    reasons.append("default_build")
    return "build", tuple(reasons)


def _plan_reroll_budget(
    state: GameState,
    profile: _BuildProfile,
    pressure: _ShopPressure,
    posture: str,
    commitment: str,
) -> int:
    missing_critical = _missing_critical(profile)
    last_shop_before_boss = not _shop_has_followup_big_blind_shop(state)

    if state.ante <= 1:
        return 2 if (not profile.has_chips or not profile.has_mult) else 1
    if state.ante <= 2:
        if posture in {"spend", "panic"} or missing_critical:
            return 2
        return 1
    if state.ante <= 4:
        if posture == "panic":
            return 4
        if posture == "spend" or missing_critical:
            return 3 if last_shop_before_boss else 2
        return 1

    if posture == "conserve":
        return 0
    if posture == "build":
        return 1 if not missing_critical else 2
    if posture == "spend":
        budget = 3 if profile.rich else 2
        if last_shop_before_boss:
            budget += 1
        if commitment == "committed":
            budget += 1
        return min(5, budget)
    if posture == "panic":
        if pressure.ratio >= 3.0 or pressure.raw_ratio >= 1.75:
            return 8 if profile.spendable_money >= 30 else 6
        return 5 if last_shop_before_boss else 4
    return 1


def _desired_score_buffer(
    state: GameState,
    pressure: _ShopPressure,
    commitment: str,
    posture: str,
) -> float:
    if posture == "panic":
        return 1.0
    base = 1.12 if commitment == "exploratory" else 1.22 if commitment == "leaning" else 1.35
    if state.ante >= 5:
        base += 0.12
    if pressure.boss_name in FINAL_BOSS_BLINDS:
        base += 0.10
    return min(1.65, base)


def _pack_priorities(
    state: GameState,
    profile: _BuildProfile,
    pressure: _ShopPressure,
    archetype: str,
    primary_hands: tuple[HandType, ...],
    posture: str,
) -> tuple[str, ...]:
    priorities: list[str] = []
    missing = set(profile.missing_roles)
    names = _active_joker_names(state)

    if profile.open_joker_slots > 0 and (missing & {"chips", "mult", "xmult", "scaling"}):
        priorities.append("buffoon")
    if primary_hands and (not profile.has_scaling or "Constellation" in names or _has_planet_investment(state)):
        priorities.append("celestial")
    if archetype in {"flush", "rank", "joker_scaling"} or primary_hands:
        priorities.append("arcana")
    if _standard_pack_payoff_names(names) or any(hand_type in RARE_HAND_TYPES for hand_type in primary_hands):
        priorities.append("standard")
    if state.ante <= 2 or posture in {"spend", "panic"} or archetype in {"flush", "rank"}:
        priorities.append("spectral")
    if pressure.ratio >= 1.05 and "buffoon" not in priorities and profile.open_joker_slots > 0:
        priorities.insert(0, "buffoon")
    return _dedupe(priorities)


def _voucher_priorities(
    state: GameState,
    profile: _BuildProfile,
    pressure: _ShopPressure,
    archetype: str,
    primary_hands: tuple[HandType, ...],
    posture: str,
) -> tuple[str, ...]:
    priorities: list[str] = []
    names = _active_joker_names(state)
    dangerous_boss = bool(pressure.boss_name in DANGEROUS_BOSS_BLINDS or pressure.boss_name in FINAL_BOSS_BLINDS)
    boss_control_target = pressure.boss_name == "Violet Vessel"

    if boss_control_target and dangerous_boss and (posture in {"spend", "panic"} or state.money >= 25):
        if "Director's Cut" in state.vouchers or "Retcon" in state.vouchers:
            priorities.append("Retcon")
        elif state.ante >= 5 or state.money >= 30:
            priorities.append("Director's Cut")
    if "Constellation" in names or (primary_hands and _has_planet_investment(state) and state.ante <= 5):
        priorities.append("Planet Tycoon")
    if _standard_pack_payoff_names(names):
        priorities.append("Illusion")
    if archetype in {"flush", "rank", "joker_scaling"}:
        priorities.append("Tarot Merchant")
    return _dedupe(priorities)


def _spectral_permissions(
    state: GameState,
    profile: _BuildProfile,
    archetype: str,
    primary_hands: tuple[HandType, ...],
) -> tuple[str, ...]:
    permissions = [
        "Talisman",
        "Aura",
        "Deja Vu",
        "Trance",
        "Medium",
        "Cryptid",
        "The Soul",
        "Black Hole",
        "Immolate",
    ]
    hand_size = _observed_hand_size(state)
    names = _active_joker_names(state)
    if archetype == "flush" or any(hand_type in _FLUSH_HANDS for hand_type in primary_hands):
        permissions.append("Sigil")
    if archetype == "rank" or any(hand_type in _RANK_HANDS for hand_type in primary_hands):
        permissions.append("Ouija")
    if hand_size >= 7 and state.jokers:
        permissions.append("Ectoplasm")
    if _has_single_dominant_joker(profile, state):
        permissions.extend(("Ankh", "Hex"))
    if state.ante <= 3 or names & {"Hologram", "Vampire", "Midas Mask"}:
        permissions.extend(("Familiar", "Grim", "Incantation"))
    return _dedupe(permissions)


def _missing_critical(profile: _BuildProfile) -> set[str]:
    return set(profile.missing_roles) & {"chips", "mult", "xmult", "scaling"}


def _has_hand_investment(state: GameState, primary_hands: tuple[HandType, ...]) -> bool:
    if _has_planet_investment(state):
        return True
    for hand_type in primary_hands:
        try:
            if int(state.hand_levels.get(hand_type.value, 1)) > 1:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _standard_pack_payoff_names(joker_names: set[str]) -> bool:
    return bool(joker_names & {"Hologram", "Vampire", "Midas Mask", "DNA", "Lucky Cat"})


def _observed_hand_size(state: GameState) -> int:
    if state.hand:
        return len(state.hand)
    raw = state.modifiers.get("hand_size", state.modifiers.get("current_hand_size", 8))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 8


def _has_single_dominant_joker(profile: _BuildProfile, state: GameState) -> bool:
    if len(state.jokers) == 1:
        return True
    names = _active_joker_names(state)
    if not names:
        return False
    if names & {"Cavendish", "Hologram", "Constellation", "Vampire", "Baron", "Campfire", "Lucky Cat"}:
        return True
    return profile.xmult_score >= 42.0 and profile.xmult_score >= profile.mult_score + profile.chip_score


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)
