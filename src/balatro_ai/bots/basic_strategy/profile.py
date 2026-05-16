"""Build profile and shop pressure data models for BasicStrategyBot."""

from __future__ import annotations

from dataclasses import dataclass

from balatro_ai.rules.hand_evaluator import HandType


CRITICAL_BUILD_ROLES = {"chips", "mult", "xmult", "scaling"}


@dataclass(frozen=True, slots=True)
class _ShopPressure:
    target_score: float
    build_capacity: float
    ratio: float
    raw_ratio: float
    safety_multiplier: float
    capacity_safety_factor: float
    boss_name: str | None = None
    boss_target_multiplier: float = 1.0
    boss_capacity_factor: float = 1.0

    @property
    def danger(self) -> float:
        return max(0.0, min(2.0, self.ratio - 1.0))

    @property
    def safe_margin(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.ratio))


@dataclass(frozen=True, slots=True)
class _BuildProfile:
    preferred_hand: HandType | None
    archetype: str
    chip_score: float
    mult_score: float
    xmult_score: float
    scaling_score: float
    economy_score: float
    has_chips: bool
    has_mult: bool
    has_xmult: bool
    has_scaling: bool
    has_economy: bool
    open_joker_slots: int
    money: int
    spendable_money: int
    ante: int

    @property
    def missing_roles(self) -> tuple[str, ...]:
        roles: list[str] = []
        if not self.has_chips:
            roles.append("chips")
        if not self.has_mult:
            roles.append("mult")
        if not self.has_xmult:
            roles.append("xmult")
        if not self.has_scaling:
            roles.append("scaling")
        if not self.has_economy:
            roles.append("economy")
        return tuple(roles)

    @property
    def rich(self) -> bool:
        return self.spendable_money >= 20

    @property
    def late(self) -> bool:
        return self.ante >= 5

    def role_score(self, role: str) -> float:
        return {
            "chips": self.chip_score,
            "mult": self.mult_score,
            "xmult": self.xmult_score,
            "scaling": self.scaling_score,
            "economy": self.economy_score,
        }.get(role, 0.0)

    def role_requirement(self, role: str) -> float:
        return _role_requirement(role, self.ante)

    def role_deficit_ratio(self, role: str) -> float:
        requirement = self.role_requirement(role)
        if requirement <= 0:
            return 0.0
        return max(0.0, min(1.0, (requirement - self.role_score(role)) / requirement))


@dataclass(frozen=True, slots=True)
class _ShopContext:
    rerolls_in_shop: int = 0
    packs_opened_in_shop: int = 0
    filled_last_joker_slot: bool = False


def _pressure_payload(pressure: _ShopPressure) -> dict[str, object]:
    payload: dict[str, float | str | None] = {
        "target_score": round(pressure.target_score, 2),
        "build_capacity": round(pressure.build_capacity, 2),
        "ratio": round(pressure.ratio, 4),
        "raw_ratio": round(pressure.raw_ratio, 4),
        "danger": round(pressure.danger, 4),
        "safe_margin": round(pressure.safe_margin, 4),
        "safety_multiplier": round(pressure.safety_multiplier, 3),
        "capacity_safety_factor": round(pressure.capacity_safety_factor, 3),
        "boss_name": pressure.boss_name,
        "boss_target_multiplier": round(pressure.boss_target_multiplier, 3),
        "boss_capacity_factor": round(pressure.boss_capacity_factor, 3),
    }
    return payload


def _build_profile_payload(profile: _BuildProfile) -> dict[str, object]:
    return {
        "preferred_hand": profile.preferred_hand.value if profile.preferred_hand is not None else None,
        "archetype": profile.archetype,
        "role_scores": {
            "chips": round(profile.chip_score, 2),
            "mult": round(profile.mult_score, 2),
            "xmult": round(profile.xmult_score, 2),
            "scaling": round(profile.scaling_score, 2),
            "economy": round(profile.economy_score, 2),
        },
        "role_requirements": {
            "chips": round(profile.role_requirement("chips"), 2),
            "mult": round(profile.role_requirement("mult"), 2),
            "xmult": round(profile.role_requirement("xmult"), 2),
            "scaling": round(profile.role_requirement("scaling"), 2),
            "economy": round(profile.role_requirement("economy"), 2),
        },
        "has_chips": profile.has_chips,
        "has_mult": profile.has_mult,
        "has_xmult": profile.has_xmult,
        "has_scaling": profile.has_scaling,
        "has_economy": profile.has_economy,
        "missing_roles": list(profile.missing_roles),
        "open_joker_slots": profile.open_joker_slots,
        "money": profile.money,
        "spendable_money": profile.spendable_money,
        "ante": profile.ante,
    }


def _role_requirement(role: str, ante: int) -> float:
    if role == "chips":
        if ante <= 2:
            return 28.0
        if ante <= 4:
            return 55.0
        return 85.0
    if role == "mult":
        if ante <= 2:
            return 8.0
        if ante <= 4:
            return 16.0
        return 24.0
    if role == "xmult":
        return 26.0 if ante <= 4 else 34.0
    if role == "scaling":
        return 22.0 if ante <= 4 else 30.0
    if role == "economy":
        return 18.0
    return 0.0
