"""Future blind and boss-pressure forecast helpers."""

from __future__ import annotations

from dataclasses import replace

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.cards import _normalize_suit
from balatro_ai.bots.basic_strategy.data import (
    ANTE_SMALL_BLIND_SCORES,
    FINAL_BOSS_BLINDS,
    FINAL_BOSS_FRAGILE_JOKERS,
    FLUSH_ARCHETYPE_HANDS,
    ORDER_SENSITIVE_JOKERS,
    PREFERRED_HAND_HUNT_TYPES,
)
from balatro_ai.bots.basic_strategy.hand_preferences import _dominant_suit, _preferred_hand_type
from balatro_ai.bots.basic_strategy.jokers import _active_joker_names
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind


def _shop_pressure_boss_capacity_factor(state: GameState, boss_name: str | None) -> float:
    if _shop_pressure_uses_exact_needle_hand(state, boss_name):
        return 1.0
    return _weighted_boss_capacity_factor(state, boss_name) * _weighted_final_boss_fragility_factor(state, boss_name)


def _shop_pressure_effective_hands(state: GameState, boss_name: str | None) -> float:
    if _shop_pressure_uses_exact_needle_hand(state, boss_name):
        return 1.0
    hands = float(state.hands_remaining or 4)
    return max(2.0, min(4.0, hands) - 0.5)


def _shop_pressure_score_state(state: GameState, boss_name: str | None, *, raw_target: float) -> GameState:
    if not _shop_pressure_uses_boss_score_state(state, boss_name):
        return state
    return replace(
        state,
        blind=str(boss_name),
        required_score=int(raw_target),
        hands_remaining=_shop_pressure_boss_hands_remaining(state, boss_name),
        discards_remaining=_shop_pressure_boss_discards_remaining(state, boss_name),
    )


def _shop_pressure_uses_boss_score_state(state: GameState, boss_name: str | None) -> bool:
    return bool(boss_name) and _shop_cleared_blind_kind(state) == "BIG"


def _shop_pressure_boss_hands_remaining(state: GameState, boss_name: str | None) -> int:
    if boss_name == "The Needle":
        return 1
    return state.hands_remaining


def _shop_pressure_boss_discards_remaining(state: GameState, boss_name: str | None) -> int:
    if boss_name == "The Water":
        return 0
    return state.discards_remaining


def _shop_pressure_uses_exact_needle_hand(state: GameState, boss_name: str | None) -> bool:
    return boss_name == "The Needle" and _shop_cleared_blind_kind(state) == "BIG"


def _has_planet_investment(state: GameState) -> bool:
    return any(level > 1 for level in state.hand_levels.values())


def _estimated_next_required_score(state: GameState) -> float:
    ante = max(1, state.ante)
    small = ANTE_SMALL_BLIND_SCORES.get(ante, _extrapolated_small_blind_score(ante))
    next_small = ANTE_SMALL_BLIND_SCORES.get(ante + 1, _extrapolated_small_blind_score(ante + 1))
    cleared_kind = _shop_cleared_blind_kind(state)

    if state.blind == "Small Blind" and cleared_kind != "BIG":
        return small * 1.5
    if state.blind == "Big Blind" or cleared_kind == "BIG":
        boss_base = small * 2.0
        boss_score = _upcoming_boss_score(state)
        if cleared_kind == "BIG" and boss_score > 0:
            return boss_score
        if boss_score > 0:
            return max(boss_base, boss_score)
        boss_name = _upcoming_boss_blind_name(state)
        return boss_base * _boss_score_target_multiplier(boss_name)
    if state.required_score > 0 and state.blind:
        return max(next_small, state.required_score * 1.25)
    if state.required_score > 0:
        return state.required_score * 1.5
    return small


def _estimated_shop_planning_required_score(state: GameState) -> float:
    next_required = _estimated_next_required_score(state)
    final_weight = _final_win_target_weight(state)
    if final_weight <= 0.0:
        return next_required
    final_required = _estimated_final_win_required_score(state) * final_weight
    return max(next_required, final_required)


def _estimated_final_win_required_score(state: GameState) -> float:
    final_small = ANTE_SMALL_BLIND_SCORES.get(8, _extrapolated_small_blind_score(8))
    target = final_small * 2.0
    if state.ante >= 8:
        boss_score = _upcoming_boss_score(state)
        if boss_score > 0:
            target = max(target, boss_score)
        elif state.blind == "Big Blind" or _shop_cleared_blind_kind(state) == "BIG":
            target *= _boss_score_target_multiplier(_upcoming_boss_blind_name(state))
    return target


def _final_win_target_weight(state: GameState) -> float:
    if state.ante >= 8:
        return 1.0
    cleared_kind = _shop_cleared_blind_kind(state)
    if state.ante == 7:
        if cleared_kind == "BIG" or state.blind == "Big Blind":
            return 1.0
        if cleared_kind == "SMALL" or state.blind == "Small Blind":
            return 0.75
        return 0.65
    return 0.0


def _upcoming_boss_blind_name(state: GameState) -> str | None:
    direct_keys = ("upcoming_boss", "next_boss", "boss_blind", "boss")
    for key in direct_keys:
        name = _blind_name_from_mapping(state.modifiers.get(key))
        if name:
            return name

    blinds = state.modifiers.get("blinds")
    if isinstance(blinds, dict):
        for key in ("boss", "Boss", "boss_blind"):
            name = _blind_name_from_mapping(blinds.get(key))
            if name:
                return name
        for value in blinds.values():
            if not isinstance(value, dict):
                continue
            blind_type = str(value.get("type", value.get("kind", ""))).upper()
            if blind_type == "BOSS":
                name = _blind_name_from_mapping(value)
                if name:
                    return name
    return None


def _upcoming_boss_score(state: GameState) -> float:
    direct_keys = ("upcoming_boss", "next_boss", "boss_blind", "boss")
    for key in direct_keys:
        score = _blind_score_from_mapping(state.modifiers.get(key))
        if score > 0:
            return score

    blinds = state.modifiers.get("blinds")
    if isinstance(blinds, dict):
        for key in ("boss", "Boss", "boss_blind"):
            score = _blind_score_from_mapping(blinds.get(key))
            if score > 0:
                return score
        for value in blinds.values():
            if not isinstance(value, dict):
                continue
            blind_type = str(value.get("type", value.get("kind", ""))).upper()
            if blind_type == "BOSS":
                score = _blind_score_from_mapping(value)
                if score > 0:
                    return score
    return 0.0


def _blind_name_from_mapping(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if not isinstance(value, dict):
        return None
    name = str(value.get("name", value.get("label", "")))
    return name or None


def _blind_score_from_mapping(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    for key in ("score", "required_score", "score_required", "chips"):
        try:
            raw = value.get(key)
            if raw is not None:
                return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _effective_boss_target_multiplier(state: GameState, boss_name: str | None) -> float:
    if (state.blind != "Big Blind" and _shop_cleared_blind_kind(state) != "BIG") or _upcoming_boss_score(state) > 0:
        return 1.0
    return _boss_score_target_multiplier(boss_name)


def _boss_score_target_multiplier(boss_name: str | None) -> float:
    if boss_name == "The Wall":
        return 2.0
    if boss_name == "Violet Vessel":
        return 3.0
    return 1.0


def _boss_target_safety_bonus(boss_name: str) -> float:
    if boss_name in {"Violet Vessel", "Verdant Leaf", "Crimson Heart"}:
        return 0.18
    if boss_name in {"The Wall", "The Needle", "Amber Acorn", "Cerulean Bell"}:
        return 0.14
    if boss_name in {"The Eye", "The Mouth", "The Pillar", "The Psychic", "The Flint"}:
        return 0.10
    if boss_name in {"The Tooth", "The Water", "The Arm", "The Manacle"}:
        return 0.07
    if boss_name in {"The Club", "The Goad", "The Head", "The Window"}:
        return 0.06
    return 0.04


def _weighted_boss_capacity_factor(state: GameState, boss_name: str | None) -> float:
    if boss_name is None:
        return 1.0
    weight = _boss_preview_weight(state)
    if weight <= 0:
        return 1.0
    factor = _boss_capacity_factor(state, boss_name)
    return 1.0 - ((1.0 - factor) * weight)


def _boss_preview_weight(state: GameState) -> float:
    cleared_kind = _shop_cleared_blind_kind(state)
    if cleared_kind == "BIG":
        return 1.0
    if cleared_kind == "SMALL":
        return 0.45
    if state.blind == "Big Blind":
        return 1.0
    if state.blind == "Small Blind":
        return 0.45
    return 0.0


def _shop_has_followup_big_blind_shop(state: GameState) -> bool:
    cleared_kind = _shop_cleared_blind_kind(state)
    return cleared_kind == "SMALL" or (state.blind == "Small Blind" and cleared_kind != "BIG")


def _shop_cleared_blind_kind(state: GameState) -> str:
    raw = state.modifiers.get("cleared_blind")
    if not isinstance(raw, dict):
        return ""
    kind = str(raw.get("kind", "")).upper()
    return kind if kind in {"SMALL", "BIG", "BOSS"} else ""


def _boss_capacity_factor(state: GameState, boss_name: str) -> float:
    if boss_name == "Verdant Leaf":
        return 0.58
    if boss_name == "Crimson Heart":
        return 0.60
    if boss_name == "Amber Acorn":
        return 0.72
    if boss_name == "Cerulean Bell":
        return 0.72
    if boss_name == "The Flint":
        return 0.66
    if boss_name == "The Needle":
        return 0.34
    if boss_name in {"The Eye", "The Mouth"}:
        return 0.8
    if boss_name == "The Pillar":
        return 0.82
    if boss_name == "The Psychic":
        return 0.86
    if boss_name in {"The Water", "The Manacle"}:
        return 0.88
    if boss_name == "The Arm":
        return 0.9
    if boss_name == "The Tooth":
        return 0.95
    if boss_name in {"The Club", "The Goad", "The Head", "The Window"}:
        return _suit_boss_capacity_factor(state, boss_name)
    return 1.0


def _suit_boss_capacity_factor(state: GameState, boss_name: str) -> float:
    debuffed = debuffed_suits_for_blind(boss_name)
    if not debuffed:
        return 1.0
    dominant = _dominant_suit(state)
    if dominant is not None and _normalize_suit(dominant) in debuffed:
        return 0.76
    if _preferred_hand_type(state) in FLUSH_ARCHETYPE_HANDS:
        return 0.86
    return 0.92


def _weighted_final_boss_fragility_factor(state: GameState, boss_name: str | None) -> float:
    if boss_name not in FINAL_BOSS_BLINDS:
        return 1.0
    factor = _final_boss_fragility_factor(state, boss_name)
    weight = _boss_preview_weight(state)
    return 1.0 - ((1.0 - factor) * weight)


def _final_boss_fragility_factor(state: GameState, boss_name: str | None) -> float:
    names = _active_joker_names(state)
    if boss_name == "Crimson Heart":
        factor = 1.0
        if len(state.jokers) >= 4:
            factor *= 0.84
        if names & FINAL_BOSS_FRAGILE_JOKERS:
            factor *= 0.88
        return factor
    if boss_name == "Amber Acorn" and names & ORDER_SENSITIVE_JOKERS:
        return 0.90
    if boss_name == "Cerulean Bell" and _preferred_hand_type(state) in PREFERRED_HAND_HUNT_TYPES:
        return 0.92
    return 1.0


def _extrapolated_small_blind_score(ante: int) -> float:
    if ante <= 8:
        return ANTE_SMALL_BLIND_SCORES.get(ante, 300)
    return 50000 * (1.6 ** (ante - 8))
