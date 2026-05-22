"""Joker metadata, effect-text, and lightweight state helpers."""

from __future__ import annotations

import re

from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.bots.basic_strategy.cards import _edition_chips_value, _edition_mult_value, _edition_xmult_value
from balatro_ai.bots.basic_strategy.data import (
    CHIP_JOKERS,
    JOKER_ECONOMY_VALUES,
    JOKER_SCALING_VALUES,
    MULT_JOKERS,
    SCALING_JOKERS,
    XMULT_JOKERS,
)


def _truthy_modifier(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "nil"}
    return bool(value)


def _joker_with_current_xmult(joker: Joker, xmult: float) -> Joker:
    metadata = dict(joker.metadata)
    effect = _joker_effect_text(joker)
    replacement = f"Currently X{_format_xmult(xmult)}"
    if effect:
        effect = re.sub(
            r"currently\s+x\s*[0-9]+(?:\.[0-9]+)?",
            replacement,
            effect,
            flags=re.IGNORECASE,
        )
        if replacement not in effect:
            effect = f"{effect} ({replacement})"
    else:
        effect = replacement

    value = metadata.get("value")
    if isinstance(value, dict):
        value = dict(value)
        value["effect"] = effect
        metadata["value"] = value
    else:
        metadata["effect"] = effect
    return Joker(joker.name, edition=joker.edition, sell_value=joker.sell_value, metadata=metadata)


def _joker_with_added_current_plus(joker: Joker, amount: int, *, suffix: str) -> Joker:
    metadata = dict(joker.metadata)
    effect = _joker_effect_text(joker)
    current = _joker_current_plus_value(joker, suffix=suffix)
    new_value = max(0, current + amount)
    sign = "+" if new_value >= 0 else ""
    replacement = f"Currently {sign}{new_value} {suffix.title()}"
    if effect:
        effect = re.sub(
            rf"currently\s+[+-]?\d+(?:\s+{re.escape(suffix)})?",
            replacement,
            effect,
            flags=re.IGNORECASE,
        )
        if replacement not in effect:
            effect = f"{effect} ({replacement})"
    else:
        effect = replacement

    value = metadata.get("value")
    if isinstance(value, dict):
        value = dict(value)
        value["effect"] = effect
        metadata["value"] = value
    else:
        metadata["effect"] = effect
    return Joker(joker.name, edition=joker.edition, sell_value=joker.sell_value, metadata=metadata)


def _joker_with_added_current_xmult(joker: Joker, amount: float, *, minimum: float = 1.0) -> Joker:
    current = _joker_current_xmult_value(joker)
    return _joker_with_current_xmult(joker, max(minimum, current + amount))


def _joker_current_plus_value(joker: Joker, *, suffix: str) -> int:
    metadata_value = _joker_metadata_current_plus_value(joker, suffix=suffix)
    if metadata_value is not None:
        return metadata_value
    return joker.effect.current_plus_value(suffix)


def _joker_current_xmult_value(joker: Joker) -> float:
    metadata_value = _joker_metadata_current_xmult_value(joker)
    if metadata_value is not None:
        return metadata_value
    return joker.effect.current_xmult_visible if joker.effect.current_xmult_visible is not None else 1.0


def _joker_effect_text(joker: Joker) -> str:
    return joker.effect.text


def _format_xmult(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def _joker_is_disabled_for_build(joker: Joker) -> bool:
    if joker.effect.disabled:
        return True
    for source in _joker_metadata_sources(joker.metadata):
        state = source.get("state")
        if isinstance(state, dict) and _truthy_modifier(state.get("debuff")):
            return True
        if _truthy_modifier(source.get("debuff")) or _truthy_modifier(source.get("disabled")):
            return True
    return False


def _joker_has_sticker(joker: Joker, sticker: str) -> bool:
    wanted = sticker.lower()
    for source in _joker_metadata_sources(joker.metadata):
        for key in (wanted, sticker, f"is_{wanted}"):
            if _truthy_modifier(source.get(key)):
                return True
        stickers = source.get("stickers")
        if isinstance(stickers, dict):
            if _truthy_modifier(stickers.get(wanted)) or _truthy_modifier(stickers.get(sticker)):
                return True
        elif isinstance(stickers, list | tuple | set):
            if any(str(item).strip().lower() == wanted for item in stickers):
                return True
        elif isinstance(stickers, str):
            normalized = stickers.lower().replace("_", " ")
            if wanted in normalized:
                return True
    return False


def _joker_remaining_count(joker: Joker) -> int | None:
    value = _joker_metadata_value(
        joker,
        (
            "current_remaining",
            "remaining",
            "uses",
            "hands_remaining",
            "hands_left",
            "cards_remaining",
        ),
    )
    if value is None:
        value = joker.effect.remaining or joker.effect.cards_remaining or joker.effect.next_hands
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _joker_metadata_value(joker: Joker, keys: tuple[str, ...]) -> object | None:
    for source in _joker_metadata_sources(joker.metadata):
        for key in keys:
            if key in source:
                return source[key]
    return None


def _joker_metadata_sources(metadata: dict[str, object]) -> tuple[dict[str, object], ...]:
    sources: list[dict[str, object]] = [metadata]
    for key in ("ability", "config", "extra", "modifier", "stickers"):
        value = metadata.get(key)
        if isinstance(value, dict):
            sources.append(value)
            nested_extra = value.get("extra")
            if isinstance(nested_extra, dict):
                sources.append(nested_extra)
    return tuple(sources)


def _joker_metadata_numeric_value(joker: Joker, keys: tuple[str, ...]) -> object | None:
    sources: list[dict[str, object]] = [joker.metadata]
    for key in ("ability", "config", "extra", "value"):
        value = joker.metadata.get(key)
        if isinstance(value, dict):
            sources.append(value)
            nested = value.get("extra")
            if isinstance(nested, dict):
                sources.append(nested)
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
    return None


def _joker_metadata_current_plus_value(joker: Joker, *, suffix: str) -> int | None:
    keys = {
        "chips": ("current_chips", "chips"),
        "mult": ("current_mult", "mult"),
    }.get(suffix, (f"current_{suffix}", suffix))
    value = _joker_metadata_numeric_value(joker, keys)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _joker_metadata_current_xmult_value(joker: Joker) -> float | None:
    value = _joker_metadata_numeric_value(joker, ("current_xmult", "xmult", "current_x_mult", "x_mult"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _joker_metadata_int_value(joker: Joker, keys: tuple[str, ...], *, default: int) -> int:
    value = _joker_metadata_numeric_value(joker, keys)
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _driver_license_enhanced_count(state: GameState, joker: Joker | None = None) -> int:
    if joker is not None and joker.name == "Driver's License":
        visible_count = _joker_metadata_numeric_value(
            joker,
            ("driver_tally", "enhanced_count", "current_enhanced", "enhancement_tally"),
        )
        try:
            if visible_count is not None:
                return max(0, int(visible_count))
        except (TypeError, ValueError):
            pass
        if joker.effect.current_int is not None:
            return max(0, joker.effect.current_int)
    return sum(1 for card in _known_full_deck_cards(state) if _card_has_driver_license_enhancement(card))


def _driver_license_readiness_factor(state: GameState, joker: Joker) -> float:
    if joker.name != "Driver's License":
        return 1.0

    enhanced_count = _driver_license_enhanced_count(state, joker)
    if enhanced_count >= 16:
        return 1.0
    if enhanced_count <= 0:
        return 0.0
    if enhanced_count < 8:
        return min(0.08, enhanced_count * 0.01)
    if enhanced_count < 12:
        return 0.08 + ((enhanced_count - 8) * 0.04)
    return min(0.85, 0.35 + ((enhanced_count - 12) * 0.15))


def _known_full_deck_cards(state: GameState) -> tuple[Card, ...]:
    return state.hand + state.known_deck + _zone_cards(state, "played_pile") + _zone_cards(state, "discard_pile")


def _zone_cards(state: GameState, key: str) -> tuple[Card, ...]:
    raw = state.modifiers.get(key, ())
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(item if isinstance(item, Card) else Card.from_mapping(item) for item in raw)


def _card_has_driver_license_enhancement(card: Card) -> bool:
    enhancement = str(card.enhancement or "").strip().lower().replace("_", " ").replace("-", " ")
    return enhancement not in {"", "none", "base", "base card"}


def _joker_names(state: GameState) -> set[str]:
    return {joker.name for joker in state.jokers}


def _active_joker_names(state: GameState) -> set[str]:
    return {joker.name for joker in state.jokers if not _joker_is_disabled_for_build(joker)}


def _current_plus_for_joker(state: GameState, joker_name: str, *, suffix: str) -> int:
    for joker in state.jokers:
        if joker.name == joker_name and not joker.effect.disabled:
            return _joker_current_plus_value(joker, suffix=suffix)
    return 0


def _castle_target_suit(state: GameState) -> str | None:
    for joker in state.jokers:
        if joker.name != "Castle" or joker.effect.disabled:
            continue
        if joker.effect.discarded_suit or joker.effect.target_suit:
            return joker.effect.discarded_suit or joker.effect.target_suit
    return None


def _mail_in_rebate_rank(state: GameState) -> str | None:
    for joker in state.jokers:
        if joker.name != "Mail-In Rebate" or joker.effect.disabled:
            continue
        return joker.effect.discarded_rank
    return None


def _joker_sell_total(state: GameState) -> int:
    return sum(joker.sell_value or 0 for joker in state.jokers)


def _static_chip_role_score(name: str) -> float:
    if name == "Stuntman":
        return 240.0
    if name == "Bull":
        return 95.0
    if name == "Blue Joker":
        return 70.0
    if name in {"Banner", "Ice Cream", "Runner", "Square Joker", "Wee Joker", "Castle"}:
        return 55.0
    if name in CHIP_JOKERS:
        return 34.0
    return 0.0


def _static_mult_role_score(name: str) -> float:
    if name == "Joker":
        return 4.0
    if name in {"Gros Michel", "Popcorn", "Mystic Summit", "Half Joker"}:
        return 16.0
    if name in {"Abstract Joker", "Swashbuckler", "Bootstraps"}:
        return 14.0
    if name in {"Green Joker", "Ride the Bus", "Spare Trousers", "Red Card", "Flash Card"}:
        return 10.0
    if name in MULT_JOKERS:
        return 8.0
    return 0.0


def _static_xmult_role_score(name: str) -> float:
    if name == "Loyalty Card":
        return 0.0
    if name in {"Cavendish", "Joker Stencil"}:
        return 70.0
    if name in {"The Duo", "The Trio", "The Family", "The Order", "The Tribe", "Seeing Double", "Blackboard"}:
        return 42.0
    if name in {"Acrobat", "Card Sharp", "Driver's License", "Ancient Joker", "The Idol", "Flower Pot"}:
        return 36.0
    if name in XMULT_JOKERS:
        return 30.0
    return 0.0


def _joker_roles(joker: Joker) -> frozenset[str]:
    if _joker_is_disabled_for_build(joker):
        return frozenset()

    roles: set[str] = set()
    name = joker.name
    if name in CHIP_JOKERS:
        roles.add("chips")
    if name in MULT_JOKERS:
        roles.add("mult")
    if name in XMULT_JOKERS and name != "Loyalty Card":
        roles.add("xmult")
    if name == "Red Card":
        if _joker_current_plus_value(joker, suffix="mult") > 0:
            roles.add("mult")
            roles.add("scaling")
    elif name in JOKER_SCALING_VALUES or name in SCALING_JOKERS:
        roles.add("scaling")
    if name in JOKER_ECONOMY_VALUES:
        roles.add("economy")

    edition = (joker.edition or "").lower()
    if "foil" in edition:
        roles.add("chips")
    if "holo" in edition or "holographic" in edition:
        roles.add("mult")
    if "polychrome" in edition:
        roles.add("xmult")
    if "negative" in edition:
        roles.add("economy")
    return frozenset(roles)
