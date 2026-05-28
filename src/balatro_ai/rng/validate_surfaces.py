"""Validate high-level RNG surface fixtures.

This module compares opened booster-pack bridge fixtures from
``capture_surfaces.py`` against the offline predictors in ``surfaces.py``.
The tests use the same helpers, so fixture captures can be added incrementally
without requiring the bridge in CI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Mapping

from balatro_ai.rng.capture import DEFAULT_FIXTURE_DIR
from balatro_ai.rng.surfaces import SHOP_POOL_DATA_PATH, PredictedCard, predict_pack_contents


PLAYING_KEY_RE = re.compile(r"^[CDHS]_(?:[2-9TJQKA]|10)$")

SET_NAMES = {
    "JOKER": "Joker",
    "TAROT": "Tarot",
    "PLANET": "Planet",
    "SPECTRAL": "Spectral",
    "VOUCHER": "Voucher",
    "BOOSTER": "Booster",
    "DEFAULT": "Default",
    "ENHANCED": "Enhanced",
}

ENHANCEMENT_KEYS = {
    "BONUS": "m_bonus",
    "MULT": "m_mult",
    "WILD": "m_wild",
    "GLASS": "m_glass",
    "STEEL": "m_steel",
    "STONE": "m_stone",
    "GOLD": "m_gold",
    "LUCKY": "m_lucky",
}

ENHANCEMENT_LABELS = {
    "Bonus Card": "m_bonus",
    "Mult Card": "m_mult",
    "Wild Card": "m_wild",
    "Glass Card": "m_glass",
    "Steel Card": "m_steel",
    "Stone Card": "m_stone",
    "Gold Card": "m_gold",
    "Lucky Card": "m_lucky",
}

HAND_ORDER = (
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
)


@dataclass(frozen=True, slots=True)
class CardSignature:
    set: str
    key: str
    front_key: str | None = None
    edition: str | None = None
    seal: str | None = None

    def compact(self) -> str:
        parts = [self.set, self.key]
        if self.front_key is not None:
            parts.append(self.front_key)
        if self.edition is not None:
            parts.append("edition=" + self.edition)
        if self.seal is not None:
            parts.append("seal=" + self.seal)
        return ":".join(parts)


@dataclass(frozen=True, slots=True)
class SurfaceCheckResult:
    name: str
    status: str  # "ok", "mismatch", or "unsupported"
    detail: str = ""


def load_pack_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(fixture, dict) and not fixture.get("seed"):
        seed = _seed_from_pack_path(path)
        if seed:
            fixture = {**fixture, "seed": seed}
    return fixture


def iter_pack_fixture_paths(root: Path = DEFAULT_FIXTURE_DIR) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("pack_seed_*.json")))


def check_pack_fixture(fixture: Mapping[str, Any]) -> SurfaceCheckResult:
    seed = fixture_seed(fixture)
    pack_key = fixture_pack_key(fixture)
    pack_cards = fixture_pack_cards(fixture)
    if seed is None:
        return SurfaceCheckResult("pack_contents", "unsupported", "fixture has no seed")
    if pack_key is None:
        return SurfaceCheckResult("pack_contents", "unsupported", "fixture has no pack_key")
    if not pack_cards:
        return SurfaceCheckResult("pack_contents", "unsupported", "fixture has no opened pack cards")

    prediction_state = fixture_prediction_state(fixture)
    used_jokers, used_consumables = visible_used_keys(prediction_state)
    predicted = predict_pack_contents(
        seed,
        ante=fixture_ante(fixture),
        pack_key=pack_key,
        vouchers=fixture_owned_vouchers(fixture),
        played_hand_types=fixture_played_hand_types(fixture),
        used_jokers=used_jokers,
        used_consumables=used_consumables,
        telescope_planet_key=fixture_telescope_planet_key(fixture),
        edition_rate=fixture_edition_rate(fixture),
    )

    actual_sig = tuple(actual_card_signature(card) for card in pack_cards)
    predicted_sig = tuple(predicted_card_signature(card) for card in predicted)
    if actual_sig == predicted_sig:
        return SurfaceCheckResult("pack_contents", "ok", f"matched {len(actual_sig)} cards")
    return SurfaceCheckResult(
        "pack_contents",
        "mismatch",
        "predicted="
        + repr(tuple(sig.compact() for sig in predicted_sig))
        + " actual="
        + repr(tuple(sig.compact() for sig in actual_sig)),
    )


def fixture_seed(fixture: Mapping[str, Any]) -> str | None:
    raw = fixture.get("seed")
    if raw is None:
        opened = fixture_opened_state(fixture)
        raw = opened.get("seed")
    return str(raw) if raw is not None else None


def fixture_ante(fixture: Mapping[str, Any]) -> int:
    for source in (fixture, fixture_opened_state(fixture), fixture_shop_state(fixture)):
        for key in ("ante", "ante_num"):
            raw = source.get(key)
            try:
                if raw is not None:
                    return max(1, int(raw))
            except (TypeError, ValueError):
                pass
    return 1


def fixture_pack_key(fixture: Mapping[str, Any]) -> str | None:
    raw = fixture.get("pack_key")
    if raw:
        return str(raw)
    shop_state = fixture_shop_state(fixture)
    pack_index = fixture.get("pack_index", 0)
    try:
        index = int(pack_index)
    except (TypeError, ValueError):
        index = 0
    boosters = extract_area_cards(shop_state, "packs")
    if 0 <= index < len(boosters):
        key = boosters[index].get("key")
        return str(key) if key is not None else None
    return None


def fixture_pack_cards(fixture: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return extract_area_cards(fixture_opened_state(fixture), "pack")


def fixture_opened_state(fixture: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("opened_state", "opened", "post_state"):
        state = fixture.get(key)
        if isinstance(state, Mapping):
            return state
    return fixture


def fixture_shop_state(fixture: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("shop_state", "pre_shop_state", "before_state"):
        state = fixture.get(key)
        if isinstance(state, Mapping):
            return state
    return {}


def fixture_prediction_state(fixture: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the pre-open state whose visible cards constrain pack pools."""

    shop_state = fixture_shop_state(fixture)
    if shop_state:
        return shop_state
    return fixture_opened_state(fixture)


def fixture_owned_vouchers(fixture: Mapping[str, Any]) -> frozenset[str]:
    vouchers: set[str] = set()
    for source in (fixture_opened_state(fixture), fixture_shop_state(fixture)):
        for key in ("owned_vouchers", "used_vouchers"):
            _collect_vouchers(vouchers, source.get(key))
    _collect_vouchers(vouchers, fixture.get("vouchers"))
    return frozenset(vouchers)


def fixture_played_hand_types(fixture: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(name for name, count in fixture_played_hand_counts(fixture).items() if count > 0)


def fixture_played_hand_counts(fixture: Mapping[str, Any]) -> dict[str, int]:
    played: dict[str, int] = {}
    for source in (fixture, fixture_opened_state(fixture), fixture_shop_state(fixture)):
        raw_played_hands = source.get("played_hands")
        if isinstance(raw_played_hands, Mapping):
            for name, count in raw_played_hands.items():
                try:
                    played[str(name)] = int(count)
                except (TypeError, ValueError):
                    pass
        hands = source.get("hands")
        if not isinstance(hands, Mapping):
            continue
        for name, payload in hands.items():
            if isinstance(payload, Mapping):
                try:
                    played[str(name)] = int(payload.get("played", 0))
                except (TypeError, ValueError):
                    pass
    return played


def fixture_telescope_planet_key(fixture: Mapping[str, Any]) -> str | None:
    vouchers = fixture_owned_vouchers(fixture)
    if not {"v_telescope", "Telescope"} & vouchers:
        return None
    counts = fixture_played_hand_counts(fixture)
    best_hand = None
    tally = 0
    for hand_name in HAND_ORDER:
        count = counts.get(hand_name, 0)
        if count > tally:
            best_hand = hand_name
            tally = count
    return _planet_key_for_hand(best_hand) if best_hand is not None else None


def fixture_edition_rate(fixture: Mapping[str, Any]) -> float:
    for source in (fixture, fixture_opened_state(fixture), fixture_shop_state(fixture)):
        raw = source.get("edition_rate")
        try:
            if raw is not None:
                return float(raw)
        except (TypeError, ValueError):
            pass
    vouchers = fixture_owned_vouchers(fixture)
    if "v_glow_up" in vouchers or "Glow Up" in vouchers:
        return 4.0
    if "v_hone" in vouchers or "Hone" in vouchers:
        return 2.0
    return 1.0


def visible_used_keys(state: Mapping[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    jokers: set[str] = set()
    consumables: set[str] = set()
    for area_name in ("shop", "jokers", "consumables"):
        for card in extract_area_cards(state, area_name):
            key = str(card.get("key", ""))
            set_name = normalize_set_name(str(card.get("set", "")), key=key)
            if key.startswith("j_") or set_name == "Joker":
                jokers.add(key)
            elif key.startswith("c_") or set_name in {"Tarot", "Planet", "Spectral"}:
                consumables.add(key)
    return frozenset(jokers), frozenset(consumables)


def extract_area_cards(state: Mapping[str, Any], area_name: str) -> tuple[dict[str, Any], ...]:
    area = state.get(area_name)
    if isinstance(area, Mapping):
        cards = area.get("cards", ())
    elif isinstance(area, list):
        cards = area
    else:
        cards = ()
    return tuple(card for card in cards if isinstance(card, dict))


def actual_card_signature(card: Mapping[str, Any]) -> CardSignature:
    key = str(card.get("key", ""))
    front_key = normalize_front_key(key)
    enhancement = actual_enhancement_key(card) if front_key is not None else None
    if front_key is not None:
        center_key = enhancement or "c_base"
        set_name = "Enhanced" if enhancement else "Default"
    else:
        center_key = key
        set_name = normalize_set_name(str(card.get("set", "")), key=key)
    return CardSignature(
        set=set_name,
        key=center_key,
        front_key=front_key,
        edition=normalize_edition(_card_attr(card, "edition")),
        seal=normalize_seal(_card_attr(card, "seal")),
    )


def predicted_card_signature(card: PredictedCard) -> CardSignature:
    return CardSignature(
        set=card.set,
        key=card.key,
        front_key=card.front_key,
        edition=normalize_edition(card.edition),
        seal=normalize_seal(card.seal),
    )


def normalize_set_name(raw_set: str, *, key: str = "") -> str:
    upper = raw_set.upper()
    if upper in SET_NAMES:
        return SET_NAMES[upper]
    if key.startswith("j_"):
        return "Joker"
    if key.startswith("c_"):
        if key in {"c_soul", "c_black_hole"}:
            return "Spectral"
        return "Consumable"
    return raw_set or "Unknown"


def normalize_front_key(key: str) -> str | None:
    if not PLAYING_KEY_RE.match(key):
        return None
    suit, rank = key.split("_", 1)
    if rank == "10":
        rank = "T"
    return suit + "_" + rank


def actual_enhancement_key(card: Mapping[str, Any]) -> str | None:
    for raw in (
        card.get("enhancement"),
        _mapping(card.get("modifier")).get("enhancement"),
        _mapping(card.get("ability")).get("effect"),
    ):
        enhancement = normalize_enhancement(raw)
        if enhancement is not None:
            return enhancement
    label = str(card.get("label", ""))
    return ENHANCEMENT_LABELS.get(label)


def normalize_enhancement(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        raw = raw.get("key") or raw.get("type") or raw.get("name") or raw.get("enhancement")
    token = str(raw).upper().replace("TYPE ", "").replace(" CARD", "").replace("-", "_")
    token = token.removeprefix("M_")
    for name, key in ENHANCEMENT_KEYS.items():
        if token == name or token.endswith("_" + name) or name in token.split():
            return key
    return None


def normalize_edition(raw: Any) -> str | None:
    value = _normalize_modifier_value(raw, "edition")
    if value is None:
        return None
    value = value.removeprefix("e_")
    if value == "holo":
        return "holographic"
    return value


def normalize_seal(raw: Any) -> str | None:
    value = _normalize_modifier_value(raw, "seal")
    if value is None:
        return None
    return value.removesuffix("_seal")


def _normalize_modifier_value(raw: Any, field: str) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        raw = raw.get("key") or raw.get("type") or raw.get("name") or raw.get(field)
    value = str(raw).strip().lower().replace(" ", "_")
    return value or None


def _card_attr(card: Mapping[str, Any], name: str) -> Any:
    if name in card:
        return card.get(name)
    modifier = _mapping(card.get("modifier"))
    if name in modifier:
        return modifier.get(name)
    state = _mapping(card.get("state"))
    return state.get(name)


def _mapping(raw: Any) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def _collect_vouchers(vouchers: set[str], raw: Any) -> None:
    if isinstance(raw, list):
        vouchers.update(str(item) for item in raw)
    elif isinstance(raw, Mapping):
        vouchers.update(str(key) for key in raw)


def _planet_key_for_hand(hand_name: str | None) -> str | None:
    if hand_name is None:
        return None
    return _planet_keys_by_hand().get(hand_name)


@lru_cache(maxsize=1)
def _planet_keys_by_hand() -> Mapping[str, str]:
    data = json.loads(SHOP_POOL_DATA_PATH.read_text(encoding="utf-8"))
    planets: dict[str, str] = {}
    for record in data.get("planets", ()):
        config = record.get("config")
        if isinstance(config, Mapping):
            hand_type = config.get("hand_type")
            key = record.get("key")
            if hand_type is not None and key is not None:
                planets[str(hand_type)] = str(key)
    return planets


def _seed_from_pack_path(path: Path) -> str | None:
    if not path.stem.startswith("pack_seed_"):
        return None
    remainder = path.stem.removeprefix("pack_seed_")
    return remainder.split("_", 1)[0] if remainder else None


def report_for_pack_fixture(path: Path) -> str:
    try:
        fixture = load_pack_fixture(path)
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path.name}: failed to load - {exc!r}"
    result = check_pack_fixture(fixture)
    seed = fixture_seed(fixture) or "?"
    pack_key = fixture_pack_key(fixture) or "?"
    prefix = f"{path.name}: seed={seed} pack={pack_key} {result.status.upper()}"
    return prefix if not result.detail else prefix + f" - {result.detail}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate captured RNG surface fixtures.")
    parser.add_argument("fixture", nargs="?", type=Path, help="Path to one pack fixture.")
    parser.add_argument("--all", action="store_true", help=f"Validate every pack fixture under {DEFAULT_FIXTURE_DIR}/.")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        fixtures = iter_pack_fixture_paths(args.fixture_dir)
        if not fixtures:
            print(f"No pack_seed_*.json fixtures in {args.fixture_dir}. Run capture_surfaces first.")
            return 2
    elif args.fixture is None:
        print("Specify a fixture path or --all.")
        return 2
    else:
        fixtures = (args.fixture,)

    failed = False
    for path in fixtures:
        report = report_for_pack_fixture(path)
        print(report)
        failed = failed or "MISMATCH" in report or "UNSUPPORTED" in report
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
