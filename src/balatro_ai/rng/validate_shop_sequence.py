"""Validate captured no-purchase shop sequences against RNG predictors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from balatro_ai.rng.balatro_rng import BalatroRNG
from balatro_ai.rng.capture import DEFAULT_FIXTURE_DIR
from balatro_ai.rng.surfaces import (
    DEFAULT_SHOP_RATES,
    PredictedCard,
    STEAMODDED_SHOP_TYPE_ORDER,
    VANILLA_SHOP_TYPE_ORDER,
    predict_initial_surface,
    predict_shop_boosters,
    predict_shop_cards,
    predict_voucher,
)
from balatro_ai.rng.validate_surfaces import extract_area_cards, fixture_edition_rate, fixture_owned_vouchers, normalize_edition


@dataclass(frozen=True, slots=True)
class ShopSequenceCheckResult:
    name: str
    status: str  # "ok", "mismatch", or "unsupported"
    detail: str = ""


def iter_shop_sequence_fixture_paths(root: Path = DEFAULT_FIXTURE_DIR) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("shop_sequence_seed_*.json")))


def load_shop_sequence_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_shop_sequence_fixture(fixture: Mapping[str, Any]) -> tuple[ShopSequenceCheckResult, ...]:
    seed = fixture.get("seed")
    if seed is None:
        return (ShopSequenceCheckResult("shop_sequence", "unsupported", "fixture has no seed"),)
    shops = fixture.get("shops")
    if not isinstance(shops, list) or not shops:
        return (ShopSequenceCheckResult("shop_sequence", "unsupported", "fixture has no shops"),)

    rng = BalatroRNG(str(seed))
    initial_surface = predict_initial_surface(rng, ante=1)
    sticker_options = _shop_sticker_options(fixture.get("stake"))
    vouchers = fixture_owned_vouchers(fixture)
    voucher_effective_shop_index = _voucher_effective_shop_index(fixture)
    # Per-ante shop voucher: ante 1 from the initial surface; ante 2+ from
    # predict_voucher rolled once per ante on the persistent rng (the
    # "Voucher"+ante stream is independent of the shop-card streams). Only
    # asserted for fixtures with no pre-owned vouchers — the case validated
    # against the captured no-purchase sequences.
    check_voucher = not vouchers
    voucher_by_ante: dict[int, str] = {1: initial_surface.voucher_key}
    results: list[ShopSequenceCheckResult] = []
    for index, entry in enumerate(shops):
        if not isinstance(entry, Mapping):
            results.append(ShopSequenceCheckResult(f"shop_{index}", "unsupported", "shop entry is not a mapping"))
            continue
        state = entry.get("state")
        if not isinstance(state, Mapping):
            results.append(ShopSequenceCheckResult(f"shop_{index}", "unsupported", "shop entry has no raw state"))
            continue
        ante = _shop_ante(entry, state)
        shop_cards = extract_area_cards(state, "shop")
        actual_shop = tuple(_actual_shop_signature(card) for card in shop_cards)
        actual_boosters = tuple(str(card.get("key", "")) for card in extract_area_cards(state, "packs"))
        active_vouchers = vouchers if index >= voucher_effective_shop_index else frozenset()
        predicted_shop = tuple(
            _predicted_shop_signature(card)
            for card in predict_shop_cards(
                rng,
                ante=ante,
                n_slots=len(actual_shop),
                rates=_shop_rates_for_vouchers(active_vouchers),
                shop_type_order=_shop_type_order_for_vouchers(active_vouchers),
                vouchers=active_vouchers,
                edition_rate=fixture_edition_rate(fixture) if active_vouchers else 1.0,
                **sticker_options,
            )
        )
        predicted_boosters = tuple(
            predict_shop_boosters(
                rng,
                ante=ante,
                n_slots=len(actual_boosters),
                first_shop_buffoon=index == 0,
            )
        )

        mismatches: list[str] = []
        if predicted_shop != actual_shop:
            mismatches.append(f"shop predicted={predicted_shop} actual={actual_shop}")
        if predicted_boosters != actual_boosters:
            mismatches.append(f"boosters predicted={predicted_boosters} actual={actual_boosters}")
        if check_voucher:
            if ante not in voucher_by_ante:
                voucher_by_ante[ante] = predict_voucher(rng, ante=ante)
            predicted_voucher = voucher_by_ante[ante]
            actual_voucher = _actual_voucher(state)
            if actual_voucher is not None and predicted_voucher != actual_voucher:
                mismatches.append(f"voucher predicted={predicted_voucher} actual={actual_voucher}")
        if mismatches:
            results.append(ShopSequenceCheckResult(f"shop_{index}", "mismatch", "; ".join(mismatches)))
        else:
            results.append(
                ShopSequenceCheckResult(
                    f"shop_{index}",
                    "ok",
                    f"ante={ante} shop={tuple(sig[0] for sig in actual_shop)} boosters={actual_boosters}",
                )
            )
    return tuple(results)


def report_for_shop_sequence_fixture(path: Path) -> str:
    try:
        fixture = load_shop_sequence_fixture(path)
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path.name}: failed to load - {exc!r}"
    results = check_shop_sequence_fixture(fixture)
    seed = fixture.get("seed", "?") if isinstance(fixture, Mapping) else "?"
    lines = [f"{path.name}: seed={seed} checks={len(results)}"]
    for result in results:
        line = f"  {result.name}: {result.status.upper()}"
        if result.detail:
            line += " - " + result.detail
        lines.append(line)
    return "\n".join(lines)


def _shop_ante(entry: Mapping[str, Any], state: Mapping[str, Any]) -> int:
    for source in (entry, state):
        for key in ("ante", "ante_num"):
            raw = source.get(key)
            try:
                if raw is not None:
                    return max(1, int(raw))
            except (TypeError, ValueError):
                pass
    return 1


def _shop_sticker_options(stake: Any) -> dict[str, bool]:
    level = _stake_level(stake)
    return {
        "enable_eternals": level >= 4,
        "enable_perishables": level >= 7,
        "enable_rentals": level >= 8,
    }


def _stake_level(stake: Any) -> int:
    if stake is None:
        return 1
    if isinstance(stake, int):
        return max(1, stake)
    text = str(stake).strip().lower()
    if text.isdigit():
        return max(1, int(text))
    return {
        "white": 1,
        "red": 2,
        "green": 3,
        "black": 4,
        "blue": 5,
        "purple": 6,
        "orange": 7,
        "gold": 8,
    }.get(text.removeprefix("stake_"), 1)


def _voucher_effective_shop_index(fixture: Mapping[str, Any]) -> int:
    raw = fixture.get("voucher_effective_shop_index")
    try:
        if raw is not None:
            return max(0, int(raw))
    except (TypeError, ValueError):
        pass
    return 0


def _shop_rates_for_vouchers(vouchers: frozenset[str]) -> Mapping[str, int | float]:
    rates: dict[str, int | float] = dict(DEFAULT_SHOP_RATES)
    if "v_tarot_tycoon" in vouchers or "Tarot Tycoon" in vouchers:
        rates["Tarot"] = 32.0
    elif "v_tarot_merchant" in vouchers or "Tarot Merchant" in vouchers:
        rates["Tarot"] = 9.6
    if "v_planet_tycoon" in vouchers or "Planet Tycoon" in vouchers:
        rates["Planet"] = 32.0
    elif "v_planet_merchant" in vouchers or "Planet Merchant" in vouchers:
        rates["Planet"] = 9.6
    if (
        "v_magic_trick" in vouchers
        or "Magic Trick" in vouchers
        or "v_illusion" in vouchers
        or "Illusion" in vouchers
    ):
        rates["playing_card"] = 4
    return rates


def _shop_type_order_for_vouchers(vouchers: frozenset[str]) -> tuple[str, ...]:
    if (
        "v_magic_trick" in vouchers
        or "Magic Trick" in vouchers
        or "v_illusion" in vouchers
        or "Illusion" in vouchers
    ):
        return VANILLA_SHOP_TYPE_ORDER
    return STEAMODDED_SHOP_TYPE_ORDER


def _predicted_shop_signature(card: PredictedCard) -> tuple[str, str | None, bool, bool, bool]:
    return (card.key, normalize_edition(card.edition), card.eternal, card.perishable, card.rental)


def _actual_shop_signature(card: Mapping[str, Any]) -> tuple[str, str | None, bool, bool, bool]:
    modifier = _mapping(card.get("modifier"))
    ability = _mapping(card.get("ability"))
    state = _mapping(card.get("state"))
    edition = normalize_edition(card.get("edition") or modifier.get("edition") or ability.get("edition") or state.get("edition"))
    return (
        str(card.get("key", "")),
        edition,
        _truthy(card.get("eternal") or modifier.get("eternal") or ability.get("eternal") or state.get("eternal")),
        _truthy(card.get("perishable") or modifier.get("perishable") or ability.get("perishable") or state.get("perishable")),
        _truthy(card.get("rental") or modifier.get("rental") or ability.get("rental") or state.get("rental")),
    )


def _actual_voucher(state: Mapping[str, Any]) -> str | None:
    area = state.get("vouchers")
    cards = area.get("cards") if isinstance(area, Mapping) else None
    if isinstance(cards, list) and cards and isinstance(cards[0], Mapping):
        key = str(cards[0].get("key", ""))
        return key or None
    return None


def _mapping(raw: Any) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def _truthy(raw: Any) -> bool:
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    return bool(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate no-purchase shop sequence fixtures.")
    parser.add_argument("fixture", nargs="?", type=Path, help="Path to one shop sequence fixture.")
    parser.add_argument("--all", action="store_true", help=f"Validate every shop sequence fixture under {DEFAULT_FIXTURE_DIR}/.")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        fixtures = iter_shop_sequence_fixture_paths(args.fixture_dir)
        if not fixtures:
            print(f"No shop_sequence_seed_*.json fixtures in {args.fixture_dir}. Run capture_shop_sequence first.")
            return 2
    elif args.fixture is None:
        print("Specify a fixture path or --all.")
        return 2
    else:
        fixtures = (args.fixture,)

    failed = False
    for path in fixtures:
        report = report_for_shop_sequence_fixture(path)
        print(report)
        failed = failed or "MISMATCH" in report or "UNSUPPORTED" in report
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
