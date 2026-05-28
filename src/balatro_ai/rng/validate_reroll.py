"""Validate captured shop-reroll sequences against RNG predictors.

A reroll re-rolls the rerollable shop-card slots only (voucher and booster
packs are untouched), advancing the same per-ante shop-card RNG streams. So the
prediction for reroll ``k`` is the ``(k+1)``-th call to ``predict_shop_cards``
on a persistent RNG that has already consumed the run-opening surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from balatro_ai.rng.balatro_rng import BalatroRNG
from balatro_ai.rng.capture import DEFAULT_FIXTURE_DIR
from balatro_ai.rng.surfaces import predict_initial_surface, predict_shop_cards
from balatro_ai.rng.validate_shop_sequence import (
    _actual_shop_signature,
    _predicted_shop_signature,
    _shop_sticker_options,
)


@dataclass(frozen=True, slots=True)
class RerollCheckResult:
    name: str
    status: str  # "ok", "mismatch", or "unsupported"
    detail: str = ""


def iter_reroll_fixture_paths(root: Path = DEFAULT_FIXTURE_DIR) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("shop_reroll_seed_*.json")))


def load_reroll_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_reroll_fixture(fixture: Mapping[str, Any]) -> tuple[RerollCheckResult, ...]:
    seed = fixture.get("seed")
    if seed is None:
        return (RerollCheckResult("reroll", "unsupported", "fixture has no seed"),)
    rerolls = fixture.get("rerolls")
    if not isinstance(rerolls, list) or not rerolls:
        return (RerollCheckResult("reroll", "unsupported", "fixture has no rerolls"),)
    ante = max(1, int(fixture.get("ante", 1)))

    rng = BalatroRNG(str(seed))
    predict_initial_surface(rng, ante=1)
    sticker_options = _shop_sticker_options(fixture.get("stake"))

    results: list[RerollCheckResult] = []
    for entry in rerolls:
        if not isinstance(entry, Mapping):
            results.append(RerollCheckResult("reroll_?", "unsupported", "entry is not a mapping"))
            continue
        index = entry.get("reroll_index")
        state = entry.get("state")
        if not isinstance(state, Mapping):
            results.append(RerollCheckResult(f"reroll_{index}", "unsupported", "entry has no raw state"))
            continue
        shop_cards = _extract_shop_cards(state)
        actual = tuple(_actual_shop_signature(card) for card in shop_cards)
        predicted = tuple(
            _predicted_shop_signature(card)
            for card in predict_shop_cards(rng, ante=ante, n_slots=len(actual), **sticker_options)
        )
        if predicted != actual:
            results.append(
                RerollCheckResult(
                    f"reroll_{index}",
                    "mismatch",
                    f"predicted={predicted} actual={actual}",
                )
            )
        else:
            results.append(
                RerollCheckResult(
                    f"reroll_{index}",
                    "ok",
                    f"shop={tuple(sig[0] for sig in actual)}",
                )
            )
    return tuple(results)


def report_for_reroll_fixture(path: Path) -> str:
    try:
        fixture = load_reroll_fixture(path)
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path.name}: failed to load - {exc!r}"
    results = check_reroll_fixture(fixture)
    seed = fixture.get("seed", "?") if isinstance(fixture, Mapping) else "?"
    lines = [f"{path.name}: seed={seed} checks={len(results)}"]
    for result in results:
        line = f"  {result.name}: {result.status.upper()}"
        if result.detail:
            line += " - " + result.detail
        lines.append(line)
    return "\n".join(lines)


def _extract_shop_cards(state: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    shop = state.get("shop")
    if isinstance(shop, Mapping):
        cards = shop.get("cards", ())
    elif isinstance(shop, list):
        cards = shop
    else:
        cards = ()
    return tuple(card for card in cards if isinstance(card, Mapping))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate shop reroll fixtures.")
    parser.add_argument("fixture", nargs="?", type=Path, help="Path to one shop reroll fixture.")
    parser.add_argument("--all", action="store_true", help=f"Validate every reroll fixture under {DEFAULT_FIXTURE_DIR}/.")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        fixtures = iter_reroll_fixture_paths(args.fixture_dir)
        if not fixtures:
            print(f"No shop_reroll_seed_*.json fixtures in {args.fixture_dir}. Run capture_reroll first.")
            return 2
    elif args.fixture is None:
        print("Specify a fixture path or --all.")
        return 2
    else:
        fixtures = (args.fixture,)

    failed = False
    for path in fixtures:
        report = report_for_reroll_fixture(path)
        print(report)
        failed = failed or "MISMATCH" in report or "UNSUPPORTED" in report
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
