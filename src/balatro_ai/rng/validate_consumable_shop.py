"""Validate that consumable use does not desync the shop-card RNG stream.

For each captured trial: a fresh persistent rng (initial surface consumed)
predicts the first shop's cards (roll #1) and then the next roll (#2). The
captured ``shop0`` must equal roll #1 and the captured ``post_use_reroll``
(taken AFTER using a consumable) must equal roll #2. If it does, the
consumable consumed no shop-stream RNG and shops stay seed-faithful through
consumable use.
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


@dataclass(frozen=True, slots=True)
class ConsumableShopCheckResult:
    name: str
    status: str  # "ok", "mismatch", or "unsupported"
    detail: str = ""


def iter_consumable_shop_fixture_paths(root: Path = DEFAULT_FIXTURE_DIR) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("consumable_shop_seed_*.json")))


def load_consumable_shop_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_consumable_shop_fixture(fixture: Mapping[str, Any]) -> tuple[ConsumableShopCheckResult, ...]:
    seed = fixture.get("seed")
    trials = fixture.get("trials")
    if seed is None:
        return (ConsumableShopCheckResult("consumable_shop", "unsupported", "fixture has no seed"),)
    if not isinstance(trials, list) or not trials:
        return (ConsumableShopCheckResult("consumable_shop", "unsupported", "fixture has no trials"),)
    ante = max(1, int(fixture.get("ante", 1)))

    results: list[ConsumableShopCheckResult] = []
    for trial in trials:
        if not isinstance(trial, Mapping):
            results.append(ConsumableShopCheckResult("trial", "unsupported", "trial is not a mapping"))
            continue
        key = str(trial.get("consumable_key", "?"))
        shop0 = tuple(str(c) for c in trial.get("shop0", ()))
        post_use = tuple(str(c) for c in trial.get("post_use_reroll", ()))

        rng = BalatroRNG(str(seed))
        predict_initial_surface(rng, ante=1)
        roll1 = tuple(card.key for card in predict_shop_cards(rng, ante=ante, n_slots=len(shop0)))
        roll2 = tuple(card.key for card in predict_shop_cards(rng, ante=ante, n_slots=len(post_use)))

        if roll1 != shop0:
            results.append(ConsumableShopCheckResult(key, "mismatch", f"shop0 predicted={roll1} actual={shop0}"))
        elif roll2 != post_use:
            results.append(
                ConsumableShopCheckResult(
                    key, "mismatch", f"post-use reroll shifted: predicted #2={roll2} actual={post_use}"
                )
            )
        else:
            results.append(ConsumableShopCheckResult(key, "ok", f"no shift (reroll={post_use})"))
    return tuple(results)


def report_for_consumable_shop_fixture(path: Path) -> str:
    try:
        fixture = load_consumable_shop_fixture(path)
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path.name}: failed to load - {exc!r}"
    results = check_consumable_shop_fixture(fixture)
    seed = fixture.get("seed", "?") if isinstance(fixture, Mapping) else "?"
    lines = [f"{path.name}: seed={seed} checks={len(results)}"]
    for result in results:
        line = f"  {result.name}: {result.status.upper()}"
        if result.detail:
            line += " - " + result.detail
        lines.append(line)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate consumable-use shop fixtures.")
    parser.add_argument("fixture", nargs="?", type=Path, help="Path to one consumable-shop fixture.")
    parser.add_argument("--all", action="store_true", help=f"Validate every fixture under {DEFAULT_FIXTURE_DIR}/.")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        fixtures = iter_consumable_shop_fixture_paths(args.fixture_dir)
        if not fixtures:
            print(f"No consumable_shop_seed_*.json fixtures in {args.fixture_dir}. Run capture_consumable_shop first.")
            return 2
    elif args.fixture is None:
        print("Specify a fixture path or --all.")
        return 2
    else:
        fixtures = (args.fixture,)

    failed = False
    for path in fixtures:
        report = report_for_consumable_shop_fixture(path)
        print(report)
        failed = failed or "MISMATCH" in report or "UNSUPPORTED" in report
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
