"""Validate captured mid-hand probability rolls against the predictors.

Each effect's per-hand outcome sequence must match a fresh persistent rng
rolling that effect's key in order: ``pseudorandom_float(key) < normal/odds``
for hit/miss effects, or ``pseudorandom_int(key, lo, hi)`` for Misprint. Nothing
rolls these keys before the first relevant hand, so a fresh ``BalatroRNG(seed)``
reproduces the sequence from the start.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from balatro_ai.rng.balatro_rng import BalatroRNG
from balatro_ai.rng.capture import DEFAULT_FIXTURE_DIR
from balatro_ai.rng.surfaces import pseudorandom_float, pseudorandom_int


@dataclass(frozen=True, slots=True)
class PlayRngCheckResult:
    name: str
    status: str  # "ok", "mismatch", or "unsupported"
    detail: str = ""


def iter_play_rng_fixture_paths(root: Path = DEFAULT_FIXTURE_DIR) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("play_rng_seed_*.json")))


def load_play_rng_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_play_rng_fixture(fixture: Mapping[str, Any]) -> tuple[PlayRngCheckResult, ...]:
    seed = fixture.get("seed")
    effects = fixture.get("effects")
    if seed is None:
        return (PlayRngCheckResult("play_rng", "unsupported", "fixture has no seed"),)
    if not isinstance(effects, list) or not effects:
        return (PlayRngCheckResult("play_rng", "unsupported", "fixture has no effects"),)

    results: list[PlayRngCheckResult] = []
    for effect in effects:
        if not isinstance(effect, Mapping):
            results.append(PlayRngCheckResult("effect", "unsupported", "effect is not a mapping"))
            continue
        key = str(effect.get("key", "?"))
        observed = effect.get("observed")
        if not isinstance(observed, list) or not observed:
            results.append(PlayRngCheckResult(key, "unsupported", "no observed samples"))
            continue

        rng = BalatroRNG(str(seed))
        if effect.get("kind") == "int":
            lo, hi = int(effect.get("odds_lo", 0)), int(effect.get("odds_hi", 0))
            predicted = [pseudorandom_int(rng, key, lo, hi) for _ in observed]
            actual = [int(v) for v in observed]
        else:
            odds = float(effect.get("odds", 1))
            predicted = [int(pseudorandom_float(rng, key) < 1.0 / odds) for _ in observed]
            actual = [int(v) for v in observed]

        if predicted == actual:
            results.append(PlayRngCheckResult(key, "ok", f"{len(actual)} hands"))
        else:
            results.append(PlayRngCheckResult(key, "mismatch", f"predicted={predicted} actual={actual}"))
    return tuple(results)


def report_for_play_rng_fixture(path: Path) -> str:
    try:
        fixture = load_play_rng_fixture(path)
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path.name}: failed to load - {exc!r}"
    results = check_play_rng_fixture(fixture)
    seed = fixture.get("seed", "?") if isinstance(fixture, Mapping) else "?"
    lines = [f"{path.name}: seed={seed} checks={len(results)}"]
    for result in results:
        line = f"  {result.name}: {result.status.upper()}"
        if result.detail:
            line += " - " + result.detail
        lines.append(line)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate mid-hand probability-roll fixtures.")
    parser.add_argument("fixture", nargs="?", type=Path, help="Path to one play-rng fixture.")
    parser.add_argument("--all", action="store_true", help=f"Validate every fixture under {DEFAULT_FIXTURE_DIR}/.")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        fixtures = iter_play_rng_fixture_paths(args.fixture_dir)
        if not fixtures:
            print(f"No play_rng_seed_*.json fixtures in {args.fixture_dir}. Run capture_play_rng first.")
            return 2
    elif args.fixture is None:
        print("Specify a fixture path or --all.")
        return 2
    else:
        fixtures = (args.fixture,)

    failed = False
    for path in fixtures:
        report = report_for_play_rng_fixture(path)
        print(report)
        failed = failed or "MISMATCH" in report or "UNSUPPORTED" in report
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
