"""Validate Spectral helper RNG fixtures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from balatro_ai.rng.capture import DEFAULT_FIXTURE_DIR
from balatro_ai.rng.capture_surfaces import extract_area_cards
from balatro_ai.rng.surfaces import predict_spectral_created_cards
from balatro_ai.rng.validate_surfaces import actual_enhancement_key, normalize_front_key


@dataclass(frozen=True, slots=True)
class SpectralHelperCheckResult:
    name: str
    status: str  # "ok", "mismatch", or "unsupported"
    detail: str = ""


def iter_spectral_helper_fixture_paths(root: Path = DEFAULT_FIXTURE_DIR) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("spectral_seed_*.json")))


def load_spectral_helper_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_spectral_helper_fixture(fixture: Mapping[str, Any]) -> SpectralHelperCheckResult:
    seed = fixture.get("seed")
    spectral_key = fixture.get("spectral_key")
    if seed is None:
        return SpectralHelperCheckResult("spectral_helper", "unsupported", "fixture has no seed")
    if spectral_key is None:
        return SpectralHelperCheckResult("spectral_helper", "unsupported", "fixture has no spectral_key")
    after_state = fixture.get("after_state")
    if not isinstance(after_state, Mapping):
        return SpectralHelperCheckResult("spectral_helper", "unsupported", "fixture has no after_state")

    actual = actual_created_cards(after_state)
    if not actual:
        return SpectralHelperCheckResult("spectral_helper", "unsupported", "fixture has no created enhanced cards")
    predicted = predict_spectral_created_cards(str(seed), str(spectral_key))
    if actual == predicted:
        return SpectralHelperCheckResult(
            "spectral_helper",
            "ok",
            f"{spectral_key} created={actual}",
        )
    return SpectralHelperCheckResult(
        "spectral_helper",
        "mismatch",
        f"predicted={predicted} actual={actual}",
    )


def actual_created_cards(state: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    created: list[tuple[str, str, str]] = []
    for card in extract_area_cards(state, "hand"):
        enhancement = actual_enhancement_key(card)
        if enhancement is None:
            continue
        front_key = normalize_front_key(str(card.get("key", "")))
        if front_key is None:
            continue
        suit, rank = front_key.split("_", 1)
        created.append((rank, suit, enhancement))
    return tuple(created)


def report_for_spectral_helper_fixture(path: Path) -> str:
    try:
        fixture = load_spectral_helper_fixture(path)
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path.name}: failed to load - {exc!r}"
    result = check_spectral_helper_fixture(fixture)
    seed = fixture.get("seed", "?") if isinstance(fixture, Mapping) else "?"
    spectral_key = fixture.get("spectral_key", "?") if isinstance(fixture, Mapping) else "?"
    line = f"{path.name}: seed={seed} spectral={spectral_key} {result.status.upper()}"
    if result.detail:
        line += " - " + result.detail
    return line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Spectral helper RNG fixtures.")
    parser.add_argument("fixture", nargs="?", type=Path, help="Path to one spectral helper fixture.")
    parser.add_argument("--all", action="store_true", help=f"Validate every spectral_seed_*.json under {DEFAULT_FIXTURE_DIR}/.")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        fixtures = iter_spectral_helper_fixture_paths(args.fixture_dir)
        if not fixtures:
            print(f"No spectral_seed_*.json fixtures in {args.fixture_dir}. Run capture_spectral_helpers first.")
            return 2
    elif args.fixture is None:
        print("Specify a fixture path or --all.")
        return 2
    else:
        fixtures = (args.fixture,)

    failed = False
    for path in fixtures:
        report = report_for_spectral_helper_fixture(path)
        print(report)
        failed = failed or "MISMATCH" in report or "UNSUPPORTED" in report
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
