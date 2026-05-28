"""Validate ``BalatroRNG`` predictions against captured bridge fixtures.

A fixture is the initial state the bridge returned for a known seed (saved by
`capture.py`). Our prediction code is built on several unverified assumptions
(shuffle key name, deck insertion order, whether ``mix_hashed_seed`` is on
in the captured Balatro version) — so the CLI doesn't just check one
prediction. It runs a small grid search over plausible configurations and
reports any that match. The first config that matches is the empirically
verified ground truth for that game version.

Usage:
    # After capturing a fixture, see if any configuration matches:
    python -m balatro_ai.rng.validate .data/rng-validation/seed_AAAAAAA_red_white.json

    # Same but for every captured fixture:
    python -m balatro_ai.rng.validate --all
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from balatro_ai.rng.balatro_rng import BalatroRNG, pseudorandom_shuffle
from balatro_ai.rng.capture import DEFAULT_FIXTURE_DIR
from balatro_ai.rng.deck import build_standard_red_deck_short_names
from balatro_ai.rng.luajit_prng import luajit_pseudoshuffle
from balatro_ai.rng.xoroshiro import SeedStrategy, pseudoshuffle_with_xoroshiro


# Candidate shuffle keys observed in community decoders or documented in the
# Lua source. Empirical match against fixtures will pick the right one.
# 'nr1' shows up in CardArea:shuffle defaults; others are gleaned from
# community decoders. We cast a wide net here because the cost of an extra
# key in the sweep is tiny vs. missing the right one.
CANDIDATE_SHUFFLE_KEYS: tuple[str, ...] = (
    "shuffle",
    "front",
    "nr",
    "nr1",
    "nr2",
    "deck",
    "card",
    "new_card",
    "init",
    "starting",
)

CANDIDATE_SEED_STRATEGIES: tuple[SeedStrategy, ...] = tuple(SeedStrategy)

INITIAL_HAND_SIZE = 8


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str  # "ok", "mismatch", "unsupported"
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    shuffle_key: str
    mix_hashed_seed: bool
    # ``algorithm`` selects between the legacy per-swap pseudoseed shuffle
    # (kept for non-deck use cases) and the actual Balatro deck shuffle:
    # xoroshiro128+ seeded from one pseudoseed call.
    algorithm: str = "per_swap_pseudoseed"  # or "xoroshiro_after_pseudoseed"
    seed_strategy: SeedStrategy | None = None

    def label(self) -> str:
        base = f"key={self.shuffle_key!r} mix_hashed_seed={self.mix_hashed_seed}"
        if self.algorithm == "xoroshiro_after_pseudoseed":
            strat = self.seed_strategy.value if self.seed_strategy else "none"
            return f"algo=xoroshiro {base} seed_strategy={strat}"
        if self.algorithm == "luajit_after_pseudoseed":
            return f"algo=luajit_tw223 {base}"
        return f"algo=per_swap_pseudoseed {base}"


def extract_initial_hand_short_names(state: dict[str, Any]) -> tuple[str, ...]:
    """Return the 8 starting-hand cards as e.g. ('QC', 'JH', '9C', ...).

    Reads directly from the bridge's raw payload rather than parsing through
    ``GameState`` so we preserve the exact ordering the game returned.
    """

    cards = state.get("hand", {}).get("cards", []) if isinstance(state.get("hand"), dict) else []
    short: list[str] = []
    for card in cards:
        value = card.get("value", {}) if isinstance(card, dict) else {}
        rank = value.get("rank", "?")
        suit = value.get("suit", "?")
        short.append(f"{rank}{suit}")
    return tuple(short)


def extract_deck_order_short_names(state: dict[str, Any]) -> tuple[str, ...]:
    """Return the full 52-card deck order as the bridge reports it.

    At BLIND_SELECT (run start, no blind picked yet) the bridge exposes the
    entire post-shuffle deck under ``cards.cards`` — that's our richest
    signal for validating the shuffle math. Order in the list matches the
    game's internal deck order.
    """

    payload = state.get("cards")
    if not isinstance(payload, dict):
        return ()
    cards = payload.get("cards", [])
    short: list[str] = []
    for card in cards:
        value = card.get("value", {}) if isinstance(card, dict) else {}
        rank = value.get("rank", "?")
        suit = value.get("suit", "?")
        short.append(f"{rank}{suit}")
    return tuple(short)


def predict_starting_hand(
    seed: str,
    config: PredictionConfig,
    *,
    hand_size: int = INITIAL_HAND_SIZE,
) -> tuple[str, ...]:
    """Predict the first ``hand_size`` cards drawn for ``seed``.

    Builds the standard Red deck in our assumed insertion order, shuffles
    with the given ``config``, and returns the first ``hand_size`` cards.
    For ``hand_size=8`` this matches the starting-hand draw at run begin
    (assuming no jokers or special decks alter the draw count).
    """

    return predict_full_shuffled_deck(seed, config)[:hand_size]


def predict_full_shuffled_deck(seed: str, config: PredictionConfig) -> tuple[str, ...]:
    """Predict the full post-shuffle deck order for ``seed`` under ``config``."""

    deck = list(build_standard_red_deck_short_names())
    rng = BalatroRNG(seed=seed, mix_hashed_seed=config.mix_hashed_seed)
    if config.algorithm == "luajit_after_pseudoseed":
        # Verified Balatro behavior: pseudoseed(key) yields one float that
        # seeds LuaJIT's TW223 via the (d * pi + e) bit-reinterpret transform,
        # then Fisher-Yates with math.random(i).
        seed_float = rng.random(config.shuffle_key)
        luajit_pseudoshuffle(deck, seed_float)
    elif config.algorithm == "xoroshiro_after_pseudoseed":
        if config.seed_strategy is None:
            raise ValueError("xoroshiro_after_pseudoseed requires a seed_strategy")
        seed_float = rng.random(config.shuffle_key)
        pseudoshuffle_with_xoroshiro(deck, seed_float, config.seed_strategy)
    else:
        pseudorandom_shuffle(deck, rng, config.shuffle_key)
    return tuple(deck)


def check_initial_hand_matches_prediction(
    fixture: dict[str, Any],
    predicted_hand: tuple[str, ...],
) -> CheckResult:
    """Compare a predicted starting-hand sequence to the fixture's hand."""

    actual = extract_initial_hand_short_names(fixture)
    if not actual:
        return CheckResult(
            name="initial_hand",
            status="unsupported",
            detail="fixture has no hand cards (capture may have run mid-blind)",
        )
    if predicted_hand == actual:
        return CheckResult(name="initial_hand", status="ok", detail=f"matched {len(actual)} cards")
    return CheckResult(
        name="initial_hand",
        status="mismatch",
        detail=f"predicted={predicted_hand} actual={actual}",
    )


def search_matching_configs(
    fixture: dict[str, Any],
    seed: str,
    *,
    shuffle_keys: Iterable[str] = CANDIDATE_SHUFFLE_KEYS,
    mix_flags: Iterable[bool] = (False, True),
) -> tuple[PredictionConfig, ...]:
    """Return every ``(shuffle_key, mix_hashed_seed)`` that matches the fixture's starting hand."""

    actual = extract_initial_hand_short_names(fixture)
    if not actual:
        return ()
    matches: list[PredictionConfig] = []
    for shuffle_key, mix_flag in product(shuffle_keys, mix_flags):
        config = PredictionConfig(shuffle_key=shuffle_key, mix_hashed_seed=mix_flag)
        predicted = predict_starting_hand(seed, config, hand_size=len(actual))
        if predicted == actual:
            matches.append(config)
    return tuple(matches)


def search_matching_configs_against_deck(
    fixture: dict[str, Any],
    seed: str,
    *,
    shuffle_keys: Iterable[str] = CANDIDATE_SHUFFLE_KEYS,
    mix_flags: Iterable[bool] = (False, True),
    seed_strategies: Iterable[SeedStrategy] = CANDIDATE_SEED_STRATEGIES,
) -> tuple[tuple[PredictionConfig, bool], ...]:
    """Search configs against the full 52-card deck (richest signal).

    Sweeps both algorithm variants:
      * per_swap_pseudoseed (legacy): one pseudoseed call per Fisher-Yates swap.
      * xoroshiro_after_pseudoseed: one pseudoseed call to seed xoroshiro128+,
        then Fisher-Yates with the xoroshiro stream — this is what Balatro's
        ``pseudoshuffle`` actually does.

    Returns tuples of (config, reversed) where ``reversed=True`` means our
    prediction matches the fixture order reversed.
    """

    actual = extract_deck_order_short_names(fixture)
    if not actual:
        return ()
    matches: list[tuple[PredictionConfig, bool]] = []
    for shuffle_key, mix_flag in product(shuffle_keys, mix_flags):
        # Verified Balatro algorithm (LuaJIT TW223).
        luajit_cfg = PredictionConfig(
            shuffle_key=shuffle_key,
            mix_hashed_seed=mix_flag,
            algorithm="luajit_after_pseudoseed",
        )
        predicted = predict_full_shuffled_deck(seed, luajit_cfg)
        if predicted == actual:
            matches.append((luajit_cfg, False))
        elif predicted == tuple(reversed(actual)):
            matches.append((luajit_cfg, True))

        # Legacy algorithm.
        legacy = PredictionConfig(
            shuffle_key=shuffle_key,
            mix_hashed_seed=mix_flag,
            algorithm="per_swap_pseudoseed",
        )
        predicted = predict_full_shuffled_deck(seed, legacy)
        if predicted == actual:
            matches.append((legacy, False))
        elif predicted == tuple(reversed(actual)):
            matches.append((legacy, True))

        # Xoroshiro algorithm (LOVE 11.x's love.math, NOT used by stock math.random).
        for strategy in seed_strategies:
            cfg = PredictionConfig(
                shuffle_key=shuffle_key,
                mix_hashed_seed=mix_flag,
                algorithm="xoroshiro_after_pseudoseed",
                seed_strategy=strategy,
            )
            predicted = predict_full_shuffled_deck(seed, cfg)
            if predicted == actual:
                matches.append((cfg, False))
            elif predicted == tuple(reversed(actual)):
                matches.append((cfg, True))
    return tuple(matches)


def _seed_from_fixture_path(path: Path) -> str:
    # Fixture filenames look like seed_AAAAAAA_red_white.json.
    stem = path.stem
    if not stem.startswith("seed_"):
        raise ValueError(f"Cannot extract seed from {path.name}; expected seed_<SEED>_<deck>_<stake>.json")
    return stem.split("_", 2)[1]


def report_for_fixture(path: Path) -> str:
    """Return a human-readable validation report for one fixture file."""

    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path.name}: failed to load — {exc!r}"

    try:
        seed = _seed_from_fixture_path(path)
    except ValueError as exc:
        return f"{path.name}: {exc}"

    actual_deck = extract_deck_order_short_names(fixture)
    if actual_deck:
        return _report_against_deck(path, seed, actual_deck)

    actual_hand = extract_initial_hand_short_names(fixture)
    if not actual_hand:
        return f"{path.name}: no deck order or starting hand in fixture"

    return _report_against_hand(path, seed, fixture, actual_hand)


def _report_against_deck(path: Path, seed: str, actual_deck: tuple[str, ...]) -> str:
    matches = search_matching_configs_against_deck({"cards": {"cards": [
        {"value": {"rank": name[:-1], "suit": name[-1]}} for name in actual_deck
    ]}}, seed)
    lines = [
        f"{path.name}: seed={seed} signal=full_deck ({len(actual_deck)} cards)",
        f"  actual[0:13]={list(actual_deck[:13])}",
    ]
    if matches:
        lines.append(f"  MATCHED {len(matches)} config(s):")
        for config, reversed_flag in matches:
            tag = " (REVERSED)" if reversed_flag else ""
            lines.append(f"    - {config.label()}{tag}")
    else:
        lines.append("  NO MATCH across candidate configs.")
        lines.append("  Assumption likely wrong: shuffle key, deck insertion order, or mix_hashed_seed.")
        lines.append("  Reference predictions (first 13 cards):")
        for shuffle_key in CANDIDATE_SHUFFLE_KEYS[:3]:
            for mix_flag in (False, True):
                config = PredictionConfig(shuffle_key=shuffle_key, mix_hashed_seed=mix_flag)
                predicted = predict_full_shuffled_deck(seed, config)
                lines.append(f"    {config.label()}: {list(predicted[:13])}")
    return "\n".join(lines)


def _report_against_hand(path: Path, seed: str, fixture: dict[str, Any], actual: tuple[str, ...]) -> str:
    matches = search_matching_configs(fixture, seed)
    lines = [
        f"{path.name}: seed={seed} signal=starting_hand actual={list(actual)}",
    ]
    if matches:
        lines.append(f"  MATCHED {len(matches)} config(s):")
        for config in matches:
            lines.append(f"    - {config.label()}")
    else:
        lines.append("  NO MATCH across candidate configs.")
        lines.append("  Reference predictions:")
        for shuffle_key in CANDIDATE_SHUFFLE_KEYS[:3]:
            config = PredictionConfig(shuffle_key=shuffle_key, mix_hashed_seed=False)
            predicted = predict_starting_hand(seed, config, hand_size=len(actual))
            lines.append(f"    {config.label()}: {list(predicted)}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate BalatroRNG predictions against captured fixtures.")
    parser.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        help="Path to a single fixture JSON. Omit with --all to validate every fixture.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Validate every fixture under {DEFAULT_FIXTURE_DIR}/.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="Directory containing seed_*.json fixtures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        if not args.fixture_dir.exists():
            print(f"No fixture directory at {args.fixture_dir}. Run capture first.")
            return 2
        fixtures = sorted(args.fixture_dir.glob("seed_*.json"))
        if not fixtures:
            print(f"No seed_*.json fixtures in {args.fixture_dir}. Run capture first.")
            return 2
    elif args.fixture is None:
        print("Specify a fixture path or --all.")
        return 2
    else:
        if not args.fixture.exists():
            print(f"Fixture not found: {args.fixture}")
            return 2
        fixtures = [args.fixture]

    any_unmatched = False
    for path in fixtures:
        report = report_for_fixture(path)
        print(report)
        print()
        if "NO MATCH" in report:
            any_unmatched = True
    return 1 if any_unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
