"""Run deterministic score-edge fixtures against the evaluator and simulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, Stake
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards
from balatro_ai.search.forward_sim import simulate_discard, simulate_play

DEFAULT_FIXTURE_DIR = Path("tests/fixtures/score_edges")


@dataclass(frozen=True, slots=True)
class FixtureResult:
    path: Path
    name: str
    kind: str
    passed: bool
    message: str = ""

    def to_text(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        suffix = f": {self.message}" if self.message else ""
        return f"[{status}] {self.path.name}::{self.name} ({self.kind}){suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run score-edge fixtures.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_FIXTURE_DIR],
        help="Fixture JSON files or directories. Defaults to tests/fixtures/score_edges.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_fixture_paths(args.paths)
    for result in results:
        print(result.to_text())
    return 0 if all(result.passed for result in results) else 1


def run_fixture_paths(paths: Iterable[Path]) -> tuple[FixtureResult, ...]:
    results: list[FixtureResult] = []
    for path in _fixture_files(paths):
        results.extend(_run_fixture_file(path))
    return tuple(results)


def _fixture_files(paths: Iterable[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        else:
            files.append(path)
    return tuple(files)


def _run_fixture_file(path: Path) -> tuple[FixtureResult, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw.get("fixtures", raw)) if isinstance(raw, dict) else raw
    if not isinstance(cases, list):
        return (FixtureResult(path, path.stem, "schema", False, "fixture file must contain a case list"),)

    results: list[FixtureResult] = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            results.append(FixtureResult(path, f"case_{index}", "schema", False, "case must be an object"))
            continue
        results.append(_run_case(path, case, index))
    return tuple(results)


def _run_case(path: Path, case: dict[str, Any], index: int) -> FixtureResult:
    name = str(case.get("name", f"case_{index}"))
    kind = str(case.get("kind", "score"))
    try:
        if kind == "score":
            return _run_score_case(path, name, case)
        if kind in {"transition", "play_transition", "discard_transition"}:
            return _run_transition_case(path, name, kind, case)
        if kind == "known_gap":
            reason = str(case.get("known_gap_reason", "")).strip()
            if not reason:
                return FixtureResult(path, name, kind, False, "known_gap cases require known_gap_reason")
            return FixtureResult(path, name, kind, True, f"known gap: {reason}")
        return FixtureResult(path, name, kind, False, f"unknown kind {kind!r}")
    except Exception as exc:  # pragma: no cover - surfaced as fixture failure text.
        return FixtureResult(path, name, kind, False, f"{type(exc).__name__}: {exc}")


def _run_score_case(path: Path, name: str, case: dict[str, Any]) -> FixtureResult:
    blind = str(case.get("blind", ""))
    evaluation = evaluate_played_cards(
        _parse_cards(case.get("cards", "")),
        _parse_hand_levels(case.get("hand_levels", {})),
        debuffed_suits=debuffed_suits_for_blind(blind),
        blind_name=blind,
        jokers=_parse_jokers(case.get("jokers", ())),
        discards_remaining=_int_value(case.get("discards_remaining")),
        hands_remaining=_int_value(case.get("hands_remaining")),
        held_cards=_parse_cards(case.get("held", ())),
        deck_size=_int_value(case.get("deck_size")),
        money=_int_value(case.get("money")),
        played_hand_types_this_round=tuple(str(item) for item in case.get("played_hand_types_this_round", ())),
        played_hand_counts=_parse_hand_levels(case.get("played_hand_counts", {})),
    )
    actual = {
        "score": evaluation.score,
        "chips": evaluation.chips,
        "mult": evaluation.mult,
        "xmult": evaluation.effect_xmult,
        "hand_type": evaluation.hand_type.value,
        "scoring_indices": list(evaluation.scoring_indices),
    }
    return _compare_expected(path, name, "score", actual, _mapping(case.get("expected")))


def _run_transition_case(path: Path, name: str, kind: str, case: dict[str, Any]) -> FixtureResult:
    state = _parse_state(_mapping(case.get("pre_state")))
    action = Action.from_mapping(_mapping(case.get("action")))
    drawn_cards = _parse_cards(case.get("drawn_cards", ()))
    if action.action_type == ActionType.PLAY_HAND:
        simulated = simulate_play(state, action, drawn_cards)
    elif action.action_type == ActionType.DISCARD:
        simulated = simulate_discard(state, action, drawn_cards)
    else:
        return FixtureResult(path, name, kind, False, f"unsupported transition action {action.action_type.value}")

    actual = _state_projection(simulated, _mapping(case.get("expected_state")).keys())
    return _compare_expected(path, name, kind, actual, _mapping(case.get("expected_state")))


def _compare_expected(
    path: Path,
    name: str,
    kind: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> FixtureResult:
    if not expected:
        return FixtureResult(path, name, kind, False, "missing expected values")
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if not _values_match(actual_value, expected_value):
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    if mismatches:
        return FixtureResult(path, name, kind, False, "; ".join(mismatches))
    return FixtureResult(path, name, kind, True)


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return abs(float(actual) - float(expected)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return actual == expected


def _state_projection(state: GameState, keys: Iterable[str]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for key in keys:
        if key == "phase":
            projected[key] = state.phase.value
        elif key == "hand":
            projected[key] = [card.short_name for card in state.hand]
        elif key == "jokers":
            projected[key] = [joker.name for joker in state.jokers]
        else:
            projected[key] = getattr(state, key)
    return projected


def _parse_state(raw: dict[str, Any]) -> GameState:
    return GameState(
        phase=_parse_phase(raw.get("phase", GamePhase.UNKNOWN.value)),
        stake=_parse_stake(raw.get("stake", Stake.UNKNOWN.value)),
        seed=raw.get("seed"),
        ante=_int_value(raw.get("ante")),
        blind=str(raw.get("blind", "")),
        required_score=_int_value(raw.get("required_score")),
        current_score=_int_value(raw.get("current_score")),
        hands_remaining=_int_value(raw.get("hands_remaining")),
        discards_remaining=_int_value(raw.get("discards_remaining")),
        money=_int_value(raw.get("money")),
        deck_size=_int_value(raw.get("deck_size")),
        hand=_parse_cards(raw.get("hand", ())),
        known_deck=_parse_cards(raw.get("known_deck", ())),
        jokers=_parse_jokers(raw.get("jokers", ())),
        consumables=tuple(str(item) for item in raw.get("consumables", ())),
        vouchers=tuple(str(item) for item in raw.get("vouchers", ())),
        shop=tuple(str(item) for item in raw.get("shop", ())),
        pack=tuple(str(item) for item in raw.get("pack", ())),
        hand_levels=_parse_hand_levels(raw.get("hand_levels", {})),
        modifiers=dict(_mapping(raw.get("modifiers"))),
        run_over=bool(raw.get("run_over", False)),
        won=bool(raw.get("won", False)),
    )


def _parse_cards(raw: Any) -> tuple[Card, ...]:
    if isinstance(raw, list | tuple):
        return tuple(Card.from_mapping(card) for card in raw)
    if raw is None or not str(raw).strip():
        return ()
    return tuple(Card.from_mapping(token) for token in str(raw).replace(",", " ").split())


def _parse_jokers(raw: Any) -> tuple[Joker, ...]:
    if isinstance(raw, list | tuple):
        return tuple(Joker.from_mapping(joker) for joker in raw)
    if raw is None or not str(raw).strip():
        return ()
    return tuple(Joker.from_mapping(token.strip()) for token in str(raw).split(",") if token.strip())


def _parse_hand_levels(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {str(name): _int_value(value) for name, value in raw.items()}


def _parse_phase(raw: Any) -> GamePhase:
    try:
        return GamePhase(str(raw).lower())
    except ValueError:
        return GamePhase.UNKNOWN


def _parse_stake(raw: Any) -> Stake:
    try:
        return Stake(str(raw).lower())
    except ValueError:
        return Stake.UNKNOWN


def _mapping(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _int_value(raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
