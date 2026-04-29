"""Evaluate a hand-scoring scenario without launching Balatro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from balatro_ai.api.state import Card, Joker
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a deterministic evaluator scenario.")
    parser.add_argument("--scenario-file", help="JSON file containing one or more scoring scenarios.")
    parser.add_argument("--cards", help="Played cards, e.g. 'KS QS JS TS 9H'.")
    parser.add_argument("--jokers", default="", help="Comma-separated joker names.")
    parser.add_argument("--held", default="", help="Held cards after play, e.g. 'QH KD'.")
    parser.add_argument("--blind", default="", help="Blind name, e.g. 'The Club'.")
    parser.add_argument("--hands-remaining", type=int, default=0, help="Hands remaining before play.")
    parser.add_argument("--discards-remaining", type=int, default=0, help="Discards remaining before play.")
    parser.add_argument("--deck-size", type=int, default=0, help="Live deck size for Blue Joker.")
    parser.add_argument("--money", type=int, default=0, help="Current money for money-scaled jokers.")
    parser.add_argument(
        "--hand-level",
        action="append",
        default=[],
        help="Hand level override like 'Pair=2'. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scenario_file:
        return _run_scenario_file(Path(args.scenario_file))
    if not args.cards:
        raise SystemExit("--cards is required unless --scenario-file is provided.")

    evaluation = _evaluate_scenario(
        {
            "cards": args.cards,
            "hand_levels": _parse_hand_levels(args.hand_level),
            "blind": args.blind,
            "jokers": args.jokers,
            "discards_remaining": args.discards_remaining,
            "hands_remaining": args.hands_remaining,
            "held": args.held,
            "deck_size": args.deck_size,
            "money": args.money,
        }
    )
    print(f"Hand type: {evaluation.hand_type.value}")
    print(f"Scoring indices: {', '.join(str(index) for index in evaluation.scoring_indices) or '-'}")
    print(f"Chips: {evaluation.chips}")
    print(f"Mult: {evaluation.mult}")
    print(f"XMult: {evaluation.effect_xmult:.6g}")
    print(f"Score: {evaluation.score}")
    return 0


def _run_scenario_file(path: Path) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = raw.get("scenarios", raw) if isinstance(raw, dict) else raw
    if not isinstance(scenarios, list):
        raise SystemExit("Scenario file must be a JSON list or an object with a 'scenarios' list.")

    failures = 0
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            raise SystemExit(f"Scenario {index} must be an object.")
        evaluation = _evaluate_scenario(scenario)
        expected = scenario.get("expected", {})
        expected_score = expected.get("score") if isinstance(expected, dict) else None
        name = str(scenario.get("name", f"scenario_{index}"))
        ok = expected_score is None or evaluation.score == int(expected_score)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: score={evaluation.score}", end="")
        if expected_score is not None:
            print(f" expected={expected_score}", end="")
        print(f" hand={evaluation.hand_type.value}")
        if not ok:
            failures += 1
    return 1 if failures else 0


def _evaluate_scenario(scenario: dict[str, Any]):
    blind = str(scenario.get("blind", ""))
    return evaluate_played_cards(
        _parse_cards(scenario.get("cards", "")),
        _parse_hand_levels_from_mapping(scenario.get("hand_levels", {})),
        debuffed_suits=debuffed_suits_for_blind(blind),
        blind_name=blind,
        jokers=_parse_jokers(scenario.get("jokers", "")),
        discards_remaining=int(scenario.get("discards_remaining", 0)),
        hands_remaining=int(scenario.get("hands_remaining", 0)),
        held_cards=_parse_cards(scenario.get("held", "")),
        deck_size=int(scenario.get("deck_size", 0)),
        money=int(scenario.get("money", 0)),
    )


def _parse_cards(raw: Any) -> tuple[Card, ...]:
    if isinstance(raw, list):
        return tuple(Card.from_mapping(card) for card in raw)
    if raw is None or not str(raw).strip():
        return ()
    tokens = str(raw).replace(",", " ").split()
    return tuple(Card.from_mapping(token) for token in tokens)


def _parse_jokers(raw: Any) -> tuple[Joker, ...]:
    if isinstance(raw, list):
        return tuple(Joker.from_mapping(joker) for joker in raw)
    if raw is None or not str(raw).strip():
        return ()
    jokers: list[Joker] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if "::" in token:
            name, effect = token.split("::", 1)
            jokers.append(Joker(name.strip(), metadata={"value": {"effect": effect.strip()}}))
        else:
            jokers.append(Joker(token))
    return tuple(jokers)


def _parse_hand_levels(raw_levels: list[str]) -> dict[str, int]:
    levels: dict[str, int] = {}
    for raw in raw_levels:
        if "=" not in raw:
            raise SystemExit(f"Invalid --hand-level value {raw!r}; expected Name=Level.")
        name, value = raw.split("=", 1)
        levels[name.strip()] = int(value)
    return levels


def _parse_hand_levels_from_mapping(raw: Any) -> dict[str, int]:
    if isinstance(raw, dict):
        return {str(name): int(value) for name, value in raw.items()}
    if isinstance(raw, list):
        return _parse_hand_levels([str(item) for item in raw])
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
