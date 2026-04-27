"""Evaluate a hand-scoring scenario without launching Balatro."""

from __future__ import annotations

import argparse

from balatro_ai.api.state import Card, Joker
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a deterministic evaluator scenario.")
    parser.add_argument("--cards", required=True, help="Played cards, e.g. 'KS QS JS TS 9H'.")
    parser.add_argument("--jokers", default="", help="Comma-separated joker names.")
    parser.add_argument("--held", default="", help="Held cards after play, e.g. 'QH KD'.")
    parser.add_argument("--blind", default="", help="Blind name, e.g. 'The Club'.")
    parser.add_argument("--hands-remaining", type=int, default=0, help="Hands remaining before play.")
    parser.add_argument("--discards-remaining", type=int, default=0, help="Discards remaining before play.")
    parser.add_argument("--deck-size", type=int, default=0, help="Live deck size for Blue Joker.")
    parser.add_argument(
        "--hand-level",
        action="append",
        default=[],
        help="Hand level override like 'Pair=2'. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation = evaluate_played_cards(
        _parse_cards(args.cards),
        _parse_hand_levels(args.hand_level),
        debuffed_suits=debuffed_suits_for_blind(args.blind),
        blind_name=args.blind,
        jokers=_parse_jokers(args.jokers),
        discards_remaining=args.discards_remaining,
        hands_remaining=args.hands_remaining,
        held_cards=_parse_cards(args.held),
        deck_size=args.deck_size,
    )
    print(f"Hand type: {evaluation.hand_type.value}")
    print(f"Scoring indices: {', '.join(str(index) for index in evaluation.scoring_indices) or '-'}")
    print(f"Chips: {evaluation.chips}")
    print(f"Mult: {evaluation.mult}")
    print(f"XMult: {evaluation.effect_xmult:.6g}")
    print(f"Score: {evaluation.score}")
    return 0


def _parse_cards(raw: str) -> tuple[Card, ...]:
    if not raw.strip():
        return ()
    tokens = raw.replace(",", " ").split()
    return tuple(Card.from_mapping(token) for token in tokens)


def _parse_jokers(raw: str) -> tuple[Joker, ...]:
    if not raw.strip():
        return ()
    jokers: list[Joker] = []
    for token in raw.split(","):
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


if __name__ == "__main__":
    raise SystemExit(main())
