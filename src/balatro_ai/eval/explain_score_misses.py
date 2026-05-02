"""Explain the largest predicted-vs-actual score misses in replay logs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from balatro_ai.api.state import Card, Joker
from balatro_ai.eval.score_audit import audit_replays
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards


@dataclass(frozen=True, slots=True)
class MissExplanation:
    replay_path: Path
    line_number: int
    seed: int | None
    ante: int
    blind: str
    cards: tuple[Card, ...]
    card_indices: tuple[int, ...]
    hand_before: tuple[Card, ...]
    held_cards: tuple[Card, ...]
    jokers: tuple[Joker, ...]
    stored_predicted_score: int
    actual_score_delta: int
    recomputed_score: int
    recomputed_chips: int
    recomputed_mult: int
    recomputed_xmult: float
    hand_type: str
    stored_hand_type: str
    suspected_causes: tuple[str, ...]

    @property
    def error(self) -> int:
        return self.actual_score_delta - self.recomputed_score

    @property
    def stored_error(self) -> int:
        return self.actual_score_delta - self.stored_predicted_score

    @property
    def absolute_error(self) -> int:
        return abs(self.error)


def explain_replays(
    paths: Iterable[Path],
    *,
    worst_count: int = 10,
    include_uncertain: bool = False,
) -> str:
    explanations = collect_score_explanations(paths, include_uncertain=include_uncertain)
    miss_explanations = tuple(item for item in explanations if item.error != 0)
    selected = sorted(miss_explanations, key=lambda item: item.absolute_error, reverse=True)[:worst_count]

    lines = [
        "Score miss explanations",
        f"Rows explained: {len(explanations)}",
        f"Rows with misses: {len(miss_explanations)}",
    ]
    if miss_explanations:
        lines.append(f"Mean absolute miss: {mean(item.absolute_error for item in miss_explanations):.1f}")
    if not selected:
        return "\n".join(lines)

    lines.append("")
    for index, item in enumerate(selected, start=1):
        lines.extend(_format_explanation(index, item))
        lines.append("")
    return "\n".join(lines).rstrip()


def collect_score_explanations(
    paths: Iterable[Path],
    *,
    include_uncertain: bool = False,
) -> tuple[MissExplanation, ...]:
    """Recompute replay score-audit rows with the current evaluator."""

    return tuple(_explanations_from_paths(paths, include_uncertain=include_uncertain))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain replay score prediction misses.")
    parser.add_argument("paths", nargs="*", type=Path, help="Replay JSONL file(s) or directories.")
    parser.add_argument("--replay-dir", type=Path, help="Replay directory to scan recursively.")
    parser.add_argument("--worst", type=int, default=10, help="Number of largest misses to explain.")
    parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Include rows marked as known-uncertain by score_audit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = list(args.paths)
    if args.replay_dir is not None:
        paths.append(args.replay_dir)
    if not paths:
        raise SystemExit("Provide at least one replay path or --replay-dir.")

    print(
        explain_replays(
            paths,
            worst_count=max(0, args.worst),
            include_uncertain=args.include_uncertain,
        )
    )
    return 0


def _explanations_from_paths(paths: Iterable[Path], *, include_uncertain: bool) -> Iterable[MissExplanation]:
    audit_summary = audit_replays(paths)
    uncertain_keys = {
        (record.replay_path.resolve(), record.line_number)
        for record in audit_summary.uncertain_records
    }
    for replay_path in _expand_paths(paths):
        with replay_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                if not include_uncertain and (replay_path.resolve(), line_number) in uncertain_keys:
                    continue
                payload = json.loads(line)
                audit = payload.get("extra", {}).get("score_audit")
                if not isinstance(audit, dict):
                    continue
                explanation = _explanation_from_payload(replay_path, line_number, payload, audit)
                if explanation is not None:
                    yield explanation


def _explanation_from_payload(
    replay_path: Path,
    line_number: int,
    payload: dict[str, Any],
    audit: dict[str, Any],
) -> MissExplanation | None:
    card_details = _list_of_dicts(audit.get("card_details"))
    if not card_details:
        return None

    cards = tuple(_card_from_detail(detail) for detail in card_details)
    held_cards = tuple(_card_from_detail(detail) for detail in _list_of_dicts(audit.get("held_card_details")))
    hand_before = tuple(_card_from_detail(detail) for detail in _list_of_dicts(audit.get("hand_before_details")))
    jokers = tuple(_joker_from_detail(detail) for detail in _list_of_dicts(audit.get("joker_details")))
    hand_levels = _int_mapping(audit.get("hand_levels"))
    blind = str(audit.get("blind", ""))
    evaluation = evaluate_played_cards(
        cards,
        hand_levels,
        debuffed_suits=debuffed_suits_for_blind(blind),
        blind_name=blind,
        jokers=jokers,
        discards_remaining=int(audit.get("discards_remaining", 0)),
        hands_remaining=int(audit.get("hands_remaining", 0)),
        held_cards=held_cards,
        deck_size=int(audit.get("deck_size", 0)),
        money=int(audit.get("money", 0)),
        played_hand_counts=_played_hand_counts(audit),
    )

    stored_predicted = int(audit.get("predicted_score", 0))
    actual = int(audit.get("actual_score_delta", 0))
    return MissExplanation(
        replay_path=replay_path,
        line_number=line_number,
        seed=_optional_int(payload.get("seed")),
        ante=int(audit.get("ante", 0)),
        blind=blind,
        cards=cards,
        card_indices=_card_indices(payload),
        hand_before=hand_before,
        held_cards=held_cards,
        jokers=jokers,
        stored_predicted_score=stored_predicted,
        actual_score_delta=actual,
        recomputed_score=evaluation.score,
        recomputed_chips=evaluation.chips,
        recomputed_mult=evaluation.mult,
        recomputed_xmult=evaluation.effect_xmult,
        hand_type=evaluation.hand_type.value,
        stored_hand_type=str(audit.get("hand_type", "")),
        suspected_causes=_suspected_causes(
            audit=audit,
            cards=cards,
            hand_before=hand_before,
            held_cards=held_cards,
            jokers=jokers,
            evaluation_score=evaluation.score,
            stored_predicted=stored_predicted,
            actual=actual,
        ),
    )


def _format_explanation(index: int, item: MissExplanation) -> list[str]:
    lines = [
        (
            f"{index}. seed={item.seed} ante={item.ante} blind={item.blind or '-'} "
            f"file={item.replay_path.name}:{item.line_number}"
        ),
        (
            f"   cards={_card_names(item.cards)} indexes={_index_text(item.card_indices)} "
            f"hand_type={item.hand_type}"
        ),
        f"   hand_before={_card_names(item.hand_before) or '-'}",
        f"   held={_card_names(item.held_cards) or '-'}",
        f"   jokers={_joker_text(item.jokers) or '-'}",
        (
            f"   score stored_pred={item.stored_predicted_score} "
            f"recomputed={item.recomputed_score} actual={item.actual_score_delta} "
            f"current_error={item.error} stored_error={item.stored_error}"
        ),
        (
            f"   recomputed pieces: chips={item.recomputed_chips} "
            f"mult={item.recomputed_mult} xmult={item.recomputed_xmult:.3g}"
        ),
    ]
    if item.stored_hand_type and item.stored_hand_type != item.hand_type:
        lines.append(f"   stored_hand_type={item.stored_hand_type}")
    lines.append(f"   suspects={'; '.join(item.suspected_causes) or 'none'}")
    return lines


def _suspected_causes(
    *,
    audit: dict[str, Any],
    cards: tuple[Card, ...],
    hand_before: tuple[Card, ...],
    held_cards: tuple[Card, ...],
    jokers: tuple[Joker, ...],
    evaluation_score: int,
    stored_predicted: int,
    actual: int,
) -> tuple[str, ...]:
    causes: list[str] = []
    joker_names = {joker.name for joker in jokers}
    if stored_predicted != evaluation_score:
        causes.append("stored prediction differs from current evaluator; replay was likely generated before latest code")
    if any(card.debuffed for card in cards):
        causes.append("played hand includes debuffed cards")
    if any(card.debuffed for card in held_cards):
        causes.append("held hand includes debuffed cards")
    if audit.get("blind") == "The Pillar":
        causes.append("The Pillar can debuff previously played cards")
    if "Green Joker" in joker_names:
        causes.append("Green Joker current counter timing may differ between gamestate and scoring")
    if "Ride the Bus" in joker_names:
        causes.append("Ride the Bus counter only scores on hands without scoring face cards")
    if "Certificate" in joker_names:
        causes.append("Certificate can create sealed cards; check raw hand/card metadata")
    if not hand_before:
        causes.append("older replay row has no hand-before details")
    if not held_cards and any(name in joker_names for name in {"Shoot the Moon", "Raised Fist", "Baron", "Blackboard"}):
        causes.append("held-card joker present but held cards were not logged")
    if actual == 0:
        causes.append("actual score delta is zero; bridge may have returned terminal/round transition state")
    return tuple(causes)


def _expand_paths(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            continue
        candidates = resolved.rglob("*.jsonl") if resolved.is_dir() else (resolved,)
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _list_of_dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _card_from_detail(detail: dict[str, Any]) -> Card:
    return Card(
        rank=str(detail.get("rank", "")),
        suit=str(detail.get("suit", "")),
        enhancement=detail.get("enhancement"),
        seal=detail.get("seal"),
        edition=detail.get("edition"),
        debuffed=bool(detail.get("debuffed", False)),
        metadata=dict(detail.get("metadata", {})) if isinstance(detail.get("metadata"), dict) else {},
    )


def _joker_from_detail(detail: dict[str, Any]) -> Joker:
    return Joker(
        name=str(detail.get("name", "Unknown Joker")),
        edition=detail.get("edition"),
        sell_value=_optional_int(detail.get("sell_value")),
        metadata=dict(detail.get("metadata", {})) if isinstance(detail.get("metadata"), dict) else {},
    )


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(item) for key, item in value.items()}


def _played_hand_counts(audit: dict[str, Any]) -> dict[str, int]:
    direct = audit.get("played_hand_counts")
    if isinstance(direct, dict):
        return {str(key): _int_value(value) for key, value in direct.items()}

    hands = audit.get("hands")
    if not isinstance(hands, dict):
        return {}
    counts: dict[str, int] = {}
    for name, value in hands.items():
        if isinstance(value, dict):
            counts[str(name)] = _hand_play_count(value)
        else:
            counts[str(name)] = _int_value(value)
    return counts


def _hand_play_count(value: dict[str, Any]) -> int:
    for key in ("played", "played_this_run", "played_count", "times_played", "count"):
        if key in value:
            return _int_value(value[key])
    return _int_value(value.get("played_this_round"))


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _card_indices(payload: dict[str, Any]) -> tuple[int, ...]:
    action = payload.get("chosen_action", {})
    if not isinstance(action, dict):
        return ()
    indices = action.get("card_indices", ())
    if not isinstance(indices, list | tuple):
        return ()
    return tuple(int(index) for index in indices)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _card_names(cards: tuple[Card, ...]) -> str:
    labels: list[str] = []
    for card in cards:
        label = card.short_name
        extras = []
        if card.debuffed:
            extras.append("debuff")
        if card.enhancement:
            extras.append(str(card.enhancement))
        if card.edition:
            extras.append(str(card.edition))
        if extras:
            label += f"({','.join(extras)})"
        labels.append(label)
    return " ".join(labels)


def _joker_text(jokers: tuple[Joker, ...]) -> str:
    labels: list[str] = []
    for joker in jokers:
        effect = _joker_effect(joker)
        labels.append(f"{joker.name}: {effect}" if effect else joker.name)
    return " | ".join(labels)


def _joker_effect(joker: Joker) -> str:
    value = joker.metadata.get("value")
    if isinstance(value, dict):
        return str(value.get("effect", ""))
    return ""


def _index_text(indices: tuple[int, ...]) -> str:
    return ",".join(str(index) for index in indices) if indices else "-"


if __name__ == "__main__":
    raise SystemExit(main())
