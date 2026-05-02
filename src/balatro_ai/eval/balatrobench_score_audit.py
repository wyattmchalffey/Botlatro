"""Audit score predictions against extracted BalatroBench game states."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from balatro_ai.api.state import Card, GamePhase, GameState
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards


UNCERTAIN_JOKERS = {
    "Misprint": "random joker effect",
    "Bloodstone": "probabilistic XMult trigger",
    "Space Joker": "probabilistic hand-level upgrade",
    "Ramen": "fractional XMult / rounding sensitive",
}

UNCERTAIN_BLINDS = {
    "The Hook": "held-card discard changes score context",
    "Cerulean Bell": "forced-selection can alter requested cards",
    "The Mouth": "can zero repeated hand types; needs careful round-history alignment",
}


@dataclass(frozen=True, slots=True)
class BalatroBenchScoreRecord:
    run_path: Path
    transition_index: int
    ante: int
    round_num: int
    blind: str
    hand_type: str
    cards: tuple[str, ...]
    jokers: tuple[str, ...]
    predicted_score: int
    actual_score_delta: int
    uncertainty_reason: str | None = None

    @property
    def error(self) -> int:
        return self.actual_score_delta - self.predicted_score

    @property
    def absolute_error(self) -> int:
        return abs(self.error)

    @property
    def is_uncertain(self) -> bool:
        return self.uncertainty_reason is not None

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["run_path"] = str(self.run_path)
        payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class BalatroBenchScoreAudit:
    records: tuple[BalatroBenchScoreRecord, ...]
    runs_scanned: int
    skips: Counter[str]
    errors: tuple[str, ...] = ()

    @property
    def exact_matches(self) -> int:
        return sum(1 for record in self.records if record.error == 0)

    @property
    def supported_records(self) -> tuple[BalatroBenchScoreRecord, ...]:
        return tuple(record for record in self.records if not record.is_uncertain)

    @property
    def supported_exact_matches(self) -> int:
        return sum(1 for record in self.supported_records if record.error == 0)

    @property
    def supported_misses(self) -> tuple[BalatroBenchScoreRecord, ...]:
        return tuple(record for record in self.supported_records if record.error != 0)

    @property
    def uncertain_records(self) -> tuple[BalatroBenchScoreRecord, ...]:
        return tuple(record for record in self.records if record.is_uncertain)

    @property
    def mean_absolute_error(self) -> float:
        if not self.records:
            return 0.0
        return mean(record.absolute_error for record in self.records)

    def to_text(self, *, worst_count: int = 10) -> str:
        lines = [
            "BalatroBench score audit",
            f"Runs scanned: {self.runs_scanned}",
            f"Inferred play records: {len(self.records)}",
            f"Exact matches: {self.exact_matches}",
            f"Mean absolute error: {self.mean_absolute_error:.1f}",
            f"Supported records: {len(self.supported_records)}",
            f"Supported exact matches: {self.supported_exact_matches}",
            f"Supported misses: {len(self.supported_misses)}",
            f"Known uncertain records: {len(self.uncertain_records)}",
            f"Errors: {len(self.errors)}",
        ]
        if self.skips:
            lines.append("Skips: " + json.dumps(dict(sorted(self.skips.items())), sort_keys=True))
        if self.supported_misses:
            lines.append("")
            lines.append(f"Worst supported misses ({min(worst_count, len(self.supported_misses))}):")
            for record in sorted(self.supported_misses, key=lambda item: item.absolute_error, reverse=True)[:worst_count]:
                lines.append(
                    f"- err={record.error} predicted={record.predicted_score} actual={record.actual_score_delta} "
                    f"ante={record.ante} blind={record.blind or '-'} hand={record.hand_type} "
                    f"cards={' '.join(record.cards)} jokers={list(record.jokers)} "
                    f"file={record.run_path.name}:{record.transition_index}"
                )
        return "\n".join(lines)


def audit_balatrobench_runs(paths: Iterable[Path]) -> BalatroBenchScoreAudit:
    run_paths = tuple(_expand_run_paths(paths))
    records: list[BalatroBenchScoreRecord] = []
    skips: Counter[str] = Counter()
    errors: list[str] = []
    for run_path in run_paths:
        try:
            rows = _load_jsonl(run_path / "gamestates.jsonl")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{run_path}: {exc!r}")
            continue
        for index in range(1, len(rows)):
            record = _record_from_transition(run_path, index, rows[index - 1], rows[index], skips, errors)
            if record is not None:
                records.append(record)
    return BalatroBenchScoreAudit(records=tuple(records), runs_scanned=len(run_paths), skips=skips, errors=tuple(errors))


def _record_from_transition(
    run_path: Path,
    index: int,
    before: dict,
    after: dict,
    skips: Counter[str],
    errors: list[str],
) -> BalatroBenchScoreRecord | None:
    try:
        state = GameState.from_mapping(before)
        next_state = GameState.from_mapping(after)
    except (TypeError, ValueError) as exc:
        errors.append(f"{run_path}:{index + 1}: state parse failed: {exc!r}")
        return None
    if state.phase not in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND}:
        skips[f"pre_phase_{state.phase.value}"] += 1
        return None
    if next_state.ante != state.ante or after.get("round_num") != before.get("round_num"):
        skips["round_or_ante_changed"] += 1
        return None
    actual_delta = next_state.current_score - state.current_score
    if actual_delta <= 0:
        skips["non_score_transition"] += 1
        return None
    if not next_state.hand:
        skips["post_hand_empty"] += 1
        return None

    selected_cards, held_cards = _infer_selected_cards(state.hand, next_state.hand)
    if not 1 <= len(selected_cards) <= 5:
        skips[f"selected_count_{len(selected_cards)}"] += 1
        return None

    try:
        evaluation = evaluate_played_cards(
            selected_cards,
            state.hand_levels,
            debuffed_suits=debuffed_suits_for_blind(state.blind),
            blind_name=state.blind,
            jokers=state.jokers,
            discards_remaining=state.discards_remaining,
            hands_remaining=state.hands_remaining,
            held_cards=held_cards,
            deck_size=state.deck_size,
            money=state.money,
            played_hand_types_this_round=_played_hand_types_this_round(state),
            played_hand_counts=_played_hand_counts(state),
        )
    except (TypeError, ValueError) as exc:
        skips["evaluation_error"] += 1
        errors.append(f"{run_path}:{index + 1}: evaluation failed: {exc!r}")
        return None

    return BalatroBenchScoreRecord(
        run_path=run_path,
        transition_index=index + 1,
        ante=state.ante,
        round_num=_int_value(before.get("round_num")),
        blind=state.blind,
        hand_type=evaluation.hand_type.value,
        cards=tuple(card.short_name for card in selected_cards),
        jokers=tuple(joker.name for joker in state.jokers),
        predicted_score=evaluation.score,
        actual_score_delta=actual_delta,
        uncertainty_reason=_uncertainty_reason(state, selected_cards),
    )


def _infer_selected_cards(before: tuple[Card, ...], after: tuple[Card, ...]) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    after_ids = Counter(_card_id(card) for card in after if _card_id(card) is not None)
    selected: list[Card] = []
    held: list[Card] = []
    for card in before:
        card_id = _card_id(card)
        if card_id is not None and after_ids[card_id] > 0:
            after_ids[card_id] -= 1
            held.append(card)
        else:
            selected.append(card)
    return tuple(selected), tuple(held)


def _card_id(card: Card) -> object:
    return card.metadata.get("id")


def _played_hand_types_this_round(state: GameState) -> tuple[str, ...]:
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, dict):
        return ()
    played: list[tuple[int, str]] = []
    for name, value in hands.items():
        if not isinstance(value, dict):
            continue
        played.extend((_int_value(value.get("order")), str(name)) for _ in range(_int_value(value.get("played_this_round"))))
    return tuple(name for _, name in sorted(played))


def _played_hand_counts(state: GameState) -> dict[str, int]:
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, dict):
        return {}
    return {str(name): _int_value(value.get("played")) for name, value in hands.items() if isinstance(value, dict)}


def _uncertainty_reason(state: GameState, selected_cards: tuple[Card, ...]) -> str | None:
    if any(str(card.enhancement or "").upper() == "LUCKY" for card in selected_cards):
        return "Lucky Card: probabilistic +20 Mult trigger"
    for joker in state.jokers:
        if joker.name in UNCERTAIN_JOKERS:
            return f"{joker.name}: {UNCERTAIN_JOKERS[joker.name]}"
    if state.blind in UNCERTAIN_BLINDS:
        return f"{state.blind}: {UNCERTAIN_BLINDS[state.blind]}"
    return None


def _load_jsonl(path: Path) -> tuple[dict, ...]:
    with path.open("r", encoding="utf-8") as file:
        return tuple(json.loads(line) for line in file if line.strip())


def _expand_run_paths(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            continue
        candidates = (resolved,) if (resolved / "gamestates.jsonl").exists() else tuple(parent for parent in resolved.rglob("gamestates.jsonl"))
        for candidate in candidates:
            run_path = candidate.parent if candidate.name == "gamestates.jsonl" else candidate
            if run_path not in seen:
                seen.add(run_path)
                yield run_path


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit extracted BalatroBench gamestates against the local score evaluator.")
    parser.add_argument("paths", nargs="+", type=Path, help="Extracted BalatroBench run directories or parent directories.")
    parser.add_argument("--worst", type=int, default=10, help="Number of largest supported misses to print.")
    parser.add_argument("--jsonl-out", type=Path, help="Optional path for per-record JSONL output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_balatrobench_runs(args.paths)
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_out.write_text(
            "\n".join(json.dumps(record.to_json()) for record in audit.records) + ("\n" if audit.records else ""),
            encoding="utf-8",
        )
    print(audit.to_text(worst_count=max(0, args.worst)))
    return 0 if not audit.supported_misses else 1


if __name__ == "__main__":
    raise SystemExit(main())
