"""Audit replay logs for predicted-vs-actual hand score gaps."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ScoreAuditRecord:
    replay_path: Path
    line_number: int
    seed: int | None
    cards: tuple[str, ...]
    hand_type: str
    predicted_score: int
    actual_score_delta: int
    ante: int
    blind: str
    jokers: tuple[str, ...]
    held_cards_known: bool = False
    deck_size_known: bool = False

    @property
    def error(self) -> int:
        return self.actual_score_delta - self.predicted_score

    @property
    def absolute_error(self) -> int:
        return abs(self.error)

    @property
    def uncertainty_reason(self) -> str | None:
        joker_text = " ".join(self.jokers)
        if "Misprint" in joker_text:
            return "Misprint has random mult"
        if "Popcorn" in joker_text:
            return "Popcorn's current mult is dynamic and not exposed in replay state"
        if "Ice Cream" in joker_text:
            return "Ice Cream's current chip counter is dynamic and not exposed in replay state"
        if "Shoot the Moon" in joker_text and not self.held_cards_known:
            return "Shoot the Moon depends on held Queens, missing from older score audit rows"
        if "Loyalty Card" in joker_text:
            return "Loyalty Card's hand counter is dynamic and not exposed in replay state"
        if "Card Sharp" in joker_text:
            return "Card Sharp depends on previous hand types this round, not included in score audit rows yet"
        if "Ceremonial Dagger" in joker_text:
            return "Ceremonial Dagger's current mult is dynamic and not exposed in replay state"
        if "Square Joker" in joker_text:
            return "Square Joker's current chip counter is dynamic and not exposed in replay state"
        if "Supernova" in joker_text:
            return "Supernova's current per-hand counter is dynamic and not exposed in replay state"
        if "Raised Fist" in joker_text and not self.held_cards_known:
            return "Raised Fist depends on held cards, missing from older score audit rows"
        if "Blue Joker" in joker_text and not self.deck_size_known:
            return "Blue Joker depends on live deck size, missing from older score audit rows"
        if self.blind == "The Pillar":
            return "The Pillar debuffs previously played cards not yet represented in card state"
        return None

    @property
    def is_uncertain(self) -> bool:
        return self.uncertainty_reason is not None


@dataclass(frozen=True, slots=True)
class ScoreAuditSummary:
    records: tuple[ScoreAuditRecord, ...]
    files_scanned: int

    @property
    def played_hands(self) -> int:
        return len(self.records)

    @property
    def exact_matches(self) -> int:
        return sum(1 for record in self.records if record.error == 0)

    @property
    def supported_records(self) -> tuple[ScoreAuditRecord, ...]:
        return tuple(record for record in self.records if not record.is_uncertain)

    @property
    def uncertain_records(self) -> tuple[ScoreAuditRecord, ...]:
        return tuple(record for record in self.records if record.is_uncertain)

    @property
    def mean_absolute_error(self) -> float:
        if not self.records:
            return 0.0
        return mean(record.absolute_error for record in self.records)

    def to_text(self, *, worst_count: int = 10) -> str:
        lines = [
            "Score audit",
            f"Files scanned: {self.files_scanned}",
            f"Played hands: {self.played_hands}",
            f"Exact matches: {self.exact_matches}",
            f"Mean absolute error: {self.mean_absolute_error:.1f}",
            f"Supported hands: {len(self.supported_records)}",
            f"Supported exact matches: {sum(1 for record in self.supported_records if record.error == 0)}",
            f"Known uncertain hands: {len(self.uncertain_records)}",
        ]
        if not self.records:
            return "\n".join(lines)

        lines.append("")
        lines.append("By hand type:")
        for hand_type, records in _group_by_hand_type(self.records):
            lines.append(
                f"- {hand_type}: n={len(records)} "
                f"avg_pred={mean(record.predicted_score for record in records):.1f} "
                f"avg_actual={mean(record.actual_score_delta for record in records):.1f} "
                f"avg_abs_err={mean(record.absolute_error for record in records):.1f}"
            )

        lines.append("")
        _append_worst_section(lines, "Worst supported", self.supported_records, worst_count)
        _append_worst_section(lines, "Worst known-uncertain", self.uncertain_records, worst_count)
        return "\n".join(lines)


def audit_replays(paths: Iterable[Path]) -> ScoreAuditSummary:
    replay_paths = tuple(_expand_paths(paths))
    records: list[ScoreAuditRecord] = []
    for replay_path in replay_paths:
        records.extend(_records_from_file(replay_path))
    return ScoreAuditSummary(records=tuple(records), files_scanned=len(replay_paths))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit predicted-vs-actual score gaps from replay JSONL files.")
    parser.add_argument("paths", nargs="*", type=Path, help="Replay JSONL file(s) or directories.")
    parser.add_argument("--replay-dir", type=Path, help="Replay directory to scan recursively.")
    parser.add_argument("--worst", type=int, default=10, help="Number of largest misses to print.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = list(args.paths)
    if args.replay_dir is not None:
        paths.append(args.replay_dir)
    if not paths:
        raise SystemExit("Provide at least one replay path or --replay-dir.")

    summary = audit_replays(paths)
    print(summary.to_text(worst_count=max(0, args.worst)))
    return 0


def _expand_paths(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            continue
        candidates = resolved.rglob("*.jsonl") if resolved.is_dir() else (resolved,)
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _records_from_file(path: Path) -> list[ScoreAuditRecord]:
    records: list[ScoreAuditRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            audit = payload.get("extra", {}).get("score_audit")
            if not audit:
                continue
            records.append(
                ScoreAuditRecord(
                    replay_path=path,
                    line_number=line_number,
                    seed=payload.get("seed"),
                    cards=tuple(str(card) for card in audit.get("cards", ())),
                    hand_type=str(audit.get("hand_type", "unknown")),
                    predicted_score=int(audit.get("predicted_score", 0)),
                    actual_score_delta=int(audit.get("actual_score_delta", 0)),
                    ante=int(audit.get("ante", 0)),
                    blind=str(audit.get("blind", "")),
                    jokers=_joker_labels(audit),
                    held_cards_known="held_card_details" in audit or "held_cards" in audit,
                    deck_size_known="deck_size" in audit,
                )
            )
    return records


def _joker_labels(audit: dict) -> tuple[str, ...]:
    details = audit.get("joker_details", ())
    if details:
        labels: list[str] = []
        for joker in details:
            if not isinstance(joker, dict):
                continue
            name = str(joker.get("name", "Unknown Joker"))
            edition = joker.get("edition")
            labels.append(f"{name} ({edition})" if edition else name)
        return tuple(labels)
    return tuple(str(joker) for joker in audit.get("jokers", ()))


def _group_by_hand_type(records: tuple[ScoreAuditRecord, ...]) -> Iterable[tuple[str, tuple[ScoreAuditRecord, ...]]]:
    hand_types = sorted({record.hand_type for record in records})
    for hand_type in hand_types:
        yield hand_type, tuple(record for record in records if record.hand_type == hand_type)


def _append_worst_section(
    lines: list[str],
    title: str,
    records: tuple[ScoreAuditRecord, ...],
    worst_count: int,
) -> None:
    if not records or worst_count <= 0:
        return

    lines.append("")
    lines.append(f"{title} {min(worst_count, len(records))}:")
    for record in sorted(records, key=lambda item: item.absolute_error, reverse=True)[:worst_count]:
        joker_text = ", ".join(record.jokers) or "-"
        reason = f" reason={record.uncertainty_reason}" if record.uncertainty_reason else ""
        lines.append(
            f"- seed={record.seed} ante={record.ante} blind={record.blind or '-'} {record.hand_type} "
            f"cards={' '.join(record.cards)} predicted={record.predicted_score} "
            f"actual={record.actual_score_delta} error={record.error} "
            f"jokers=[{joker_text}] file={record.replay_path.name}:{record.line_number}{reason}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
