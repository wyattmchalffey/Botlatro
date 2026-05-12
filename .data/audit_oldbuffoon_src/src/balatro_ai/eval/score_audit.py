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
    known_current_jokers: tuple[str, ...] = ()
    hand_counts_known: bool = False

    @property
    def error(self) -> int:
        return self.actual_score_delta - self.predicted_score

    @property
    def absolute_error(self) -> int:
        return abs(self.error)

    @property
    def uncertainty_reason(self) -> str | None:
        joker_text = " ".join(self.jokers)
        money_scaled_score = "Bull" in joker_text or "Bootstraps" in joker_text
        if "Misprint" in joker_text:
            return "Misprint has random mult"
        if money_scaled_score and "Business Card" in joker_text:
            return "Business Card can randomly add scoring-time dollars before money-scaled jokers"
        if money_scaled_score and "Reserved Parking" in joker_text:
            return "Reserved Parking can randomly add scoring-time dollars before money-scaled jokers"
        if money_scaled_score and "Matador" in joker_text:
            return "Matador can add boss-triggered scoring-time dollars before money-scaled jokers"
        if money_scaled_score and "To Do List" in joker_text:
            return "To Do List depends on the current target hand before money-scaled jokers"
        if "Popcorn" in joker_text and "Popcorn" not in self.known_current_jokers:
            return "Popcorn's visible current mult was not logged in this replay row"
        if "Ice Cream" in joker_text and "Ice Cream" not in self.known_current_jokers:
            return "Ice Cream's visible current chip counter was not logged in this replay row"
        if "Ramen" in joker_text and "Ramen" not in self.known_current_jokers:
            return "Ramen's visible current XMult was not logged in this replay row"
        if "Shoot the Moon" in joker_text and not self.held_cards_known:
            return "Shoot the Moon depends on held Queens, missing from older score audit rows"
        if "Loyalty Card" in joker_text and "Loyalty Card" not in self.known_current_jokers:
            return "Loyalty Card's visible countdown was not logged in this replay row"
        if "Bloodstone" in joker_text:
            return "Bloodstone has probabilistic XMult triggers"
        if "Space Joker" in joker_text:
            return "Space Joker can randomly upgrade hand level before scoring"
        if "Obelisk" in joker_text:
            return "Obelisk depends on prior hand-type history and can reset before scoring"
        if "Green Joker" in joker_text and "Green Joker" not in self.known_current_jokers:
            return "Green Joker's visible mult counter was not logged in this replay row"
        if "Card Sharp" in joker_text:
            return "Card Sharp depends on previous hand types this round, not included in score audit rows yet"
        if "Ceremonial Dagger" in joker_text:
            return "Ceremonial Dagger's current mult is dynamic and not exposed in replay state"
        if "Certificate" in joker_text:
            return "Certificate-created sealed cards can differ in older replay hand/deck snapshots"
        if "Square Joker" in joker_text and "Square Joker" not in self.known_current_jokers:
            return "Square Joker's visible chip counter was not logged in this replay row"
        if "Supernova" in joker_text and not self.hand_counts_known and "Supernova" not in self.known_current_jokers:
            return "Supernova depends on visible hand play counts that were not logged in this replay row"
        if "Raised Fist" in joker_text and not self.held_cards_known:
            return "Raised Fist depends on held cards, missing from older score audit rows"
        if "Blue Joker" in joker_text and not self.deck_size_known:
            return "Blue Joker depends on live deck size, missing from older score audit rows"
        if self.blind == "The Pillar":
            return "The Pillar debuffs previously played cards not yet represented in card state"
        if self.blind == "The Eye":
            return "The Eye depends on prior hand-type history, not included in isolated score audit rows"
        if self.blind == "The Mouth" and not self.hand_counts_known:
            return "The Mouth can zero hands after the first played hand type this round"
        if self.blind == "The Hook":
            return "The Hook discards held cards before scoring, changing held-card effects"
        if self.blind == "Cerulean Bell":
            return "Cerulean Bell forced selection can make bridge-scored cards differ from requested action cards"
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
                    known_current_jokers=_known_current_jokers(audit),
                    hand_counts_known="played_hand_counts" in audit or "hands" in audit,
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


def _known_current_jokers(audit: dict) -> tuple[str, ...]:
    known: list[str] = []
    for joker in audit.get("joker_details", ()):
        if not isinstance(joker, dict):
            continue
        name = str(joker.get("name", ""))
        metadata = joker.get("metadata")
        if name in {"Ice Cream", "Square Joker", "Stone Joker", "Castle"} and _has_current_value(metadata, "chips"):
            known.append(name)
        elif name in {"Popcorn", "Supernova", "Green Joker", "Ride the Bus"} and _has_current_value(metadata, "mult"):
            known.append(name)
        elif name == "Ramen" and (_has_current_value(metadata, "xmult") or _has_ramen_effect_value(metadata)):
            known.append(name)
        elif name == "Loyalty Card" and _has_countdown_value(metadata):
            known.append(name)
    return tuple(known)


def _has_current_value(metadata: object, suffix: str) -> bool:
    if not isinstance(metadata, dict):
        return False
    keys = {
        "chips": ("current_chips", "chips"),
        "mult": ("current_mult", "mult"),
    }.get(suffix, (f"current_{suffix}", suffix))
    for source in _metadata_sources(metadata):
        if any(key in source for key in keys):
            return True
        value = source.get("effect")
        if isinstance(value, str) and "currently" in value.lower():
            return True
    value = metadata.get("value")
    if isinstance(value, dict):
        effect = value.get("effect")
        if isinstance(effect, str) and "currently" in effect.lower():
            return True
    return False


def _has_countdown_value(metadata: object) -> bool:
    if not isinstance(metadata, dict):
        return False
    for source in _metadata_sources(metadata):
        if any(key in source for key in ("current_remaining", "remaining", "hands_remaining", "hands_left")):
            return True
        value = source.get("effect")
        if isinstance(value, str) and ("remaining" in value.lower() or "active" in value.lower()):
            return True
    value = metadata.get("value")
    if isinstance(value, dict):
        effect = value.get("effect")
        if isinstance(effect, str) and ("remaining" in effect.lower() or "active" in effect.lower()):
            return True
    return False


def _has_ramen_effect_value(metadata: object) -> bool:
    if not isinstance(metadata, dict):
        return False
    for source in _metadata_sources(metadata):
        value = source.get("effect")
        if isinstance(value, str) and "x" in value.lower() and "mult" in value.lower():
            return True
    value = metadata.get("value")
    if isinstance(value, dict):
        effect = value.get("effect")
        return isinstance(effect, str) and "x" in effect.lower() and "mult" in effect.lower()
    return False


def _metadata_sources(metadata: dict) -> tuple[dict, ...]:
    sources = [metadata]
    for key in ("ability", "config", "extra"):
        value = metadata.get(key)
        if isinstance(value, dict):
            sources.append(value)
            nested_extra = value.get("extra")
            if isinstance(nested_extra, dict):
                sources.append(nested_extra)
    return tuple(sources)


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
