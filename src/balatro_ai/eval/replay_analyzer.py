"""Summarize replay logs for bot behavior and decision quality."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from statistics import mean
from typing import Iterable


ANTE_PATTERN = re.compile(r"ante=(\d+)")
BLIND_PATTERN = re.compile(r"ante=(\d+) blind=(.*?) score=")
PRESSURE_PATTERN = re.compile(r"pressure=([0-9]+(?:\.[0-9]+)?)")
STATE_PATTERN = re.compile(
    r"phase=(?P<phase>\S+) ante=(?P<ante>\d+) blind=(?P<blind>.*?) "
    r"score=(?P<score>\d+)/(?P<required>\d+) money=(?P<money>-?\d+) "
    r"hands=(?P<hands>\d+) discards=(?P<discards>\d+) .* jokers=\[(?P<jokers>.*?)\]"
)


@dataclass(frozen=True, slots=True)
class RunReplaySummary:
    path: Path
    seed: str
    max_ante: int
    row_count: int
    action_counts: Counter[str]
    shop_reason_counts: Counter[str]
    pressure_values: tuple[float, ...]
    hands_per_blind: tuple[int, ...]
    sell_actions: int
    observed_win: bool = False
    outcome: str | None = None
    final_phase: str = ""
    final_blind: str = ""
    final_score: int = 0
    final_required_score: int = 0
    final_money: int = 0
    final_hands: int = 0
    final_jokers: tuple[str, ...] = ()
    shop_action_count: int = 0


@dataclass(frozen=True, slots=True)
class ReplayAnalysis:
    runs: tuple[RunReplaySummary, ...]
    malformed_rows: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    shop_reason_counts: Counter[str] = field(default_factory=Counter)
    pressure_values: tuple[float, ...] = ()
    hands_per_blind: tuple[int, ...] = ()

    @property
    def files_scanned(self) -> int:
        return len(self.runs)

    @property
    def average_ante(self) -> float:
        return mean(run.max_ante for run in self.runs) if self.runs else 0.0

    @property
    def ante_distribution(self) -> Counter[int]:
        return Counter(run.max_ante for run in self.runs)

    @property
    def observed_wins(self) -> int:
        return sum(1 for run in self.runs if run.observed_win)

    @property
    def outcome_distribution(self) -> Counter[str]:
        return Counter(run.outcome for run in self.runs if run.outcome)

    @property
    def average_hands_per_blind(self) -> float:
        return mean(self.hands_per_blind) if self.hands_per_blind else 0.0

    @property
    def hands_per_blind_distribution(self) -> Counter[int]:
        return Counter(self.hands_per_blind)

    @property
    def average_pressure(self) -> float:
        return mean(self.pressure_values) if self.pressure_values else 0.0

    @property
    def early_failures(self) -> tuple[RunReplaySummary, ...]:
        return tuple(run for run in self.runs if not run.observed_win and run.max_ante <= 2)

    @property
    def deep_losses(self) -> tuple[RunReplaySummary, ...]:
        return tuple(run for run in self.runs if not run.observed_win and 6 <= run.max_ante < 8)

    def to_text(self) -> str:
        lines = [
            "Replay analysis",
            f"Files scanned: {self.files_scanned}",
            f"Malformed rows: {self.malformed_rows}",
            f"Observed wins in replay rows: {self.observed_wins}",
            f"Outcome distribution: {_format_counter(self.outcome_distribution)}",
            f"Average max ante: {self.average_ante:.2f}",
            f"Ante distribution: {_format_counter(self.ante_distribution)}",
        ]
        if self.runs:
            for threshold in (2, 3, 4, 5, 6, 8):
                reached = sum(1 for run in self.runs if run.max_ante >= threshold)
                lines.append(f"Reached ante >= {threshold}: {reached}/{self.files_scanned}")

        lines.extend(
            (
                "",
                f"Action counts: {_format_counter(self.action_counts)}",
                f"Shop reason counts: {_format_counter(self.shop_reason_counts)}",
                f"Sell actions: {self.action_counts.get('sell', 0)}",
            )
        )
        if self.pressure_values:
            lines.append(
                "Pressure: "
                f"avg={self.average_pressure:.2f} "
                f"min={min(self.pressure_values):.2f} "
                f"max={max(self.pressure_values):.2f} "
                f">=1.0={sum(1 for value in self.pressure_values if value >= 1.0)}"
            )
        if self.hands_per_blind:
            lines.extend(
                (
                    "",
                    f"Average played hands per blind: {self.average_hands_per_blind:.2f}",
                    f"Played hands per blind: {_format_counter(self.hands_per_blind_distribution)}",
                    f"One-hand clears: {self.hands_per_blind_distribution.get(1, 0)}/{len(self.hands_per_blind)}",
                    f"Four-hand clears/deaths: {self.hands_per_blind_distribution.get(4, 0)}/{len(self.hands_per_blind)}",
                )
            )

        if self.early_failures:
            early_jokers = Counter(joker for run in self.early_failures for joker in run.final_jokers)
            early_play_counts = Counter(
                hand_count
                for run in self.early_failures
                for hand_count in run.hands_per_blind
            )
            lines.extend(
                (
                    "",
                    "Early failures:",
                    f"Ante <= 2 losses: {len(self.early_failures)}/{self.files_scanned}",
                    f"Average shop actions: {mean(run.shop_action_count for run in self.early_failures):.2f}",
                    f"Final joker counts: {_format_counter(early_jokers)}",
                    f"Played hands per blind: {_format_counter(early_play_counts)}",
                )
            )
            for run in sorted(self.early_failures, key=lambda item: (item.max_ante, item.seed))[:5]:
                lines.append(f"- {_failure_line(run)}")

        if self.deep_losses:
            deep_jokers = Counter(joker for run in self.deep_losses for joker in run.final_jokers)
            deep_play_counts = Counter(
                hand_count
                for run in self.deep_losses
                for hand_count in run.hands_per_blind
            )
            lines.extend(
                (
                    "",
                    "Deep losses:",
                    f"Ante 6-7 losses: {len(self.deep_losses)}/{self.files_scanned}",
                    f"Final joker counts: {_format_counter(deep_jokers)}",
                    f"Played hands per blind: {_format_counter(deep_play_counts)}",
                )
            )
            for run in sorted(self.deep_losses, key=lambda item: (-item.max_ante, item.seed))[:5]:
                lines.append(f"- {_failure_line(run)}")

        if self.runs:
            lines.extend(("", "Deepest runs:"))
            for run in sorted(self.runs, key=lambda item: item.max_ante, reverse=True)[:5]:
                lines.append(f"- seed={run.seed} ante={run.max_ante} file={run.path.name}")
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "malformed_rows": self.malformed_rows,
            "observed_wins": self.observed_wins,
            "outcome_distribution": _counter_to_json_dict(self.outcome_distribution),
            "average_max_ante": self.average_ante,
            "ante_distribution": _counter_to_json_dict(self.ante_distribution),
            "reach_counts": {
                str(threshold): sum(1 for run in self.runs if run.max_ante >= threshold)
                for threshold in (2, 3, 4, 5, 6, 8)
            },
            "action_counts": _counter_to_json_dict(self.action_counts),
            "shop_reason_counts": _counter_to_json_dict(self.shop_reason_counts),
            "pressure": {
                "count": len(self.pressure_values),
                "average": self.average_pressure,
                "minimum": min(self.pressure_values) if self.pressure_values else 0.0,
                "maximum": max(self.pressure_values) if self.pressure_values else 0.0,
                "at_least_1": sum(1 for value in self.pressure_values if value >= 1.0),
            },
            "hands_per_blind": {
                "average": self.average_hands_per_blind,
                "distribution": _counter_to_json_dict(self.hands_per_blind_distribution),
                "one_hand": self.hands_per_blind_distribution.get(1, 0),
                "four_hand": self.hands_per_blind_distribution.get(4, 0),
                "total": len(self.hands_per_blind),
            },
            "early_failures": {
                "count": len(self.early_failures),
                "sample": [_run_json_summary(run) for run in sorted(self.early_failures, key=lambda item: (item.max_ante, item.seed))[:5]],
            },
            "deep_losses": {
                "count": len(self.deep_losses),
                "sample": [_run_json_summary(run) for run in sorted(self.deep_losses, key=lambda item: (-item.max_ante, item.seed))[:5]],
            },
            "deepest_runs": [
                {"seed": run.seed, "max_ante": run.max_ante, "file": run.path.name}
                for run in sorted(self.runs, key=lambda item: item.max_ante, reverse=True)[:5]
            ],
        }


def analyze_replays(paths: Iterable[Path]) -> ReplayAnalysis:
    run_summaries: list[RunReplaySummary] = []
    malformed_rows = 0
    for replay_path in _expand_paths(paths):
        summary, bad_rows = _analyze_file(replay_path)
        malformed_rows += bad_rows
        if summary is not None:
            run_summaries.append(summary)

    action_counts: Counter[str] = Counter()
    shop_reason_counts: Counter[str] = Counter()
    pressure_values: list[float] = []
    hands_per_blind: list[int] = []
    for run in run_summaries:
        action_counts.update(run.action_counts)
        shop_reason_counts.update(run.shop_reason_counts)
        pressure_values.extend(run.pressure_values)
        hands_per_blind.extend(run.hands_per_blind)

    return ReplayAnalysis(
        runs=tuple(run_summaries),
        malformed_rows=malformed_rows,
        action_counts=action_counts,
        shop_reason_counts=shop_reason_counts,
        pressure_values=tuple(pressure_values),
        hands_per_blind=tuple(hands_per_blind),
    )


def _analyze_file(path: Path) -> tuple[RunReplaySummary | None, int]:
    row_count = 0
    malformed_rows = 0
    max_ante = 0
    seed = path.stem
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    pressure_values: list[float] = []
    hands_per_blind: list[int] = []
    current_blind: str | None = None
    current_play_count = 0
    observed_win = False
    outcome: str | None = None
    last_state: dict[str, object] = {}
    shop_action_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue

            row_count += 1
            seed = str(row.get("seed", seed))
            if row.get("record_type") == "run_summary":
                max_ante = max(max_ante, _int_value(row.get("ante")))
                summary_state = _state_from_summary_row(row)
                if summary_state:
                    last_state = summary_state
                row_outcome = row.get("outcome")
                if row_outcome is not None:
                    outcome = str(row_outcome)
                if row.get("outcome") == "win" or row.get("won") is True:
                    observed_win = True
                continue

            state_text = str(row.get("state", ""))
            max_ante = max(max_ante, _ante_from_state(state_text))
            parsed_state = _parse_state_text(state_text)
            if parsed_state:
                last_state = parsed_state
            if row.get("outcome") == "win" or row.get("won") is True:
                observed_win = True

            action = row.get("chosen_action", {})
            action_type = str(action.get("type", "unknown")) if isinstance(action, dict) else "unknown"
            action_counts[action_type] += 1
            if action_type in {"buy", "sell", "reroll", "open_pack", "choose_pack_card"}:
                shop_action_count += 1

            metadata = action.get("metadata", {}) if isinstance(action, dict) else {}
            reason = metadata.get("reason") if isinstance(metadata, dict) else None
            if reason:
                reason_text = str(reason)
                reason_counts[reason_text.split()[0]] += 1
                pressure_values.extend(_pressure_values(reason_text))

            blind_key = _blind_key_from_state(state_text)
            if action_type == "play_hand":
                if current_blind is None:
                    current_blind = blind_key
                    current_play_count = 0
                if blind_key != current_blind:
                    if current_play_count:
                        hands_per_blind.append(current_play_count)
                    current_blind = blind_key
                    current_play_count = 0
                current_play_count += 1
            elif action_type in {"cash_out", "select_blind", "skip_blind"}:
                if current_play_count:
                    hands_per_blind.append(current_play_count)
                    current_play_count = 0
                    current_blind = None

    if current_play_count:
        hands_per_blind.append(current_play_count)
    if row_count == 0:
        return None, malformed_rows

    return (
        RunReplaySummary(
            path=path,
            seed=seed,
            max_ante=max_ante,
            row_count=row_count,
            action_counts=action_counts,
            shop_reason_counts=reason_counts,
            pressure_values=tuple(pressure_values),
            hands_per_blind=tuple(hands_per_blind),
            sell_actions=action_counts.get("sell", 0),
            observed_win=observed_win,
            outcome=outcome,
            final_phase=str(last_state.get("phase", "")),
            final_blind=str(last_state.get("blind", "")),
            final_score=_int_value(last_state.get("score")),
            final_required_score=_int_value(last_state.get("required")),
            final_money=_int_value(last_state.get("money")),
            final_hands=_int_value(last_state.get("hands")),
            final_jokers=tuple(last_state.get("jokers", ())),
            shop_action_count=shop_action_count,
        ),
        malformed_rows,
    )


def _expand_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.jsonl")))
        elif path.exists():
            expanded.append(path)
    return tuple(expanded)


def _ante_from_state(state_text: str) -> int:
    match = ANTE_PATTERN.search(state_text)
    return int(match.group(1)) if match else 0


def _parse_state_text(state_text: str) -> dict[str, object]:
    match = STATE_PATTERN.search(state_text)
    if not match:
        return {}
    values = match.groupdict()
    joker_text = values["jokers"]
    jokers = () if joker_text == "-" else tuple(item.strip() for item in joker_text.split(",") if item.strip())
    return {
        "phase": values["phase"],
        "ante": int(values["ante"]),
        "blind": values["blind"],
        "score": int(values["score"]),
        "required": int(values["required"]),
        "money": int(values["money"]),
        "hands": int(values["hands"]),
        "discards": int(values["discards"]),
        "jokers": jokers,
    }


def _state_from_summary_row(row: dict[str, object]) -> dict[str, object]:
    detail = row.get("final_state_detail")
    if isinstance(detail, dict):
        return {
            "phase": str(detail.get("phase", "")),
            "ante": _int_value(detail.get("ante")),
            "blind": str(detail.get("blind", "")),
            "score": _int_value(detail.get("current_score")),
            "required": _int_value(detail.get("required_score")),
            "money": _int_value(detail.get("money", row.get("final_money"))),
            "hands": _int_value(detail.get("hands_remaining")),
            "discards": _int_value(detail.get("discards_remaining")),
            "jokers": tuple(
                str(joker.get("name", ""))
                for joker in detail.get("jokers", ())
                if isinstance(joker, dict) and joker.get("name")
            ),
        }
    return _parse_state_text(str(row.get("final_state", "")))


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _blind_key_from_state(state_text: str) -> str:
    match = BLIND_PATTERN.search(state_text)
    return match.group(0) if match else ""


def _pressure_values(reason: str) -> tuple[float, ...]:
    return tuple(float(match.group(1)) for match in PRESSURE_PATTERN.finditer(reason))


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "{}"
    return "{" + ", ".join(f"{key}: {counter[key]}" for key in sorted(counter)) + "}"


def _counter_to_json_dict(counter: Counter) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def _failure_line(run: RunReplaySummary) -> str:
    jokers = ", ".join(run.final_jokers) if run.final_jokers else "-"
    return (
        f"seed={run.seed} ante={run.max_ante} blind={run.final_blind or '-'} "
        f"score={run.final_score}/{run.final_required_score} hands={run.final_hands} "
        f"money={run.final_money} jokers=[{jokers}] file={run.path.name}"
    )


def _run_json_summary(run: RunReplaySummary) -> dict[str, object]:
    return {
        "seed": run.seed,
        "file": run.path.name,
        "max_ante": run.max_ante,
        "final_blind": run.final_blind,
        "final_score": run.final_score,
        "final_required_score": run.final_required_score,
        "final_hands": run.final_hands,
        "final_money": run.final_money,
        "final_jokers": list(run.final_jokers),
        "shop_action_count": run.shop_action_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Botlatro replay behavior.")
    parser.add_argument("paths", nargs="*", type=Path, help="Replay JSONL file(s) or directories.")
    parser.add_argument("--replay-dir", type=Path, help="Replay directory to scan recursively.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = list(args.paths)
    if args.replay_dir is not None:
        paths.append(args.replay_dir)
    if not paths:
        raise SystemExit("Provide at least one replay path or --replay-dir.")
    analysis = analyze_replays(paths)
    if args.json:
        print(json.dumps(analysis.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(analysis.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
