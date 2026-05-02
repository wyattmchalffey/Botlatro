"""Calibration checks for Phase 7 state-value predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.search import replay_diff
from balatro_ai.search.state_value import clear_probability


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    path: Path
    row_number: int
    seed: str
    ante: int
    blind: str
    required_score: int
    predicted_clear: float
    actual_clear: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "file": str(self.path),
            "row_number": self.row_number,
            "seed": self.seed,
            "ante": self.ante,
            "blind": self.blind,
            "required_score": self.required_score,
            "predicted_clear": self.predicted_clear,
            "actual_clear": self.actual_clear,
        }


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    average_prediction: float
    actual_clear_rate: float

    @property
    def label(self) -> str:
        return f"{int(self.lower * 100):02d}-{int(self.upper * 100):02d}%"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "average_prediction": self.average_prediction,
            "actual_clear_rate": self.actual_clear_rate,
        }


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    files_scanned: int
    examples: tuple[CalibrationExample, ...]
    skipped_rows: int = 0

    @property
    def blind_count(self) -> int:
        return len(self.examples)

    @property
    def bins(self) -> tuple[CalibrationBin, ...]:
        grouped: dict[int, list[CalibrationExample]] = defaultdict(list)
        for example in self.examples:
            index = min(9, max(0, int(example.predicted_clear * 10)))
            grouped[index].append(example)

        bins: list[CalibrationBin] = []
        for index in range(10):
            items = grouped.get(index, [])
            lower = index / 10
            upper = (index + 1) / 10
            if not items:
                bins.append(CalibrationBin(lower, upper, 0, 0.0, 0.0))
                continue
            average_prediction = sum(item.predicted_clear for item in items) / len(items)
            actual_rate = sum(1 for item in items if item.actual_clear) / len(items)
            bins.append(CalibrationBin(lower, upper, len(items), average_prediction, actual_rate))
        return tuple(bins)

    def to_text(self) -> str:
        lines = [
            "State-value calibration",
            f"Files scanned: {self.files_scanned}",
            f"Blind starts scored: {self.blind_count}",
            f"Skipped rows: {self.skipped_rows}",
            "",
            "Bin        Count  Avg pred  Actual",
        ]
        for item in self.bins:
            if item.count <= 0:
                continue
            lines.append(
                f"{item.label:<10} {item.count:>5}  {item.average_prediction:>7.1%}  {item.actual_clear_rate:>7.1%}"
            )
        return "\n".join(lines)

    def to_json_dict(self, *, example_limit: int = 20) -> dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "blind_count": self.blind_count,
            "skipped_rows": self.skipped_rows,
            "bins": [item.to_json_dict() for item in self.bins],
            "examples": [item.to_json_dict() for item in self.examples[:example_limit]],
        }


def calibrate_clear_probability(
    paths: tuple[Path, ...] | list[Path],
    *,
    samples: int = 32,
    seed: int = 0,
    limit: int | None = None,
) -> CalibrationSummary:
    replay_paths = replay_diff._expand_paths(paths)
    examples: list[CalibrationExample] = []
    skipped_rows = 0
    for path in replay_paths:
        rows, bad_rows = replay_diff._load_rows(path)
        skipped_rows += bad_rows
        active_key: tuple[int, str, int] | None = None
        for index, row in enumerate(rows):
            state_key = replay_diff._blind_key_from_detail(row.get("state_detail"))
            if state_key is None:
                if _row_ends_blind(row):
                    active_key = None
                continue
            if state_key == active_key:
                continue
            active_key = state_key
            state = replay_diff._state_from_detail(row.get("state_detail"), audit=replay_diff._score_audit_from_row(row))
            if state is None or not _is_blind_start_state(state):
                continue
            try:
                predicted = clear_probability(state, samples=samples, seed=seed + len(examples))
            except ValueError:
                skipped_rows += 1
                continue
            examples.append(
                CalibrationExample(
                    path=path,
                    row_number=index + 1,
                    seed=str(row.get("seed", path.stem)),
                    ante=state.ante,
                    blind=state.blind,
                    required_score=state.required_score,
                    predicted_clear=predicted,
                    actual_clear=_blind_cleared_after(rows, index, state_key),
                )
            )
            if limit is not None and len(examples) >= limit:
                return CalibrationSummary(files_scanned=len(replay_paths), examples=tuple(examples), skipped_rows=skipped_rows)
    return CalibrationSummary(files_scanned=len(replay_paths), examples=tuple(examples), skipped_rows=skipped_rows)


def _is_blind_start_state(state: GameState) -> bool:
    return (
        state.phase in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND}
        and state.current_score == 0
        and state.required_score > 0
        and state.hands_remaining > 0
        and bool(state.hand)
    )


def _blind_cleared_after(
    rows: tuple[dict[str, object], ...],
    start_index: int,
    state_key: tuple[int, str, int],
) -> bool:
    for row in rows[start_index + 1 :]:
        action = replay_diff._action_from_row(row)
        if action is not None and action.action_type.value == "cash_out":
            return True
        detail = row.get("state_detail")
        if not isinstance(detail, dict):
            if row.get("record_type") == "run_summary":
                return bool(row.get("won", False))
            continue
        phase = replay_diff._phase_from_value(detail.get("phase"))
        if phase == GamePhase.RUN_OVER:
            return False
        if phase == GamePhase.ROUND_EVAL:
            return True
        next_key = replay_diff._blind_key_from_detail(detail)
        if next_key is not None and next_key != state_key:
            return False
    return False


def _row_ends_blind(row: dict[str, object]) -> bool:
    action = replay_diff._action_from_row(row)
    if action is not None and action.action_type.value in {"cash_out", "select_blind", "skip_blind"}:
        return True
    detail = row.get("state_detail")
    if isinstance(detail, dict):
        phase = replay_diff._phase_from_value(detail.get("phase"))
        return phase in {GamePhase.SHOP, GamePhase.BLIND_SELECT, GamePhase.RUN_OVER}
    return row.get("record_type") == "run_summary"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate state_value.clear_probability against replay outcomes.")
    parser.add_argument("paths", nargs="*", type=Path, help="Replay JSONL file(s) or directories.")
    parser.add_argument("--replay-dir", type=Path, help="Replay directory to scan recursively.")
    parser.add_argument("--samples", type=int, default=32, help="Rollout samples per blind-start state.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for rollout sampling.")
    parser.add_argument("--limit", type=int, help="Maximum blind starts to score.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary.")
    parser.add_argument("--examples", type=int, default=20, help="Maximum JSON examples to include.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = list(args.paths)
    if args.replay_dir is not None:
        paths.append(args.replay_dir)
    if not paths:
        raise SystemExit("Provide at least one replay path or --replay-dir.")
    summary = calibrate_clear_probability(paths, samples=args.samples, seed=args.seed, limit=args.limit)
    if args.json:
        print(json.dumps(summary.to_json_dict(example_limit=args.examples), indent=2, sort_keys=True))
    else:
        print(summary.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
