"""Validate the score dataset used by Phase 7 search work."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from balatro_ai.eval.explain_score_misses import MissExplanation, collect_score_explanations
from balatro_ai.eval.score_audit import ScoreAuditSummary, audit_replays
from balatro_ai.eval.score_edge_fixtures import DEFAULT_FIXTURE_DIR, FixtureResult, run_fixture_paths


@dataclass(frozen=True, slots=True)
class ScoreDatasetCheck:
    fixture_results: tuple[FixtureResult, ...]
    replay_audit: ScoreAuditSummary | None
    replay_rows_checked: tuple[MissExplanation, ...]

    @property
    def fixture_failures(self) -> tuple[FixtureResult, ...]:
        return tuple(result for result in self.fixture_results if not result.passed)

    @property
    def replay_misses(self) -> tuple[MissExplanation, ...]:
        return tuple(row for row in self.replay_rows_checked if row.error != 0)

    @property
    def passed(self) -> bool:
        return not self.fixture_failures and not self.replay_misses

    def to_text(self, *, worst_count: int = 10, verbose_fixtures: bool = False) -> str:
        lines = [
            "Score dataset check",
            f"Fixture cases: {len(self.fixture_results)}",
            f"Fixture failures: {len(self.fixture_failures)}",
        ]
        if self.replay_audit is None:
            lines.extend(
                [
                    "Replay files scanned: 0",
                    "Replay rows checked: 0",
                    "Replay misses: 0",
                ]
            )
        else:
            lines.extend(
                [
                    f"Replay files scanned: {self.replay_audit.files_scanned}",
                    f"Replay audit rows found: {self.replay_audit.played_hands}",
                    f"Replay rows checked: {len(self.replay_rows_checked)}",
                    f"Replay misses: {len(self.replay_misses)}",
                    f"Known uncertain replay rows: {len(self.replay_audit.uncertain_records)}",
                ]
            )
        lines.append(f"Result: {'PASS' if self.passed else 'FAIL'}")

        if verbose_fixtures and self.fixture_results:
            lines.append("")
            lines.append("Fixtures:")
            lines.extend(result.to_text() for result in self.fixture_results)
        elif self.fixture_failures:
            lines.append("")
            lines.append("Fixture failures:")
            lines.extend(result.to_text() for result in self.fixture_failures)

        if self.replay_misses:
            lines.append("")
            lines.append(f"Worst replay misses ({min(worst_count, len(self.replay_misses))}):")
            selected = sorted(self.replay_misses, key=lambda row: row.absolute_error, reverse=True)[:worst_count]
            for row in selected:
                lines.append(
                    f"- seed={row.seed} ante={row.ante} blind={row.blind or '-'} "
                    f"hand={row.hand_type} recomputed={row.recomputed_score} "
                    f"actual={row.actual_score_delta} error={row.error} "
                    f"file={row.replay_path.name}:{row.line_number}"
                )
        return "\n".join(lines)


def check_score_dataset(
    *,
    fixture_paths: Iterable[Path] = (DEFAULT_FIXTURE_DIR,),
    replay_paths: Iterable[Path] = (),
    include_uncertain: bool = False,
) -> ScoreDatasetCheck:
    fixture_path_tuple = tuple(fixture_paths)
    replay_path_tuple = tuple(replay_paths)
    fixture_results = run_fixture_paths(fixture_path_tuple) if fixture_path_tuple else ()
    replay_audit = audit_replays(replay_path_tuple) if replay_path_tuple else None
    replay_rows = (
        collect_score_explanations(replay_path_tuple, include_uncertain=include_uncertain)
        if replay_path_tuple
        else ()
    )
    return ScoreDatasetCheck(
        fixture_results=fixture_results,
        replay_audit=replay_audit,
        replay_rows_checked=replay_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate score fixtures and recomputed replay score-audit rows.")
    parser.add_argument(
        "--fixtures",
        nargs="*",
        type=Path,
        default=[DEFAULT_FIXTURE_DIR],
        help="Fixture JSON files or directories. Defaults to tests/fixtures/score_edges.",
    )
    parser.add_argument(
        "--replay-dir",
        action="append",
        type=Path,
        default=[],
        help="Replay directory to scan recursively. Can be passed more than once.",
    )
    parser.add_argument(
        "--replay-paths",
        nargs="*",
        type=Path,
        default=[],
        help="Additional replay JSONL files or directories.",
    )
    parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Also recompute rows marked as uncertain by the score-audit classifier.",
    )
    parser.add_argument("--worst", type=int, default=10, help="Number of replay misses to print.")
    parser.add_argument("--verbose-fixtures", action="store_true", help="Print every fixture result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    replay_paths = tuple(args.replay_dir) + tuple(args.replay_paths)
    check = check_score_dataset(
        fixture_paths=tuple(args.fixtures),
        replay_paths=replay_paths,
        include_uncertain=bool(args.include_uncertain),
    )
    print(check.to_text(worst_count=max(0, args.worst), verbose_fixtures=bool(args.verbose_fixtures)))
    return 0 if check.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
