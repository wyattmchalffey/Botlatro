"""Import opt-in human gameplay logs into the local replay dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True, slots=True)
class UserLogImportSummary:
    files_imported: int
    rows_imported: int
    summary_rows: int
    malformed_rows: int
    output_files: tuple[Path, ...]

    def to_text(self) -> str:
        lines = [
            "User replay import",
            f"Files imported: {self.files_imported}",
            f"Rows imported: {self.rows_imported}",
            f"Run summaries: {self.summary_rows}",
            f"Malformed rows skipped: {self.malformed_rows}",
        ]
        if self.output_files:
            lines.append("Output files:")
            lines.extend(f"- {path}" for path in self.output_files)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "files_imported": self.files_imported,
            "rows_imported": self.rows_imported,
            "summary_rows": self.summary_rows,
            "malformed_rows": self.malformed_rows,
            "output_files": [str(path) for path in self.output_files],
        }


def import_user_logs(
    sources: Iterable[Path],
    *,
    dest: Path,
    player_id: str = "human",
    overwrite: bool = False,
) -> UserLogImportSummary:
    dest.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    rows_imported = 0
    summary_rows = 0
    malformed_rows = 0
    imported_files = 0

    for source in _expand_sources(sources):
        rows: list[dict[str, object]] = []
        with source.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_rows += 1
                    continue
                if not isinstance(row, dict):
                    malformed_rows += 1
                    continue
                row["dataset_source"] = {
                    "player_id": player_id,
                    "source_file": str(source),
                    "source_line": line_number,
                    "imported_by": "balatro_ai.eval.import_user_logs",
                }
                rows.append(row)

        if not rows:
            continue

        output = _destination_path(dest, player_id=player_id, source=source, overwrite=overwrite)
        if output.exists() and overwrite:
            output.unlink()
        with output.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")

        imported_files += 1
        rows_imported += len(rows)
        summary_rows += sum(1 for row in rows if row.get("record_type") == "run_summary")
        output_files.append(output)

    return UserLogImportSummary(
        files_imported=imported_files,
        rows_imported=rows_imported,
        summary_rows=summary_rows,
        malformed_rows=malformed_rows,
        output_files=tuple(output_files),
    )


def _expand_sources(sources: Iterable[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for source in sources:
        if source.is_dir():
            files.extend(sorted(source.rglob("*.jsonl")))
        elif source.exists():
            files.append(source)
    return tuple(files)


def _destination_path(dest: Path, *, player_id: str, source: Path, overwrite: bool) -> Path:
    safe_player = _safe_name(player_id)
    safe_stem = _safe_name(source.stem)
    candidate = dest / f"{safe_player}_{safe_stem}.jsonl"
    if overwrite or not candidate.exists():
        return candidate
    for suffix in range(2, 10000):
        candidate = dest / f"{safe_player}_{safe_stem}_{suffix}.jsonl"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find a free destination filename for {source}")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "human"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Botlatro User Logger JSONL files.")
    parser.add_argument("--source", nargs="+", type=Path, required=True, help="Source JSONL file(s) or directories.")
    parser.add_argument("--dest", type=Path, required=True, help="Destination replay dataset directory.")
    parser.add_argument("--player-id", default="human", help="Dataset player/source identifier to attach to rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite matching imported files.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable import summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = import_user_logs(args.source, dest=args.dest, player_id=args.player_id, overwrite=args.overwrite)
    if args.json:
        print(json.dumps(summary.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(summary.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
