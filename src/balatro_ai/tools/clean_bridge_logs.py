"""Clean noisy BalatroBot/Lovely bridge logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DROP_SUBSTRINGS = (
    "LONG DT @",
    ":: DEBUG :: BB.DISPATCHER :: Loading endpoint",
    ":: DEBUG :: BB.DISPATCHER :: Registered endpoint",
    ":: DEBUG :: BB.REQUEST :: gamestate",
    ":: DEBUG :: BB.RESPONSE :: gamestate OK",
    "Init gamestate()",
    "Return gamestate()",
)


@dataclass(frozen=True, slots=True)
class CleanLogResult:
    path: Path
    output_path: Path
    original_lines: int
    kept_lines: int
    dropped_lines: int


def is_noisy_bridge_log_line(line: str, drop_substrings: tuple[str, ...] = DEFAULT_DROP_SUBSTRINGS) -> bool:
    """Return true for high-volume bridge diagnostics that obscure real failures."""

    return any(pattern in line for pattern in drop_substrings)


def clean_log_file(
    path: Path,
    *,
    output_path: Path | None = None,
    replace: bool = False,
    drop_substrings: tuple[str, ...] = DEFAULT_DROP_SUBSTRINGS,
) -> CleanLogResult:
    """Filter a single log file.

    By default this writes ``*.clean.log`` next to the source file. With
    ``replace=True``, the source file is rewritten in place.
    """

    path = Path(path)
    if output_path is None:
        output_path = path if replace else path.with_name(f"{path.stem}.clean{path.suffix}")
    else:
        output_path = Path(output_path)

    original_lines = 0
    kept_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            original_lines += 1
            if not is_noisy_bridge_log_line(line, drop_substrings):
                kept_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(kept_lines), encoding="utf-8")
    return CleanLogResult(
        path=path,
        output_path=output_path,
        original_lines=original_lines,
        kept_lines=len(kept_lines),
        dropped_lines=original_lines - len(kept_lines),
    )


def clean_logs(
    paths: list[Path],
    *,
    replace: bool = False,
    drop_substrings: tuple[str, ...] = DEFAULT_DROP_SUBSTRINGS,
) -> list[CleanLogResult]:
    """Clean all ``*.log`` files under the given files or directories."""

    results: list[CleanLogResult] = []
    for root in paths:
        root = Path(root)
        log_files = [root] if root.is_file() else sorted(root.rglob("*.log"))
        for log_file in log_files:
            if log_file.name.endswith(".clean.log"):
                continue
            results.append(clean_log_file(log_file, replace=replace, drop_substrings=drop_substrings))
    return results


def summarize_results(results: list[CleanLogResult]) -> str:
    dropped = sum(result.dropped_lines for result in results)
    kept = sum(result.kept_lines for result in results)
    return f"Cleaned {len(results)} log file(s): dropped {dropped} noisy line(s), kept {kept} line(s)."


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean noisy BalatroBot/Lovely bridge logs.")
    parser.add_argument("paths", nargs="*", type=Path, help="Log files or directories to clean.")
    parser.add_argument("--logs-root", type=Path, help="Convenience directory to clean, e.g. .logs.")
    parser.add_argument("--replace", action="store_true", help="Rewrite log files in place instead of writing .clean.log files.")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.logs_root is not None:
        paths.append(args.logs_root)
    if not paths:
        paths.append(Path(".logs"))

    results = clean_logs(paths, replace=args.replace)
    print(summarize_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
