"""JSONL + seed-list readers.

Two responsibilities:

- `JsonlSeedReader.completed_seeds(path)` — fast scan of an existing
  output JSONL to determine which seeds are already done. Used by the
  CLI's resume logic to skip them on a second run.

- `read_seed_file(path)` — tolerant seed-list parser. Accepts:
    - one seed per line (the natural format)
    - comma-separated on a single line (the format used by older
      `.data/*-seeds.txt` files)
    - mixed (multi-line files with commas inside each line)
  Lines starting with `#` are treated as comments and skipped.
  Numeric seeds are normalized to their `str()` form so resume-
  deduplication is a pure string match.
"""

from __future__ import annotations

import json
from pathlib import Path

from balatro_ai.dataset.schema import SeedResult


class JsonlSeedReader:
    """Reads `SeedResult` rows back from a JSONL output file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_rows(self):
        """Yield each parsed `SeedResult` from the file.

        Skips blank lines. Raises on malformed JSON — better to surface
        corruption loudly than silently drop rows.
        """

        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}:{lineno}: malformed JSONL row — {exc}"
                    ) from exc
                yield SeedResult.from_json_dict(data)

    def completed_seeds(self) -> set[str]:
        """Return the set of seed strings that already have a row.

        Used by the dataset CLI's resume logic. A seed counts as
        "completed" whether it succeeded or errored — re-running
        won't retry failures by default. To force a retry, delete
        the relevant rows from the output file first.
        """

        return {row.seed for row in self.iter_rows()}


def read_seed_file(path: str | Path) -> list[str]:
    """Parse a seed file. Returns a deduplicated list, preserving input order.

    Accepted formats:
    - One seed per line: `AAAAAAA\\nBBBBBBB\\n...`
    - Comma-separated on one line: `1,2,3,4,5`
    - Mixed: lines that contain commas are split
    - Lines starting with `#` are comments
    """

    path = Path(path)
    seeds: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for token in line.split(","):
                seed = token.strip()
                if not seed or seed in seen:
                    continue
                seen.add(seed)
                seeds.append(seed)
    return seeds
