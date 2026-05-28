"""Streaming JSONL writer for dataset rows.

`JsonlSeedWriter` appends one `SeedResult` per line. Each write is
fsync'd so a kill during a long batch doesn't lose completed seeds.
This is the resume mechanism — the reader will see every committed
seed and skip it on the next run.

The writer is designed to be created in the main process (NOT the
workers) so we don't have to worry about cross-process file locks.
The worker pool sends `SeedResult` objects back to the main process
via the pool's result queue; the main process holds the writer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import IO

from balatro_ai.dataset.schema import SeedResult


class JsonlSeedWriter:
    """Append-only JSONL writer for `SeedResult` rows.

    Usage:
        with JsonlSeedWriter(path) as w:
            for result in results:
                w.write(result)

    The context-manager close flushes + closes the file. `write` also
    fsyncs after each row, so even if the process is killed mid-batch
    everything up to the last completed write is on disk.

    The writer does NOT deduplicate against existing rows. If you
    want resume semantics, ask `JsonlSeedReader.completed_seeds(path)`
    for the set of seeds already in the file and skip them in your
    work-feeding loop.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = None

    def __enter__(self) -> JsonlSeedWriter:
        # Open in append mode so resume works automatically.
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                # fsync isn't supported on every filesystem (e.g. some
                # network mounts); the flush above is the best we can
                # do in that case.
                pass
            self._fh.close()
            self._fh = None

    def write(self, result: SeedResult) -> None:
        """Append one row. Flushes + fsyncs after the write."""

        if self._fh is None:
            raise RuntimeError("Writer is not open. Use as a context manager.")
        line = json.dumps(result.to_json_dict(), separators=(",", ":"))
        self._fh.write(line + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass
