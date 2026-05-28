"""Dataset CLI + writers (Tier 1 #2 of SOLVER_OPTIMIZATION_PLAN.md).

This package owns the multi-seed trajectory generation pipeline that
feeds Phase 8 imitation training. It's deliberately thin — most of the
work is delegated to `solver.multi_archetype` or `solver.trajectory`;
this package is responsible only for:

- Reading a seed list (one-per-line or comma-separated).
- Fanning out across workers via `multiprocessing.Pool`.
- Writing each completed seed's trajectory to a streaming JSONL file.
- Skipping seeds that are already present in the output file (resume).
- Enforcing a per-seed wall-clock timeout so one pathological seed
  can't stall the whole batch.

Public surface:
- `SeedResult` — dataclass for one row of the output JSONL.
- `read_seed_file` — tolerant seed-file parser.
- `JsonlSeedWriter` — append-only writer with fsync.
- `JsonlSeedReader` — read existing rows; used by the resume logic.
- `solve_seed` — per-seed entrypoint (pickleable; the worker function).
- `main` / CLI — `python -m balatro_ai.dataset.cli ...`

See `SOLVER_OPTIMIZATION_PLAN.md` §3 #2 for the full design rationale.
"""

from balatro_ai.dataset.reader import JsonlSeedReader, read_seed_file
from balatro_ai.dataset.schema import SeedResult
from balatro_ai.dataset.worker import solve_seed
from balatro_ai.dataset.writer import JsonlSeedWriter

__all__ = [
    "JsonlSeedReader",
    "JsonlSeedWriter",
    "SeedResult",
    "read_seed_file",
    "solve_seed",
]
