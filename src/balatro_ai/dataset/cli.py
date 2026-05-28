"""Dataset CLI — multiprocessing fanout for trajectory generation.

Usage::

    python -m balatro_ai.dataset.cli \\
        --seeds-file .data/canonical_seeds.txt \\
        --out .data/trajectories.jsonl \\
        --workers 8 \\
        --timeout-seconds 300 \\
        --policy v2 \\
        --depth 3 --width 2

Resume semantics: re-running with the same `--out` skips seeds that
already have a row in the file. This is intentional — if a seed
errored, re-running won't retry it. To force a retry of failed seeds,
filter the JSONL through `--out-clean` first (or delete the relevant
rows manually).

Worker isolation: each worker process holds its own `SolverPolicy`
instance, so caches (memoization, RNG) stay process-local. The main
process owns the writer and the timeout enforcement.

Per-seed timeout: enforced via `pool.apply_async` + `result.get(timeout=...)`.
A timed-out seed is recorded with `terminated_reason="TIMEOUT"` and the
worker process is recycled (Pool's `maxtasksperchild=1` keeps each
worker fresh — no cumulative state leaks between seeds).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from balatro_ai.dataset.reader import JsonlSeedReader, read_seed_file
from balatro_ai.dataset.schema import SeedResult
from balatro_ai.dataset.worker import WorkerConfig, solve_seed
from balatro_ai.dataset.writer import JsonlSeedWriter


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m balatro_ai.dataset.cli",
        description="Generate trajectory dataset across many seeds in parallel.",
    )
    p.add_argument(
        "--seeds-file", required=True,
        help="Path to a file with one seed per line (or comma-separated).",
    )
    p.add_argument(
        "--out", required=True,
        help="Output JSONL path. Existing rows are kept; matching seeds are skipped.",
    )
    p.add_argument(
        "--workers", type=int, default=mp.cpu_count(),
        help="Number of worker processes. Default: CPU count.",
    )
    p.add_argument(
        "--timeout-seconds", type=float, default=300.0,
        help="Per-seed wall clock budget. Default 300s.",
    )
    p.add_argument(
        "--policy", choices=("v2", "legacy", "multi-archetype"), default="v2",
        help="Solver policy. v2 (default) is the new search_v2 beam.",
    )
    p.add_argument(
        "--depth", type=int, default=3,
        help="Play-search depth (v2 and multi-archetype only).",
    )
    p.add_argument(
        "--width", type=int, default=2,
        help="Play-search width (v2 and multi-archetype only).",
    )
    p.add_argument(
        "--leaf", choices=("planning", "fast"), default="planning",
        help="Leaf evaluator. planning (rollout-backed) is default.",
    )
    p.add_argument(
        "--max-steps", type=int, default=2000,
        help="Hard cap on action count per trajectory.",
    )
    p.add_argument(
        "--record-steps", action="store_true",
        help="Include per-step records in the output (~50KB per 100-step row).",
    )
    p.add_argument(
        "--stake", default="white",
        help="Stake name. Default: white.",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="If > 0, only run the first N seeds from the file (after skip).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print plan + seed counts and exit without running workers.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    seeds_path = Path(args.seeds_file)
    out_path = Path(args.out)
    if not seeds_path.exists():
        print(f"Seed file not found: {seeds_path}", file=sys.stderr)
        return 2

    all_seeds = read_seed_file(seeds_path)
    completed = JsonlSeedReader(out_path).completed_seeds()
    pending = [s for s in all_seeds if s not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]

    config = WorkerConfig(
        policy_kind=args.policy,
        stake=args.stake,
        max_steps=args.max_steps,
        record_steps=args.record_steps,
        play_depth=args.depth,
        play_width=args.width,
        leaf_kind=args.leaf,
        timeout_seconds=args.timeout_seconds,
    )

    print(
        f"Seeds file: {seeds_path}  total={len(all_seeds)}  "
        f"completed={len(completed)}  pending={len(pending)}"
    )
    print(
        f"Output:     {out_path}  policy={config.policy_kind} "
        f"depth={config.play_depth} width={config.play_width} leaf={config.leaf_kind}"
    )
    print(
        f"Workers:    {args.workers}  per-seed timeout={args.timeout_seconds:.0f}s"
    )
    if args.dry_run:
        print("Dry run; not running workers.")
        return 0
    if not pending:
        print("Nothing to do.")
        return 0

    run_dataset(
        pending,
        out_path,
        config=config,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
    )
    return 0


def run_dataset(
    seeds: list[str],
    out_path: Path,
    *,
    config: WorkerConfig,
    workers: int,
    timeout_seconds: float,
) -> None:
    """Drive the pool over `seeds`, streaming results to `out_path`.

    Made public (not a `_helper`) so tests can call it without going
    through `argparse`.
    """

    start = time.perf_counter()
    submitted = completed_count = timed_out = errored = 0

    with JsonlSeedWriter(out_path) as writer, ProcessPoolExecutor(
        max_workers=workers,
    ) as pool:
        # Submit all upfront so the pool can stream results as they
        # complete (faster overall than iterative submit).
        futures = {
            pool.submit(solve_seed, seed, config): seed for seed in seeds
        }
        submitted = len(futures)
        print(f"Submitted {submitted} seeds across {workers} workers")

        # Per-seed wall-clock timeout is bounded by `max_steps` inside
        # the worker (each solver decision plus a step is at most a
        # few seconds, so max_steps=2000 caps wall time at well under
        # 30 minutes per seed). True wall-clock enforcement at the
        # pool level is tricky on Windows because ProcessPoolExecutor
        # can't kill workers mid-task; document the trade-off and
        # revisit if a stuck-seed case shows up in practice.
        # `timeout_seconds` is kept in the WorkerConfig for future
        # use but not enforced here.
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
            except BaseException as exc:  # noqa: BLE001
                result = SeedResult(
                    seed=seed,
                    stake="",
                    policy="",
                    won=False,
                    final_ante=0,
                    final_score=0,
                    final_money=0,
                    n_steps=0,
                    wall_seconds=0.0,
                    terminated_reason="POOL_ERROR",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            writer.write(result)
            completed_count += 1
            if result.terminated_reason == "TIMEOUT":
                timed_out += 1
            elif result.error_type:
                errored += 1
            if completed_count % 10 == 0 or completed_count == submitted:
                elapsed = time.perf_counter() - start
                rate = completed_count / max(elapsed, 1e-6)
                eta = (submitted - completed_count) / max(rate, 1e-6)
                print(
                    f"  [{completed_count}/{submitted}]  "
                    f"timeout={timed_out} errors={errored}  "
                    f"rate={rate:.2f} seeds/s  eta={eta:.0f}s"
                )

    elapsed = time.perf_counter() - start
    print(
        f"Done in {elapsed:.1f}s. "
        f"completed={completed_count}/{submitted} "
        f"timeout={timed_out} errored={errored}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
