"""Benchmark bots through the pure-Python local simulator."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from balatro_ai.bots.registry import create_bot
from balatro_ai.eval.metrics import RunResult, summarize_runs
from balatro_ai.eval.seed_sets import make_explicit_seed_set, make_seed_set, parse_seed_values
from balatro_ai.sim.local_runner import LocalSimOptions, run_local_seed, run_local_seed_with_trace


DETERMINISTIC_HASH_SEED = "0"
_HASH_REEXEC_GUARD = "BOTLATRO_LOCAL_BENCHMARK_HASH_REEXEC"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fast local-simulator benchmark.")
    parser.add_argument("--bot", required=True, help="Bot name to benchmark.")
    parser.add_argument("--seeds", type=int, default=100, help="Number of deterministic seeds.")
    parser.add_argument("--seed-list", default="", help="Comma/space-separated exact seeds; overrides --seeds.")
    parser.add_argument("--stake", default="white", help="Stake label; first pass is tuned for white.")
    parser.add_argument("--label", default="local", help="Seed-set label.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum simulator steps per run.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N seeds; 0 disables.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel local simulator workers; 1 keeps serial behavior.")
    parser.add_argument("--jsonl-out", type=Path, help="Optional path for per-seed run_summary JSONL output.")
    parser.add_argument("--trace-jsonl-out", type=Path, help="Optional path for per-decision local simulator trace JSONL output.")
    return parser


def _run_seed_worker(payload: tuple[int, str, int, str, int, bool]) -> tuple[int, RunResult, tuple[dict[str, object], ...]]:
    index, bot_name, seed, stake, max_steps, trace_enabled = payload
    bot = create_bot(bot_name, seed=seed)
    options = LocalSimOptions(seed=seed, stake=stake, max_steps=max_steps)
    if trace_enabled:
        result, trace_rows = run_local_seed_with_trace(bot=bot, options=options)
        return index, result, trace_rows
    result = run_local_seed(bot=bot, options=options)
    return index, result, ()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        _ensure_deterministic_hash_seed()
    args = build_parser().parse_args(argv)
    seed_values = parse_seed_values(args.seed_list)
    seed_set = (
        make_explicit_seed_set(label=f"{args.stake}:{args.label}:explicit", seeds=seed_values)
        if seed_values
        else make_seed_set(label=f"{args.stake}:{args.label}", size=args.seeds)
    )
    indexed_results: dict[int, RunResult] = {}
    indexed_traces: dict[int, tuple[dict[str, object], ...]] = {}
    worker_count = max(1, int(args.workers))
    trace_enabled = args.trace_jsonl_out is not None
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_out.write_text("", encoding="utf-8")
    if args.trace_jsonl_out is not None:
        args.trace_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_jsonl_out.write_text("", encoding="utf-8")
    if worker_count == 1:
        for index, seed in enumerate(seed_set.seeds, start=1):
            _, result, trace_rows = _run_seed_worker((index, args.bot, seed, args.stake, args.max_steps, trace_enabled))
            indexed_results[index] = result
            if trace_enabled:
                indexed_traces[index] = trace_rows
            if args.jsonl_out is not None:
                _append_result_json(args.jsonl_out, result)
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(seed_set.seeds)):
                print(f"Progress: {index}/{len(seed_set.seeds)}")
    else:
        payloads = [
            (index, args.bot, seed, args.stake, args.max_steps, trace_enabled)
            for index, seed in enumerate(seed_set.seeds, start=1)
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_seed_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                index, result, trace_rows = future.result()
                indexed_results[index] = result
                if trace_enabled:
                    indexed_traces[index] = trace_rows
                if args.jsonl_out is not None:
                    _append_result_json(args.jsonl_out, result)
                completed += 1
                if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == len(seed_set.seeds)):
                    print(f"Progress: {completed}/{len(seed_set.seeds)}")
    results = [indexed_results[index] for index in range(1, len(seed_set.seeds) + 1)]
    if args.jsonl_out is not None:
        args.jsonl_out.write_text(
            "\n".join(_result_json(result) for result in results) + ("\n" if results else ""),
            encoding="utf-8",
        )
    if args.trace_jsonl_out is not None:
        trace_json_lines = [
            json.dumps(trace_row, sort_keys=True)
            for index in range(1, len(seed_set.seeds) + 1)
            for trace_row in indexed_traces.get(index, ())
        ]
        args.trace_jsonl_out.write_text(
            "\n".join(trace_json_lines) + ("\n" if trace_json_lines else ""),
            encoding="utf-8",
        )
    summary = summarize_runs(tuple(results))
    print(summary.to_text())
    return 0


def _ensure_deterministic_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == DETERMINISTIC_HASH_SEED:
        return
    if os.environ.get(_HASH_REEXEC_GUARD) == "1":
        raise RuntimeError(
            f"local_benchmark requires PYTHONHASHSEED={DETERMINISTIC_HASH_SEED}, "
            "but the deterministic relaunch did not apply it."
        )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = DETERMINISTIC_HASH_SEED
    env[_HASH_REEXEC_GUARD] = "1"
    module_name = __spec__.name if __spec__ is not None and __spec__.name else "balatro_ai.eval.local_benchmark"
    completed = subprocess.run([sys.executable, "-m", module_name, *sys.argv[1:]], env=env, check=False)
    raise SystemExit(completed.returncode)


def _result_json(result: RunResult) -> str:
    payload = asdict(result)
    payload["record_type"] = "run_summary"
    payload["ante"] = result.ante_reached
    return json.dumps(payload, sort_keys=True)


def _append_result_json(path: Path, result: RunResult) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(_result_json(result) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
