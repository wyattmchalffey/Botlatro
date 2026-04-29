"""Benchmark command entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from balatro_ai.eval.runner import BenchmarkOptions, endpoint_urls, run_benchmark
from balatro_ai.eval.run_seed import REPLAY_MODES
from balatro_ai.eval.seed_sets import make_explicit_seed_set, make_seed_set, parse_seed_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a Botlatro benchmark run.")
    parser.add_argument("--bot", required=True, help="Bot name to benchmark.")
    parser.add_argument("--seeds", type=int, default=100, help="Number of deterministic seeds.")
    parser.add_argument("--seed-list", default="", help="Comma/space-separated exact seeds; overrides --seeds.")
    parser.add_argument("--stake", default="white", help="Stake name.")
    parser.add_argument("--deck", default="RED", help="Deck enum to start with, e.g. RED.")
    parser.add_argument("--profile-name", default="P1", help="Balatro profile used by the benchmark.")
    parser.add_argument("--unlock-state", default="all", help="Unlock pool description, e.g. all or profile-default.")
    parser.add_argument("--label", default="default", help="Seed-set label.")
    parser.add_argument("--execute", action="store_true", help="Run the benchmark against a live bridge.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346", help="JSON-RPC endpoint.")
    parser.add_argument("--host", default="127.0.0.1", help="Host used with --base-port/--workers.")
    parser.add_argument("--base-port", type=int, default=12346, help="First worker port.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker endpoints.")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="JSON-RPC request timeout.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Step cap per run.")
    parser.add_argument("--replay-dir", type=Path, help="Optional directory for per-run replay JSONL files.")
    parser.add_argument("--start-retries", type=int, default=1, help="Retries for bridge start/reset failures.")
    parser.add_argument(
        "--fast-benchmark",
        action="store_true",
        help="Use low-overhead benchmark defaults; currently selects summary replay unless --replay-mode is set.",
    )
    parser.add_argument(
        "--replay-mode",
        choices=REPLAY_MODES,
        default=None,
        help="Replay detail: off, summary-only JSONL, light JSONL, or score-audit JSONL.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_values = parse_seed_values(args.seed_list)
    seed_set = (
        make_explicit_seed_set(label=f"{args.stake}:{args.label}:explicit", seeds=seed_values)
        if seed_values
        else make_seed_set(label=f"{args.stake}:{args.label}", size=args.seeds)
    )
    print(f"Bot: {args.bot}")
    print(f"Stake: {args.stake}")
    print(f"Deck: {args.deck.upper()}")
    print(f"Profile: {args.profile_name}")
    print(f"Unlocks: {args.unlock_state}")
    print(f"Seeds: {len(seed_set.seeds)}")
    print(f"First seed: {seed_set.seeds[0] if seed_set.seeds else '-'}")

    if not args.execute:
        print("Status: benchmark prepared; pass --execute to run against a live BalatroBot bridge.")
        return 0

    replay_mode = args.replay_mode or ("summary" if args.fast_benchmark else "score_audit")
    endpoints = (
        (args.endpoint,)
        if args.workers == 1 and args.endpoint
        else endpoint_urls(args.host, args.base_port, args.workers)
    )
    run_benchmark(
        BenchmarkOptions(
            bot=args.bot,
            stake=args.stake,
            deck=args.deck,
            profile_name=args.profile_name,
            unlock_state=args.unlock_state,
            seeds=args.seeds,
            seed_values=seed_values or None,
            label=args.label,
            endpoints=endpoints,
            timeout_seconds=args.timeout_seconds,
            max_steps=args.max_steps,
            replay_dir=args.replay_dir,
            replay_mode=replay_mode,
            start_retries=args.start_retries,
        ),
        progress=print,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
