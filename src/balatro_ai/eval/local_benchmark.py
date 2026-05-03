"""Benchmark bots through the pure-Python local simulator."""

from __future__ import annotations

import argparse

from balatro_ai.bots.registry import create_bot
from balatro_ai.eval.metrics import summarize_runs
from balatro_ai.eval.seed_sets import make_explicit_seed_set, make_seed_set, parse_seed_values
from balatro_ai.sim.local_runner import LocalSimOptions, run_local_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fast local-simulator benchmark.")
    parser.add_argument("--bot", required=True, help="Bot name to benchmark.")
    parser.add_argument("--seeds", type=int, default=100, help="Number of deterministic seeds.")
    parser.add_argument("--seed-list", default="", help="Comma/space-separated exact seeds; overrides --seeds.")
    parser.add_argument("--stake", default="white", help="Stake label; first pass is tuned for white.")
    parser.add_argument("--label", default="local", help="Seed-set label.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum simulator steps per run.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N seeds; 0 disables.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_values = parse_seed_values(args.seed_list)
    seed_set = (
        make_explicit_seed_set(label=f"{args.stake}:{args.label}:explicit", seeds=seed_values)
        if seed_values
        else make_seed_set(label=f"{args.stake}:{args.label}", size=args.seeds)
    )
    results = []
    for index, seed in enumerate(seed_set.seeds, start=1):
        results.append(
            run_local_seed(
                bot=create_bot(args.bot, seed=seed),
                options=LocalSimOptions(seed=seed, stake=args.stake, max_steps=args.max_steps),
            )
        )
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(seed_set.seeds)):
            print(f"Progress: {index}/{len(seed_set.seeds)}")
    summary = summarize_runs(tuple(results))
    print(summary.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
