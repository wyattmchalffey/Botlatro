"""Compare two bots on the same seed set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import comb, erf, sqrt
from pathlib import Path
import random
import re
from statistics import mean

from balatro_ai.eval.metrics import RunResult
from balatro_ai.eval.runner import BenchmarkOptions, endpoint_urls, run_benchmark
from balatro_ai.eval.run_seed import REPLAY_MODES
from balatro_ai.eval.seed_sets import make_explicit_seed_set, make_seed_set, parse_seed_values

PROGRESS_RESULT_PATTERN = re.compile(
    r"^\[(?:\d+/\d+|retry \d+)\]\s+"
    r"seed=(?P<seed>-?\d+)\s+"
    r"won=(?P<won>True|False)\s+"
    r"ante=(?P<ante>-?\d+)\s+"
    r"score=(?P<score>-?\d+)\s+"
    r"money=(?P<money>-?\d+)\s+"
    r"runtime=(?P<runtime>[0-9]+(?:\.[0-9]+)?)s"
    r"(?:\s+death=(?P<death>.*))?$"
)


@dataclass(frozen=True, slots=True)
class ComparisonSummary:
    bot_a: str
    bot_b: str
    stake: str
    seed_count: int
    bot_a_win_rate: float
    bot_b_win_rate: float
    win_rate_delta: float
    wins_flipped: int
    wins_lost: int
    mcnemar_p_value: float
    bot_a_average_ante: float
    bot_b_average_ante: float
    average_ante_delta: float
    wilcoxon_ante_p_value: float
    bot_a_average_score: float
    bot_b_average_score: float
    average_score_delta: float
    score_delta_ci_low: float
    score_delta_ci_high: float

    def to_text(self) -> str:
        return "\n".join(
            (
                "Paired bot comparison",
                f"Bot A: {self.bot_a}",
                f"Bot B: {self.bot_b}",
                f"Stake: {self.stake}",
                f"Paired seeds: {self.seed_count}",
                "",
                f"Bot A win rate: {self.bot_a_win_rate:.1%}",
                f"Bot B win rate: {self.bot_b_win_rate:.1%}",
                f"Win rate delta: {self.win_rate_delta:+.1%}",
                f"Wins flipped (A lost, B won): {self.wins_flipped}",
                f"Wins lost (A won, B lost): {self.wins_lost}",
                f"McNemar p-value: {self.mcnemar_p_value:.6g}",
                "",
                f"Bot A average ante: {self.bot_a_average_ante:.2f}",
                f"Bot B average ante: {self.bot_b_average_ante:.2f}",
                f"Average ante delta: {self.average_ante_delta:+.2f}",
                f"Wilcoxon signed-rank p-value: {self.wilcoxon_ante_p_value:.6g}",
                "",
                f"Bot A average score: {self.bot_a_average_score:.1f}",
                f"Bot B average score: {self.bot_b_average_score:.1f}",
                f"Average score delta: {self.average_score_delta:+.1f}",
                "Score delta bootstrap 95% CI: "
                f"[{self.score_delta_ci_low:+.1f}, {self.score_delta_ci_high:+.1f}]",
            )
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "bot_a": self.bot_a,
            "bot_b": self.bot_b,
            "stake": self.stake,
            "seed_count": self.seed_count,
            "win_rate": {
                "bot_a": self.bot_a_win_rate,
                "bot_b": self.bot_b_win_rate,
                "delta": self.win_rate_delta,
                "wins_flipped": self.wins_flipped,
                "wins_lost": self.wins_lost,
                "mcnemar_p_value": self.mcnemar_p_value,
            },
            "ante": {
                "bot_a_average": self.bot_a_average_ante,
                "bot_b_average": self.bot_b_average_ante,
                "average_delta": self.average_ante_delta,
                "wilcoxon_signed_rank_p_value": self.wilcoxon_ante_p_value,
            },
            "score": {
                "bot_a_average": self.bot_a_average_score,
                "bot_b_average": self.bot_b_average_score,
                "average_delta": self.average_score_delta,
                "bootstrap_95_ci": [self.score_delta_ci_low, self.score_delta_ci_high],
            },
        }


def compare_paired_results(
    bot_a_results: tuple[RunResult, ...] | list[RunResult],
    bot_b_results: tuple[RunResult, ...] | list[RunResult],
    *,
    bootstrap_samples: int = 5000,
    rng_seed: int = 0,
) -> ComparisonSummary:
    by_seed_a = {result.seed: result for result in bot_a_results}
    by_seed_b = {result.seed: result for result in bot_b_results}
    common_seeds = tuple(sorted(set(by_seed_a) & set(by_seed_b)))
    if not common_seeds:
        raise ValueError("Cannot compare results without overlapping seeds")

    paired_a = tuple(by_seed_a[seed] for seed in common_seeds)
    paired_b = tuple(by_seed_b[seed] for seed in common_seeds)
    first_a = paired_a[0]
    first_b = paired_b[0]
    wins_flipped = sum(1 for a, b in zip(paired_a, paired_b, strict=True) if not a.won and b.won)
    wins_lost = sum(1 for a, b in zip(paired_a, paired_b, strict=True) if a.won and not b.won)
    ante_deltas = tuple(b.ante_reached - a.ante_reached for a, b in zip(paired_a, paired_b, strict=True))
    score_deltas = tuple(b.final_score - a.final_score for a, b in zip(paired_a, paired_b, strict=True))
    score_low, score_high = _bootstrap_mean_ci(score_deltas, samples=bootstrap_samples, rng_seed=rng_seed)

    return ComparisonSummary(
        bot_a=first_a.bot_version,
        bot_b=first_b.bot_version,
        stake=first_a.stake,
        seed_count=len(common_seeds),
        bot_a_win_rate=sum(1 for result in paired_a if result.won) / len(paired_a),
        bot_b_win_rate=sum(1 for result in paired_b if result.won) / len(paired_b),
        win_rate_delta=(sum(1 for result in paired_b if result.won) - sum(1 for result in paired_a if result.won))
        / len(paired_a),
        wins_flipped=wins_flipped,
        wins_lost=wins_lost,
        mcnemar_p_value=_mcnemar_exact_p_value(wins_flipped, wins_lost),
        bot_a_average_ante=mean(result.ante_reached for result in paired_a),
        bot_b_average_ante=mean(result.ante_reached for result in paired_b),
        average_ante_delta=mean(ante_deltas),
        wilcoxon_ante_p_value=_wilcoxon_signed_rank_p_value(ante_deltas),
        bot_a_average_score=mean(result.final_score for result in paired_a),
        bot_b_average_score=mean(result.final_score for result in paired_b),
        average_score_delta=mean(score_deltas),
        score_delta_ci_low=score_low,
        score_delta_ci_high=score_high,
    )


def load_run_results(
    paths: tuple[Path, ...] | list[Path],
    *,
    default_bot: str,
    default_stake: str,
) -> tuple[RunResult, ...]:
    """Load completed run summaries from replay JSONL files or benchmark stdout logs."""

    by_seed: dict[int, RunResult] = {}
    for path in _expand_result_paths(paths):
        loader = _load_jsonl_results if path.suffix.lower() == ".jsonl" else _load_progress_log_results
        for result in loader(path, default_bot=default_bot, default_stake=default_stake):
            by_seed[result.seed] = result
    if not by_seed:
        raise ValueError("No run results found in provided path(s)")
    return tuple(by_seed[seed] for seed in sorted(by_seed))


def _expand_result_paths(paths: tuple[Path, ...] | list[Path]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.jsonl")))
        elif path.exists():
            expanded.append(path)
    return tuple(expanded)


def _load_jsonl_results(path: Path, *, default_bot: str, default_stake: str) -> tuple[RunResult, ...]:
    results: list[RunResult] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("record_type") != "run_summary":
                continue
            results.append(
                RunResult(
                    bot_version=str(row.get("bot_version") or default_bot),
                    seed=_int_value(row.get("seed")),
                    stake=str(row.get("stake") or default_stake),
                    won=bool(row.get("won")),
                    ante_reached=_int_value(row.get("ante", row.get("ante_reached"))),
                    final_score=_int_value(row.get("final_score")),
                    final_money=_int_value(row.get("final_money")),
                    runtime_seconds=_float_value(row.get("runtime_seconds")),
                    death_reason=_optional_string(row.get("death_reason")),
                )
            )
    return tuple(results)


def _load_progress_log_results(path: Path, *, default_bot: str, default_stake: str) -> tuple[RunResult, ...]:
    bot = default_bot
    stake = default_stake
    results: list[RunResult] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text.startswith("Bot:"):
                bot = text.split(":", 1)[1].strip() or default_bot
                continue
            if text.startswith("Stake:"):
                stake = text.split(":", 1)[1].strip() or default_stake
                continue
            match = PROGRESS_RESULT_PATTERN.match(text)
            if not match:
                continue
            values = match.groupdict()
            results.append(
                RunResult(
                    bot_version=bot,
                    seed=int(values["seed"]),
                    stake=stake,
                    won=values["won"] == "True",
                    ante_reached=int(values["ante"]),
                    final_score=int(values["score"]),
                    final_money=int(values["money"]),
                    runtime_seconds=float(values["runtime"]),
                    death_reason=values.get("death"),
                )
            )
    return tuple(results)


def _mcnemar_exact_p_value(wins_flipped: int, wins_lost: int) -> float:
    discordant = wins_flipped + wins_lost
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(wins_flipped, wins_lost) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _wilcoxon_signed_rank_p_value(differences: tuple[int | float, ...]) -> float:
    nonzero = tuple(float(value) for value in differences if value != 0)
    n = len(nonzero)
    if n == 0:
        return 1.0

    ranks = _absolute_ranks(nonzero)
    positive_rank_sum = sum(rank for rank, value in zip(ranks, nonzero, strict=True) if value > 0)
    mean_rank_sum = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0:
        return 1.0
    continuity = 0.5 if positive_rank_sum > mean_rank_sum else -0.5
    z_score = (positive_rank_sum - mean_rank_sum - continuity) / sqrt(variance)
    tail = 1.0 - _standard_normal_cdf(abs(z_score))
    return max(0.0, min(1.0, 2.0 * tail))


def _absolute_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(enumerate(abs(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return tuple(ranks)


def _standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _bootstrap_mean_ci(
    values: tuple[int | float, ...],
    *,
    samples: int,
    rng_seed: int,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1 or samples <= 0:
        value = float(mean(values))
        return (value, value)

    rng = random.Random(rng_seed)
    size = len(values)
    means = sorted(mean(rng.choice(values) for _ in range(size)) for _ in range(samples))
    low_index = max(0, int(samples * 0.025) - 1)
    high_index = min(samples - 1, int(samples * 0.975))
    return (float(means[low_index]), float(means[high_index]))


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _filter_seed_values(results: tuple[RunResult, ...], seed_values: tuple[int, ...]) -> tuple[RunResult, ...]:
    if not seed_values:
        return results
    allowed = set(seed_values)
    return tuple(result for result in results if result.seed in allowed)


def _emit_comparison(
    comparison: ComparisonSummary,
    *,
    save_json: Path | None,
    emit_json: bool,
) -> None:
    payload = comparison.to_json_dict()
    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if emit_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("")
        print(comparison.to_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two bots on the same Botlatro seed set.")
    parser.add_argument("--bot-a", required=True, help="Baseline bot name.")
    parser.add_argument("--bot-b", required=True, help="Candidate bot name.")
    parser.add_argument(
        "--bot-a-results",
        nargs="+",
        type=Path,
        help="Saved replay JSONL file(s), replay directory, or benchmark stdout log for Bot A.",
    )
    parser.add_argument(
        "--bot-b-results",
        nargs="+",
        type=Path,
        help="Saved replay JSONL file(s), replay directory, or benchmark stdout log for Bot B.",
    )
    parser.add_argument("--seeds", type=int, default=100, help="Number of deterministic seeds.")
    parser.add_argument("--seed-list", default="", help="Comma/space-separated exact seeds; overrides --seeds.")
    parser.add_argument("--stake", default="white", help="Stake name.")
    parser.add_argument("--deck", default="RED", help="Deck enum to start with, e.g. RED.")
    parser.add_argument("--profile-name", default="P1", help="Balatro profile used by the benchmark.")
    parser.add_argument("--unlock-state", default="all", help="Unlock pool description, e.g. all or profile-default.")
    parser.add_argument("--label", default="phase7-compare", help="Seed-set label.")
    parser.add_argument("--execute", action="store_true", help="Run both benchmarks against live bridge endpoints.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346", help="JSON-RPC endpoint.")
    parser.add_argument("--host", default="127.0.0.1", help="Host used with --base-port/--workers.")
    parser.add_argument("--base-port", type=int, default=12346, help="First worker port.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker endpoints.")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="JSON-RPC request timeout.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Step cap per run.")
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=1800.0,
        help="Wall-clock cap for one full seed; use 0 to disable.",
    )
    parser.add_argument("--replay-dir", type=Path, help="Optional parent directory for per-bot replay JSONL files.")
    parser.add_argument("--start-retries", type=int, default=1, help="Retries for bridge start/reset failures.")
    parser.add_argument("--retry-failed-seeds", type=int, default=1, help="Retries for bridge/client error seeds.")
    parser.add_argument("--replay-mode", choices=REPLAY_MODES, default="summary", help="Replay detail for both runs.")
    parser.add_argument("--bootstrap-samples", type=int, default=5000, help="Bootstrap resamples for score delta CI.")
    parser.add_argument("--save-json", type=Path, help="Write the comparison summary to this JSON file.")
    parser.add_argument("--json", action="store_true", help="Emit the comparison summary as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_values = parse_seed_values(args.seed_list)
    using_saved_results = bool(args.bot_a_results or args.bot_b_results)

    print(f"Bot A: {args.bot_a}")
    print(f"Bot B: {args.bot_b}")
    print(f"Stake: {args.stake}")
    print(f"Deck: {args.deck.upper()}")
    print(f"Profile: {args.profile_name}")
    print(f"Unlocks: {args.unlock_state}")
    if using_saved_results:
        if not args.bot_a_results or not args.bot_b_results:
            raise SystemExit("Provide both --bot-a-results and --bot-b-results, or neither.")
        if seed_values:
            print(f"Seed filter: {len(seed_values)} explicit seed(s)")
            print(f"First seed: {seed_values[0] if seed_values else '-'}")
        else:
            print("Seed filter: all overlapping saved seeds")
        bot_a_results = _filter_seed_values(
            load_run_results(args.bot_a_results, default_bot=args.bot_a, default_stake=args.stake),
            seed_values,
        )
        bot_b_results = _filter_seed_values(
            load_run_results(args.bot_b_results, default_bot=args.bot_b, default_stake=args.stake),
            seed_values,
        )
        print(f"Loaded Bot A results: {len(bot_a_results)}")
        print(f"Loaded Bot B results: {len(bot_b_results)}")
        comparison = compare_paired_results(
            bot_a_results,
            bot_b_results,
            bootstrap_samples=args.bootstrap_samples,
        )
        _emit_comparison(comparison, save_json=args.save_json, emit_json=args.json)
        return 0

    seed_set = (
        make_explicit_seed_set(label=f"{args.stake}:{args.label}:explicit", seeds=seed_values)
        if seed_values
        else make_seed_set(label=f"{args.stake}:{args.label}", size=args.seeds)
    )
    print(f"Seeds: {len(seed_set.seeds)}")
    print(f"First seed: {seed_set.seeds[0] if seed_set.seeds else '-'}")
    if not args.execute:
        print("Status: comparison prepared; pass --execute to run both bots against live BalatroBot bridge(s).")
        return 0

    endpoints = (
        (args.endpoint,)
        if args.workers == 1 and args.endpoint
        else endpoint_urls(args.host, args.base_port, args.workers)
    )
    options = {
        "stake": args.stake,
        "deck": args.deck,
        "profile_name": args.profile_name,
        "unlock_state": args.unlock_state,
        "seeds": args.seeds,
        "seed_values": seed_values or None,
        "label": args.label,
        "endpoints": endpoints,
        "timeout_seconds": args.timeout_seconds,
        "max_steps": args.max_steps,
        "run_timeout_seconds": _optional_positive_float(args.run_timeout_seconds),
        "replay_mode": args.replay_mode,
        "start_retries": args.start_retries,
        "retry_failed_seeds": args.retry_failed_seeds,
    }
    replay_parent = args.replay_dir
    bot_a_run = run_benchmark(
        BenchmarkOptions(
            bot=args.bot_a,
            replay_dir=(replay_parent / args.bot_a) if replay_parent is not None else None,
            **options,
        ),
        progress=print,
    )
    bot_b_run = run_benchmark(
        BenchmarkOptions(
            bot=args.bot_b,
            replay_dir=(replay_parent / args.bot_b) if replay_parent is not None else None,
            **options,
        ),
        progress=print,
    )
    comparison = compare_paired_results(
        bot_a_run.results,
        bot_b_run.results,
        bootstrap_samples=args.bootstrap_samples,
    )
    _emit_comparison(comparison, save_json=args.save_json, emit_json=args.json)
    return 0


def _optional_positive_float(value: float) -> float | None:
    return value if value > 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
