"""Sweep a single BotConfig parameter across a value list and report results.

Each (config_value, seed) is one local-sim job. All jobs share a single
ProcessPoolExecutor so slow seeds in one config don't block other configs.

Usage example:
    python -m balatro_ai.eval.config_sweep \\
        --param joker_sample_coefficient \\
        --values 0.05,0.08,0.12,0.16,0.20 \\
        --workers 8

Default seed list is the canonical 200-seed benchmark set. Override with
--seeds-file or --seed-list for a deliberate custom comparison.

The value matching BotConfig's default for the chosen param is treated as the
baseline for paired McNemar comparison. Pass --baseline-value to force a
specific row to be the baseline.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from math import comb
from pathlib import Path
from statistics import mean

from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.bots.config import BotConfig
from balatro_ai.eval.metrics import RunResult, summarize_runs
from balatro_ai.eval.seed_sets import make_benchmark_seed_set
from balatro_ai.sim.local_runner import LocalSimOptions, run_local_seed


DETERMINISTIC_HASH_SEED = "0"
_HASH_REEXEC_GUARD = "BOTLATRO_CONFIG_SWEEP_HASH_REEXEC"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep one BotConfig parameter over a value list.")
    parser.add_argument("--param", required=True, help="BotConfig field name to sweep.")
    parser.add_argument(
        "--values",
        required=True,
        help="Comma-separated values to test (parsed as floats).",
    )
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=None,
        help="Optional JSONL file with a 'seed' field per line. Overrides the canonical 200-seed default.",
    )
    parser.add_argument(
        "--seed-list",
        default="",
        help="Comma-separated seeds; overrides --seeds-file when set.",
    )
    parser.add_argument(
        "--seeds-label",
        default="",
        help="Deterministic seed set label. Overrides --seeds-file when set.",
    )
    parser.add_argument(
        "--seeds-count",
        type=int,
        default=200,
        help="Number of seeds to generate when --seeds-label is set.",
    )
    parser.add_argument(
        "--seed-window",
        type=int,
        default=1,
        help="Canonical 200-seed window: 1 = seeds 1-200, 2 = seeds 201-400, etc.",
    )
    parser.add_argument(
        "--fixed-overrides",
        default="",
        help=(
            "Comma-separated key=value pairs applied to every BotConfig in the "
            "sweep (e.g. 'panic_discard_cap=0.95,joker_sample_coefficient=0.20'). "
            "Useful for combined parameter tests."
        ),
    )
    parser.add_argument("--stake", default="white", help="Stake label.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Simulator step cap per run.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel process workers.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory for per-value JSONL output and the sweep CSV.",
    )
    parser.add_argument(
        "--baseline-value",
        type=float,
        default=None,
        help="Value to treat as the McNemar baseline. Defaults to the BotConfig default for --param.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N completed jobs; 0 disables.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        _ensure_deterministic_hash_seed()
    args = build_parser().parse_args(argv)

    _validate_param(args.param)
    values = _parse_values(args.values)
    seeds, seeds_source = _load_seeds(
        args.seeds_file,
        args.seed_list,
        args.seeds_label,
        args.seeds_count,
        args.seed_window,
    )
    fixed_overrides = _parse_fixed_overrides(args.fixed_overrides)
    if args.param in fixed_overrides:
        raise SystemExit(
            f"--fixed-overrides may not include the swept parameter '{args.param}'"
        )

    baseline_value = args.baseline_value
    if baseline_value is None:
        baseline_value = float(getattr(BotConfig(), args.param))
    if baseline_value not in values:
        print(
            f"warning: baseline value {baseline_value} for '{args.param}' is not in --values; "
            "McNemar comparisons will fall back to the closest value.",
            file=sys.stderr,
        )

    print(f"Param:          {args.param}")
    print(f"Values:         {', '.join(str(v) for v in values)}")
    print(f"Baseline value: {baseline_value}")
    print(f"Seeds:          {len(seeds)} from {seeds_source}")
    print(f"Workers:        {args.workers}")
    if fixed_overrides:
        overrides_text = ", ".join(f"{k}={v}" for k, v in fixed_overrides.items())
        print(f"Fixed overrides: {overrides_text}")
    print()

    out_dir = args.out_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Flat job queue: (value_index, value, seed_index, seed)
    jobs: list[tuple[int, float, int, int]] = []
    for vi, value in enumerate(values):
        for si, seed in enumerate(seeds):
            jobs.append((vi, value, si, seed))

    fixed_overrides_tuple = tuple(sorted(fixed_overrides.items()))
    payloads = [
        (vi, value, si, seed, args.param, args.stake, args.max_steps, fixed_overrides_tuple)
        for vi, value, si, seed in jobs
    ]

    # results[vi][si] = RunResult
    results: list[list[RunResult | None]] = [[None] * len(seeds) for _ in values]
    completed = 0
    total = len(payloads)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_run_one_job, payload) for payload in payloads]
        for future in as_completed(futures):
            vi, si, result = future.result()
            results[vi][si] = result
            completed += 1
            if args.progress_every > 0 and (
                completed % args.progress_every == 0 or completed == total
            ):
                print(f"Progress: {completed}/{total}")

    print()
    summaries: list[tuple[float, tuple[RunResult, ...]]] = []
    for vi, value in enumerate(values):
        run_tuple = tuple(r for r in results[vi] if r is not None)
        if len(run_tuple) != len(seeds):
            print(f"warning: value {value} got {len(run_tuple)}/{len(seeds)} results", file=sys.stderr)
        summaries.append((value, run_tuple))
        if out_dir is not None:
            _write_value_jsonl(out_dir, args.param, value, run_tuple)

    _print_sweep_table(args.param, summaries, baseline_value)

    if out_dir is not None:
        _write_sweep_csv(out_dir, args.param, summaries, baseline_value)
        print(f"\nWrote per-value JSONL and sweep.csv to {out_dir}")

    return 0


def _run_one_job(
    payload: tuple[int, float, int, int, str, str, int, tuple[tuple[str, float], ...]],
) -> tuple[int, int, RunResult]:
    vi, value, si, seed, param, stake, max_steps, fixed_overrides = payload
    overrides: dict[str, float] = dict(fixed_overrides)
    overrides[param] = value
    cfg = BotConfig(**overrides)
    bot = BasicStrategyBot(seed=seed, config=cfg)
    options = LocalSimOptions(seed=seed, stake=stake, max_steps=max_steps)
    result = run_local_seed(bot=bot, options=options)
    return vi, si, result


def _validate_param(name: str) -> None:
    fields = {f.name for f in dataclasses.fields(BotConfig)}
    if name not in fields:
        raise SystemExit(
            f"Unknown BotConfig field: {name!r}. Available fields:\n  "
            + "\n  ".join(sorted(fields))
        )


def _parse_values(raw: str) -> tuple[float, ...]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise SystemExit("--values must contain at least one number")
    try:
        return tuple(float(t) for t in tokens)
    except ValueError as exc:
        raise SystemExit(f"failed to parse --values: {exc}") from exc


def _load_seeds(
    seeds_file: Path | None,
    seed_list: str,
    seeds_label: str,
    seeds_count: int,
    seed_window: int,
) -> tuple[tuple[int, ...], str]:
    label = seeds_label.strip()
    if label:
        if seeds_count <= 0:
            raise SystemExit("--seeds-count must be positive when --seeds-label is set")
        seed_set = make_benchmark_seed_set(
            label=label,
            size=seeds_count,
            seed_window=seed_window,
        )
        return seed_set.seeds, f"label={label!r} (n={seeds_count})"

    text = seed_list.strip()
    if text:
        import re
        tokens = [t for t in re.split(r"[\s,;]+", text) if t]
        return tuple(int(t) for t in tokens), f"--seed-list ({len(tokens)} seeds)"

    if seeds_file is None:
        seed_set = make_benchmark_seed_set(
            label="config-sweep",
            size=200,
            seed_window=seed_window,
        )
        return seed_set.seeds, seed_set.label

    if not seeds_file.exists():
        raise SystemExit(f"seeds file not found: {seeds_file}")
    seeds: list[int] = []
    for line in seeds_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        seed = record.get("seed")
        if seed is None:
            continue
        seeds.append(int(seed))
    if not seeds:
        raise SystemExit(f"no seeds found in {seeds_file}")
    return tuple(seeds), str(seeds_file)


def _parse_fixed_overrides(raw: str) -> dict[str, float]:
    text = raw.strip()
    if not text:
        return {}
    fields = {f.name for f in dataclasses.fields(BotConfig)}
    overrides: dict[str, float] = {}
    for token in (t.strip() for t in text.split(",") if t.strip()):
        if "=" not in token:
            raise SystemExit(f"invalid --fixed-overrides token (expected key=value): {token!r}")
        key, _, value = token.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in fields:
            raise SystemExit(
                f"unknown BotConfig field in --fixed-overrides: {key!r}. "
                f"Available: {', '.join(sorted(fields))}"
            )
        try:
            overrides[key] = float(value)
        except ValueError as exc:
            raise SystemExit(f"could not parse {key}={value!r} as float: {exc}") from exc
    return overrides


def _mcnemar_exact_p_value(wins_flipped: int, wins_lost: int) -> float:
    discordant = wins_flipped + wins_lost
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(wins_flipped, wins_lost) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _holm_adjusted_p_values(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Holm-Bonferroni correction across a family of paired tests.

    Returns adjusted p-values in the same order as the input. Use when you
    test N non-baseline parameter values against the same baseline so the
    family-wise error rate stays controlled instead of leaking ~5% per test.
    """

    m = len(p_values)
    if m == 0:
        return ()
    indexed = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, original_index in enumerate(indexed):
        scaled = min(1.0, (m - rank) * p_values[original_index])
        running_max = max(running_max, scaled)
        adjusted[original_index] = running_max
    return tuple(adjusted)


def _select_baseline(
    summaries: list[tuple[float, tuple[RunResult, ...]]],
    requested: float,
) -> tuple[float, tuple[RunResult, ...]]:
    for value, runs in summaries:
        if value == requested:
            return value, runs
    # fallback: nearest by absolute distance
    return min(summaries, key=lambda item: abs(item[0] - requested))


def _print_sweep_table(
    param: str,
    summaries: list[tuple[float, tuple[RunResult, ...]]],
    baseline_value: float,
) -> None:
    baseline_val, baseline_runs = _select_baseline(summaries, baseline_value)
    baseline_won = {r.seed: r.won for r in baseline_runs}

    # First pass: compute raw stats so we can derive Holm-adjusted p-values
    # across the non-baseline values as a family before printing.
    raw_rows: list[dict] = []
    for value, runs in summaries:
        if not runs:
            raw_rows.append({"value": value, "empty": True})
            continue
        wins = sum(1 for r in runs if r.won)
        winrate = wins / len(runs)
        avg_ante = mean(r.ante_reached for r in runs)
        a5 = sum(1 for r in runs if r.ante_reached >= 5)
        a6 = sum(1 for r in runs if r.ante_reached >= 6)
        a7 = sum(1 for r in runs if r.ante_reached >= 7)
        sec = mean(r.runtime_seconds for r in runs)
        if value == baseline_val:
            flips_plus = flips_minus = 0
            p_value: float | None = None
        else:
            flips_plus = sum(1 for r in runs if r.won and not baseline_won.get(r.seed, False))
            flips_minus = sum(1 for r in runs if (not r.won) and baseline_won.get(r.seed, False))
            p_value = _mcnemar_exact_p_value(flips_plus, flips_minus)
        raw_rows.append(
            {
                "empty": False,
                "value": value,
                "wins": wins,
                "winrate": winrate,
                "avg_ante": avg_ante,
                "a5": a5,
                "a6": a6,
                "a7": a7,
                "flips_plus": flips_plus,
                "flips_minus": flips_minus,
                "p_value": p_value,
                "sec": sec,
            }
        )
    non_baseline = [row for row in raw_rows if not row["empty"] and row["p_value"] is not None]
    holm = _holm_adjusted_p_values(tuple(row["p_value"] for row in non_baseline))
    for row, adjusted in zip(non_baseline, holm, strict=True):
        row["holm_p"] = adjusted

    print(f"Sweep results for {param} (baseline = {baseline_val}):")
    print()
    header = (
        f"{'value':>8}  {'wins':>5}  {'winrate':>8}  {'avg ante':>8}  "
        f"{'a5+':>5}  {'a6+':>5}  {'a7+':>5}  {'flips +':>7}  {'flips -':>7}  "
        f"{'McNemar p':>10}  {'Holm p':>8}  {'sec/run':>8}"
    )
    print(header)
    print("-" * len(header))

    rows: list[tuple] = []
    for row in raw_rows:
        if row["empty"]:
            print(f"{row['value']:>8}  (no results)")
            continue
        is_baseline = row["p_value"] is None
        p_value = 1.0 if is_baseline else float(row["p_value"])
        holm_p = float(row.get("holm_p", 1.0)) if not is_baseline else 1.0
        marker = " *" if is_baseline else ""
        print(
            f"{row['value']:>8}  {row['wins']:>5}  {row['winrate']:>7.1%}  {row['avg_ante']:>8.2f}  "
            f"{row['a5']:>5}  {row['a6']:>5}  {row['a7']:>5}  {row['flips_plus']:>7}  {row['flips_minus']:>7}  "
            f"{p_value:>10.4f}  {holm_p:>8.4f}  {row['sec']:>7.2f}s{marker}"
        )
        rows.append(
            (
                row["value"], row["wins"], row["winrate"], row["avg_ante"],
                row["a5"], row["a6"], row["a7"], row["flips_plus"], row["flips_minus"],
                p_value, row["sec"],
            )
        )

    print()
    print("* = baseline value (paired McNemar reference)")
    print("Holm p = family-wise-corrected McNemar p across non-baseline values. Use this to gate \"significant\" claims.")
    if rows:
        best = max(rows, key=lambda r: (r[1], r[3]))  # by wins, tiebreak avg ante
        print(f"Best winrate: {best[0]} -> {best[1]} wins ({best[2]:.1%}), avg ante {best[3]:.2f}")


def _write_value_jsonl(
    out_dir: Path, param: str, value: float, runs: tuple[RunResult, ...]
) -> None:
    path = out_dir / f"{param}_{_safe_value_token(value)}.jsonl"
    lines = []
    for r in runs:
        payload = asdict(r)
        payload["record_type"] = "run_summary"
        payload["ante"] = r.ante_reached
        payload["config_param"] = param
        payload["config_value"] = value
        lines.append(json.dumps(payload, sort_keys=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_sweep_csv(
    out_dir: Path,
    param: str,
    summaries: list[tuple[float, tuple[RunResult, ...]]],
    baseline_value: float,
) -> None:
    baseline_val, baseline_runs = _select_baseline(summaries, baseline_value)
    baseline_won = {r.seed: r.won for r in baseline_runs}

    # Precompute non-baseline McNemar p-values so we can attach Holm-adjusted
    # values in the same order they appear in the CSV.
    non_baseline_raw_p: list[float] = []
    non_baseline_positions: list[int] = []
    raw_p_by_index: dict[int, float] = {}
    for index, (value, runs) in enumerate(summaries):
        if not runs or value == baseline_val:
            continue
        flips_plus = sum(1 for r in runs if r.won and not baseline_won.get(r.seed, False))
        flips_minus = sum(1 for r in runs if (not r.won) and baseline_won.get(r.seed, False))
        p_value = _mcnemar_exact_p_value(flips_plus, flips_minus)
        raw_p_by_index[index] = p_value
        non_baseline_positions.append(index)
        non_baseline_raw_p.append(p_value)
    holm = _holm_adjusted_p_values(tuple(non_baseline_raw_p))
    holm_by_index: dict[int, float] = dict(zip(non_baseline_positions, holm, strict=True))

    path = out_dir / "sweep.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "param",
                "value",
                "is_baseline",
                "runs",
                "wins",
                "winrate",
                "avg_ante",
                "ante5plus",
                "ante6plus",
                "ante7plus",
                "wins_flipped_plus",
                "wins_flipped_minus",
                "mcnemar_p_value",
                "holm_p_value",
                "avg_runtime_sec",
                "avg_final_money",
            ]
        )
        for index, (value, runs) in enumerate(summaries):
            if not runs:
                continue
            wins = sum(1 for r in runs if r.won)
            winrate = wins / len(runs)
            avg_ante = mean(r.ante_reached for r in runs)
            a5 = sum(1 for r in runs if r.ante_reached >= 5)
            a6 = sum(1 for r in runs if r.ante_reached >= 6)
            a7 = sum(1 for r in runs if r.ante_reached >= 7)
            money = mean(r.final_money for r in runs)
            sec = mean(r.runtime_seconds for r in runs)
            if value == baseline_val:
                flips_plus = flips_minus = 0
                p_value = 1.0
                holm_p = 1.0
            else:
                flips_plus = sum(
                    1 for r in runs if r.won and not baseline_won.get(r.seed, False)
                )
                flips_minus = sum(
                    1 for r in runs if (not r.won) and baseline_won.get(r.seed, False)
                )
                p_value = raw_p_by_index.get(index, _mcnemar_exact_p_value(flips_plus, flips_minus))
                holm_p = holm_by_index.get(index, 1.0)
            writer.writerow(
                [
                    param,
                    value,
                    "yes" if value == baseline_val else "no",
                    len(runs),
                    wins,
                    f"{winrate:.4f}",
                    f"{avg_ante:.4f}",
                    a5,
                    a6,
                    a7,
                    flips_plus,
                    flips_minus,
                    f"{p_value:.6f}",
                    f"{holm_p:.6f}",
                    f"{sec:.4f}",
                    f"{money:.4f}",
                ]
            )


def _safe_value_token(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "neg")


def _ensure_deterministic_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == DETERMINISTIC_HASH_SEED:
        return
    if os.environ.get(_HASH_REEXEC_GUARD) == "1":
        raise RuntimeError(
            f"config_sweep requires PYTHONHASHSEED={DETERMINISTIC_HASH_SEED}, "
            "but the deterministic relaunch did not apply it."
        )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = DETERMINISTIC_HASH_SEED
    env[_HASH_REEXEC_GUARD] = "1"
    module = __spec__.name if __spec__ is not None and __spec__.name else "balatro_ai.eval.config_sweep"
    completed = subprocess.run([sys.executable, "-m", module, *sys.argv[1:]], env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
