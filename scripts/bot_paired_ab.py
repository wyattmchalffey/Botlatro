"""THE canonical paired same-seed A/B bench (P0 instrument, upgraded 2026-07-02).

Runs each seed once with bot A and once with bot B inside the same worker,
cancelling seed variance. Every run of this script is:

  * HONEST by default — BALATRO_NO_FORESIGHT is forced to "shuffle" unless
    already set; a clairvoyant run requires --allow-clairvoyant and is stamped
    into the result JSON. (The pre-2026-06-10 record certified clairvoyant
    players; this is the guard against repeating that.)
  * DETERMINISTIC — per-run seeding of random/numpy and single-thread torch
    (the pre-2026-06-16 local gates read the same checkpoint at 26/38/42
    wins/1024 from multi-threaded FP reduction noise). `--self-test` certifies
    it end-to-end: A vs A must be bit-identical on every pair.
  * FAITHFUL-mode by default (real Balatro seed RNG); --generic opts out.
  * ENFORCING — --expect improves|non-inferior turns the verdict into the exit
    code (0 pass, 3 fail), so pipelines cannot proceed past a failed gate.

Config knobs A/B in paired form (supersedes winrate_bench_config for A/Bs):
--config-a/--config-b take a JSON dict of BotConfig field overrides (or
@path/to/file.json). Neural arms: --ckpt-a/--ckpt-b set BALATRO_POLICY_CKPT.

    PYTHONPATH=src python scripts/bot_paired_ab.py \
        --bot-a basic_strategy_bot --bot-b basic_strategy_bot \
        --config-b '{"shop_target_safety_base": 1.15}' \
        --seeds 1024 --jobs 12 --expect non-inferior --margin 0.02 \
        --metrics .data/ab_safety_base.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path


def _determinism_guard() -> None:
    """Per-run determinism: fixed global seeds + single-thread torch (when
    present). Multi-threaded torch FP reductions flip ~1-2% of argmax-boundary
    decisions run-to-run — the bug that made every pre-2026-06-16 local gate
    irreproducible. Heuristic-only benches pay one lazy torch import per
    worker process; workers persist across seeds so the cost amortizes."""
    import random

    random.seed(0)
    os.environ["BALATRO_DEVICE"] = "cpu"
    os.environ.setdefault("BALATRO_NO_FORESIGHT", "shuffle")
    try:
        import numpy as np

        np.random.seed(0)
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(0)
    except ImportError:
        pass


def _run_one(bot_name: str, seed: str, faithful: bool,
             overrides: dict | None = None, ckpt: str | None = None) -> dict:
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase, with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    _determinism_guard()
    kwargs = {"seed": _stable_seed_int(seed), "stake": "white"}
    if faithful:
        kwargs["balatro_seed"] = seed
    old_run_seed = os.environ.get("BALATRO_RUN_SEED")
    os.environ["BALATRO_RUN_SEED"] = seed
    old_ckpt = os.environ.get("BALATRO_POLICY_CKPT")
    if ckpt:
        os.environ["BALATRO_POLICY_CKPT"] = ckpt
    try:
        sim = LocalBalatroSimulator(**kwargs)
        sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
        bot = create_bot(bot_name, seed=0)
        termination = "max_steps"
        started = time.process_time()
        cfg = replace(DEFAULT_CONFIG, shop_audit_enabled=False, **(overrides or {}))
        with bot_config_scope(cfg):
            for steps in range(1, 4001):
                state = sim.state
                if state.run_over or state.phase == GamePhase.RUN_OVER:
                    termination = "run_over"
                    break
                action = bot.choose_action(state)
                if action is None or action.action_type == ActionType.NO_OP:
                    termination = "no_action"
                    break
                try:
                    sim.step(action)
                except (ValueError, IndexError, KeyError, TypeError, AttributeError) as exc:
                    termination = f"sim_error:{type(exc).__name__}"
                    break
            else:
                steps = 4000
    finally:
        if old_run_seed is None:
            os.environ.pop("BALATRO_RUN_SEED", None)
        else:
            os.environ["BALATRO_RUN_SEED"] = old_run_seed
        if ckpt:
            if old_ckpt is None:
                os.environ.pop("BALATRO_POLICY_CKPT", None)
            else:
                os.environ["BALATRO_POLICY_CKPT"] = old_ckpt

    final = sim.state
    won = bool(final.won)
    run_over = bool(final.run_over or final.phase == GamePhase.RUN_OVER)
    return {
        "bot": bot_name,
        "won": won,
        "run_over": run_over,
        "termination": "run_over" if run_over else termination,
        "ante": int(final.ante),
        "score": int(final.current_score),
        "required_score": int(final.required_score),
        "loss_frac": (final.current_score / final.required_score)
        if (not won and final.required_score > 0)
        else None,
        "steps": steps,
        "cpu_s": round(time.process_time() - started, 3),
    }


def _run_pair(args: tuple) -> dict:
    seed, bot_a, bot_b, faithful, ov_a, ov_b, ckpt_a, ckpt_b = args
    a = _run_one(bot_a, seed, faithful, ov_a, ckpt_a)
    b = _run_one(bot_b, seed, faithful, ov_b, ckpt_b)
    return {
        "seed": seed,
        "a": a,
        "b": b,
        "d_ante": b["ante"] - a["ante"],
        "d_score": b["score"] - a["score"],
        "d_win": int(b["won"]) - int(a["won"]),
    }


def _write_pair(path: Path, task: tuple) -> None:
    row = _run_pair(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(row, fh)
    tmp_path.replace(path)


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def summarize(rows: list[dict], *, bot_a: str, bot_b: str, faithful: bool, wall_s: float) -> dict:
    a_rows = [r["a"] for r in rows]
    b_rows = [r["b"] for r in rows]
    d_ante = [r["d_ante"] for r in rows]
    d_score_same_ante_losses = [
        r["d_score"]
        for r in rows
        if not r["a"]["won"] and not r["b"]["won"] and r["a"]["ante"] == r["b"]["ante"]
    ]
    d_loss_frac = [
        r["b"]["loss_frac"] - r["a"]["loss_frac"]
        for r in rows
        if r["a"]["loss_frac"] is not None and r["b"]["loss_frac"] is not None
    ]
    from balatro_ai.bench_stats import mcnemar_exact_p, paired_delta_ci, paired_mean_diff_ci

    flips_for_b = sum(1 for r in rows if not r["a"]["won"] and r["b"]["won"])
    flips_for_a = sum(1 for r in rows if r["a"]["won"] and not r["b"]["won"])
    better = sum(1 for d in d_ante if d > 0)
    worse = sum(1 for d in d_ante if d < 0)
    ante_mean, ante_lo, ante_hi = (
        paired_mean_diff_ci([float(d) for d in d_ante]) if d_ante else (0.0, 0.0, 0.0)
    )
    return {
        "bot_a": bot_a,
        "bot_b": bot_b,
        "faithful": faithful,
        "no_foresight": os.environ.get("BALATRO_NO_FORESIGHT", ""),
        "n": len(rows),
        "wall_s": round(wall_s, 2),
        "a": _arm_summary(a_rows),
        "b": _arm_summary(b_rows),
        "paired": {
            "d_ante_mean": _mean(d_ante),
            "d_ante_mean_ci95": [round(ante_lo, 4), round(ante_hi, 4)],
            "d_ante_median": _median(d_ante),
            "d_score_same_ante_loss_mean": _mean(d_score_same_ante_losses),
            "d_loss_frac_mean": _mean(d_loss_frac),
            "better": better,
            "worse": worse,
            "same": sum(1 for d in d_ante if d == 0),
            "win_flips_for_b": flips_for_b,
            "win_flips_for_a": flips_for_a,
            "win_mcnemar_exact_p": round(mcnemar_exact_p(flips_for_b, flips_for_a), 5),
            "d_winrate": round((flips_for_b - flips_for_a) / max(1, len(rows)), 4),
            "d_winrate_ci95": [
                round(x, 4) for x in paired_delta_ci(flips_for_b, flips_for_a, len(rows))
            ],
            "d_ante_sign_p": round(mcnemar_exact_p(better, worse), 5),
        },
        "rows": rows,
    }


def write_metrics(
    path: Path,
    rows: list[dict],
    *,
    bot_a: str,
    bot_b: str,
    faithful: bool,
    started: float,
    expected_n: int,
) -> dict:
    ordered_rows = sorted(rows, key=lambda row: row["seed"])
    result = summarize(
        ordered_rows,
        bot_a=bot_a,
        bot_b=bot_b,
        faithful=faithful,
        wall_s=time.perf_counter() - started,
    )
    result["expected_n"] = expected_n
    result["complete"] = len(ordered_rows) == expected_n
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    tmp_path.replace(path)
    return result


def _run_pairs_process(
    tasks: list[tuple[str, str, str, bool]],
    *,
    metrics_path: Path,
    bot_a: str,
    bot_b: str,
    faithful: bool,
    started: float,
    expected_n: int,
    jobs: int,
) -> list[dict]:
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run_pair, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            write_metrics(
                metrics_path,
                rows,
                bot_a=bot_a,
                bot_b=bot_b,
                faithful=faithful,
                started=started,
                expected_n=expected_n,
            )
    return rows


def _run_pairs_subprocess(
    tasks: list[tuple[str, str, str, bool]],
    *,
    metrics_path: Path,
    bot_a: str,
    bot_b: str,
    faithful: bool,
    started: float,
    expected_n: int,
    jobs: int,
) -> list[dict]:
    rows: list[dict] = []
    pair_dir = metrics_path.with_suffix(metrics_path.suffix + f".pairs.{time.time_ns()}")
    pair_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    pending = list(tasks)
    running: dict[subprocess.Popen, tuple[Path, Path]] = {}

    def stop_running() -> None:
        for other in list(running):
            if other.poll() is None:
                other.terminate()
        for other in list(running):
            if other.poll() is not None:
                continue
            try:
                other.wait(timeout=5)
            except subprocess.TimeoutExpired:
                other.kill()

    def launch(task: tuple) -> None:
        seed, task_bot_a, task_bot_b, task_faithful, ov_a, ov_b, ckpt_a, ckpt_b = task
        row_path = pair_dir / f"{seed}.json"
        err_path = pair_dir / f"{seed}.err.txt"
        command = [
            sys.executable,
            str(script_path),
            "--bot-a",
            task_bot_a,
            "--bot-b",
            task_bot_b,
            "--metrics",
            str(metrics_path),
            "--worker-seed",
            seed,
            "--worker-row",
            str(row_path),
        ]
        if not task_faithful:
            command.append("--generic")
        if ov_a:
            command.extend(["--config-a", json.dumps(ov_a)])
        if ov_b:
            command.extend(["--config-b", json.dumps(ov_b)])
        if ckpt_a:
            command.extend(["--ckpt-a", ckpt_a])
        if ckpt_b:
            command.extend(["--ckpt-b", ckpt_b])
        with (pair_dir / f"{seed}.out.txt").open("w", encoding="utf-8") as stdout:
            with err_path.open("w", encoding="utf-8") as stderr:
                proc = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                )
        running[proc] = (row_path, err_path)

    while pending or running:
        while pending and len(running) < max(1, jobs):
            launch(pending.pop(0))
        time.sleep(0.25)
        for proc in list(running):
            code = proc.poll()
            if code is None:
                continue
            row_path, err_path = running.pop(proc)
            if code != 0:
                stop_running()
                detail = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""
                raise RuntimeError(f"subprocess pair worker failed with exit code {code}: {detail}")
            with row_path.open("r", encoding="utf-8") as fh:
                rows.append(json.load(fh))
            write_metrics(
                metrics_path,
                rows,
                bot_a=bot_a,
                bot_b=bot_b,
                faithful=faithful,
                started=started,
                expected_n=expected_n,
            )
    return rows


def _run_pairs_serial(
    tasks: list[tuple[str, str, str, bool]],
    *,
    metrics_path: Path,
    bot_a: str,
    bot_b: str,
    faithful: bool,
    started: float,
    expected_n: int,
) -> list[dict]:
    rows: list[dict] = []
    for task in tasks:
        rows.append(_run_pair(task))
        write_metrics(
            metrics_path,
            rows,
            bot_a=bot_a,
            bot_b=bot_b,
            faithful=faithful,
            started=started,
            expected_n=expected_n,
        )
    return rows


def _arm_summary(rows: list[dict]) -> dict:
    losses = [r for r in rows if not r["won"]]
    aborts = [r for r in rows if not r["won"] and not r["run_over"]]
    from balatro_ai.bench_stats import wilson_ci

    wins = sum(1 for r in rows if r["won"])
    return {
        "wins": wins,
        "winrate": round(wins / max(1, len(rows)), 4),
        "winrate_ci95": [round(x, 4) for x in wilson_ci(wins, len(rows))],
        "mean_ante": _mean([r["ante"] for r in rows]),
        "median_ante": _median([r["ante"] for r in rows]),
        "ante_hist": dict(sorted(Counter(r["ante"] for r in rows).items())),
        "loss_frac_median": _median([r["loss_frac"] for r in losses if r["loss_frac"] is not None]),
        "loss_frac_mean": _mean([r["loss_frac"] for r in losses if r["loss_frac"] is not None]),
        "mean_cpu_s": _mean([r["cpu_s"] for r in rows]),
        "aborts": len(aborts),
        "abort_reasons": dict(sorted(Counter(r["termination"] for r in aborts).items())),
    }


def _parse_overrides(raw: str | None) -> dict:
    """JSON dict of BotConfig field overrides, or @path to a JSON file."""
    if not raw:
        return {}
    if raw.startswith("@"):
        with open(raw[1:], encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot-a", default="basic_strategy_bot")
    parser.add_argument("--bot-b", default="solver_policy_bot")
    parser.add_argument("--config-a", default=None,
                        help="JSON dict of BotConfig overrides for arm A (or @file.json)")
    parser.add_argument("--config-b", default=None,
                        help="JSON dict of BotConfig overrides for arm B (or @file.json)")
    parser.add_argument("--ckpt-a", default=None, help="BALATRO_POLICY_CKPT for arm A (neural bots)")
    parser.add_argument("--ckpt-b", default=None, help="BALATRO_POLICY_CKPT for arm B (neural bots)")
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--backend", choices=("auto", "process", "subprocess", "serial"), default="auto")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--faithful", action="store_true",
                        help="deprecated no-op: faithful is now the DEFAULT; see --generic")
    parser.add_argument("--generic", action="store_true",
                        help="opt out of faithful real-seed RNG (the pre-2026-07 default)")
    parser.add_argument("--allow-clairvoyant", action="store_true",
                        help="permit BALATRO_NO_FORESIGHT values other than shuffle/hide "
                             "(diagnostics ONLY; the result JSON is stamped)")
    parser.add_argument("--expect", choices=("report", "improves", "non-inferior"), default="report",
                        help="enforcement: improves = McNemar p<0.05 AND d_winrate>0 (adopt gate); "
                             "non-inferior = d_winrate CI lower bound >= -margin; exit 3 on FAIL")
    parser.add_argument("--margin", type=float, default=0.02,
                        help="non-inferiority margin on d_winrate (default 0.02)")
    parser.add_argument("--self-test", action="store_true",
                        help="run arm A vs itself; every pair must be bit-identical (exit 4 if not)")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--worker-seed")
    parser.add_argument("--worker-row")
    args = parser.parse_args()

    # ---- honest-mode guard (THE contamination source of the old record) ---- #
    nf = os.environ.get("BALATRO_NO_FORESIGHT", "").strip()
    if not nf:
        nf = "shuffle"
        os.environ["BALATRO_NO_FORESIGHT"] = nf  # inherited by all worker processes
    if nf not in ("shuffle", "hide") and not args.allow_clairvoyant:
        print(f"[bench] REFUSING to run: BALATRO_NO_FORESIGHT={nf!r} is a CLAIRVOYANT regime. "
              f"Unset it (honest default) or pass --allow-clairvoyant for diagnostics.",
              file=sys.stderr)
        return 2

    faithful = not args.generic
    ov_a = _parse_overrides(args.config_a)
    ov_b = _parse_overrides(args.config_b)
    ckpt_a, ckpt_b = args.ckpt_a, args.ckpt_b
    bot_b = args.bot_b
    if args.self_test:
        bot_b, ov_b, ckpt_b = args.bot_a, dict(ov_a), ckpt_a

    if args.worker_row:
        if not args.worker_seed:
            parser.error("--worker-row requires --worker-seed")
        _write_pair(Path(args.worker_row),
                    (args.worker_seed, args.bot_a, bot_b, faithful, ov_a, ov_b, ckpt_a, ckpt_b))
        return 0

    seeds = [f"{i:07d}" for i in range(args.offset + 1, args.offset + args.seeds + 1)]
    tasks = [(seed, args.bot_a, bot_b, faithful, ov_a, ov_b, ckpt_a, ckpt_b) for seed in seeds]
    started = time.perf_counter()
    metrics_path = Path(args.metrics)
    if args.jobs <= 1 or args.backend == "serial":
        rows = _run_pairs_serial(
            tasks,
            metrics_path=metrics_path,
            bot_a=args.bot_a,
            bot_b=bot_b,
            faithful=faithful,
            started=started,
            expected_n=args.seeds,
        )
    elif args.backend == "subprocess":
        rows = _run_pairs_subprocess(
            tasks,
            metrics_path=metrics_path,
            bot_a=args.bot_a,
            bot_b=bot_b,
            faithful=faithful,
            started=started,
            expected_n=args.seeds,
            jobs=args.jobs,
        )
    else:
        try:
            rows = _run_pairs_process(
                tasks,
                metrics_path=metrics_path,
                bot_a=args.bot_a,
                bot_b=bot_b,
                faithful=faithful,
                started=started,
                expected_n=args.seeds,
                jobs=args.jobs,
            )
        except PermissionError:
            if args.backend == "process":
                raise
            rows = _run_pairs_subprocess(
                tasks,
                metrics_path=metrics_path,
                bot_a=args.bot_a,
                bot_b=bot_b,
                faithful=faithful,
                started=started,
                expected_n=args.seeds,
                jobs=args.jobs,
            )
    result = write_metrics(
        metrics_path,
        rows,
        bot_a=args.bot_a,
        bot_b=bot_b,
        faithful=faithful,
        started=started,
        expected_n=args.seeds,
    )

    # ---- reproducibility metadata into the result JSON --------------------- #
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        head = ""
    result["meta"] = {
        "git_head": head,
        "no_foresight": os.environ.get("BALATRO_NO_FORESIGHT", ""),
        "faithful": faithful,
        "config_a": ov_a, "config_b": ov_b,
        "ckpt_a": ckpt_a, "ckpt_b": ckpt_b,
        "offset": args.offset, "expect": args.expect, "self_test": args.self_test,
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    mode = f"honest({os.environ['BALATRO_NO_FORESIGHT']})" \
        if os.environ["BALATRO_NO_FORESIGHT"] in ("shuffle", "hide") else "CLAIRVOYANT"
    print(f"=== paired bot A/B: {args.bot_a} -> {bot_b}, {args.seeds} seeds "
          f"[{mode}, {'faithful' if faithful else 'generic'}, HEAD {head}] ===")
    print(f"  {args.bot_a}: wins {result['a']['wins']}/{args.seeds} "
          f"ante {result['a']['mean_ante']:.2f} cpu/run {result['a']['mean_cpu_s']:.1f}s")
    print(f"  {bot_b}: wins {result['b']['wins']}/{args.seeds} "
          f"ante {result['b']['mean_ante']:.2f} cpu/run {result['b']['mean_cpu_s']:.1f}s")
    p = result["paired"]
    alo, ahi = p["d_ante_mean_ci95"]
    print(f"  paired d_ante mean={p['d_ante_mean']:+.3f} (95% CI {alo:+.3f}..{ahi:+.3f}) "
          f"median={p['d_ante_median']:+.1f} "
          f"(better {p['better']} / worse {p['worse']} / same {p['same']})")
    if p["d_loss_frac_mean"] is not None:
        print(f"  paired d_loss_frac mean={p['d_loss_frac_mean']:+.3f} (non-win pairs; higher is closer to clear)")
    print(f"  win flips for {bot_b}: {p['win_flips_for_b']} | for {args.bot_a}: {p['win_flips_for_a']} "
          f"| McNemar exact p={p['win_mcnemar_exact_p']:.4f}")
    lo, hi = p["d_winrate_ci95"]
    print(f"  d_winrate {p['d_winrate']:+.4f} (95% CI {lo:+.4f}..{hi:+.4f}) | d_ante sign-test p={p['d_ante_sign_p']:.4f}")

    # ---- self-test: A vs A must be bit-identical on every pair ------------- #
    if args.self_test:
        divergent = [r["seed"] for r in rows
                     if r["a"]["won"] != r["b"]["won"] or r["a"]["ante"] != r["b"]["ante"]
                     or r["a"]["score"] != r["b"]["score"] or r["a"]["steps"] != r["b"]["steps"]]
        if divergent:
            print(f"  SELF-TEST FAIL: {len(divergent)}/{len(rows)} pairs diverged "
                  f"(nondeterminism!): seeds {divergent[:10]}", file=sys.stderr)
            return 4
        print(f"  SELF-TEST PASS: {len(rows)}/{len(rows)} pairs bit-identical — "
              f"harness determinism CERTIFIED")
        return 0

    # ---- enforcement: the verdict IS the exit code -------------------------- #
    if args.expect == "improves":
        ok = p["win_mcnemar_exact_p"] < 0.05 and p["d_winrate"] > 0
        print(f"  VERDICT (expect improves): {'PASS — ADOPT' if ok else 'FAIL — DO NOT ADOPT'} "
              f"(need McNemar p<0.05 AND d_winrate>0)")
        return 0 if ok else 3
    if args.expect == "non-inferior":
        ok = lo >= -args.margin
        print(f"  VERDICT (expect non-inferior, margin {args.margin}): "
              f"{'PASS' if ok else 'FAIL'} (d_winrate CI lower bound {lo:+.4f} vs -{args.margin})")
        return 0 if ok else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
