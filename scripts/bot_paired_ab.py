"""Paired same-seed A/B for two registered bots.

Runs each seed once with bot A and once with bot B inside the same worker. This
cancels seed variance and gives a cleaner signal than two independent winrate
benchmarks. Shop audit payloads are disabled by default because this script only
uses final outcomes.

    PYTHONPATH=src python scripts/bot_paired_ab.py \
        --bot-a basic_strategy_bot --bot-b solver_policy_bot \
        --seeds 48 --jobs 8 --metrics .data/bot_paired_basic_solver.json
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


def _run_one(bot_name: str, seed: str, faithful: bool) -> dict:
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase, with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    kwargs = {"seed": _stable_seed_int(seed), "stake": "white"}
    if faithful:
        kwargs["balatro_seed"] = seed
    old_run_seed = os.environ.get("BALATRO_RUN_SEED")
    os.environ["BALATRO_RUN_SEED"] = seed
    try:
        sim = LocalBalatroSimulator(**kwargs)
        sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
        bot = create_bot(bot_name, seed=0)
        termination = "max_steps"
        started = time.process_time()
        with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
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


def _run_pair(args: tuple[str, str, str, bool]) -> dict:
    seed, bot_a, bot_b, faithful = args
    a = _run_one(bot_a, seed, faithful)
    b = _run_one(bot_b, seed, faithful)
    return {
        "seed": seed,
        "a": a,
        "b": b,
        "d_ante": b["ante"] - a["ante"],
        "d_score": b["score"] - a["score"],
        "d_win": int(b["won"]) - int(a["won"]),
    }


def _write_pair(path: Path, task: tuple[str, str, str, bool]) -> None:
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
    return {
        "bot_a": bot_a,
        "bot_b": bot_b,
        "faithful": faithful,
        "n": len(rows),
        "wall_s": round(wall_s, 2),
        "a": _arm_summary(a_rows),
        "b": _arm_summary(b_rows),
        "paired": {
            "d_ante_mean": _mean(d_ante),
            "d_ante_median": _median(d_ante),
            "d_score_same_ante_loss_mean": _mean(d_score_same_ante_losses),
            "d_loss_frac_mean": _mean(d_loss_frac),
            "better": sum(1 for d in d_ante if d > 0),
            "worse": sum(1 for d in d_ante if d < 0),
            "same": sum(1 for d in d_ante if d == 0),
            "win_flips_for_b": sum(1 for r in rows if not r["a"]["won"] and r["b"]["won"]),
            "win_flips_for_a": sum(1 for r in rows if r["a"]["won"] and not r["b"]["won"]),
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

    def launch(task: tuple[str, str, str, bool]) -> None:
        seed, task_bot_a, task_bot_b, task_faithful = task
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
        if task_faithful:
            command.append("--faithful")
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
    return {
        "wins": sum(1 for r in rows if r["won"]),
        "winrate": round(sum(1 for r in rows if r["won"]) / max(1, len(rows)), 4),
        "mean_ante": _mean([r["ante"] for r in rows]),
        "median_ante": _median([r["ante"] for r in rows]),
        "ante_hist": dict(sorted(Counter(r["ante"] for r in rows).items())),
        "loss_frac_median": _median([r["loss_frac"] for r in losses if r["loss_frac"] is not None]),
        "loss_frac_mean": _mean([r["loss_frac"] for r in losses if r["loss_frac"] is not None]),
        "mean_cpu_s": _mean([r["cpu_s"] for r in rows]),
        "aborts": len(aborts),
        "abort_reasons": dict(sorted(Counter(r["termination"] for r in aborts).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot-a", default="basic_strategy_bot")
    parser.add_argument("--bot-b", default="solver_policy_bot")
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--backend", choices=("auto", "process", "subprocess", "serial"), default="auto")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--faithful", action="store_true")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--worker-seed")
    parser.add_argument("--worker-row")
    args = parser.parse_args()

    if args.worker_row:
        if not args.worker_seed:
            parser.error("--worker-row requires --worker-seed")
        _write_pair(Path(args.worker_row), (args.worker_seed, args.bot_a, args.bot_b, args.faithful))
        return 0

    seeds = [f"{i:07d}" for i in range(args.offset + 1, args.offset + args.seeds + 1)]
    tasks = [(seed, args.bot_a, args.bot_b, args.faithful) for seed in seeds]
    started = time.perf_counter()
    metrics_path = Path(args.metrics)
    if args.jobs <= 1 or args.backend == "serial":
        rows = _run_pairs_serial(
            tasks,
            metrics_path=metrics_path,
            bot_a=args.bot_a,
            bot_b=args.bot_b,
            faithful=args.faithful,
            started=started,
            expected_n=args.seeds,
        )
    elif args.backend == "subprocess":
        rows = _run_pairs_subprocess(
            tasks,
            metrics_path=metrics_path,
            bot_a=args.bot_a,
            bot_b=args.bot_b,
            faithful=args.faithful,
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
                bot_b=args.bot_b,
                faithful=args.faithful,
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
                bot_b=args.bot_b,
                faithful=args.faithful,
                started=started,
                expected_n=args.seeds,
                jobs=args.jobs,
            )
    result = write_metrics(
        metrics_path,
        rows,
        bot_a=args.bot_a,
        bot_b=args.bot_b,
        faithful=args.faithful,
        started=started,
        expected_n=args.seeds,
    )

    print(f"=== paired bot A/B: {args.bot_a} -> {args.bot_b}, {args.seeds} seeds ===")
    print(f"  {args.bot_a}: wins {result['a']['wins']}/{args.seeds} "
          f"ante {result['a']['mean_ante']:.2f} cpu/run {result['a']['mean_cpu_s']:.1f}s")
    print(f"  {args.bot_b}: wins {result['b']['wins']}/{args.seeds} "
          f"ante {result['b']['mean_ante']:.2f} cpu/run {result['b']['mean_cpu_s']:.1f}s")
    p = result["paired"]
    print(f"  paired d_ante mean={p['d_ante_mean']:+.3f} median={p['d_ante_median']:+.1f} "
          f"(better {p['better']} / worse {p['worse']} / same {p['same']})")
    if p["d_loss_frac_mean"] is not None:
        print(f"  paired d_loss_frac mean={p['d_loss_frac_mean']:+.3f} (non-win pairs; higher is closer to clear)")
    print(f"  win flips for {args.bot_b}: {p['win_flips_for_b']} | for {args.bot_a}: {p['win_flips_for_a']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
