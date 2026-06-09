"""Run the local full-sim verification gate before neural data generation.

This gate composes the no-live-bridge checks that protect neural labels:
forward-sim unit coverage, replay/score validators, seed-game/RNG tests, and
captured RNG fixture validators. It writes a JSON report so data-generation runs
can record exactly which sim gate they trusted.

    python scripts/full_sim_verification_gate.py --jobs 4 --metrics .data/full_sim_gate.json

Use ``--quick`` while iterating; use the default/full task set before producing
training labels that should be trusted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time


_TEST_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "forward_sim_tests",
        (
            "tests/test_forward_sim.py",
            "tests/test_state_value.py",
            "tests/test_solver_seed_game.py",
        ),
    ),
    (
        "replay_and_score_tests",
        (
            "tests/test_local_sim_replay_validator.py",
            "tests/test_replay_diff.py",
            "tests/test_replay_analyzer.py",
            "tests/test_score_audit.py",
            "tests/test_score_dataset.py",
            "tests/test_balatrobench_score_audit.py",
            "tests/test_sim_divergence_audit.py",
        ),
    ),
    (
        "rng_tests",
        (
            "tests/test_rng_deck_and_validate.py",
            "tests/test_rng_pack_surfaces.py",
            "tests/test_rng_surfaces.py",
        ),
    ),
)

_VALIDATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rng_surfaces_all", ("-m", "balatro_ai.rng.validate_surfaces", "--all")),
    ("rng_shop_sequence_all", ("-m", "balatro_ai.rng.validate_shop_sequence", "--all")),
    ("rng_spectral_helpers_all", ("-m", "balatro_ai.rng.validate_spectral_helpers", "--all")),
    (
        "score_edge_fixtures",
        ("-m", "balatro_ai.eval.score_dataset", "--fixtures", "tests/fixtures/score_edges"),
    ),
)


@dataclass(frozen=True, slots=True)
class GateTask:
    name: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateTaskResult:
    name: str
    command: tuple[str, ...]
    exit_code: int
    wall_s: float
    stdout_tail: str
    stderr_tail: str

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


def _tail(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run_task(task: GateTask, cwd: Path) -> GateTaskResult:
    started = time.perf_counter()
    completed = subprocess.run(
        task.command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return GateTaskResult(
        name=task.name,
        command=task.command,
        exit_code=completed.returncode,
        wall_s=round(time.perf_counter() - started, 3),
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def _tasks(*, quick: bool) -> list[GateTask]:
    tasks: list[GateTask] = []
    for name, paths in _TEST_GROUPS:
        if quick and name == "replay_and_score_tests":
            paths = ("tests/test_local_sim_replay_validator.py", "tests/test_replay_diff.py")
        tasks.append(GateTask(name=name, command=(sys.executable, "-m", "pytest", "-q", *paths)))
    validators = _VALIDATORS[:2] if quick else _VALIDATORS
    for name, args in validators:
        tasks.append(GateTask(name=name, command=(sys.executable, *args)))
    return tasks


def run_gate(*, cwd: Path, jobs: int, quick: bool) -> dict:
    tasks = _tasks(quick=quick)
    started = time.perf_counter()
    results: list[GateTaskResult] = []
    max_workers = max(1, min(int(jobs), len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_task, task, cwd): task.name for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    ordered = sorted(results, key=lambda item: item.name)
    return {
        "passed": all(item.passed for item in ordered),
        "quick": quick,
        "jobs": max_workers,
        "wall_s": round(time.perf_counter() - started, 3),
        "tasks": [asdict(item) | {"passed": item.passed} for item in ordered],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Botlatro full-sim verification gate.")
    parser.add_argument("--jobs", type=int, default=4, help="Independent verifier tasks to run concurrently.")
    parser.add_argument("--quick", action="store_true", help="Run the smaller iteration gate.")
    parser.add_argument("--metrics", type=Path, required=True, help="JSON report path.")
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parents[1]
    report = run_gate(cwd=cwd, jobs=args.jobs, quick=args.quick)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.metrics.with_suffix(args.metrics.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(args.metrics)
    print(json.dumps({k: report[k] for k in ("passed", "quick", "jobs", "wall_s")}, indent=2), flush=True)
    for task in report["tasks"]:
        status = "PASS" if task["passed"] else "FAIL"
        print(f"[{status}] {task['name']} ({task['wall_s']}s)", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
