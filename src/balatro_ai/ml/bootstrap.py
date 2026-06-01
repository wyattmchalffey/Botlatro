"""Bootstrap dataset generation (Stage 1.1).

Runs a teacher bot (default `basic_strategy_bot` — the stronger, faster of the
two bootstraps) across many seeds in parallel and persists each run as a thin,
replay-complete `RunCapture` (one JSONL line). Training loads these back and
re-expands them into encoded `(state, action, value)` examples via
`dataset.examples_from_capture` — so storage stays small (action logs, not
states) while every per-step state is faithfully reconstructable.

Resumable: re-running with the same `--out` skips seeds already in the file.

CLI::

    python -m balatro_ai.ml.bootstrap --bot basic_strategy_bot \\
        --seeds 64 --workers 14 --out .data/phase8-bootstrap-basic-64.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from balatro_ai.ml.dataset import RunCapture, TrainingExample, examples_from_capture


def _capture_seed_json(args: tuple[str, str, str, int]) -> dict:
    """Worker: construct the bot in-process and capture one run."""
    seed, bot_name, stake, max_steps = args
    # Local imports keep the worker self-contained under spawn.
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.ml.dataset import capture_run
    from balatro_ai.solver.trajectory import _stable_seed_int

    bot = create_bot(bot_name, _stable_seed_int(seed))
    capture = capture_run(seed, bot.choose_action, stake=stake, max_steps=max_steps)
    return capture.to_json_dict()


@dataclass(frozen=True)
class BootstrapStats:
    bot: str
    n_runs: int
    wins: int
    win_rate: float
    mean_final_ante: float
    by_reason: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def read_captures(path: str | Path) -> Iterator[RunCapture]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield RunCapture.from_json_dict(json.loads(line))


def summarize(captures: Sequence[RunCapture], *, bot: str = "") -> BootstrapStats:
    n = len(captures)
    wins = sum(1 for c in captures if c.won)
    by_reason: dict[str, int] = {}
    for c in captures:
        by_reason[c.terminated_reason] = by_reason.get(c.terminated_reason, 0) + 1
    mean_ante = sum(c.final_ante for c in captures) / n if n else 0.0
    return BootstrapStats(
        bot=bot,
        n_runs=n,
        wins=wins,
        win_rate=wins / n if n else 0.0,
        mean_final_ante=mean_ante,
        by_reason=by_reason,
    )


def generate_captures(
    seeds: Sequence[str],
    out_path: str | Path,
    *,
    bot_name: str = "basic_strategy_bot",
    stake: str = "white",
    max_steps: int = 2000,
    workers: int | None = None,
    resume: bool = True,
) -> BootstrapStats:
    """Generate captures for `seeds` in parallel, appending to `out_path` JSONL.

    Returns stats over *all* captures in the file (including any from a prior
    resumed run). A seed whose worker raises is skipped (and logged to stderr),
    not written, so a single bad seed can't poison the shard.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = {c.seed for c in read_captures(out)} if resume else set()
    pending = [s for s in seeds if s not in done]
    workers = workers or (os.cpu_count() or 1)

    errors = 0
    if pending:
        with out.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(
            max_workers=workers
        ) as pool:
            futures = {
                pool.submit(_capture_seed_json, (s, bot_name, stake, max_steps)): s
                for s in pending
            }
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    payload = fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    print(f"[bootstrap] seed {seed} failed: {type(exc).__name__}: {exc}")
                    continue
                fh.write(json.dumps(payload) + "\n")
                fh.flush()

    stats = summarize(list(read_captures(out)), bot=bot_name)
    if errors:
        print(f"[bootstrap] {errors} seed(s) failed and were skipped")
    return stats


def load_examples(
    path: str | Path,
    *,
    limit_runs: int | None = None,
) -> list[TrainingExample]:
    """Re-expand persisted captures into encoded training examples."""
    examples: list[TrainingExample] = []
    for i, capture in enumerate(read_captures(path)):
        if limit_runs is not None and i >= limit_runs:
            break
        examples.extend(examples_from_capture(capture))
    return examples


def _numeric_seeds(n: int, *, start: int = 1) -> list[str]:
    """The project's canonical winnable seed family (`0000001`...)."""
    return [f"{i:07d}" for i in range(start, start + n)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m balatro_ai.ml.bootstrap")
    p.add_argument("--bot", default="basic_strategy_bot")
    p.add_argument("--seeds", type=int, default=64, help="Count of numeric seeds (0000001..).")
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--stake", default="white")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    seeds = _numeric_seeds(args.seeds, start=args.seed_start)
    stats = generate_captures(
        seeds, args.out, bot_name=args.bot, stake=args.stake,
        max_steps=args.max_steps, workers=args.workers,
    )
    print(json.dumps(stats.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
