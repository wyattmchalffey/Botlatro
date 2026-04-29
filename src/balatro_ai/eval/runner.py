"""Reusable benchmark runner with optional endpoint parallelism."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Queue
from threading import Lock
from time import perf_counter
from typing import Callable

from balatro_ai.api.client import JsonRpcBalatroClient
from balatro_ai.bots.registry import create_bot
from balatro_ai.data.replay_logger import ReplayLogger
from balatro_ai.eval.metrics import BenchmarkSummary, RunResult, summarize_runs
from balatro_ai.eval.run_seed import REPLAY_MODES, RunSeedOptions, run_single_seed
from balatro_ai.eval.seed_sets import SeedSet, make_explicit_seed_set, make_seed_set

ProgressCallback = Callable[[str], None]
StopCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    bot: str
    stake: str = "white"
    deck: str = "RED"
    profile_name: str = "P1"
    unlock_state: str = "all"
    seeds: int = 100
    seed_values: tuple[int, ...] | None = None
    label: str = "default"
    endpoints: tuple[str, ...] = ("http://127.0.0.1:12346",)
    timeout_seconds: float = 10.0
    max_steps: int = 1000
    replay_dir: Path | None = None
    replay_mode: str = "score_audit"
    start_retries: int = 1


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    seed_set_label: str
    results: tuple[RunResult, ...]
    summary: BenchmarkSummary


def endpoint_urls(host: str, base_port: int, workers: int) -> tuple[str, ...]:
    if workers < 1:
        raise ValueError("Worker count must be at least 1")
    return tuple(f"http://{host}:{base_port + index}" for index in range(workers))


def run_benchmark(
    options: BenchmarkOptions,
    progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
) -> BenchmarkRun:
    if not options.endpoints:
        raise ValueError("At least one endpoint is required")

    seed_set = _seed_set_from_options(options)
    _emit(progress, f"Bot: {options.bot}")
    _emit(progress, f"Stake: {options.stake}")
    _emit(progress, f"Deck: {options.deck.upper()}")
    _emit(progress, f"Profile: {options.profile_name}")
    _emit(progress, f"Unlocks: {options.unlock_state}")
    _emit(progress, f"Seeds: {len(seed_set.seeds)}")
    _emit(progress, f"Endpoints: {', '.join(options.endpoints)}")
    _emit(progress, f"Replay mode: {options.replay_mode}")
    _emit(progress, f"Start retries: {options.start_retries}")
    if options.replay_mode not in REPLAY_MODES:
        raise ValueError(f"Unknown replay mode: {options.replay_mode}")

    results: list[RunResult] = []
    completed = 0
    completed_lock = Lock()

    if len(options.endpoints) == 1:
        endpoint = options.endpoints[0]
        for seed in seed_set.seeds:
            if _stop_requested(should_stop):
                break
            result = _run_seed(seed=seed, endpoint=endpoint, options=options)
            results.append(result)
            completed += 1
            _emit_result(progress, completed, len(seed_set.seeds), result)
    else:
        endpoint_queue: Queue[str] = Queue()
        for endpoint in options.endpoints:
            endpoint_queue.put(endpoint)

        def worker(seed: int) -> RunResult:
            endpoint = endpoint_queue.get()
            try:
                return _run_seed(seed=seed, endpoint=endpoint, options=options)
            finally:
                endpoint_queue.put(endpoint)

        with ThreadPoolExecutor(max_workers=len(options.endpoints)) as executor:
            seed_iter = iter(seed_set.seeds)
            pending: set[Future[RunResult]] = set()

            def submit_next() -> bool:
                if _stop_requested(should_stop):
                    return False
                try:
                    seed = next(seed_iter)
                except StopIteration:
                    return False
                pending.add(executor.submit(worker, seed))
                return True

            for _ in range(len(options.endpoints)):
                if not submit_next():
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    results.append(result)
                    with completed_lock:
                        completed += 1
                        count = completed
                    _emit_result(progress, count, len(seed_set.seeds), result)

                while len(pending) < len(options.endpoints):
                    if not submit_next():
                        break

    seed_order = {seed: index for index, seed in enumerate(seed_set.seeds)}
    sorted_results = tuple(sorted(results, key=lambda result: seed_order[result.seed]))
    if _stop_requested(should_stop):
        _emit(progress, f"Stopped after {len(sorted_results)}/{len(seed_set.seeds)} seeds.")

    summary = (
        _summary_with_options_metadata(summarize_runs(sorted_results), options)
        if sorted_results
        else _empty_summary(options)
    )
    _emit(progress, "")
    _emit(progress, summary.to_text())
    return BenchmarkRun(seed_set_label=seed_set.label, results=sorted_results, summary=summary)


def _run_seed(*, seed: int, endpoint: str, options: BenchmarkOptions) -> RunResult:
    started_at = perf_counter()
    replay_path = None
    if options.replay_dir is not None:
        replay_path = options.replay_dir / f"{options.bot}_{options.stake}_{seed}.jsonl"

    try:
        bot = create_bot(options.bot, seed=seed)
        client = JsonRpcBalatroClient(
            endpoint=endpoint,
            deck=options.deck.upper(),
            timeout_seconds=options.timeout_seconds,
        )
        return run_single_seed(
            bot=bot,
            client=client,
            options=RunSeedOptions(
                seed=seed,
                stake=options.stake,
                max_steps=options.max_steps,
                replay_path=replay_path,
                replay_mode=options.replay_mode,
                start_retries=options.start_retries,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - benchmarks should report failed seeds and keep moving.
        result = RunResult(
            bot_version=options.bot,
            seed=seed,
            stake=options.stake,
            won=False,
            ante_reached=0,
            final_score=0,
            final_money=0,
            runtime_seconds=perf_counter() - started_at,
            death_reason=f"error:{type(exc).__name__}: {exc}",
        )
        if replay_path is not None and options.replay_mode != "off":
            ReplayLogger(replay_path).log_summary(
                bot_version=result.bot_version,
                seed=result.seed,
                stake=result.stake,
                won=result.won,
                ante_reached=result.ante_reached,
                final_score=result.final_score,
                final_money=result.final_money,
                runtime_seconds=result.runtime_seconds,
                death_reason=result.death_reason,
            )
        return result


def _seed_set_from_options(options: BenchmarkOptions) -> SeedSet:
    if options.seed_values is not None:
        return make_explicit_seed_set(label=f"{options.stake}:{options.label}:explicit", seeds=options.seed_values)
    return make_seed_set(label=f"{options.stake}:{options.label}", size=options.seeds)


def _empty_summary(options: BenchmarkOptions) -> BenchmarkSummary:
    return BenchmarkSummary(
        bot_version=options.bot,
        stake=options.stake,
        deck=options.deck.upper(),
        profile_name=options.profile_name,
        unlock_state=options.unlock_state,
        run_count=0,
        win_rate=0.0,
        average_ante=0.0,
        average_final_score=0.0,
        average_final_money=0.0,
        average_runtime_seconds=0.0,
    )


def _summary_with_options_metadata(summary: BenchmarkSummary, options: BenchmarkOptions) -> BenchmarkSummary:
    return replace(
        summary,
        deck=options.deck.upper(),
        profile_name=options.profile_name,
        unlock_state=options.unlock_state,
    )


def _stop_requested(should_stop: StopCallback | None) -> bool:
    return bool(should_stop and should_stop())


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _emit_result(
    progress: ProgressCallback | None,
    count: int,
    total: int,
    result: RunResult,
) -> None:
    message = (
        f"[{count}/{total}] seed={result.seed} won={result.won} "
        f"ante={result.ante_reached} score={result.final_score} "
        f"money={result.final_money} runtime={result.runtime_seconds:.2f}s"
    )
    if result.death_reason and result.death_reason.startswith("error:"):
        message += f" death={result.death_reason[:140]}"
    _emit(progress, message)
