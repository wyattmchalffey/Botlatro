"""Capture neural shop-ranker overrides on baseline trajectories.

The existing shop labels are useful but still off-policy: they label generic
candidate sets, while the live bot only fails on the small set of actions the
ranker would actually override. This script follows a baseline bot, asks the
ranker what it would do at each gated shop state, and writes only disagreements.
The resulting JSONL can be passed directly to ``phase8_sequential_baseline_probe``
with ``--focus-deepening-candidate`` for solver-confirmed labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterator


def _configure_rust_bestplay(enabled: bool) -> None:
    os.environ["BALATRO_RUST_BESTPLAY"] = "1" if enabled else "0"
    module = sys.modules.get("balatro_ai.rules.hand_evaluator")
    if module is not None:
        setattr(module, "_RUST_BESTPLAY_ENABLED", enabled)


def _capture_worker(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        seeds,
        baseline_bot,
        stake,
        ranker_ckpt,
        max_steps,
        min_capture_ante,
        max_capture_ante,
        per_seed,
        max_actions,
        candidate_action_types,
        candidate_action_kinds,
        min_margin,
        min_baseline_margin,
        max_ranker_actions_per_shop,
        max_ranker_actions_per_run,
    ) = args
    records: list[dict[str, Any]] = []
    for seed in seeds:
        records.extend(
            _capture_seed_overrides(
                seed,
                baseline_bot=baseline_bot,
                stake=stake,
                ranker_ckpt=ranker_ckpt,
                max_steps=max_steps,
                min_capture_ante=min_capture_ante,
                max_capture_ante=max_capture_ante,
                per_seed=per_seed,
                max_actions=max_actions,
                candidate_action_types=candidate_action_types,
                candidate_action_kinds=candidate_action_kinds,
                min_margin=min_margin,
                min_baseline_margin=min_baseline_margin,
                max_ranker_actions_per_shop=max_ranker_actions_per_shop,
                max_ranker_actions_per_run=max_ranker_actions_per_run,
            )
        )
    return records


def _capture_seed_overrides(
    seed: str,
    *,
    baseline_bot: str,
    stake: str,
    ranker_ckpt: Path,
    max_steps: int,
    min_capture_ante: int,
    max_capture_ante: int,
    per_seed: int,
    max_actions: int,
    candidate_action_types: tuple[str, ...],
    candidate_action_kinds: tuple[str, ...],
    min_margin: float,
    min_baseline_margin: float,
    max_ranker_actions_per_shop: int,
    max_ranker_actions_per_run: int,
) -> list[dict[str, Any]]:
    from dataclasses import replace

    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.neural_shop import _ranker_shop_action
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.ml.shop_candidate_dataset import action_key
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    records: list[dict[str, Any]] = []
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake=stake)
    sim.state = SeedGame(seed, stake=stake).initial_state()
    bot = create_bot(baseline_bot, seed=0)
    ranker_actions_in_shop = 0
    ranker_actions_in_run = 0
    was_in_shop = False

    with (
        bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)),
        _ranker_env(
            ranker_ckpt=ranker_ckpt,
            max_actions=max_actions,
            candidate_action_types=candidate_action_types,
            candidate_action_kinds=candidate_action_kinds,
            min_margin=min_margin,
            min_baseline_margin=min_baseline_margin,
        ),
    ):
        for step in range(max(0, int(max_steps))):
            state = sim.state
            if state.run_over or state.phase == GamePhase.RUN_OVER:
                break

            if state.phase != GamePhase.SHOP:
                ranker_actions_in_shop = 0
                was_in_shop = False
                action = bot.choose_action(state)
                if action is None or action.action_type == ActionType.NO_OP:
                    break
                try:
                    sim.step(action)
                except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                    break
                continue

            if not was_in_shop:
                ranker_actions_in_shop = 0
                was_in_shop = True

            baseline_action = bot.choose_action(state)
            if baseline_action is None or baseline_action.action_type == ActionType.NO_OP:
                break

            ante = int(state.ante)
            if (
                ante >= min_capture_ante
                and ante <= max_capture_ante
                and len(records) < per_seed
                and ranker_actions_in_shop < max_ranker_actions_per_shop
                and ranker_actions_in_run < max_ranker_actions_per_run
            ):
                ranker_action = _ranker_shop_action(
                    state,
                    ckpt=ranker_ckpt,
                    baseline_action_getter=lambda: baseline_action,
                )
                if ranker_action is not None and action_key(ranker_action) != action_key(baseline_action):
                    records.append(
                        _proposal_record(
                            seed=seed,
                            state_index=step,
                            source_bot=baseline_bot,
                            state=state,
                            baseline_action=baseline_action,
                            ranker_action=ranker_action,
                            ranker_ckpt=ranker_ckpt,
                            max_actions=max_actions,
                            candidate_action_types=candidate_action_types,
                            candidate_action_kinds=candidate_action_kinds,
                            min_margin=min_margin,
                            min_baseline_margin=min_baseline_margin,
                        )
                    )
                    ranker_actions_in_shop += 1
                    ranker_actions_in_run += 1

            try:
                sim.step(baseline_action)
            except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                break
            if len(records) >= per_seed:
                break
    return records


def _proposal_record(
    *,
    seed: str,
    state_index: int,
    source_bot: str,
    state,
    baseline_action,
    ranker_action,
    ranker_ckpt: Path,
    max_actions: int,
    candidate_action_types: tuple[str, ...],
    candidate_action_kinds: tuple[str, ...],
    min_margin: float,
    min_baseline_margin: float,
) -> dict[str, Any]:
    from balatro_ai.ml.encoding import ENCODING_VERSION, encode_state
    from balatro_ai.ml.shop_candidate_dataset import action_key, state_snapshot

    payload = _ranker_payload(ranker_action)
    return {
        "seed": str(seed),
        "state_index": int(state_index),
        "source_bot": str(source_bot),
        "proposal_source": "shop_ranker_override_capture",
        "phase": state.phase.value,
        "ante": int(state.ante),
        "money": int(state.money),
        "encoding_version": ENCODING_VERSION,
        "encoded_state": asdict(encode_state(state)),
        "state_snapshot": state_snapshot(state),
        "heuristic_action_key": action_key(baseline_action),
        "deepening_candidate_action_key": action_key(ranker_action),
        "deepening_candidate_action": ranker_action.to_json(),
        "deepening_candidate_action_type": ranker_action.action_type.value,
        "deepening_candidate_action_kind": _action_kind(ranker_action),
        "deepening_heuristic_action_key": action_key(baseline_action),
        "deepening_heuristic_action": baseline_action.to_json(),
        "deepening_heuristic_action_type": baseline_action.action_type.value,
        "deepening_heuristic_action_kind": _action_kind(baseline_action),
        "ranker_ckpt": ranker_ckpt.name,
        "ranker_score": payload.get("score"),
        "ranker_margin": payload.get("margin"),
        "ranker_baseline_score": payload.get("baseline_score"),
        "ranker_baseline_margin": payload.get("baseline_margin"),
        "ranker_baseline_key": payload.get("baseline_key"),
        "ranker_max_actions": int(max_actions),
        "ranker_candidate_action_types": list(candidate_action_types),
        "ranker_candidate_action_kinds": list(candidate_action_kinds),
        "ranker_min_margin": float(min_margin),
        "ranker_min_baseline_margin": float(min_baseline_margin),
    }


def _ranker_payload(action) -> dict[str, Any]:
    metadata = action.metadata if isinstance(action.metadata, dict) else {}
    payload = metadata.get("shop_ranker", {})
    return dict(payload) if isinstance(payload, dict) else {}


def _action_kind(action) -> str:
    metadata = action.metadata if isinstance(action.metadata, dict) else {}
    return str(metadata.get("kind", "") or "").strip().lower()


@contextmanager
def _ranker_env(
    *,
    ranker_ckpt: Path,
    max_actions: int,
    candidate_action_types: tuple[str, ...],
    candidate_action_kinds: tuple[str, ...],
    min_margin: float,
    min_baseline_margin: float,
) -> Iterator[None]:
    updates = {
        "BALATRO_SHOP_RANKER_CKPT": str(ranker_ckpt),
        "BALATRO_SHOP_RANKER_MAX_ACTIONS": str(max_actions),
        "BALATRO_SHOP_RANKER_ACTION_TYPES": ",".join(candidate_action_types),
        "BALATRO_SHOP_RANKER_ACTION_KINDS": ",".join(candidate_action_kinds),
        "BALATRO_SHOP_RANKER_MIN_MARGIN": str(float(min_margin)),
        "BALATRO_SHOP_RANKER_MIN_BASELINE_MARGIN": str(float(min_baseline_margin)),
    }
    old = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _seed_strings(offset: int, count: int) -> list[str]:
    return [f"{offset + i:07d}" for i in range(1, max(0, int(count)) + 1)]


def _chunks(items: list[str], n_chunks: int) -> list[list[str]]:
    n_chunks = max(1, min(int(n_chunks), len(items) or 1))
    size = max(1, (len(items) + n_chunks - 1) // n_chunks)
    return [items[start : start + size] for start in range(0, len(items), size)]


def _parse_csv(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in str(value or "").split(",") if item.strip())


def _validate_action_types(values: tuple[str, ...]) -> tuple[str, ...]:
    from balatro_ai.api.actions import ActionType

    valid_by_value = {action_type.value for action_type in ActionType}
    valid_by_name = {action_type.name.lower(): action_type.value for action_type in ActionType}
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).strip().lower()
        normalized = key if key in valid_by_value else valid_by_name.get(key)
        if normalized is None:
            allowed = ", ".join(sorted(valid_by_value))
            raise ValueError(f"unknown action type {value!r}; expected one of: {allowed}")
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return tuple(out)


def _summarize(records: list[dict[str, Any]], *, args: argparse.Namespace, wall_s: float) -> dict[str, Any]:
    margins = [float(record["ranker_margin"]) for record in records if record.get("ranker_margin") is not None]
    baseline_margins = [
        float(record["ranker_baseline_margin"])
        for record in records
        if record.get("ranker_baseline_margin") is not None
    ]
    return {
        "records": len(records),
        "wall_s": round(wall_s, 3),
        "wall_s_per_record": round(wall_s / len(records), 4) if records else None,
        "jobs": int(args.jobs),
        "baseline_bot": args.baseline_bot,
        "stake": args.stake,
        "seed_offset": int(args.seed_offset),
        "seed_count": int(args.seed_count),
        "states": int(args.states),
        "per_seed": int(args.per_seed),
        "ranker_ckpt": str(args.ranker_ckpt),
        "min_capture_ante": int(args.min_capture_ante),
        "max_capture_ante": int(args.max_capture_ante),
        "max_steps": int(args.max_steps),
        "ranker_max_actions": int(args.max_actions),
        "ranker_candidate_action_types": list(args.candidate_action_types),
        "ranker_candidate_action_kinds": list(args.candidate_action_kinds),
        "ranker_min_margin": float(args.min_margin),
        "ranker_min_baseline_margin": float(args.min_baseline_margin),
        "max_ranker_actions_per_shop": int(args.max_ranker_actions_per_shop),
        "max_ranker_actions_per_run": int(args.max_ranker_actions_per_run),
        "rust_bestplay": bool(args.rust_bestplay),
        "records_by_ante": {
            str(key): value for key, value in sorted(Counter(int(record.get("ante", 0)) for record in records).items())
        },
        "records_by_candidate_action_type": dict(
            sorted(Counter(str(record.get("deepening_candidate_action_type", "")) for record in records).items())
        ),
        "records_by_candidate_action_kind": dict(
            sorted(Counter(str(record.get("deepening_candidate_action_kind", "")) for record in records).items())
        ),
        "records_by_heuristic_action_type": dict(
            sorted(Counter(str(record.get("deepening_heuristic_action_type", "")) for record in records).items())
        ),
        "mean_ranker_margin": round(statistics.mean(margins), 6) if margins else None,
        "min_ranker_margin": round(min(margins), 6) if margins else None,
        "mean_ranker_baseline_margin": round(statistics.mean(baseline_margins), 6) if baseline_margins else None,
        "min_ranker_baseline_margin": round(min(baseline_margins), 6) if baseline_margins else None,
    }


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture neural shop-ranker override proposals.")
    parser.add_argument("--ranker-ckpt", type=Path, required=True)
    parser.add_argument("--baseline-bot", default="solver_shop_basic_play_bot")
    parser.add_argument("--stake", default="white")
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--per-seed", type=int, default=1)
    parser.add_argument("--min-capture-ante", type=int, default=2)
    parser.add_argument("--max-capture-ante", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--candidate-action-types", default="buy,open_pack,end_shop")
    parser.add_argument("--candidate-action-kinds", default="card,pack")
    parser.add_argument("--min-margin", type=float, default=0.5)
    parser.add_argument("--min-baseline-margin", type=float, default=0.0)
    parser.add_argument("--max-ranker-actions-per-shop", type=int, default=1)
    parser.add_argument("--max-ranker-actions-per-run", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--no-rust-bestplay", dest="rust_bestplay", action="store_false", default=True)
    args = parser.parse_args(argv)

    if args.states < 0:
        parser.error("--states must be non-negative")
    if args.per_seed <= 0:
        parser.error("--per-seed must be positive")
    if args.max_capture_ante < args.min_capture_ante:
        parser.error("--max-capture-ante must be >= --min-capture-ante")
    if not args.ranker_ckpt.exists():
        parser.error(f"--ranker-ckpt does not exist: {args.ranker_ckpt}")
    try:
        args.candidate_action_types = _validate_action_types(_parse_csv(args.candidate_action_types))
    except ValueError as exc:
        parser.error(str(exc))
    args.candidate_action_kinds = _parse_csv(args.candidate_action_kinds)
    _configure_rust_bestplay(bool(args.rust_bestplay))

    started = time.perf_counter()
    seeds = _seed_strings(args.seed_offset, args.seed_count)
    chunks = _chunks(seeds, args.jobs)
    jobs = [
        (
            chunk,
            args.baseline_bot,
            args.stake,
            args.ranker_ckpt,
            args.max_steps,
            args.min_capture_ante,
            args.max_capture_ante,
            args.per_seed,
            args.max_actions,
            args.candidate_action_types,
            args.candidate_action_kinds,
            args.min_margin,
            args.min_baseline_margin,
            args.max_ranker_actions_per_shop,
            args.max_ranker_actions_per_run,
        )
        for chunk in chunks
        if chunk
    ]

    records_by_job: list[tuple[int, list[dict[str, Any]]]] = []
    max_workers = max(1, min(int(args.jobs), len(jobs) or 1))
    if max_workers > 1 and jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_capture_worker, job): index for index, job in enumerate(jobs)}
            for future in as_completed(futures):
                records_by_job.append((futures[future], future.result()))
    else:
        for index, job in enumerate(jobs):
            records_by_job.append((index, _capture_worker(job)))

    records = [
        record
        for _, batch in sorted(records_by_job)
        for record in batch
    ][: max(0, int(args.states))]
    _atomic_write_jsonl(args.out, records)
    metrics = _summarize(records, args=args, wall_s=time.perf_counter() - started)
    _atomic_write_json(args.metrics, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if records or args.states == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
