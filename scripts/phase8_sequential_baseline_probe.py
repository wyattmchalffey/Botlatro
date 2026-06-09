"""Sequential baseline-vs-candidate shop probe for Phase 8.

The fixed candidate labeler spends the same rollout budget on every candidate.
This script uses paired CRN samples against the heuristic baseline and stops
individual candidates early once their paired lower/upper confidence bound is
clearly positive or clearly negative. The output is compatible with the shop
candidate JSONL shape, with extra ``sequential_*`` fields for auditability.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import os
import time
from typing import Any


def _configure_rust_bestplay(enabled: bool) -> None:
    os.environ["BALATRO_RUST_BESTPLAY"] = "1" if enabled else "0"
    module = sys.modules.get("balatro_ai.rules.hand_evaluator")
    if module is not None:
        setattr(module, "_RUST_BESTPLAY_ENABLED", enabled)


def _worker(args: tuple[Any, ...]) -> dict | None:
    (
        record,
        rollout_bot,
        heuristic_bot,
        max_antes,
        max_steps,
        max_actions,
        candidate_action_types,
        candidate_priority,
        include_heuristic_action,
        min_rollouts,
        max_rollouts,
        z,
        positive_margin,
        negative_margin,
        max_wall_s_per_state,
        focus_action_key,
        focus_deepening_candidate,
    ) = args
    return _probe_record(
        record,
        rollout_bot=rollout_bot,
        heuristic_bot=heuristic_bot,
        max_antes=max_antes,
        max_steps=max_steps,
        max_actions=max_actions,
        candidate_action_types=candidate_action_types,
        candidate_priority=candidate_priority,
        include_heuristic_action=include_heuristic_action,
        min_rollouts=min_rollouts,
        max_rollouts=max_rollouts,
        z=z,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
        max_wall_s_per_state=max_wall_s_per_state,
        focus_action_key=focus_action_key,
        focus_deepening_candidate=focus_deepening_candidate,
    )


def _probe_record(
    record: dict[str, Any],
    *,
    rollout_bot: str,
    heuristic_bot: str,
    max_antes: int,
    max_steps: int,
    max_actions: int,
    candidate_action_types,
    candidate_priority: str,
    include_heuristic_action: bool,
    min_rollouts: int,
    max_rollouts: int,
    z: float,
    positive_margin: float,
    negative_margin: float,
    max_wall_s_per_state: float,
    focus_action_key: str,
    focus_deepening_candidate: bool,
) -> dict | None:
    from balatro_ai.api.actions import Action
    from balatro_ai.api.state import GameState
    from balatro_ai.ml.encoding import ENCODING_VERSION, encode_state
    from balatro_ai.ml.shop_candidate_dataset import (
        LABEL_VALUE_VERSION,
        _append_missing_action,
        _heuristic_action,
        _shop_token_index,
        action_key,
        candidate_shop_actions,
        paired_rollout_advantage,
        rollout_value_after_action,
        state_snapshot,
    )

    snapshot = record.get("state_snapshot")
    if not isinstance(snapshot, dict):
        return None
    state = GameState.from_mapping(snapshot)
    actions = list(
        candidate_shop_actions(
            state,
            max_actions=max_actions,
            action_types=candidate_action_types,
            priority=candidate_priority,
        )
    )
    heuristic = _heuristic_action(state, heuristic_bot)
    if heuristic is None:
        return None
    heuristic_key = action_key(heuristic)
    if include_heuristic_action:
        actions = list(_append_missing_action(tuple(actions), heuristic))
    selected_focus_key = focus_action_key.strip()
    if focus_deepening_candidate and not selected_focus_key:
        selected_focus_key = str(record.get("deepening_candidate_action_key", "")).strip()
    if selected_focus_key and not any(action_key(action) == selected_focus_key for action in actions):
        raw_focus = record.get("deepening_candidate_action")
        if isinstance(raw_focus, dict):
            try:
                parsed_focus = Action.from_mapping(raw_focus)
            except (ValueError, KeyError, TypeError):
                parsed_focus = None
            if parsed_focus is not None and action_key(parsed_focus) == selected_focus_key:
                actions.append(parsed_focus)
    if len(actions) < 2:
        return None

    action_by_key = {action_key(action): action for action in actions}
    if heuristic_key not in action_by_key:
        actions.append(heuristic)
        action_by_key[heuristic_key] = heuristic
    if selected_focus_key:
        focused = action_by_key.get(selected_focus_key)
        if focused is None:
            raw_focus = record.get("deepening_candidate_action")
            if isinstance(raw_focus, dict):
                try:
                    parsed_focus = Action.from_mapping(raw_focus)
                except (ValueError, KeyError, TypeError):
                    parsed_focus = None
                if parsed_focus is not None and action_key(parsed_focus) == selected_focus_key:
                    focused = parsed_focus
                    action_by_key[selected_focus_key] = parsed_focus
        if focused is None or selected_focus_key == heuristic_key:
            return None
        action_by_key = {
            heuristic_key: action_by_key[heuristic_key],
            selected_focus_key: focused,
        }
    values: dict[str, list[float]] = {key: [] for key in action_by_key}
    stop_reason: dict[str, str] = {key: "max_rollouts" for key in action_by_key}
    active = {key for key in action_by_key if key != heuristic_key}

    started = time.perf_counter()
    timed_out = False
    for first_seed in range(1, max_rollouts + 1, 2):
        if max_wall_s_per_state > 0 and time.perf_counter() - started > max_wall_s_per_state:
            timed_out = True
            break
        pair_seeds = [first_seed]
        if first_seed + 1 <= max_rollouts:
            pair_seeds.append(first_seed + 1)
        for seed in pair_seeds:
            if max_wall_s_per_state > 0 and time.perf_counter() - started > max_wall_s_per_state:
                timed_out = True
                break
            baseline_value = rollout_value_after_action(
                state,
                action_by_key[heuristic_key],
                seed=seed,
                rollout_bot=rollout_bot,
                max_antes=max_antes,
                max_steps=max_steps,
            )
            if baseline_value is None:
                return None
            values[heuristic_key].append(float(baseline_value))
            for key in tuple(active):
                if max_wall_s_per_state > 0 and time.perf_counter() - started > max_wall_s_per_state:
                    timed_out = True
                    break
                value = rollout_value_after_action(
                    state,
                    action_by_key[key],
                    seed=seed,
                    rollout_bot=rollout_bot,
                    max_antes=max_antes,
                    max_steps=max_steps,
                )
                if value is None:
                    active.remove(key)
                    stop_reason[key] = "invalid_action"
                    continue
                values[key].append(float(value))
            if timed_out:
                break

        for key in tuple(active):
            if len(values[key]) < min_rollouts:
                continue
            stats = paired_rollout_advantage(
                {"rollout_values": values[key]},
                {"rollout_values": values[heuristic_key]},
                z=z,
            )
            if stats is None:
                continue
            if stats.lower_bound > positive_margin:
                stop_reason[key] = "positive_lcb"
                active.remove(key)
            elif stats.upper_bound < -abs(negative_margin):
                stop_reason[key] = "negative_ucb"
                active.remove(key)
        if not active:
            break
        if timed_out:
            break

    if timed_out:
        for key in active:
            stop_reason[key] = "state_timeout"

    raw = [(action_by_key[key], tuple(vals)) for key, vals in values.items() if len(vals) >= 2]
    if len(raw) < 2 or heuristic_key not in {action_key(action) for action, _ in raw}:
        return None
    ranked = sorted(raw, key=lambda item: statistics.mean(item[1]), reverse=True)
    labels: list[dict[str, Any]] = []
    for rank, (action, rollout_values) in enumerate(ranked, start=1):
        key = action_key(action)
        labels.append(
            {
                "action_key": key,
                "action": action.to_json(),
                "shop_token_index": _shop_token_index(state, action),
                "rollout_values": list(rollout_values),
                "mean_value": float(statistics.mean(rollout_values)),
                "first_half_mean": _half_mean(rollout_values, first=True),
                "second_half_mean": _half_mean(rollout_values, first=False),
                "rank": rank,
                "is_heuristic_action": key == heuristic_key,
                "sequential_rollouts": len(rollout_values),
                "sequential_stop_reason": stop_reason.get(key, "baseline" if key == heuristic_key else "max_rollouts"),
            }
        )
    return {
        "seed": str(record.get("seed", "")),
        "state_index": int(record.get("state_index", 0) or 0),
        "source_bot": str(record.get("source_bot", "")),
        "label_value_version": LABEL_VALUE_VERSION,
        "phase": state.phase.value,
        "ante": int(state.ante),
        "money": int(state.money),
        "encoding_version": ENCODING_VERSION,
        "encoded_state": asdict(encode_state(state)),
        "state_snapshot": state_snapshot(state),
        "heuristic_action_key": heuristic_key,
        "best_action_key": labels[0]["action_key"],
        "candidates": labels,
        "sequential_probe": True,
        "sequential_min_rollouts": min_rollouts,
        "sequential_max_rollouts": max_rollouts,
        "sequential_z": z,
        "sequential_positive_margin": positive_margin,
        "sequential_negative_margin": negative_margin,
        "sequential_wall_s": round(time.perf_counter() - started, 3),
        "sequential_timed_out": timed_out,
        "sequential_focus_action_key": selected_focus_key,
    }


def _half_mean(values: tuple[float, ...], *, first: bool) -> float:
    midpoint = max(1, len(values) // 2)
    half = values[:midpoint] if first else values[midpoint:]
    if not half:
        half = values
    return float(statistics.mean(half))


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _parse_action_types_csv(value: str | None):
    if value is None or not value.strip():
        return None
    from balatro_ai.api.actions import ActionType

    parsed: list[ActionType] = []
    seen: set[ActionType] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            action_type = ActionType(token)
        except ValueError as exc:
            try:
                action_type = ActionType[token.upper()]
            except KeyError:
                allowed = ", ".join(action.value for action in ActionType)
                raise ValueError(f"unknown action type {token!r}; expected one of: {allowed}") from exc
        if action_type not in seen:
            seen.add(action_type)
            parsed.append(action_type)
    return tuple(parsed) if parsed else None


def _summarize(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    wall_s: float,
    input_record_count: int | None = None,
) -> dict[str, Any]:
    from balatro_ai.ml.shop_candidate_dataset import rollout_confidence_summary

    attempted = len(records) if input_record_count is None else int(input_record_count)
    action_counts = [len(record.get("candidates", ())) for record in records]
    stop_counts = Counter(
        candidate.get("sequential_stop_reason", "")
        for record in records
        for candidate in record.get("candidates", ())
        if not candidate.get("is_heuristic_action")
    )
    rollout_counts = [
        int(candidate.get("sequential_rollouts", len(candidate.get("rollout_values", ()))))
        for record in records
        for candidate in record.get("candidates", ())
    ]
    metrics = {
        "records": len(records),
        "input_record_count": attempted,
        "skipped_record_count": max(0, attempted - len(records)),
        "wall_s": round(wall_s, 3),
        "wall_s_per_record": round(wall_s / len(records), 4) if records else None,
        "estimated_candidate_continuations": sum(rollout_counts),
        "candidate_continuations_per_wall_s": round(sum(rollout_counts) / wall_s, 4) if wall_s > 0 else None,
        "jobs": args.jobs,
        "input_records": [str(path) for path in args.input_records],
        "rollout_bot": args.rollout_bot,
        "heuristic_bot": args.heuristic_bot,
        "candidate_action_types": [
            action_type.value for action_type in args.candidate_action_types or ()
        ],
        "candidate_priority": args.candidate_priority,
        "include_heuristic_action": bool(args.include_heuristic_action),
        "min_rollouts": args.min_rollouts,
        "max_rollouts": args.max_rollouts,
        "max_antes": args.max_antes,
        "max_steps": args.max_steps,
        "z": args.z,
        "positive_margin": args.positive_margin,
        "negative_margin": args.negative_margin,
        "max_wall_s_per_state": args.max_wall_s_per_state,
        "focus_deepening_candidate": bool(args.focus_deepening_candidate),
        "focus_action_key": args.focus_action_key,
        "rust_bestplay": bool(args.rust_bestplay),
        "mean_candidates": round(statistics.mean(action_counts), 4) if action_counts else None,
        "records_by_source_bot": dict(sorted(Counter(record.get("source_bot", "") for record in records).items())),
        "records_by_ante": {
            str(key): value for key, value in sorted(Counter(int(record.get("ante", 0)) for record in records).items())
        },
        "candidate_stop_reasons": dict(sorted(stop_counts.items())),
        "mean_candidate_rollouts": round(statistics.mean(rollout_counts), 4) if rollout_counts else None,
        "min_candidate_rollouts": min(rollout_counts) if rollout_counts else None,
        "max_candidate_rollouts": max(rollout_counts) if rollout_counts else None,
    }
    metrics.update(rollout_confidence_summary(records, z=args.z, practical_margin=args.positive_margin))
    return metrics


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential paired baseline-vs-candidate shop probe.")
    parser.add_argument("--input-records", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--rollout-bot", default="solver_shop_basic_play_bot")
    parser.add_argument("--heuristic-bot", default="solver_shop_basic_play_bot")
    parser.add_argument("--min-rollouts", type=int, default=4)
    parser.add_argument("--max-rollouts", type=int, default=8)
    parser.add_argument("--max-antes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--candidate-action-types", default="buy,open_pack,end_shop")
    parser.add_argument("--candidate-priority", default="deep_advantage", choices=("legal", "deep_advantage"))
    parser.add_argument("--include-heuristic-action", action="store_true")
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--positive-margin", type=float, default=0.10)
    parser.add_argument("--negative-margin", type=float, default=0.10)
    parser.add_argument(
        "--max-wall-s-per-state",
        type=float,
        default=0.0,
        help="Stop sampling a state between rollouts after this many seconds; 0 disables.",
    )
    parser.add_argument(
        "--focus-deepening-candidate",
        action="store_true",
        help="If input records include deepening_candidate_action_key, probe only that candidate plus the heuristic.",
    )
    parser.add_argument(
        "--focus-action-key",
        default="",
        help="Probe only this candidate action key plus the heuristic.",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--no-rust-bestplay", dest="rust_bestplay", action="store_false", default=True)
    args = parser.parse_args()

    if args.min_rollouts < 2 or args.max_rollouts < args.min_rollouts:
        parser.error("--max-rollouts must be >= --min-rollouts >= 2")
    if args.min_rollouts % 2 or args.max_rollouts % 2:
        parser.error("--min-rollouts and --max-rollouts must be even")
    try:
        args.candidate_action_types = _parse_action_types_csv(args.candidate_action_types)
    except ValueError as exc:
        parser.error(str(exc))
    _configure_rust_bestplay(bool(args.rust_bestplay))

    started = time.perf_counter()
    input_records = _load_records(args.input_records)[: max(0, int(args.states))]
    jobs = [
        (
            record,
            args.rollout_bot,
            args.heuristic_bot,
            args.max_antes,
            args.max_steps,
            args.max_actions,
            args.candidate_action_types,
            args.candidate_priority,
            bool(args.include_heuristic_action),
            args.min_rollouts,
            args.max_rollouts,
            args.z,
            args.positive_margin,
            args.negative_margin,
            args.max_wall_s_per_state,
            args.focus_action_key,
            bool(args.focus_deepening_candidate),
        )
        for record in input_records
    ]
    records_by_index: list[tuple[int, dict]] = []
    max_workers = max(1, min(int(args.jobs), len(jobs) or 1))
    if max_workers > 1 and jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_worker, job): index for index, job in enumerate(jobs)}
            for future in as_completed(futures):
                record = future.result()
                if record is not None:
                    records_by_index.append((futures[future], record))
    else:
        for index, job in enumerate(jobs):
            record = _worker(job)
            if record is not None:
                records_by_index.append((index, record))
    records = [record for _, record in sorted(records_by_index)]
    _atomic_write_jsonl(args.out, records)
    metrics = _summarize(records, args=args, wall_s=time.perf_counter() - started, input_record_count=len(input_records))
    _atomic_write_json(args.metrics, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
