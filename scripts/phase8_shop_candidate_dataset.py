"""Generate Phase 8 shop candidate-ranker training records.

Each output JSONL row is one shop state with multiple candidate actions and
common-random-number rollout labels. This is the first data artifact for the
neural action-ranker path:

    python scripts/phase8_shop_candidate_dataset.py \
        --capture-bot solver_shop_basic_play_bot --capture-bot basic_strategy_bot \
        --seed-offset 800000 --seed-count 40 --states 32 --rollouts 4 \
        --jobs 8 --out .data/phase8_shop_candidates_smoke.jsonl \
        --metrics .data/phase8_shop_candidates_smoke.metrics.json

The CLI enables the audited Rust best-play scorer by default before worker
processes import the evaluator. Use ``--no-rust-bestplay`` only for debugging
or parity probes. State selection is source-balanced by default when multiple
capture bots are used. Long label runs write partial JSONL/metrics checkpoints
by default; use ``--resume-partial`` to continue from a matching partial JSONL.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time


SOURCE_METADATA_KEYS = (
    "record_type",
    "selection_reason",
    "shops_from_terminal",
    "trajectory_shop_count",
    "terminal_won",
    "terminal_ante",
    "terminal_money",
    "terminal_score",
    "terminal_required_score",
)


def _label_worker(args) -> dict | None:
    from balatro_ai.ml.shop_candidate_dataset import label_shop_state

    (
        source_bot,
        seed,
        state_index,
        state,
        source_metadata,
        crn_seeds,
        rollout_bot,
        heuristic_bot,
        max_antes,
        max_steps,
        max_actions,
        candidate_action_types,
        candidate_priority,
        include_heuristic_action,
    ) = args
    record = label_shop_state(
        state,
        seed=seed,
        state_index=state_index,
        crn_seeds=crn_seeds,
        rollout_bot=rollout_bot,
        heuristic_bot=heuristic_bot,
        source_bot=source_bot,
        max_antes=max_antes,
        max_steps=max_steps,
        max_actions=max_actions,
        candidate_action_types=candidate_action_types,
        candidate_priority=candidate_priority,
        include_heuristic_action=bool(include_heuristic_action),
    )
    if record is None:
        return None
    out = record.to_json_dict()
    if isinstance(source_metadata, dict):
        out.update(source_metadata)
    return out


def _collect_worker(args) -> list[tuple[str, str, int, object]]:
    from balatro_ai.ml.shop_candidate_dataset import collect_shop_states

    (
        bot_name,
        seeds,
        cap,
        per_seed,
        max_collect_steps,
        min_capture_ante,
        max_capture_ante,
        balance_antes,
        candidate_action_types,
        candidate_priority,
    ) = args
    return [
        (bot_name, seed, state_index, state)
        for seed, state_index, state in collect_shop_states(
            seeds,
            bot_name=bot_name,
            cap=cap,
            per_seed=per_seed,
            max_steps=max_collect_steps,
            min_ante=min_capture_ante,
            max_ante=max_capture_ante,
            balance_antes=balance_antes,
            candidate_action_types=candidate_action_types,
            candidate_priority=candidate_priority,
        )
    ]


def _state_item_parts(item):
    if len(item) == 4:
        source_bot, seed, state_index, state = item
        return source_bot, seed, state_index, state, {}
    source_bot, seed, state_index, state, metadata = item
    return source_bot, seed, state_index, state, dict(metadata or {})


def _seed_strings(offset: int, count: int) -> list[str]:
    return [f"{offset + i:07d}" for i in range(1, count + 1)]


def _chunks(items: list[str], n_chunks: int) -> list[list[str]]:
    n_chunks = max(1, min(n_chunks, len(items) or 1))
    size = max(1, (len(items) + n_chunks - 1) // n_chunks)
    return [items[start:start + size] for start in range(0, len(items), size)]


def _configure_rust_bestplay(enabled: bool) -> None:
    os.environ["BALATRO_RUST_BESTPLAY"] = "1" if enabled else "0"
    module = sys.modules.get("balatro_ai.rules.hand_evaluator")
    if module is not None:
        setattr(module, "_RUST_BESTPLAY_ENABLED", enabled)


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


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _state_dedupe_key(seed: str, state_index: int, state) -> tuple[object, ...]:
    modifiers = state.modifiers or {}
    shop_cards = modifiers.get("shop_cards", ())
    booster_packs = modifiers.get("booster_packs", ())
    voucher_cards = modifiers.get("voucher_cards", ())
    jokers = tuple(
        (
            joker.name,
            joker.edition,
            joker.sell_value,
            _stable_json(joker.metadata),
        )
        for joker in state.jokers
    )
    return (
        seed,
        state_index,
        state.phase.value,
        state.ante,
        state.money,
        jokers,
        _stable_json(shop_cards),
        _stable_json(booster_packs),
        _stable_json(voucher_cards),
    )


def _dedupe_states(states: list[tuple[str, str, int, object]]) -> list[tuple[str, str, int, object]]:
    deduped: list[tuple[str, str, int, object]] = []
    seen: set[tuple[object, ...]] = set()
    for item in states:
        source_bot, seed, state_index, state, metadata = _state_item_parts(item)
        key = _state_dedupe_key(seed, state_index, state)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((source_bot, seed, state_index, state, metadata))
    return deduped


def _select_states(
    states: list[tuple[str, str, int, object]],
    *,
    limit: int,
    selection_seed: int,
    balance_sources: bool = False,
    balance_antes: bool = False,
) -> list[tuple[str, str, int, object]]:
    selected = list(states)
    if balance_sources or balance_antes:
        rng = random.Random(selection_seed)
        by_group: dict[tuple[object, ...], list[tuple[str, str, int, object]]] = {}
        for item in selected:
            source_bot, _, _, state, _ = _state_item_parts(item)
            group: list[object] = []
            if balance_sources:
                group.append(source_bot)
            if balance_antes:
                ante = getattr(state, "ante", None)
                group.append(int(ante) if ante is not None else -1)
            by_group.setdefault(tuple(group), []).append(item)
        for items in by_group.values():
            rng.shuffle(items)
        out: list[tuple[str, str, int, object]] = []
        groups = sorted(by_group)
        while len(out) < limit and any(by_group.values()):
            for group in groups:
                items = by_group[group]
                if not items:
                    continue
                out.append(items.pop())
                if len(out) >= limit:
                    break
        return out
    random.Random(selection_seed).shuffle(selected)
    return selected[:limit]


def _summarize(records: list[dict], *, args, wall_s: float) -> dict:
    from balatro_ai.ml.shop_candidate_dataset import LABEL_VALUE_VERSION, rollout_confidence_summary

    action_counts = [len(record["candidates"]) for record in records]
    estimated_continuations = sum(action_counts) * int(args.rollouts)
    heuristic_present = [
        any(candidate["is_heuristic_action"] for candidate in record["candidates"])
        for record in records
    ]
    heuristic_best = [
        record["heuristic_action_key"] == record["best_action_key"]
        for record in records
        if record["heuristic_action_key"] is not None
    ]
    margins = []
    top_tie_counts = []
    nonzero_margins = []
    actions_within_005 = []
    actions_within_010 = []
    heuristic_within_005 = []
    heuristic_within_010 = []
    heuristic_action_types = []
    heuristic_outside_candidate_action_types = []
    split_half_best_agreements = []
    mean_best_first_half_agreements = []
    mean_best_second_half_agreements = []
    candidate_action_type_values = {
        action_type.value for action_type in getattr(args, "candidate_action_types", ()) or ()
    }
    for record in records:
        values = [candidate["mean_value"] for candidate in record["candidates"]]
        if len(values) >= 2:
            ordered = sorted(values, reverse=True)
            margin = ordered[0] - ordered[1]
            margins.append(margin)
            nonzero_margins.append(margin > 1e-6)
            top_tie_counts.append(sum(1 for value in values if abs(value - ordered[0]) <= 1e-6))
            best = ordered[0]
            actions_within_005.append(sum(1 for value in values if best - value <= 0.05))
            actions_within_010.append(sum(1 for value in values if best - value <= 0.10))
            heuristic_candidate = next(
                (candidate for candidate in record["candidates"] if candidate.get("is_heuristic_action")),
                None,
            )
            if heuristic_candidate is not None:
                heuristic_action_type = str((heuristic_candidate.get("action") or {}).get("type", ""))
                if heuristic_action_type:
                    heuristic_action_types.append(heuristic_action_type)
                if heuristic_action_type and candidate_action_type_values:
                    heuristic_outside_candidate_action_types.append(
                        heuristic_action_type not in candidate_action_type_values
                    )
                heuristic_regret = best - float(heuristic_candidate["mean_value"])
                heuristic_within_005.append(heuristic_regret <= 0.05)
                heuristic_within_010.append(heuristic_regret <= 0.10)
        first_half_best = _best_candidate_key(record, "first_half_mean")
        second_half_best = _best_candidate_key(record, "second_half_mean")
        mean_best = record["best_action_key"]
        if first_half_best is not None and second_half_best is not None:
            split_half_best_agreements.append(first_half_best == second_half_best)
            mean_best_first_half_agreements.append(first_half_best == mean_best)
            mean_best_second_half_agreements.append(second_half_best == mean_best)
    summary = {
        "records": len(records),
        "wall_s": round(wall_s, 3),
        "estimated_candidate_continuations": estimated_continuations,
        "candidate_continuations_per_wall_s": (
            round(estimated_continuations / wall_s, 4)
            if wall_s > 0
            else None
        ),
        "wall_s_per_record": round(wall_s / len(records), 4) if records else None,
        "jobs": args.jobs,
        "capture_bots": args.capture_bot,
        "input_records": [str(path) for path in getattr(args, "input_records", ()) or ()],
        "rollout_bot": args.rollout_bot,
        "heuristic_bot": args.heuristic_bot,
        "label_value_version": LABEL_VALUE_VERSION,
        "candidate_action_types": [
            action_type.value for action_type in getattr(args, "candidate_action_types", ()) or ()
        ],
        "candidate_priority": str(getattr(args, "candidate_priority", "legal")),
        "include_heuristic_action": bool(getattr(args, "include_heuristic_action", False)),
        "seed_offset": args.seed_offset,
        "seed_count": args.seed_count,
        "captured_states": getattr(args, "captured_states", None),
        "deduped_states": getattr(args, "deduped_states", None),
        "selected_states": getattr(args, "selected_states", None),
        "collect_jobs": args.collect_jobs,
        "min_capture_ante": args.min_capture_ante,
        "max_capture_ante": args.max_capture_ante,
        "selection_seed": args.selection_seed,
        "balance_source_bots": bool(args.balance_source_bots),
        "balance_antes": bool(getattr(args, "balance_antes", False)),
        "rust_bestplay": bool(args.rust_bestplay),
        "resume_partial": bool(getattr(args, "resume_partial", False)),
        "resumed_partial_records": getattr(args, "resumed_partial_records", 0),
        "remaining_label_jobs": getattr(args, "remaining_label_jobs", None),
        "records_by_source_bot": dict(sorted(Counter(record["source_bot"] for record in records).items())),
        "records_by_label_value_version": {
            str(version): count
            for version, count in sorted(
                Counter(int(record.get("label_value_version", 1)) for record in records).items()
            )
        },
        "records_by_ante": {
            str(ante): count
            for ante, count in sorted(Counter(int(record["ante"]) for record in records).items())
        },
        "records_by_terminal_won": {
            str(key): int(value)
            for key, value in sorted(
                Counter(record.get("terminal_won") for record in records if "terminal_won" in record).items(),
                key=lambda item: str(item[0]),
            )
        },
        "records_by_selection_reason": dict(
            sorted(Counter(str(record.get("selection_reason")) for record in records if "selection_reason" in record).items())
        ),
        "rollouts_per_action": args.rollouts,
        "max_antes": args.max_antes,
        "partial_every": getattr(args, "partial_every", None),
        "mean_candidates": round(statistics.mean(action_counts), 4) if action_counts else None,
        "heuristic_present_rate": round(statistics.mean(heuristic_present), 4) if heuristic_present else None,
        "heuristic_best_rate": round(statistics.mean(heuristic_best), 4) if heuristic_best else None,
        "heuristic_action_types": dict(sorted(Counter(heuristic_action_types).items())),
        "heuristic_outside_candidate_action_types_rate": (
            round(statistics.mean(heuristic_outside_candidate_action_types), 4)
            if heuristic_outside_candidate_action_types
            else None
        ),
        "heuristic_outside_candidate_action_types_count": (
            sum(heuristic_outside_candidate_action_types)
            if heuristic_outside_candidate_action_types
            else None
        ),
        "mean_best_margin": round(statistics.mean(margins), 4) if margins else None,
        "nonzero_best_margin_rate": round(statistics.mean(nonzero_margins), 4) if nonzero_margins else None,
        "mean_top_tie_count": round(statistics.mean(top_tie_counts), 4) if top_tie_counts else None,
        "mean_actions_within_0_05": round(statistics.mean(actions_within_005), 4) if actions_within_005 else None,
        "mean_actions_within_0_10": round(statistics.mean(actions_within_010), 4) if actions_within_010 else None,
        "heuristic_within_0_05_rate": (
            round(statistics.mean(heuristic_within_005), 4) if heuristic_within_005 else None
        ),
        "heuristic_within_0_10_rate": (
            round(statistics.mean(heuristic_within_010), 4) if heuristic_within_010 else None
        ),
        "split_half_best_agreement_rate": (
            round(statistics.mean(split_half_best_agreements), 4)
            if split_half_best_agreements
            else None
        ),
        "mean_best_first_half_agreement_rate": (
            round(statistics.mean(mean_best_first_half_agreements), 4)
            if mean_best_first_half_agreements
            else None
        ),
        "mean_best_second_half_agreement_rate": (
            round(statistics.mean(mean_best_second_half_agreements), 4)
            if mean_best_second_half_agreements
            else None
        ),
    }
    summary.update(
        rollout_confidence_summary(
            records,
            z=float(getattr(args, "confidence_z", 1.0)),
            practical_margin=float(getattr(args, "confidence_margin", 0.10)),
        )
    )
    return summary


def _best_candidate_key(record: dict, field: str) -> str | None:
    candidates = record.get("candidates", ())
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.get(field, float("-inf"))).get("action_key")


def _default_partial_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".partial")


def _ordered_records(records_by_job: list[tuple[int, dict]]) -> list[dict]:
    return [record for _, record in sorted(records_by_job, key=lambda item: item[0])]


def _job_resume_key(job: tuple) -> tuple[str, str, int]:
    source_bot, seed, state_index = job[:3]
    return str(source_bot), str(seed), int(state_index)


def _record_resume_key(record: dict) -> tuple[str, str, int] | None:
    try:
        return str(record["source_bot"]), str(record["seed"]), int(record["state_index"])
    except (KeyError, TypeError, ValueError):
        return None


def _load_partial_records_for_jobs(path: Path, jobs: list[tuple]) -> list[tuple[int, dict]]:
    if not path.exists():
        return []
    job_index_by_key = {_job_resume_key(job): index for index, job in enumerate(jobs)}
    records_by_job: list[tuple[int, dict]] = []
    seen_indices: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = _record_resume_key(record)
        if key is None:
            continue
        index = job_index_by_key.get(key)
        if index is None or index in seen_indices:
            continue
        records_by_job.append((index, record))
        seen_indices.add(index)
    return records_by_job


def _load_states_from_records(paths: list[Path]) -> list[tuple[str, str, int, object]]:
    from balatro_ai.api.state import GameState

    states: list[tuple[str, str, int, object]] = []
    missing = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                snapshot = record.get("state_snapshot")
                if not isinstance(snapshot, dict):
                    missing += 1
                    continue
                metadata = {key: record[key] for key in SOURCE_METADATA_KEYS if key in record}
                states.append(
                    (
                        str(record.get("source_bot", "input_records")),
                        str(record.get("seed", "")),
                        int(record.get("state_index", 0) or 0),
                        GameState.from_mapping(snapshot),
                        metadata,
                    )
                )
    if missing:
        raise ValueError(
            f"{missing} input records do not include state_snapshot; regenerate them with the current dataset CLI"
        )
    return states


def _state_pool_records(states: list[tuple[str, str, int, object]]) -> list[dict]:
    from balatro_ai.ml.shop_candidate_dataset import LABEL_VALUE_VERSION, state_snapshot

    records = []
    for item in states:
        source_bot, seed, state_index, state, metadata = _state_item_parts(item)
        record = {
            "source_bot": str(source_bot),
            "seed": str(seed),
            "state_index": int(state_index),
            "label_value_version": LABEL_VALUE_VERSION,
            "phase": state.phase.value,
            "ante": int(state.ante),
            "money": int(state.money),
            "state_snapshot": state_snapshot(state),
        }
        record.update(metadata)
        records.append(record)
    return records


def _summarize_state_pool(records: list[dict], *, args, wall_s: float) -> dict:
    from balatro_ai.ml.shop_candidate_dataset import LABEL_VALUE_VERSION

    return {
        "records": len(records),
        "wall_s": round(wall_s, 3),
        "wall_s_per_record": round(wall_s / len(records), 4) if records else None,
        "jobs": args.jobs,
        "capture_bots": args.capture_bot,
        "input_records": [str(path) for path in getattr(args, "input_records", ()) or ()],
        "label_value_version": LABEL_VALUE_VERSION,
        "seed_offset": args.seed_offset,
        "seed_count": args.seed_count,
        "captured_states": getattr(args, "captured_states", None),
        "deduped_states": getattr(args, "deduped_states", None),
        "selected_states": getattr(args, "selected_states", None),
        "collect_jobs": args.collect_jobs,
        "min_capture_ante": args.min_capture_ante,
        "max_capture_ante": args.max_capture_ante,
        "selection_seed": args.selection_seed,
        "balance_source_bots": bool(args.balance_source_bots),
        "balance_antes": bool(getattr(args, "balance_antes", False)),
        "rust_bestplay": bool(getattr(args, "rust_bestplay", False)),
        "records_by_source_bot": dict(sorted(Counter(str(record["source_bot"]) for record in records).items())),
        "records_by_ante": {
            str(key): int(value)
            for key, value in sorted(Counter(int(record.get("ante", 0)) for record in records).items())
        },
        "records_by_terminal_won": {
            str(key): int(value)
            for key, value in sorted(
                Counter(record.get("terminal_won") for record in records if "terminal_won" in record).items(),
                key=lambda item: str(item[0]),
            )
        },
        "records_by_selection_reason": dict(
            sorted(Counter(str(record.get("selection_reason")) for record in records if "selection_reason" in record).items())
        ),
        "capture_only": True,
    }


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_label_progress(
    records_by_job: list[tuple[int, dict]],
    *,
    args,
    started: float,
    expected_label_jobs: int,
    completed_label_jobs: int,
    partial_out: Path | None,
    partial_metrics: Path | None,
    complete: bool,
) -> None:
    records = _ordered_records(records_by_job)
    if partial_out is not None:
        _atomic_write_jsonl(partial_out, records)
    if partial_metrics is not None:
        metrics = _summarize(records, args=args, wall_s=time.perf_counter() - started)
        metrics.update(
            {
                "complete": complete,
                "expected_label_jobs": expected_label_jobs,
                "completed_label_jobs": completed_label_jobs,
                "partial_out": str(partial_out) if partial_out is not None else None,
                "partial_metrics": str(partial_metrics),
            }
        )
        _atomic_write_json(partial_metrics, metrics)


def _label_records(
    indexed_jobs: list[tuple[int, tuple]],
    *,
    max_workers: int,
    args,
    started: float,
    expected_label_jobs: int,
    initial_records_by_job: list[tuple[int, dict]] | None = None,
    partial_out: Path | None,
    partial_metrics: Path | None,
) -> list[dict]:
    records_by_job: list[tuple[int, dict]] = list(initial_records_by_job or [])
    completed_jobs = len(records_by_job)
    partial_enabled = args.partial_every > 0 and (partial_out is not None or partial_metrics is not None)

    def maybe_write_progress(*, force: bool = False) -> None:
        if not partial_enabled:
            return
        if force or completed_jobs % args.partial_every == 0:
            _write_label_progress(
                records_by_job,
                args=args,
                started=started,
                expected_label_jobs=expected_label_jobs,
                completed_label_jobs=completed_jobs,
                partial_out=partial_out,
                partial_metrics=partial_metrics,
                complete=completed_jobs >= expected_label_jobs,
            )

    if max_workers > 1 and indexed_jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_label_worker, job): index for index, job in indexed_jobs}
            for future in as_completed(futures):
                index = futures[future]
                record = future.result()
                completed_jobs += 1
                if record is not None:
                    records_by_job.append((index, record))
                maybe_write_progress()
    else:
        for index, job in indexed_jobs:
            record = _label_worker(job)
            completed_jobs += 1
            if record is not None:
                records_by_job.append((index, record))
            maybe_write_progress()

    maybe_write_progress(force=True)
    return _ordered_records(records_by_job)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate shop candidate-ranker JSONL data.")
    parser.add_argument("--capture-bot", action="append", default=[], help="Bot trajectory source; may repeat.")
    parser.add_argument(
        "--input-records",
        action="append",
        type=Path,
        default=[],
        help="Existing candidate JSONL with state_snapshot fields to relabel; may repeat.",
    )
    parser.add_argument("--rollout-bot", default="solver_shop_basic_play_bot")
    parser.add_argument("--heuristic-bot", default="solver_shop_basic_play_bot")
    parser.add_argument("--seed-offset", type=int, default=800000)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--per-seed", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-antes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument(
        "--confidence-z",
        type=float,
        default=1.0,
        help="Z multiplier for paired rollout confidence summaries.",
    )
    parser.add_argument(
        "--confidence-margin",
        type=float,
        default=0.10,
        help="Practical advantage margin for confidence summaries.",
    )
    parser.add_argument("--max-collect-steps", type=int, default=2500)
    parser.add_argument("--min-capture-ante", type=int, default=1)
    parser.add_argument("--max-capture-ante", type=int, default=None)
    parser.add_argument(
        "--candidate-action-types",
        default=None,
        help="Comma-separated candidate actions to label, e.g. buy,open_pack,end_shop. Defaults to all shop candidates.",
    )
    parser.add_argument(
        "--candidate-priority",
        default="legal",
        choices=("legal", "deep_advantage"),
        help="Candidate ordering before --max-actions truncation.",
    )
    parser.add_argument(
        "--include-heuristic-action",
        action="store_true",
        help=(
            "Always label the heuristic/rollout bot action as a comparison candidate, "
            "even when --candidate-action-types filters it out."
        ),
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Write selected state_snapshot records and skip expensive rollout labeling.",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument(
        "--no-balance-source-bots",
        dest="balance_source_bots",
        action="store_false",
        default=True,
        help="Use plain global shuffle instead of round-robin source-bot selection.",
    )
    parser.add_argument(
        "--balance-antes",
        action="store_true",
        help="Collect per-ante quotas and round-robin selected states across capture antes.",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--collect-jobs", type=int, default=0)
    parser.add_argument(
        "--no-rust-bestplay",
        dest="rust_bestplay",
        action="store_false",
        default=True,
        help="Disable the verified Rust best-play scorer for debugging/parity checks.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--partial-every",
        type=int,
        default=4,
        help="Write partial JSONL/metrics after this many completed label jobs; 0 disables checkpointing.",
    )
    parser.add_argument(
        "--partial-out",
        type=Path,
        default=None,
        help="Partial JSONL checkpoint path. Defaults to OUT.partial when partial checkpointing is enabled.",
    )
    parser.add_argument(
        "--partial-metrics",
        type=Path,
        default=None,
        help="Partial metrics checkpoint path. Defaults to METRICS.partial when partial checkpointing is enabled.",
    )
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="Reuse matching records from the partial JSONL checkpoint before labeling remaining jobs.",
    )
    args = parser.parse_args()

    _configure_rust_bestplay(bool(args.rust_bestplay))
    if args.rollouts < 2 or args.rollouts % 2:
        parser.error("--rollouts must be an even number >= 2")
    if args.partial_every < 0:
        parser.error("--partial-every must be >= 0")
    if args.resume_partial and args.partial_every <= 0:
        parser.error("--resume-partial requires partial checkpointing")
    if args.balance_antes and args.max_capture_ante is None and not args.input_records:
        parser.error("--balance-antes requires --max-capture-ante")
    try:
        args.candidate_action_types = _parse_action_types_csv(args.candidate_action_types)
    except ValueError as exc:
        parser.error(str(exc))
    capture_bots = args.capture_bot or ["solver_shop_basic_play_bot", "basic_strategy_bot"]
    args.capture_bot = capture_bots
    args.collect_jobs = args.collect_jobs or args.jobs

    started = time.perf_counter()
    if args.input_records:
        raw_states = _load_states_from_records(args.input_records)
        capture_bots = sorted({str(_state_item_parts(item)[0]) for item in raw_states})
        args.capture_bot = capture_bots
    else:
        seeds = _seed_strings(args.seed_offset, args.seed_count)
        chunks_per_bot = max(1, min(len(seeds) or 1, max(1, args.collect_jobs) // max(1, len(capture_bots))))
        collect_jobs = [
            (
                bot_name,
                chunk,
                args.states,
                args.per_seed,
                args.max_collect_steps,
                args.min_capture_ante,
                args.max_capture_ante,
                bool(args.balance_antes),
                args.candidate_action_types,
                args.candidate_priority,
            )
            for bot_name in capture_bots
            for chunk in _chunks(seeds, chunks_per_bot)
        ]
        collect_workers = max(1, min(args.collect_jobs, len(collect_jobs) or 1))
        if collect_workers > 1 and collect_jobs:
            with ProcessPoolExecutor(max_workers=collect_workers) as pool:
                collected = pool.map(_collect_worker, collect_jobs)
                raw_states = [item for items in collected for item in items]
        else:
            raw_states = [item for job in collect_jobs for item in _collect_worker(job)]
    deduped_states = _dedupe_states(raw_states)
    states = _select_states(
        deduped_states,
        limit=args.states,
        selection_seed=args.selection_seed,
        balance_sources=bool(args.balance_source_bots),
        balance_antes=bool(args.balance_antes),
    )
    args.captured_states = len(raw_states)
    args.deduped_states = len(deduped_states)
    args.selected_states = len(states)
    if args.capture_only:
        records = _state_pool_records(states)
        _atomic_write_jsonl(args.out, records)
        metrics = _summarize_state_pool(records, args=args, wall_s=time.perf_counter() - started)
        metrics.update({"complete": True, "expected_label_jobs": 0, "completed_label_jobs": 0})
        _atomic_write_json(args.metrics, metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        return 0 if records else 1
    crn_seeds = tuple(range(1, args.rollouts + 1))
    jobs = [
        (
            source_bot,
            seed,
            state_index,
            state,
            metadata,
            crn_seeds,
            args.rollout_bot,
            args.heuristic_bot,
            args.max_antes,
            args.max_steps,
            args.max_actions,
            args.candidate_action_types,
            args.candidate_priority,
            bool(args.include_heuristic_action),
        )
        for source_bot, seed, state_index, state, metadata in (_state_item_parts(item) for item in states)
    ]
    if args.partial_every > 0:
        partial_out = args.partial_out or _default_partial_path(args.out)
        partial_metrics = args.partial_metrics or _default_partial_path(args.metrics)
    else:
        partial_out = None
        partial_metrics = None
    initial_records_by_job = (
        _load_partial_records_for_jobs(partial_out, jobs)
        if args.resume_partial and partial_out is not None
        else []
    )
    done_indices = {index for index, _ in initial_records_by_job}
    indexed_jobs = [(index, job) for index, job in enumerate(jobs) if index not in done_indices]
    args.resumed_partial_records = len(initial_records_by_job)
    args.remaining_label_jobs = len(indexed_jobs)
    max_workers = max(1, min(args.jobs, len(indexed_jobs) or 1))
    records = _label_records(
        indexed_jobs,
        max_workers=max_workers,
        args=args,
        started=started,
        expected_label_jobs=len(jobs),
        initial_records_by_job=initial_records_by_job,
        partial_out=partial_out,
        partial_metrics=partial_metrics,
    )

    _atomic_write_jsonl(args.out, records)

    metrics = _summarize(records, args=args, wall_s=time.perf_counter() - started)
    metrics.update(
        {
            "complete": True,
            "expected_label_jobs": len(jobs),
            "completed_label_jobs": len(jobs),
            "partial_out": str(partial_out) if partial_out is not None else None,
            "partial_metrics": str(partial_metrics) if partial_metrics is not None else None,
        }
    )
    _atomic_write_json(args.metrics, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
