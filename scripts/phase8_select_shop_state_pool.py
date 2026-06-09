"""Select targeted shop-state snapshots from a larger capture-only pool.

This is a cheap pre-labeling step for Phase 8. It keeps state collection
decoupled from expensive rollout labels, while letting us spend rollout budget
on states where the heuristic baseline competes with focused neural candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from balatro_ai.api.actions import ActionType
from balatro_ai.api.state import GameState
from balatro_ai.ml.shop_candidate_dataset import action_key, candidate_shop_actions


@dataclass(frozen=True, slots=True)
class ScoredShopState:
    record: dict[str, Any]
    source_bot: str
    ante: int
    money: int
    heuristic_action_type: str
    heuristic_in_candidates: bool
    candidate_action_types: tuple[str, ...]
    score: float


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, str, int] | None:
    try:
        return (
            str(record.get("source_bot", "")),
            str(record["seed"]),
            int(record.get("state_index", 0) or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _exclude_keys(paths: list[Path]) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for record in _load_records(paths):
        key = _record_key(record)
        if key is not None:
            keys.add(key)
    return keys


def _filter_excluded(
    records: list[dict[str, Any]],
    excluded: set[tuple[str, str, int]],
) -> list[dict[str, Any]]:
    if not excluded:
        return records
    return [record for record in records if _record_key(record) not in excluded]


def _parse_action_types_csv(raw: str) -> tuple[ActionType, ...]:
    if not raw.strip():
        return ()
    by_value = {action_type.value: action_type for action_type in ActionType}
    by_name = {action_type.name.lower(): action_type for action_type in ActionType}
    out: list[ActionType] = []
    seen: set[ActionType] = set()
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        action_type = by_value.get(key) or by_name.get(key)
        if action_type is None:
            allowed = ", ".join(action.value for action in ActionType)
            raise ValueError(f"unknown action type {item!r}; expected one of: {allowed}")
        if action_type not in seen:
            seen.add(action_type)
            out.append(action_type)
    return tuple(out)


def _score_records(
    records: list[dict[str, Any]],
    *,
    heuristic_bot: str,
    max_actions: int,
    candidate_action_types: tuple[ActionType, ...],
    candidate_priority: str,
    preferred_heuristic_types: tuple[str, ...],
) -> list[ScoredShopState]:
    from balatro_ai.ml.shop_candidate_dataset import _heuristic_action

    preferred_rank = {name: len(preferred_heuristic_types) - index for index, name in enumerate(preferred_heuristic_types)}
    scored: list[ScoredShopState] = []
    for record in records:
        snapshot = record.get("state_snapshot")
        if not isinstance(snapshot, dict):
            continue
        state = GameState.from_mapping(snapshot)
        actions = candidate_shop_actions(
            state,
            max_actions=max_actions,
            action_types=candidate_action_types or None,
            priority=candidate_priority,
        )
        if not actions:
            continue
        heuristic = _heuristic_action(state, heuristic_bot)
        heuristic_type = heuristic.action_type.value if heuristic is not None else ""
        heuristic_in_candidates = (
            heuristic is not None and any(action_key(action) == action_key(heuristic) for action in actions)
        )
        action_types = tuple(action.action_type.value for action in actions)
        score = float(preferred_rank.get(heuristic_type, 0))
        if not heuristic_in_candidates:
            score += 2.0
        if ActionType.OPEN_PACK.value in action_types:
            score += 0.25
        if ActionType.END_SHOP.value in action_types:
            score += 0.25
        scored.append(
            ScoredShopState(
                record=record,
                source_bot=str(record.get("source_bot", "")),
                ante=int(record.get("ante", getattr(state, "ante", 0)) or 0),
                money=int(record.get("money", getattr(state, "money", 0)) or 0),
                heuristic_action_type=heuristic_type,
                heuristic_in_candidates=heuristic_in_candidates,
                candidate_action_types=action_types,
                score=score,
            )
        )
    return scored


def _select_balanced(
    items: list[ScoredShopState],
    *,
    limit: int,
    seed: int,
    balance_fields: tuple[str, ...],
) -> list[ScoredShopState]:
    rng = random.Random(seed)
    decorated = [(item.score, rng.random(), item) for item in items]
    decorated.sort(key=lambda value: (-value[0], value[1]))
    remaining = [item for _, _, item in decorated]
    if not balance_fields:
        return remaining[:limit]

    tuple_counts: Counter[tuple[Any, ...]] = Counter()
    marginal_counts: Counter[tuple[str, Any]] = Counter()
    selected: list[ScoredShopState] = []
    while len(selected) < limit and remaining:
        best_index = min(
            range(len(remaining)),
            key=lambda index: _selection_balance_key(
                remaining[index],
                balance_fields=balance_fields,
                tuple_counts=tuple_counts,
                marginal_counts=marginal_counts,
            ),
        )
        item = remaining.pop(best_index)
        selected.append(item)
        tuple_key = tuple(getattr(item, field) for field in balance_fields)
        tuple_counts[tuple_key] += 1
        for field in balance_fields:
            marginal_counts[(field, getattr(item, field))] += 1
    return selected


def _selection_balance_key(
    item: ScoredShopState,
    *,
    balance_fields: tuple[str, ...],
    tuple_counts: Counter[tuple[Any, ...]],
    marginal_counts: Counter[tuple[str, Any]],
) -> tuple[float, ...]:
    marginal = [marginal_counts[(field, getattr(item, field))] for field in balance_fields]
    tuple_key = tuple(getattr(item, field) for field in balance_fields)
    return (
        float(max(marginal, default=0)),
        float(sum(marginal)),
        float(tuple_counts[tuple_key]),
        -float(item.score),
    )


def _metrics(
    *,
    scored: list[ScoredShopState],
    selected: list[ScoredShopState],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "input_records": [str(path) for path in args.input],
        "exclude_records": [str(path) for path in args.exclude_records],
        "records_seen": len(scored),
        "records_selected": len(selected),
        "excluded_record_keys": int(args.excluded_record_keys),
        "states_requested": int(args.states),
        "selection_seed": int(args.selection_seed),
        "heuristic_bot": args.heuristic_bot,
        "max_actions": int(args.max_actions),
        "candidate_action_types": [action.value for action in args.candidate_action_types],
        "candidate_priority": args.candidate_priority,
        "prefer_heuristic_action_types": list(args.prefer_heuristic_action_types),
        "require_heuristic_outside_candidates": bool(args.require_heuristic_outside_candidates),
        "balance_fields": list(args.balance_fields),
        "seen_by_source_bot": dict(sorted(Counter(item.source_bot for item in scored).items())),
        "selected_by_source_bot": dict(sorted(Counter(item.source_bot for item in selected).items())),
        "seen_by_ante": {str(key): value for key, value in sorted(Counter(item.ante for item in scored).items())},
        "selected_by_ante": {str(key): value for key, value in sorted(Counter(item.ante for item in selected).items())},
        "seen_by_heuristic_action_type": dict(sorted(Counter(item.heuristic_action_type for item in scored).items())),
        "selected_by_heuristic_action_type": dict(
            sorted(Counter(item.heuristic_action_type for item in selected).items())
        ),
        "seen_heuristic_outside_candidate_count": sum(not item.heuristic_in_candidates for item in scored),
        "selected_heuristic_outside_candidate_count": sum(not item.heuristic_in_candidates for item in selected),
        "mean_selected_score": (
            sum(item.score for item in selected) / len(selected) if selected else None
        ),
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Select targeted shop state snapshots for expensive relabeling.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--exclude-records", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--heuristic-bot", default="solver_shop_basic_play_bot")
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--candidate-action-types", default="buy,open_pack,end_shop")
    parser.add_argument("--candidate-priority", default="deep_advantage", choices=("legal", "deep_advantage"))
    parser.add_argument("--prefer-heuristic-action-types", default="sell,reroll,end_shop,open_pack,buy")
    parser.add_argument("--require-heuristic-outside-candidates", action="store_true")
    parser.add_argument("--balance-fields", default="source_bot,ante")
    args = parser.parse_args()

    try:
        args.candidate_action_types = _parse_action_types_csv(args.candidate_action_types)
        args.prefer_heuristic_action_types = tuple(
            item.strip().lower() for item in args.prefer_heuristic_action_types.split(",") if item.strip()
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.balance_fields = tuple(item.strip() for item in args.balance_fields.split(",") if item.strip())

    excluded = _exclude_keys(args.exclude_records)
    args.excluded_record_keys = len(excluded)
    records = _filter_excluded(_load_records(args.input), excluded)
    scored = _score_records(
        records,
        heuristic_bot=args.heuristic_bot,
        max_actions=args.max_actions,
        candidate_action_types=args.candidate_action_types,
        candidate_priority=args.candidate_priority,
        preferred_heuristic_types=args.prefer_heuristic_action_types,
    )
    if args.require_heuristic_outside_candidates:
        scored = [item for item in scored if not item.heuristic_in_candidates]
    selected = _select_balanced(
        scored,
        limit=max(0, int(args.states)),
        seed=int(args.selection_seed),
        balance_fields=args.balance_fields,
    )
    out_records = []
    for rank, item in enumerate(selected, start=1):
        record = dict(item.record)
        record["selection_rank"] = rank
        record["selection_score"] = item.score
        record["selection_heuristic_action_type"] = item.heuristic_action_type
        record["selection_heuristic_in_candidates"] = item.heuristic_in_candidates
        record["selection_candidate_action_types"] = list(item.candidate_action_types)
        out_records.append(record)

    metrics = _metrics(scored=scored, selected=selected, args=args)
    _atomic_write_jsonl(args.out, out_records)
    _atomic_write_json(args.metrics, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if out_records else 1


if __name__ == "__main__":
    raise SystemExit(main())
