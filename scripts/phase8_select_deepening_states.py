"""Select shallow-labeled shop states worth deeper relabeling.

This script is the adaptive step after a cheap confidence probe. It reads
candidate records that already contain paired rollout samples, finds candidate
actions with promising candidate-minus-heuristic evidence, and writes a compact
state-snapshot JSONL that ``phase8_shop_candidate_dataset.py --input-records``
can relabel with a deeper horizon or more CRN seeds.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import random
from typing import Any

from balatro_ai.ml.shop_candidate_dataset import paired_rollout_advantage


@dataclass(frozen=True, slots=True)
class DeepeningOpportunity:
    record: dict[str, Any]
    source_bot: str
    ante: int
    money: int
    terminal_won: bool | None
    selection_reason: str
    heuristic_action_type: str
    candidate_action_type: str
    candidate_action_key: str
    heuristic_action_key: str
    n: int
    mean_advantage: float
    sem: float
    lcb: float
    positive_rate: float
    score: float


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _best_opportunities(
    records: list[dict[str, Any]],
    *,
    z: float,
    min_mean_advantage: float,
    min_lcb: float,
    min_positive_rate: float,
    min_rollouts: int,
    max_sem: float | None,
    min_lcb_sem_ratio: float | None,
    candidate_action_types: tuple[str, ...],
    exclude_candidate_action_types: tuple[str, ...],
    excluded_candidate_keys: set[tuple[str, int, str]],
    preferred_candidate_types: tuple[str, ...],
) -> list[DeepeningOpportunity]:
    allowed_candidate_types = frozenset(candidate_action_types)
    excluded_candidate_types = frozenset(exclude_candidate_action_types)
    preferred_rank = {
        action_type: len(preferred_candidate_types) - index
        for index, action_type in enumerate(preferred_candidate_types)
    }
    opportunities: list[DeepeningOpportunity] = []
    for record in records:
        snapshot = record.get("state_snapshot")
        candidates = record.get("candidates", ())
        if not isinstance(snapshot, dict) or not isinstance(candidates, list):
            continue
        heuristic = next((candidate for candidate in candidates if candidate.get("is_heuristic_action")), None)
        if not isinstance(heuristic, dict):
            continue
        heuristic_key = str(heuristic.get("action_key", ""))
        heuristic_type = _action_type(heuristic)
        best: DeepeningOpportunity | None = None
        for candidate in candidates:
            candidate_key = str(candidate.get("action_key", "")) if isinstance(candidate, dict) else ""
            record_key = (str(record.get("seed", "")), int(record.get("state_index", 0) or 0), candidate_key)
            if not isinstance(candidate, dict) or candidate_key == heuristic_key:
                continue
            if record_key in excluded_candidate_keys:
                continue
            stats = paired_rollout_advantage(candidate, heuristic, z=z)
            if stats is None:
                continue
            if stats.n < min_rollouts:
                continue
            if stats.mean + 1e-12 < min_mean_advantage:
                continue
            if stats.lower_bound + 1e-12 < min_lcb:
                continue
            if stats.positive_rate + 1e-12 < min_positive_rate:
                continue
            if max_sem is not None and stats.sem > max_sem + 1e-12:
                continue
            if min_lcb_sem_ratio is not None:
                ratio = float("inf") if stats.sem <= 0.0 else stats.lower_bound / stats.sem
                if ratio + 1e-12 < min_lcb_sem_ratio:
                    continue
            candidate_type = _action_type(candidate)
            if allowed_candidate_types and candidate_type not in allowed_candidate_types:
                continue
            if candidate_type in excluded_candidate_types:
                continue
            score = (
                stats.lower_bound
                + 0.25 * stats.mean
                + 0.05 * stats.positive_rate
                + 0.01 * preferred_rank.get(candidate_type, 0)
            )
            opportunity = DeepeningOpportunity(
                record=record,
                source_bot=str(record.get("source_bot", "")),
                ante=int(record.get("ante", 0) or 0),
                money=int(record.get("money", 0) or 0),
                terminal_won=_optional_bool(record.get("terminal_won")),
                selection_reason=str(record.get("selection_reason", "")),
                heuristic_action_type=heuristic_type,
                candidate_action_type=candidate_type,
                candidate_action_key=str(candidate.get("action_key", "")),
                heuristic_action_key=heuristic_key,
                n=int(stats.n),
                mean_advantage=float(stats.mean),
                sem=float(stats.sem),
                lcb=float(stats.lower_bound),
                positive_rate=float(stats.positive_rate),
                score=float(score),
            )
            if best is None or opportunity.score > best.score:
                best = opportunity
        if best is not None:
            opportunities.append(best)
    return opportunities


def _select_balanced(
    opportunities: list[DeepeningOpportunity],
    *,
    limit: int,
    seed: int,
    balance_fields: tuple[str, ...],
) -> list[DeepeningOpportunity]:
    rng = random.Random(seed)
    groups: dict[tuple[Any, ...], list[DeepeningOpportunity]] = {}
    for opportunity in opportunities:
        key = tuple(getattr(opportunity, field) for field in balance_fields) if balance_fields else ("all",)
        groups.setdefault(key, []).append(opportunity)
    for items in groups.values():
        decorated = [(item.score, rng.random(), item) for item in items]
        decorated.sort(key=lambda item: (-item[0], item[1]))
        items[:] = [item for _, _, item in decorated]
    selected: list[DeepeningOpportunity] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            group = groups[key]
            if not group:
                continue
            selected.append(group.pop(0))
            if len(selected) >= limit:
                break
    return selected


def _action_type(candidate: dict[str, Any]) -> str:
    action = candidate.get("action", {})
    if isinstance(action, dict):
        return str(action.get("type", ""))
    return ""


def _finite_mean(values: list[float]) -> float | None:
    items = [float(value) for value in values if isfinite(float(value))]
    return sum(items) / len(items) if items else None


def _metrics(
    *,
    records: list[dict[str, Any]],
    opportunities: list[DeepeningOpportunity],
    selected: list[DeepeningOpportunity],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "input_records": [str(path) for path in args.input],
        "records_seen": len(records),
        "opportunity_records": len(opportunities),
        "records_selected": len(selected),
        "excluded_candidate_keys": len(args.excluded_candidate_keys),
        "states_requested": int(args.states),
        "selection_seed": int(args.selection_seed),
        "confidence_z": float(args.z),
        "min_mean_advantage": float(args.min_mean_advantage),
        "min_lcb": float(args.min_lcb),
        "min_positive_rate": float(args.min_positive_rate),
        "min_rollouts": int(args.min_rollouts),
        "max_sem": args.max_sem,
        "min_lcb_sem_ratio": args.min_lcb_sem_ratio,
        "candidate_action_types": list(args.candidate_action_types),
        "exclude_candidate_action_types": list(args.exclude_candidate_action_types),
        "preferred_candidate_types": list(args.preferred_candidate_types),
        "balance_fields": list(args.balance_fields),
        "opportunities_by_heuristic_action_type": dict(
            sorted(Counter(item.heuristic_action_type for item in opportunities).items())
        ),
        "selected_by_heuristic_action_type": dict(
            sorted(Counter(item.heuristic_action_type for item in selected).items())
        ),
        "opportunities_by_candidate_action_type": dict(
            sorted(Counter(item.candidate_action_type for item in opportunities).items())
        ),
        "selected_by_candidate_action_type": dict(
            sorted(Counter(item.candidate_action_type for item in selected).items())
        ),
        "selected_by_source_bot": dict(sorted(Counter(item.source_bot for item in selected).items())),
        "selected_by_ante": {str(key): value for key, value in sorted(Counter(item.ante for item in selected).items())},
        "opportunities_by_terminal_won": {
            str(key): int(value) for key, value in sorted(Counter(item.terminal_won for item in opportunities).items(), key=lambda item: str(item[0]))
        },
        "selected_by_terminal_won": {
            str(key): int(value) for key, value in sorted(Counter(item.terminal_won for item in selected).items(), key=lambda item: str(item[0]))
        },
        "opportunities_by_selection_reason": dict(
            sorted(Counter(item.selection_reason for item in opportunities).items())
        ),
        "selected_by_selection_reason": dict(sorted(Counter(item.selection_reason for item in selected).items())),
        "mean_selected_advantage": _finite_mean([item.mean_advantage for item in selected]),
        "mean_selected_lcb": _finite_mean([item.lcb for item in selected]),
        "mean_selected_positive_rate": _finite_mean([item.positive_rate for item in selected]),
        "mean_selected_rollouts": _finite_mean([float(item.n) for item in selected]),
        "min_selected_rollouts": min((item.n for item in selected), default=None),
        "max_selected_lcb": max((item.lcb for item in selected), default=None),
    }


def _atomic_write_jsonl(path: Path, opportunities: list[DeepeningOpportunity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for rank, opportunity in enumerate(opportunities, start=1):
            candidate_action = _candidate_action_payload(opportunity.record, opportunity.candidate_action_key)
            heuristic_action = _candidate_action_payload(opportunity.record, opportunity.heuristic_action_key)
            record = {
                "source_bot": opportunity.source_bot,
                "seed": str(opportunity.record.get("seed", "")),
                "state_index": int(opportunity.record.get("state_index", 0) or 0),
                "ante": opportunity.ante,
                "money": opportunity.money,
                "terminal_won": opportunity.terminal_won,
                "selection_reason": opportunity.selection_reason,
                "state_snapshot": opportunity.record["state_snapshot"],
                "deepening_rank": rank,
                "deepening_score": opportunity.score,
                "deepening_candidate_action_key": opportunity.candidate_action_key,
                "deepening_candidate_action_type": opportunity.candidate_action_type,
                "deepening_heuristic_action_key": opportunity.heuristic_action_key,
                "deepening_heuristic_action_type": opportunity.heuristic_action_type,
                "deepening_rollouts": opportunity.n,
                "deepening_mean_advantage": opportunity.mean_advantage,
                "deepening_sem": opportunity.sem,
                "deepening_lcb": opportunity.lcb,
                "deepening_positive_rate": opportunity.positive_rate,
            }
            if candidate_action is not None:
                record["deepening_candidate_action"] = candidate_action
            if heuristic_action is not None:
                record["deepening_heuristic_action"] = heuristic_action
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _candidate_action_payload(record: dict[str, Any], action_key: str) -> dict[str, Any] | None:
    candidates = record.get("candidates", ())
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict) or str(candidate.get("action_key", "")) != action_key:
            continue
        action = candidate.get("action")
        return dict(action) if isinstance(action, dict) else None
    return None


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in str(value or "").split(",") if item.strip())


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes"}:
            return True
        if token in {"false", "0", "no"}:
            return False
    return None


def _candidate_exclusion_keys(records: list[dict[str, Any]]) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    for record in records:
        seed = str(record.get("seed", ""))
        state_index = int(record.get("state_index", 0) or 0)
        deepening_key = str(record.get("deepening_candidate_action_key", "")).strip()
        if deepening_key:
            keys.add((seed, state_index, deepening_key))
        candidates = record.get("candidates", ())
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict) or candidate.get("is_heuristic_action"):
                    continue
                action_key = str(candidate.get("action_key", "")).strip()
                if action_key:
                    keys.add((seed, state_index, action_key))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Select shallow-labeled states worth deeper shop relabeling.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--exclude-records",
        type=Path,
        action="append",
        default=[],
        help="Deepening or solver-confirmed JSONL records whose candidate keys should not be selected again.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--min-mean-advantage", type=float, default=0.0)
    parser.add_argument("--min-lcb", type=float, default=0.0)
    parser.add_argument("--min-positive-rate", type=float, default=0.5)
    parser.add_argument("--min-rollouts", type=int, default=2)
    parser.add_argument(
        "--max-sem",
        type=float,
        default=None,
        help="Optional maximum cheap paired-advantage SEM before a candidate can be selected.",
    )
    parser.add_argument(
        "--min-lcb-sem-ratio",
        type=float,
        default=None,
        help="Optional minimum cheap lower-bound/SEM ratio before a candidate can be selected.",
    )
    parser.add_argument(
        "--candidate-action-types",
        default="",
        help="Comma-separated candidate action types to allow before selecting deepening states.",
    )
    parser.add_argument(
        "--exclude-candidate-action-types",
        default="",
        help="Comma-separated candidate action types to exclude before selecting deepening states.",
    )
    parser.add_argument("--preferred-candidate-types", default="open_pack,buy,end_shop")
    parser.add_argument("--balance-fields", default="heuristic_action_type,ante")
    args = parser.parse_args()
    args.candidate_action_types = _parse_csv(args.candidate_action_types)
    args.exclude_candidate_action_types = _parse_csv(args.exclude_candidate_action_types)
    args.preferred_candidate_types = tuple(
        item.strip().lower() for item in args.preferred_candidate_types.split(",") if item.strip()
    )
    args.balance_fields = tuple(item.strip() for item in args.balance_fields.split(",") if item.strip())

    records = _load_records(args.input)
    args.excluded_candidate_keys = _candidate_exclusion_keys(_load_records(args.exclude_records))
    opportunities = _best_opportunities(
        records,
        z=float(args.z),
        min_mean_advantage=float(args.min_mean_advantage),
        min_lcb=float(args.min_lcb),
        min_positive_rate=float(args.min_positive_rate),
        min_rollouts=max(1, int(args.min_rollouts)),
        max_sem=args.max_sem,
        min_lcb_sem_ratio=args.min_lcb_sem_ratio,
        candidate_action_types=args.candidate_action_types,
        exclude_candidate_action_types=args.exclude_candidate_action_types,
        excluded_candidate_keys=args.excluded_candidate_keys,
        preferred_candidate_types=args.preferred_candidate_types,
    )
    selected = _select_balanced(
        opportunities,
        limit=max(0, int(args.states)),
        seed=int(args.selection_seed),
        balance_fields=args.balance_fields,
    )
    metrics = _metrics(records=records, opportunities=opportunities, selected=selected, args=args)
    _atomic_write_jsonl(args.out, selected)
    _atomic_write_json(args.metrics, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
