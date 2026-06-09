"""Audit cheap deepening proposals against solver-confirmed labels."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any

from balatro_ai.ml.shop_candidate_dataset import paired_rollout_advantage


@dataclass(frozen=True, slots=True)
class ConfirmationRow:
    seed: str
    state_index: int
    candidate_action_type: str
    heuristic_action_type: str
    cheap_rollouts: int
    cheap_mean: float
    cheap_sem: float
    cheap_lcb: float
    cheap_positive_rate: float
    ranker_margin: float | None
    ranker_baseline_margin: float | None
    solver_mean: float
    solver_sem: float
    solver_lcb: float
    solver_ucb: float
    solver_positive_rate: float
    solver_label: str


def _load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _join_confirmations(
    *,
    deepening_records: list[dict[str, Any]],
    solver_records: list[dict[str, Any]],
    margin: float,
) -> list[ConfirmationRow]:
    proposals: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in deepening_records:
        key = (
            str(record.get("seed", "")),
            int(record.get("state_index", 0) or 0),
            str(record.get("deepening_candidate_action_key", "")),
        )
        proposals[key] = record

    rows: list[ConfirmationRow] = []
    for record in solver_records:
        candidates = record.get("candidates", ())
        if not isinstance(candidates, list):
            continue
        heuristic = next((candidate for candidate in candidates if candidate.get("is_heuristic_action")), None)
        if not isinstance(heuristic, dict):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("is_heuristic_action"):
                continue
            key = (
                str(record.get("seed", "")),
                int(record.get("state_index", 0) or 0),
                str(candidate.get("action_key", "")),
            )
            proposal = proposals.get(key)
            if proposal is None:
                continue
            stats = paired_rollout_advantage(candidate, heuristic, z=1.0)
            if stats is None:
                continue
            label = "ambiguous"
            if stats.lower_bound > margin:
                label = "positive"
            elif stats.upper_bound < -abs(margin):
                label = "negative"
            rows.append(
                ConfirmationRow(
                    seed=key[0],
                    state_index=key[1],
                    candidate_action_type=str(proposal.get("deepening_candidate_action_type", "")),
                    heuristic_action_type=str(proposal.get("deepening_heuristic_action_type", "")),
                    cheap_rollouts=int(proposal.get("deepening_rollouts", 0) or 0),
                    cheap_mean=float(proposal.get("deepening_mean_advantage", 0.0) or 0.0),
                    cheap_sem=float(proposal.get("deepening_sem", 0.0) or 0.0),
                    cheap_lcb=float(proposal.get("deepening_lcb", 0.0) or 0.0),
                    cheap_positive_rate=float(proposal.get("deepening_positive_rate", 0.0) or 0.0),
                    ranker_margin=_optional_float(proposal.get("ranker_margin")),
                    ranker_baseline_margin=_optional_float(proposal.get("ranker_baseline_margin")),
                    solver_mean=float(stats.mean),
                    solver_sem=float(stats.sem),
                    solver_lcb=float(stats.lower_bound),
                    solver_ucb=float(stats.upper_bound),
                    solver_positive_rate=float(stats.positive_rate),
                    solver_label=label,
                )
            )
    return rows


def _optional_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def _label_counts(rows: list[ConfirmationRow]) -> dict[str, int]:
    counts = Counter(row.solver_label for row in rows)
    return {key: int(counts.get(key, 0)) for key in ("positive", "negative", "ambiguous")}


def _mean(values: list[float]) -> float | None:
    items = [float(value) for value in values if isfinite(float(value))]
    return float(sum(items) / len(items)) if items else None


def _label_stats(rows: list[ConfirmationRow]) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for label in ("positive", "negative", "ambiguous"):
        subset = [row for row in rows if row.solver_label == label]
        out[label] = {
            "count": len(subset),
            "mean_cheap_lcb": _mean([row.cheap_lcb for row in subset]),
            "mean_cheap_sem": _mean([row.cheap_sem for row in subset]),
            "mean_cheap_positive_rate": _mean([row.cheap_positive_rate for row in subset]),
            "mean_ranker_margin": _mean_optional([row.ranker_margin for row in subset]),
            "mean_ranker_baseline_margin": _mean_optional([row.ranker_baseline_margin for row in subset]),
            "mean_solver_lcb": _mean([row.solver_lcb for row in subset]),
        }
    return out


def _mean_optional(values: list[float | None]) -> float | None:
    return _mean([value for value in values if value is not None])


def _filter_summaries(rows: list[ConfirmationRow], *, max_sem_values: list[float]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for max_sem in max_sem_values:
        subset = [row for row in rows if row.cheap_sem <= max_sem]
        counts = _label_counts(subset)
        summaries[f"max_sem_{max_sem:.3f}"] = {
            "max_sem": float(max_sem),
            "kept": len(subset),
            **counts,
            "positive_precision": float(counts["positive"] / len(subset)) if subset else None,
        }
    return summaries


def _by_action_label(rows: list[ConfirmationRow]) -> dict[str, int]:
    counts = Counter(f"{row.candidate_action_type}:{row.solver_label}" for row in rows)
    return {key: int(value) for key, value in sorted(counts.items())}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit cheap deepening proposals against solver confirmations.")
    parser.add_argument("--deepening", type=Path, action="append", required=True)
    parser.add_argument("--solver", type=Path, action="append", required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--max-sem", type=float, action="append", default=[])
    args = parser.parse_args(argv)

    deepening_records = _load_jsonl(args.deepening)
    solver_records = _load_jsonl(args.solver)
    rows = _join_confirmations(
        deepening_records=deepening_records,
        solver_records=solver_records,
        margin=max(0.0, float(args.margin)),
    )
    max_sem_values = args.max_sem or [0.45, 0.55, 0.65, 0.80]
    metrics: dict[str, Any] = {
        "deepening": [str(path) for path in args.deepening],
        "solver": [str(path) for path in args.solver],
        "margin": max(0.0, float(args.margin)),
        "deepening_records": len(deepening_records),
        "solver_records": len(solver_records),
        "joined_records": len(rows),
        "label_counts": _label_counts(rows),
        "label_stats": _label_stats(rows),
        "by_candidate_action_and_label": _by_action_label(rows),
        "filter_summaries": _filter_summaries(rows, max_sem_values=[float(value) for value in max_sem_values]),
    }
    if args.metrics is not None:
        _atomic_write_json(args.metrics, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
