"""Audit paired rollout confidence for Phase 8 shop candidate labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from balatro_ai.ml.shop_candidate_dataset import rollout_confidence_summary


def _load_jsonl(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(json.loads(line))
    return records


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _count_by(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        label = "None" if value is None else str(value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _candidate_mean(candidate: dict) -> float:
    return _finite(candidate.get("mean_value"), default=0.0)


def _heuristic_mean_gap_stats(records: list[dict], practical_margin: float) -> dict[str, float]:
    covered = 0
    within_margin = 0
    positive = 0
    gaps: list[float] = []
    for record in records:
        candidates = list(record.get("candidates") or [])
        if len(candidates) < 2:
            continue
        heuristic = next((candidate for candidate in candidates if bool(candidate.get("is_heuristic_action"))), None)
        if heuristic is None:
            continue
        best = max(candidates, key=_candidate_mean)
        gap = _candidate_mean(best) - _candidate_mean(heuristic)
        covered += 1
        gaps.append(gap)
        if gap > 1e-9:
            positive += 1
        if gap <= practical_margin:
            within_margin += 1
    return {
        "heuristic_mean_gap_covered_n": covered,
        "mean_best_mean_gap_vs_heuristic": sum(gaps) / len(gaps) if gaps else float("nan"),
        "oracle_positive_mean_gap_vs_heuristic_rate": positive / covered if covered else float("nan"),
        "heuristic_within_practical_margin_rate": within_margin / covered if covered else float("nan"),
    }


def block_quality_summary(
    records: list[dict],
    confidence: dict,
    *,
    practical_margin: float = 0.10,
    min_records: int = 8,
    min_practical_rate: float = 0.20,
    min_mean_lcb: float = 0.02,
    calibration_heuristic_close_rate: float = 0.70,
) -> dict:
    """Classify a labeled block before using it as ranker training signal.

    The classification is deliberately conservative. A block can be useful
    calibration without being good positive-acquisition data, and wins are
    retained as metadata instead of being filtered out here.
    """

    practical_margin = max(0.0, float(practical_margin))
    min_records = max(1, int(min_records))
    min_practical_rate = max(0.0, float(min_practical_rate))
    min_mean_lcb = float(min_mean_lcb)
    calibration_heuristic_close_rate = max(0.0, float(calibration_heuristic_close_rate))

    terminal_won_counts = _count_by(records, "terminal_won")
    selection_reason_counts = _count_by(records, "selection_reason")
    gap_stats = _heuristic_mean_gap_stats(records, practical_margin)

    records_n = len(records)
    win_records = terminal_won_counts.get("True", 0)
    practical_best_rate = _finite(confidence.get("best_practical_high_conf_beats_heuristic_rate"), default=0.0)
    any_practical_rate = _finite(confidence.get("any_high_conf_practical_override_candidate_rate"), default=0.0)
    mean_lcb = _finite(confidence.get("mean_best_lcb_advantage_vs_heuristic"), default=0.0)
    heuristic_close_rate = _finite(gap_stats.get("heuristic_within_practical_margin_rate"), default=0.0)

    enough_records = records_n >= min_records
    strong_signal = (
        enough_records
        and mean_lcb >= min_mean_lcb
        and (practical_best_rate >= min_practical_rate or any_practical_rate >= min_practical_rate)
    )
    calibration_only = (
        enough_records
        and not strong_signal
        and heuristic_close_rate >= calibration_heuristic_close_rate
        and any_practical_rate <= 0.0
        and mean_lcb <= 0.0
    )
    if strong_signal:
        decision = "strong_signal"
        recommendation = "select_and_confirm_candidates"
    elif calibration_only:
        decision = "calibration_only"
        recommendation = "keep_for_calibration_or_holdout"
    else:
        decision = "weak_or_mixed"
        recommendation = "audit_before_training"

    return {
        "decision": decision,
        "recommendation": recommendation,
        "records": records_n,
        "min_records": min_records,
        "records_by_terminal_won": terminal_won_counts,
        "records_by_selection_reason": selection_reason_counts,
        "terminal_win_record_rate": win_records / records_n if records_n else float("nan"),
        "best_practical_high_conf_beats_heuristic_rate": practical_best_rate,
        "any_high_conf_practical_override_candidate_rate": any_practical_rate,
        "mean_best_lcb_advantage_vs_heuristic": mean_lcb,
        "heuristic_within_practical_margin_rate": heuristic_close_rate,
        "thresholds": {
            "practical_margin": practical_margin,
            "min_practical_rate": min_practical_rate,
            "min_mean_lcb": min_mean_lcb,
            "calibration_heuristic_close_rate": calibration_heuristic_close_rate,
        },
        **gap_stats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit confidence of paired CRN shop rollout labels.")
    parser.add_argument("--input", action="append", type=Path, required=True, help="Shop candidate JSONL; may repeat.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON metrics output path.")
    parser.add_argument("--z", type=float, default=1.0, help="Z multiplier for lower/upper confidence bounds.")
    parser.add_argument("--practical-margin", type=float, default=0.10)
    parser.add_argument("--quality-min-records", type=int, default=8)
    parser.add_argument("--quality-min-practical-rate", type=float, default=0.20)
    parser.add_argument("--quality-min-mean-lcb", type=float, default=0.02)
    parser.add_argument("--quality-calibration-heuristic-close-rate", type=float, default=0.70)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    records = _load_jsonl(args.input)
    metrics = rollout_confidence_summary(records, z=args.z, practical_margin=args.practical_margin)
    metrics["block_quality"] = block_quality_summary(
        records,
        metrics,
        practical_margin=args.practical_margin,
        min_records=args.quality_min_records,
        min_practical_rate=args.quality_min_practical_rate,
        min_mean_lcb=args.quality_min_mean_lcb,
        calibration_heuristic_close_rate=args.quality_calibration_heuristic_close_rate,
    )
    metrics.update(
        {
            "records": len(records),
            "input_records": [str(path) for path in args.input],
            "wall_s": round(time.perf_counter() - started, 3),
        }
    )
    if args.out is not None:
        _atomic_write_json(args.out, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
