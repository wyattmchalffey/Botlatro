"""Run repeated seed-split evaluations for Phase 8 shop rankers.

This is a lightweight guard against one lucky train/val split. It trains the
ranker across several split seeds and reports whether the neural chooser beats
the stored heuristic action on held-out regret and near-best accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from balatro_ai.ml.shop_ranker import (
    ShopRankerTrainConfig,
    confidence_advantage_label_summary,
    evaluate_advantage_overrides,
    evaluate_heuristic_baseline,
    examples_from_jsonl_paths,
    filter_examples_by_label_quality,
    label_quality_summary,
    parse_action_kinds_csv,
    parse_action_types_csv,
    split_examples,
    train_shop_ranker,
)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _field_values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _finite(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _summarize_rows(rows: list[dict]) -> dict:
    advantage_lifts = _field_values(rows, "advantage_lift_vs_baseline")
    advantage_regret_deltas = _field_values(rows, "advantage_regret_delta_vs_baseline")
    advantage_override_rates = _field_values(rows, "advantage_override_rate")
    advantage_harmful_override_rates = _field_values(rows, "advantage_harmful_override_rate")
    advantage_harmful_covered_rates = _field_values(rows, "advantage_harmful_covered_rate")
    advantage_helpful_covered_rates = _field_values(rows, "advantage_helpful_covered_rate")
    calibrated_val_lifts = _field_values(rows, "calibrated_val_lift_vs_baseline")
    calibrated_val_regret_deltas = _field_values(rows, "calibrated_val_regret_delta_vs_baseline")
    calibrated_val_override_rates = _field_values(rows, "calibrated_val_override_rate")
    calibrated_val_harmful_covered_rates = _field_values(rows, "calibrated_val_harmful_covered_rate")
    calibrated_val_helpful_covered_rates = _field_values(rows, "calibrated_val_helpful_covered_rate")
    calibrated_thresholds = _field_values(rows, "calibrated_threshold")
    return {
        "runs": len(rows),
        "mean_model_regret": _mean([float(row["model_regret"]) for row in rows]),
        "mean_heuristic_regret": _mean([float(row["heuristic_regret"]) for row in rows]),
        "model_regret_wins": sum(float(row["model_regret"]) < float(row["heuristic_regret"]) for row in rows),
        "mean_model_near_best_0_05": _mean([float(row["model_near_best_0_05"]) for row in rows]),
        "mean_heuristic_near_best_0_05": _mean([float(row["heuristic_near_best_0_05"]) for row in rows]),
        "model_near_best_0_05_wins": sum(
            float(row["model_near_best_0_05"]) > float(row["heuristic_near_best_0_05"])
            for row in rows
        ),
        "mean_model_acceptable_0_25": _mean([float(row["model_acceptable_0_25"]) for row in rows]),
        "mean_heuristic_acceptable_0_25": _mean([float(row["heuristic_acceptable_0_25"]) for row in rows]),
        "model_acceptable_0_25_wins": sum(
            float(row["model_acceptable_0_25"]) > float(row["heuristic_acceptable_0_25"])
            for row in rows
        ),
        "mean_model_top1": _mean([float(row["model_top1"]) for row in rows]),
        "mean_heuristic_top1": _mean([float(row["heuristic_top1"]) for row in rows]),
        "mean_advantage_lift_vs_baseline": _mean(advantage_lifts),
        "advantage_positive_runs": sum(value > 0.0 for value in advantage_lifts),
        "mean_advantage_regret_delta": _mean(advantage_regret_deltas),
        "advantage_regret_delta_wins": sum(value < 0.0 for value in advantage_regret_deltas),
        "mean_advantage_override_rate": _mean(advantage_override_rates),
        "mean_advantage_harmful_override_rate": _mean(advantage_harmful_override_rates),
        "mean_advantage_helpful_covered_rate": _mean(advantage_helpful_covered_rates),
        "mean_advantage_harmful_covered_rate": _mean(advantage_harmful_covered_rates),
        "mean_calibrated_threshold": _mean(calibrated_thresholds),
        "mean_calibrated_val_lift_vs_baseline": _mean(calibrated_val_lifts),
        "calibrated_val_positive_runs": sum(value > 0.0 for value in calibrated_val_lifts),
        "mean_calibrated_val_regret_delta_vs_baseline": _mean(calibrated_val_regret_deltas),
        "calibrated_val_regret_delta_wins": sum(value < 0.0 for value in calibrated_val_regret_deltas),
        "mean_calibrated_val_override_rate": _mean(calibrated_val_override_rates),
        "mean_calibrated_val_helpful_covered_rate": _mean(calibrated_val_helpful_covered_rates),
        "mean_calibrated_val_harmful_covered_rate": _mean(calibrated_val_harmful_covered_rates),
    }


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.3f}"


def _summarize_advantage_by_threshold(rows: list[dict], thresholds: list[float]) -> dict:
    summary: dict[str, dict] = {}
    for threshold in thresholds:
        key = _threshold_key(threshold)
        metrics = [
            row.get("advantage_by_threshold", {}).get(key)
            for row in rows
            if row.get("advantage_by_threshold", {}).get(key) is not None
        ]
        lifts = [float(metric["mean_lift_vs_baseline"]) for metric in metrics]
        regret_deltas = [float(metric["mean_regret_delta_vs_baseline"]) for metric in metrics]
        override_rates = [float(metric["override_rate"]) for metric in metrics]
        harmful_rates = [float(metric["harmful_override_rate"]) for metric in metrics]
        helpful_rates = [float(metric["helpful_override_rate"]) for metric in metrics]
        harmful_covered_rates = [float(metric["harmful_covered_rate"]) for metric in metrics]
        helpful_covered_rates = [float(metric["helpful_covered_rate"]) for metric in metrics]
        summary[key] = {
            "runs": len(metrics),
            "mean_lift_vs_baseline": _mean(lifts),
            "positive_runs": sum(value > 0.0 for value in lifts),
            "mean_regret_delta_vs_baseline": _mean(regret_deltas),
            "regret_delta_wins": sum(value < 0.0 for value in regret_deltas),
            "mean_override_rate": _mean(override_rates),
            "mean_helpful_override_rate": _mean(helpful_rates),
            "mean_harmful_override_rate": _mean(harmful_rates),
            "mean_helpful_covered_rate": _mean(helpful_covered_rates),
            "mean_harmful_covered_rate": _mean(harmful_covered_rates),
        }
    return summary


def _select_calibrated_threshold(
    metrics_by_threshold: dict[str, dict],
    *,
    max_harmful_covered_rate: float,
) -> tuple[str, dict]:
    """Pick a threshold from train metrics, preferring lift under a harm cap."""

    items = [
        (key, metric)
        for key, metric in metrics_by_threshold.items()
        if metric is not None and math.isfinite(_finite(metric.get("threshold"), default=float("nan")))
    ]
    if not items:
        return "", {}
    cap = max(0.0, float(max_harmful_covered_rate))
    eligible = [
        item for item in items if _finite(item[1].get("harmful_covered_rate"), default=float("inf")) <= cap
    ]
    if eligible:
        return max(
            eligible,
            key=lambda item: (
                _finite(item[1].get("mean_lift_vs_baseline"), default=-float("inf")),
                -_finite(item[1].get("harmful_covered_rate"), default=float("inf")),
                -_finite(item[1].get("override_rate"), default=float("inf")),
                -_finite(item[1].get("threshold"), default=float("inf")),
            ),
        )
    return min(
        items,
        key=lambda item: (
            _finite(item[1].get("harmful_covered_rate"), default=float("inf")),
            -_finite(item[1].get("mean_lift_vs_baseline"), default=-float("inf")),
            _finite(item[1].get("threshold"), default=float("inf")),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeated split evaluation for shop rankers.")
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--encoder", action="append", choices=("mean", "attention"), default=[])
    parser.add_argument("--split-seed", type=int, action="append", default=[])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--split-by", default="seed", choices=("seed", "random"))
    parser.add_argument(
        "--loss",
        default="soft",
        choices=(
            "soft",
            "hard",
            "acceptable",
            "pairwise",
            "advantage_mse",
            "advantage_tie_mse",
            "confidence_advantage_tie_mse",
        ),
    )
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--acceptable-margin", type=float, default=0.05)
    parser.add_argument("--pairwise-margin", type=float, default=0.10)
    parser.add_argument("--advantage-threshold", type=float, action="append", default=[])
    parser.add_argument("--calibration-max-harmful-covered-rate", type=float, default=0.05)
    parser.add_argument("--confidence-z", type=float, default=1.0)
    parser.add_argument("--confidence-margin", type=float, default=0.10)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--train-min-best-margin", type=float, default=0.0)
    parser.add_argument("--train-require-split-half-agreement", action="store_true")
    parser.add_argument("--train-max-actions-within-0-05", type=int, default=None)
    parser.add_argument("--train-max-actions-within-0-10", type=int, default=None)
    parser.add_argument("--train-min-best-runnerup-lcb", type=float, default=None)
    parser.add_argument("--train-min-best-vs-baseline-lcb", type=float, default=None)
    parser.add_argument("--train-min-high-conf-override-candidates", type=int, default=None)
    parser.add_argument("--train-min-high-conf-practical-override-candidates", type=int, default=None)
    parser.add_argument(
        "--candidate-action-types",
        default="",
        help="Comma-separated candidate action types to keep before ranking.",
    )
    parser.add_argument(
        "--candidate-action-kinds",
        default="",
        help="Comma-separated candidate metadata kinds to keep before ranking, e.g. card,pack.",
    )
    parser.add_argument(
        "--keep-heuristic-action",
        action="store_true",
        help="Keep the heuristic action as a baseline candidate even when candidate action filtering would remove it.",
    )
    args = parser.parse_args()

    allowed_action_types = parse_action_types_csv(args.candidate_action_types)
    allowed_action_kinds = parse_action_kinds_csv(args.candidate_action_kinds)
    examples = examples_from_jsonl_paths(
        args.data,
        allowed_action_types=allowed_action_types or None,
        allowed_action_kinds=allowed_action_kinds or None,
        keep_heuristic_action=bool(args.keep_heuristic_action),
        confidence_z=args.confidence_z,
        practical_margin=args.confidence_margin,
    )
    if not examples:
        raise SystemExit(f"{args.data}: no trainable examples")
    encoders = args.encoder or ["attention"]
    split_seeds = args.split_seed or [1, 2, 3, 4, 5, 7, 11]
    advantage_thresholds = args.advantage_threshold or [0.0]
    primary_advantage_threshold = float(advantage_thresholds[0])
    primary_advantage_key = _threshold_key(primary_advantage_threshold)
    rows: list[dict] = []
    for encoder in encoders:
        for split_seed in split_seeds:
            train_examples, val_examples = split_examples(
                examples,
                args.val_frac,
                seed=split_seed,
                split_by=args.split_by,
            )
            fit_examples = filter_examples_by_label_quality(
                train_examples,
                min_best_margin=args.train_min_best_margin,
                require_split_half_agreement=bool(args.train_require_split_half_agreement),
                max_actions_within_0_05=args.train_max_actions_within_0_05,
                max_actions_within_0_10=args.train_max_actions_within_0_10,
                min_best_runnerup_lcb=args.train_min_best_runnerup_lcb,
                min_best_vs_baseline_lcb=args.train_min_best_vs_baseline_lcb,
                min_high_conf_override_candidates=args.train_min_high_conf_override_candidates,
                min_high_conf_practical_override_candidates=args.train_min_high_conf_practical_override_candidates,
            )
            if not fit_examples or not val_examples:
                continue
            config = ShopRankerTrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                seed=split_seed,
                encoder=encoder,
                loss=args.loss,
                target_temperature=args.target_temperature,
                acceptable_margin=args.acceptable_margin,
                pairwise_margin=args.pairwise_margin,
                val_every=args.val_every,
            )
            result = train_shop_ranker(fit_examples, config, val_examples=val_examples)
            heuristic_val = evaluate_heuristic_baseline(val_examples)
            final_val = result.final_val or {}
            advantage_by_threshold = evaluate_advantage_overrides(
                result.model,
                val_examples,
                thresholds=advantage_thresholds,
            )
            train_advantage_by_threshold = evaluate_advantage_overrides(
                result.model,
                fit_examples,
                thresholds=advantage_thresholds,
            )
            calibrated_key, calibrated_train = _select_calibrated_threshold(
                train_advantage_by_threshold,
                max_harmful_covered_rate=args.calibration_max_harmful_covered_rate,
            )
            calibrated_val = advantage_by_threshold.get(calibrated_key, {})
            advantage_val = advantage_by_threshold[primary_advantage_key]
            rows.append(
                {
                    "encoder": encoder,
                    "split_seed": split_seed,
                    "n_train_unfiltered": len(train_examples),
                    "n_train": len(fit_examples),
                    "n_val": len(val_examples),
                    "best_epoch": result.best_epoch,
                    "model_regret": final_val.get("mean_regret"),
                    "model_near_best_0_05": final_val.get("near_best_acc_0_05"),
                    "model_near_best_0_10": final_val.get("near_best_acc_0_10"),
                    "model_acceptable_0_25": final_val.get("acceptable_acc_0_25"),
                    "model_top1": final_val.get("top1_acc"),
                    "heuristic_regret": heuristic_val.get("mean_regret"),
                    "heuristic_near_best_0_05": heuristic_val.get("near_best_acc_0_05"),
                    "heuristic_near_best_0_10": heuristic_val.get("near_best_acc_0_10"),
                    "heuristic_acceptable_0_25": heuristic_val.get("acceptable_acc_0_25"),
                    "heuristic_top1": heuristic_val.get("top1_acc"),
                    "advantage_threshold": primary_advantage_threshold,
                    "advantage_lift_vs_baseline": advantage_val.get("mean_lift_vs_baseline"),
                    "advantage_regret_delta_vs_baseline": advantage_val.get("mean_regret_delta_vs_baseline"),
                    "advantage_chosen_regret": advantage_val.get("mean_chosen_regret"),
                    "advantage_baseline_regret": advantage_val.get("mean_baseline_regret"),
                    "advantage_override_rate": advantage_val.get("override_rate"),
                    "advantage_helpful_override_rate": advantage_val.get("helpful_override_rate"),
                    "advantage_harmful_override_rate": advantage_val.get("harmful_override_rate"),
                    "advantage_helpful_covered_rate": advantage_val.get("helpful_covered_rate"),
                    "advantage_harmful_covered_rate": advantage_val.get("harmful_covered_rate"),
                    "calibrated_threshold": calibrated_train.get("threshold"),
                    "calibrated_train_lift_vs_baseline": calibrated_train.get("mean_lift_vs_baseline"),
                    "calibrated_train_override_rate": calibrated_train.get("override_rate"),
                    "calibrated_train_helpful_covered_rate": calibrated_train.get("helpful_covered_rate"),
                    "calibrated_train_harmful_covered_rate": calibrated_train.get("harmful_covered_rate"),
                    "calibrated_val_lift_vs_baseline": calibrated_val.get("mean_lift_vs_baseline"),
                    "calibrated_val_regret_delta_vs_baseline": calibrated_val.get(
                        "mean_regret_delta_vs_baseline"
                    ),
                    "calibrated_val_override_rate": calibrated_val.get("override_rate"),
                    "calibrated_val_helpful_covered_rate": calibrated_val.get("helpful_covered_rate"),
                    "calibrated_val_harmful_covered_rate": calibrated_val.get("harmful_covered_rate"),
                    "train_advantage_by_threshold": train_advantage_by_threshold,
                    "advantage_by_threshold": advantage_by_threshold,
                }
            )

    by_encoder = {
        encoder: _summarize_rows([row for row in rows if row["encoder"] == encoder])
        for encoder in encoders
    }
    advantage_by_encoder = {
        encoder: _summarize_advantage_by_threshold(
            [row for row in rows if row["encoder"] == encoder],
            [float(threshold) for threshold in advantage_thresholds],
        )
        for encoder in encoders
    }
    metrics = {
        "data": [str(path) for path in args.data],
        "n_examples": len(examples),
        "quality_all": label_quality_summary(examples),
        "confidence_labels_all": confidence_advantage_label_summary(examples, margin=args.acceptable_margin),
        "split_seeds": split_seeds,
        "encoders": encoders,
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "val_frac": args.val_frac,
            "split_by": args.split_by,
            "loss": args.loss,
            "target_temperature": args.target_temperature,
            "acceptable_margin": args.acceptable_margin,
            "pairwise_margin": args.pairwise_margin,
            "advantage_threshold": primary_advantage_threshold,
            "advantage_thresholds": [float(threshold) for threshold in advantage_thresholds],
            "calibration_max_harmful_covered_rate": args.calibration_max_harmful_covered_rate,
            "confidence_z": args.confidence_z,
            "confidence_margin": args.confidence_margin,
            "val_every": args.val_every,
        },
        "quality_filter": {
            "candidate_action_types": sorted(action_type.value for action_type in allowed_action_types),
            "candidate_action_kinds": sorted(allowed_action_kinds),
            "keep_heuristic_action": bool(args.keep_heuristic_action),
            "train_min_best_margin": args.train_min_best_margin,
            "train_require_split_half_agreement": bool(args.train_require_split_half_agreement),
            "train_max_actions_within_0_05": args.train_max_actions_within_0_05,
            "train_max_actions_within_0_10": args.train_max_actions_within_0_10,
            "train_min_best_runnerup_lcb": args.train_min_best_runnerup_lcb,
            "train_min_best_vs_baseline_lcb": args.train_min_best_vs_baseline_lcb,
            "train_min_high_conf_override_candidates": args.train_min_high_conf_override_candidates,
            "train_min_high_conf_practical_override_candidates": (
                args.train_min_high_conf_practical_override_candidates
            ),
        },
        "summary_by_encoder": by_encoder,
        "advantage_summary_by_encoder": advantage_by_encoder,
        "rows": rows,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.metrics.with_suffix(args.metrics.suffix + ".tmp")
    tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    tmp.replace(args.metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
