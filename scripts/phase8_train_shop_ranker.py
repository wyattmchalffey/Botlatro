"""Train the Phase 8 neural shop candidate ranker.

    python scripts/phase8_train_shop_ranker.py \
        --data .data/phase8_shop_candidates_smoke.jsonl \
        --epochs 50 --batch-size 8 \
        --ckpt .data/phase8_shop_ranker_smoke.pt \
        --metrics .data/phase8_shop_ranker_smoke.metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from balatro_ai.ml.shop_ranker import (
    ShopRankerTrainConfig,
    confidence_advantage_label_summary,
    evaluate_advantage_overrides,
    evaluate_heuristic_baseline,
    evaluate_shop_ranker,
    examples_from_jsonl_paths,
    filter_examples_by_label_quality,
    parse_action_kinds_csv,
    label_quality_summary,
    parse_action_types_csv,
    save_checkpoint,
    split_examples,
    train_shop_ranker,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a shop candidate-ranker model.")
    parser.add_argument("--data", type=Path, action="append", required=True, help="Candidate-ranker JSONL dataset; may repeat.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Output checkpoint path.")
    parser.add_argument("--metrics", type=Path, required=True, help="Output metrics JSON path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-trunk", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--encoder", default="mean", choices=("mean", "attention"))
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
    parser.add_argument("--target-temperature", type=float, default=0.10)
    parser.add_argument("--acceptable-margin", type=float, default=0.05)
    parser.add_argument("--pairwise-margin", type=float, default=0.10)
    parser.add_argument("--advantage-threshold", type=float, action="append", default=[])
    parser.add_argument("--confidence-z", type=float, default=1.0)
    parser.add_argument("--confidence-margin", type=float, default=0.10)
    parser.add_argument("--split-by", default="seed", choices=("seed", "random"))
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
        help="Comma-separated candidate action types to keep before ranking, e.g. buy,open_pack,end_shop.",
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
    train_examples, val_examples = split_examples(examples, args.val_frac, seed=args.seed, split_by=args.split_by)
    if not train_examples:
        train_examples, val_examples = examples, []
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
    if not fit_examples:
        raise SystemExit("label-quality filter removed every training example")
    config = ShopRankerTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        d_trunk=args.d_trunk,
        dropout=args.dropout,
        encoder=args.encoder,
        loss=args.loss,
        target_temperature=args.target_temperature,
        acceptable_margin=args.acceptable_margin,
        pairwise_margin=args.pairwise_margin,
        val_every=args.val_every,
    )
    result = train_shop_ranker(fit_examples, config, val_examples=val_examples)
    advantage_thresholds = tuple(args.advantage_threshold or [0.0, 0.05, 0.10])
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(result.model, args.ckpt)
    metrics = {
        "data": [str(path) for path in args.data],
        "ckpt": str(args.ckpt),
        "n_examples": len(examples),
        "n_train": len(fit_examples),
        "n_train_unfiltered": len(train_examples),
        "n_val": len(val_examples),
        "split_by": args.split_by,
        "n_train_seeds": len({example.seed for example in fit_examples if example.seed}),
        "n_train_unfiltered_seeds": len({example.seed for example in train_examples if example.seed}),
        "n_val_seeds": len({example.seed for example in val_examples if example.seed}),
        "quality_filter": {
            "candidate_action_types": sorted(action_type.value for action_type in allowed_action_types),
            "candidate_action_kinds": sorted(allowed_action_kinds),
            "keep_heuristic_action": bool(args.keep_heuristic_action),
            "train_min_best_margin": args.train_min_best_margin,
            "train_require_split_half_agreement": bool(args.train_require_split_half_agreement),
            "train_max_actions_within_0_05": args.train_max_actions_within_0_05,
            "train_max_actions_within_0_10": args.train_max_actions_within_0_10,
            "confidence_z": args.confidence_z,
            "confidence_margin": args.confidence_margin,
            "train_min_best_runnerup_lcb": args.train_min_best_runnerup_lcb,
            "train_min_best_vs_baseline_lcb": args.train_min_best_vs_baseline_lcb,
            "train_min_high_conf_override_candidates": args.train_min_high_conf_override_candidates,
            "train_min_high_conf_practical_override_candidates": (
                args.train_min_high_conf_practical_override_candidates
            ),
        },
        "quality_all": label_quality_summary(examples),
        "quality_train_unfiltered": label_quality_summary(train_examples),
        "quality_train": label_quality_summary(fit_examples),
        "quality_val": label_quality_summary(val_examples),
        "confidence_labels_all": confidence_advantage_label_summary(examples, margin=args.acceptable_margin),
        "confidence_labels_train_unfiltered": confidence_advantage_label_summary(
            train_examples,
            margin=args.acceptable_margin,
        ),
        "confidence_labels_train": confidence_advantage_label_summary(fit_examples, margin=args.acceptable_margin),
        "confidence_labels_val": confidence_advantage_label_summary(val_examples, margin=args.acceptable_margin),
        "config": config.__dict__,
        "final_train": result.final_train,
        "final_train_unfiltered": evaluate_shop_ranker(
            result.model,
            train_examples,
            target_temperature=args.target_temperature,
        ),
        "final_val": result.final_val,
        "best_epoch": result.best_epoch,
        "best_val": result.best_val,
        "final_all": evaluate_shop_ranker(result.model, examples, target_temperature=args.target_temperature),
        "advantage_thresholds": list(advantage_thresholds),
        "advantage_train": evaluate_advantage_overrides(
            result.model,
            fit_examples,
            thresholds=advantage_thresholds,
        ),
        "advantage_train_unfiltered": evaluate_advantage_overrides(
            result.model,
            train_examples,
            thresholds=advantage_thresholds,
        ),
        "advantage_val": evaluate_advantage_overrides(
            result.model,
            val_examples,
            thresholds=advantage_thresholds,
        ),
        "advantage_all": evaluate_advantage_overrides(
            result.model,
            examples,
            thresholds=advantage_thresholds,
        ),
        "heuristic_train": evaluate_heuristic_baseline(fit_examples),
        "heuristic_train_unfiltered": evaluate_heuristic_baseline(train_examples),
        "heuristic_val": evaluate_heuristic_baseline(val_examples),
        "heuristic_all": evaluate_heuristic_baseline(examples),
        "history_tail": result.history[-10:],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.metrics.with_suffix(args.metrics.suffix + ".tmp")
    tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    tmp.replace(args.metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
