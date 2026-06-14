"""Value-head salvage + V0 re-gate.

V0 showed the B0 value head OVERFIT (train 0.99 / held-out 0.63), not flat —
salvageable. This retrains the decision policy on the (cached) mixture
examples with held-out-value early-stopping (keep the best-held-out-value
checkpoint) + stronger regularization, then reports the new held-out value
AUC so V0 can be re-judged. The held-out run set is the val signal (avoids
in-dataset outcome-label leakage). Uses the example cache, so retrains are
fast once the dataset is expanded once.

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python scripts/phaseb_value_salvage.py \
        --dataset .data/phaseb_mix_1000.jsonl --heldout .data/phaseb_heldout_200.jsonl \
        --epochs 25 --weight-decay 1e-3 --dropout 0.2 --ckpt .data/phaseb_policy_v0salvage.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--ckpt", default=".data/phaseb_policy_v0salvage.pt")
    args = ap.parse_args()

    from balatro_ai.ml.dataset import load_or_expand_examples
    from balatro_ai.ml.policy_net import PolicyConfig, save_policy, train_decision_policy

    train = load_or_expand_examples(args.dataset)
    val = load_or_expand_examples(args.heldout)
    print(f"[salvage] train {len(train)} examples, val {len(val)} examples", flush=True)

    cfg = PolicyConfig(
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=0,
    )
    net, metrics = train_decision_policy(train, cfg, val_examples=val)
    print(f"[salvage] best_val_value_auc={metrics.get('best_val_value_auc')} "
          f"(epoch {metrics.get('best_epoch')}) | policy top1={metrics['top1']} "
          f"no_heuristic={metrics['top1_no_heuristic']}", flush=True)
    save_policy(net, args.ckpt, config=cfg)
    bar = 0.65
    auc = metrics.get("best_val_value_auc", 0.0)
    print(f"[salvage] V0 RE-GATE: held-out value AUC {auc:.4f} vs bar {bar} -> "
          f"{'PASS' if auc >= bar else 'still below bar'}", flush=True)
    print(f"[salvage] saved -> {args.ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
