"""Stage 1.1 + 1.2 driver: bootstrap a teacher dataset, train the value head.

Generates `basic_strategy_bot` captures (resumable), splits BY RUN (so train/val
share no run — avoids leakage since every step in a run carries the same outcome
label), trains `ValueNet` (win-prob + dense expected-ante heads, with dropout +
weight-decay), evaluates on held-out runs, saves a checkpoint, and writes metrics.

Re-running after the captures already exist skips generation and just retrains.

    PYTHONPATH=src python scripts/phase8_bootstrap_train.py \
        --seeds 512 --workers 14 --epochs 20 \
        --out .data/phase8-bootstrap-basic.jsonl \
        --ckpt .data/phase8_value_v3_bootstrap.pt --metrics .data/phase8_stage1_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="basic_strategy_bot")
    ap.add_argument("--seeds", type=int, default=512)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ante-loss-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    from balatro_ai.ml import bootstrap as bs
    from balatro_ai.ml.dataset import examples_from_capture
    from balatro_ai.ml.train import TrainConfig, evaluate_full, save_checkpoint, train

    seeds = bs._numeric_seeds(args.seeds)
    stats = bs.generate_captures(seeds, args.out, bot_name=args.bot, workers=args.workers)
    print("[stage1] bootstrap:", json.dumps(stats.to_dict()), flush=True)

    caps = list(bs.read_captures(args.out))
    rng = random.Random(0)
    rng.shuffle(caps)
    n_val = max(1, int(len(caps) * args.val_frac))
    val_caps, train_caps = caps[:n_val], caps[n_val:]
    train_ex = [e for c in train_caps for e in examples_from_capture(c)]
    val_ex = [e for c in val_caps for e in examples_from_capture(c)]
    print(f"[stage1] runs train/val = {len(train_caps)}/{len(val_caps)}; "
          f"examples train/val = {len(train_ex)}/{len(val_ex)}", flush=True)

    res = train(train_ex, TrainConfig(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        weight_decay=args.weight_decay, dropout=args.dropout,
        ante_loss_weight=args.ante_loss_weight), val_examples=val_ex)

    tr = evaluate_full(res.model, train_ex)
    va = evaluate_full(res.model, val_ex)
    pos_rate = sum(1 for e in val_ex if e.value.won) / max(1, len(val_ex))
    save_checkpoint(res.model, args.ckpt)

    def _round(d: dict) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "n"}

    metrics = {
        "bot": args.bot,
        "n_runs": stats.n_runs,
        "run_win_rate": round(stats.win_rate, 4),
        "mean_final_ante": round(stats.mean_final_ante, 3),
        "examples_train": len(train_ex),
        "examples_val": len(val_ex),
        "val_example_pos_rate": round(pos_rate, 4),
        "train": _round(tr),
        "val": _round(va),
        "config": {"epochs": args.epochs, "dropout": args.dropout,
                   "weight_decay": args.weight_decay, "ante_loss_weight": args.ante_loss_weight},
        "ckpt": args.ckpt,
    }
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("[stage1] metrics:", json.dumps(metrics), flush=True)

    marker = (f".data/_STAGE1_runs{stats.n_runs}_wr{stats.win_rate:.2f}"
              f"_valwinauc{va['win_auc']:.3f}_valantecorr{va['ante_corr']:.3f}")
    open(marker, "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
