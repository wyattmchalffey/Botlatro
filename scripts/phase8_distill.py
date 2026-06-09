"""Stage 1.3 Option A driver: distill the clear_probability rollout leaf.

Collects `(state, clear_probability)` pairs from the v2 beam's own leaf-state
distribution (train seeds vs held-out val seeds), trains `ValueNet.clear_head`
to regress them, reports fit, and saves a checkpoint. The downstream search A/B
runs separately: `scripts/phase8_leaf_ab.py --ckpt <ckpt> --head clear`.

    PYTHONPATH=src python scripts/phase8_distill.py --train-seeds 16 --val-seeds 6 \
        --epochs 15 --ckpt .data/phase8_clear_v3.pt --metrics .data/phase8_distill.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, default=16)
    ap.add_argument("--val-seeds", type=int, default=6)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    from balatro_ai.ml.distill import collect_distill_pairs, eval_distill, train_distill
    from balatro_ai.ml.train import save_checkpoint

    train_seeds = [f"{i:07d}" for i in range(1, args.train_seeds + 1)]
    val_seeds = [f"{i:07d}" for i in range(args.train_seeds + 1,
                                           args.train_seeds + args.val_seeds + 1)]

    t0 = time.perf_counter()
    train_pairs = collect_distill_pairs(train_seeds, depth=args.depth, width=args.width)
    val_pairs = collect_distill_pairs(val_seeds, depth=args.depth, width=args.width)
    collect_s = time.perf_counter() - t0
    print(f"[distill] collected train/val pairs = {len(train_pairs)}/{len(val_pairs)} "
          f"in {collect_s:.0f}s", flush=True)

    model = train_distill(train_pairs, epochs=args.epochs)
    tr = eval_distill(model, train_pairs)
    va = eval_distill(model, val_pairs)
    save_checkpoint(model, args.ckpt)

    metrics = {
        "train_seeds": args.train_seeds, "val_seeds": args.val_seeds,
        "depth": args.depth, "width": args.width, "epochs": args.epochs,
        "n_train_pairs": len(train_pairs), "n_val_pairs": len(val_pairs),
        "collect_seconds": round(collect_s, 1),
        "train": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tr.items()},
        "val": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in va.items()},
        "ckpt": args.ckpt,
    }
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("[distill] metrics:", json.dumps(metrics), flush=True)

    marker = (f".data/_DISTILL_valcorr{va['corr']:.3f}_valmse{va['mse']:.4f}"
              f"_npairs{len(train_pairs)}")
    open(marker, "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
