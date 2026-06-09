"""Stage 2.2 driver: train the candidate-subset play policy on bootstrap captures.

Splits BY RUN, trains the play-candidate scorer, and reports held-out top-1
(does the teacher's actual play rank highest vs N random subsets?).

    PYTHONPATH=src python scripts/phase8_play_policy_train.py \
        --captures .data/phase8-bootstrap-basic.jsonl --epochs 15 \
        --ckpt .data/phase8_playpolicy_v3.pt --metrics .data/phase8_play_policy.json
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
    ap.add_argument("--captures", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n-neg", type=int, default=15)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    from balatro_ai.ml import bootstrap as bs
    from balatro_ai.ml.dataset import examples_from_capture
    from balatro_ai.ml.play_policy import (
        PlayPolicyConfig, eval_play_policy, train_play_policy)
    from balatro_ai.ml.train import save_checkpoint

    caps = list(bs.read_captures(args.captures))
    rng = random.Random(0)
    rng.shuffle(caps)
    n_val = max(1, int(len(caps) * args.val_frac))
    val_caps, train_caps = caps[:n_val], caps[n_val:]
    train_ex = [e for c in train_caps for e in examples_from_capture(c)]
    val_ex = [e for c in val_caps for e in examples_from_capture(c)]
    print(f"[play-policy] runs train/val = {len(train_caps)}/{len(val_caps)}; "
          f"examples train/val = {len(train_ex)}/{len(val_ex)}", flush=True)

    model = train_play_policy(train_ex, PlayPolicyConfig(epochs=args.epochs, n_neg=args.n_neg))
    tr = eval_play_policy(model, train_ex)
    va = eval_play_policy(model, val_ex)
    save_checkpoint(model, args.ckpt)

    def _r(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    metrics = {"captures": args.captures, "epochs": args.epochs, "n_neg": args.n_neg,
               "train": _r(tr), "val": _r(va), "ckpt": args.ckpt}
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("[play-policy] metrics:", json.dumps(metrics), flush=True)
    open(f".data/_PLAYPOL_valtop1{va['top1_acc']:.3f}_base{va['random_baseline']:.3f}", "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
