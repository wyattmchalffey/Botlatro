"""Stage 2 driver: imitation-train the policy head on the bootstrap captures.

Loads an existing capture file (no new data-gen), splits BY RUN, trains the
policy heads, and reports held-out action-type accuracy (vs base rate) plus the
per-card play-pointer accuracy.

    PYTHONPATH=src python scripts/phase8_policy_train.py \
        --captures .data/phase8-bootstrap-basic.jsonl --epochs 15 \
        --ckpt .data/phase8_policy_v3.pt --metrics .data/phase8_policy.json
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    from balatro_ai.ml import bootstrap as bs
    from balatro_ai.ml.dataset import examples_from_capture
    from balatro_ai.ml.policy import PolicyConfig, eval_policy, train_policy
    from balatro_ai.ml.train import save_checkpoint

    caps = list(bs.read_captures(args.captures))
    rng = random.Random(0)
    rng.shuffle(caps)
    n_val = max(1, int(len(caps) * args.val_frac))
    val_caps, train_caps = caps[:n_val], caps[n_val:]
    train_ex = [e for c in train_caps for e in examples_from_capture(c)]
    val_ex = [e for c in val_caps for e in examples_from_capture(c)]
    print(f"[policy] runs train/val = {len(train_caps)}/{len(val_caps)}; "
          f"examples train/val = {len(train_ex)}/{len(val_ex)}", flush=True)

    model = train_policy(train_ex, PolicyConfig(epochs=args.epochs))
    tr = eval_policy(model, train_ex)
    va = eval_policy(model, val_ex)
    save_checkpoint(model, args.ckpt)

    def _r(d: dict) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    metrics = {"captures": args.captures, "epochs": args.epochs,
               "train": _r(tr), "val": _r(va), "ckpt": args.ckpt}
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("[policy] metrics:", json.dumps(metrics), flush=True)

    marker = (f".data/_POLICY_valtype{va['type_acc']:.3f}_base{va['type_base_rate']:.3f}"
              f"_valcard{va['card_pos_acc']:.3f}")
    open(marker, "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
