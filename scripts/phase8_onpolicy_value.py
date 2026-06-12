"""On-policy value-net loop, increment 1.

Every prior value head trained on basic_strategy / bootstrap trajectories and went
flat on the SolverPolicy shop states the beam actually visits (DISTRIBUTION SHIFT,
PROGRESS.md 2026-06-03). This captures from the SOLVER bot itself -- the policy that
will deploy the value as its shop leaf -- so training and deployment share a state
distribution. Attention encoder (not mean-pool, which the notes flag as low-std).

Capture is the long pole and is cached to --out (resumable); training reads it back.
Capture seeds use --seed-offset (default 2,000,000) so the winrate A/B on the
canonical 1..N seeds stays fully held out.

    PYTHONPATH=src py -3.12 scripts/phase8_onpolicy_value.py \
        --bot solver_shop_basic_play_bot --seeds 384 --seed-offset 2000000 --workers 8 \
        --encoder attention --epochs 60 \
        --out .data/onpolicy_solver_caps.jsonl \
        --ckpt .data/value_onpolicy_attn_v1.pt --metrics .data/value_onpolicy_attn_v1.metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _cap_worker(args):
    seed, bot_name, stake, max_steps = args
    from dataclasses import replace
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.ml.dataset import capture_run
    from balatro_ai.solver.trajectory import _stable_seed_int

    bot = create_bot(bot_name, _stable_seed_int(seed))
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        cap = capture_run(seed, bot.choose_action, stake=stake, max_steps=max_steps)
    row = cap.to_json_dict()
    # Policy provenance (Phase B): mixture accounting + per-policy advantage
    # baselines need to know which policy generated each run.
    row["bot_name"] = bot_name
    return row


def _capture(args) -> int:
    out = args.out
    # TRUE resumability (Phase B feasibility-review fix): per-run incremental
    # appends + per-seed dedupe — a crash at run 49k of 50k loses at most the
    # in-flight runs (the old cache regenerated EVERYTHING unless complete).
    # Bot MIXTURES: --bot "name1:w1,name2:w2" -> deterministic per-seed
    # assignment proportional to weights (stable across resumes).
    done_seeds: set[str] = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        done_seeds.add(json.loads(line)["seed"])
                    except (ValueError, KeyError):
                        continue
    seeds = [f"{args.seed_offset + i:07d}" for i in range(1, args.seeds + 1)]

    mix: list[tuple[str, float]] = []
    for part in str(args.bot).split(","):
        bot_name, _, weight = part.strip().partition(":")
        mix.append((bot_name, float(weight) if weight else 1.0))
    total_weight = sum(w for _, w in mix)

    def _bot_for(index: int) -> str:
        if len(mix) == 1:
            return mix[0][0]
        slot = (index * 0.6180339887498949) % 1.0  # low-discrepancy assignment
        acc = 0.0
        for bot_name, weight in mix:
            acc += weight / total_weight
            if slot < acc:
                return bot_name
        return mix[-1][0]

    jobs = [
        (s, _bot_for(i), args.stake, args.max_steps)
        for i, s in enumerate(seeds)
        if s not in done_seeds
    ]
    if not jobs:
        print(f"[capture] all {len(seeds)} seeds already present in {out}", flush=True)
        return len(seeds)
    if done_seeds:
        print(f"[capture] resume: {len(done_seeds)} present, {len(jobs)} to run", flush=True)
    t0 = time.perf_counter()
    n_new = 0
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_cap_worker, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            n_new += 1
            if n_new % 32 == 0:
                print(f"[capture] {n_new}/{len(jobs)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    with open(out, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    won = sum(1 for r in rows if r.get("won"))
    print(f"[capture] {out} now holds {len(rows)} runs | winrate {won}/{len(rows)} "
          f"({won/len(rows):.1%}) | {n_new} new in {time.perf_counter()-t0:.0f}s", flush=True)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="solver_shop_basic_play_bot")
    ap.add_argument("--seeds", type=int, default=384)
    ap.add_argument("--seed-offset", type=int, default=2000000)
    ap.add_argument("--stake", default="white")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--encoder", default="attention", choices=("mean", "attention"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ante-loss-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--capture-only", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    n_caps = _capture(args)
    if args.capture_only:
        return 0

    from balatro_ai.ml import bootstrap as bs
    from balatro_ai.ml.dataset import examples_from_capture
    from balatro_ai.ml.train import TrainConfig, evaluate_full, save_checkpoint, train

    caps = list(bs.read_captures(args.out))
    rng = random.Random(0)
    rng.shuffle(caps)
    n_val = max(1, int(len(caps) * args.val_frac))
    val_caps, train_caps = caps[:n_val], caps[n_val:]
    train_ex = [e for c in train_caps for e in examples_from_capture(c)]
    val_ex = [e for c in val_caps for e in examples_from_capture(c)]
    won = sum(1 for c in caps if c.won)
    print(f"[train] runs train/val = {len(train_caps)}/{len(val_caps)}; "
          f"examples train/val = {len(train_ex)}/{len(val_ex)}; capture winrate {won}/{len(caps)}", flush=True)

    res = train(train_ex, TrainConfig(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        weight_decay=args.weight_decay, dropout=args.dropout,
        ante_loss_weight=args.ante_loss_weight, encoder=args.encoder), val_examples=val_ex)

    tr = evaluate_full(res.model, train_ex)
    va = evaluate_full(res.model, val_ex)
    save_checkpoint(res.model, args.ckpt)

    def _round(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "n"}

    metrics = {
        "bot": args.bot, "encoder": args.encoder, "n_runs": len(caps),
        "capture_winrate": round(won / max(1, len(caps)), 4),
        "seed_offset": args.seed_offset, "n_seeds": args.seeds,
        "examples_train": len(train_ex), "examples_val": len(val_ex),
        "train": _round(tr), "val": _round(va),
        "config": {"epochs": args.epochs, "encoder": args.encoder, "dropout": args.dropout,
                   "weight_decay": args.weight_decay, "ante_loss_weight": args.ante_loss_weight},
        "ckpt": args.ckpt,
    }
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("[train] metrics:", json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
