"""Stage 2.5 validation: does an ATTENTION encoder fix joker valuation that the
mean-pool encoder can't?

Trains the value head (ante) on the SAME bootstrap data with `encoder="mean"` and
`encoder="attention"` (identical config, only the trunk differs) and compares:
- val ante correlation       (does it predict outcome better?)
- val prediction std         (dynamic range — mean-pool collapses to ~constant)
- joker-removal Δ            (THE test: remove a joker, does predicted value DROP?
                              mean-pool can't see jokers -> Δ≈0 -> it sells the build;
                              attention should -> Δ>0)

    PYTHONPATH=src python scripts/phase8_encoder_validate.py \
        --captures .data/phase8-bootstrap-basic-64.jsonl --epochs 18 --lr 5e-3 \
        --metrics .data/phase8_encoder_validate.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _collect_joker_states(seeds, min_jokers, cap):
    from balatro_ai.api.state import GamePhase
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    drv = SolverPolicy(play_backend="v2", play_depth=2, play_width=1, seed=0)
    out = []
    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        for _ in range(1500):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            if len(st.jokers) >= min_jokers and st.phase in (
                    GamePhase.SELECTING_HAND, GamePhase.SHOP):
                out.append(st)
                if len(out) >= cap:
                    return out
            sim.step(drv.choose_action(st))
    return out


def _ante_corr_std(model, val_ex):
    import torch
    from balatro_ai.ml.model import collate_states
    from balatro_ai.ml.train import _pearson

    states = [e.encoded_state for e in val_ex]
    tgt = [min(1.0, max(0.0, e.value.final_ante / 8.0)) for e in val_ex]
    model.eval()
    with torch.no_grad():
        preds = model.ante_value(collate_states(states)).tolist()
    return _pearson(preds, tgt), statistics.pstdev(preds)


def _joker_removal_delta(model, states):
    """Mean (value_with_joker - value_without) over removing each joker once."""
    import torch
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.model import collate_states

    model.eval()
    deltas = []
    with torch.no_grad():
        for st in states:
            base = float(model.ante_value(collate_states([encode_state(st)]))[0])
            for i in range(len(st.jokers)):
                cf = dataclasses.replace(
                    st, jokers=tuple(j for k, j in enumerate(st.jokers) if k != i))
                v = float(model.ante_value(collate_states([encode_state(cf)]))[0])
                deltas.append(base - v)
    return (statistics.mean(deltas), statistics.pstdev(deltas),
            sum(1 for d in deltas if d > 0) / max(1, len(deltas)), len(deltas))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=".data/phase8-bootstrap-basic-64.jsonl")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-cap", type=int, default=50)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    from balatro_ai.ml import bootstrap as bs
    from balatro_ai.ml.dataset import examples_from_capture
    from balatro_ai.ml.train import TrainConfig, save_checkpoint, train

    caps = list(bs.read_captures(args.captures))
    rng = random.Random(0)
    rng.shuffle(caps)
    n_val = max(1, int(len(caps) * args.val_frac))
    val_caps, train_caps = caps[:n_val], caps[n_val:]
    train_ex = [e for c in train_caps for e in examples_from_capture(c)]
    val_ex = [e for c in val_caps for e in examples_from_capture(c)]
    print(f"[enc-val] runs {len(train_caps)}/{len(val_caps)}; "
          f"examples {len(train_ex)}/{len(val_ex)}", flush=True)

    test_seeds = [f"{800000 + i:07d}" for i in range(1, args.test_seeds + 1)]
    test_states = _collect_joker_states(test_seeds, min_jokers=2, cap=args.test_cap)
    n_jok = sum(len(s.jokers) for s in test_states)
    print(f"[enc-val] joker-removal test: {len(test_states)} states, {n_jok} jokers", flush=True)

    results = {}
    for mode in ("mean", "attention"):
        res = train(train_ex, TrainConfig(
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            dropout=args.dropout, encoder=mode), val_examples=val_ex)
        corr, std = _ante_corr_std(res.model, val_ex)
        jmean, jstd, jpos, jn = _joker_removal_delta(res.model, test_states)
        ckpt = f".data/phase8_value_{mode}_v0.pt"
        save_checkpoint(res.model, ckpt)
        results[mode] = {
            "val_ante_corr": round(corr, 4), "val_pred_std": round(std, 4),
            "joker_removal_delta_mean": round(jmean, 4),
            "joker_removal_delta_std": round(jstd, 4),
            "joker_removal_frac_positive": round(jpos, 3),
            "joker_removals": jn, "val_loss": round(res.final_val_loss or 0.0, 4),
            "ckpt": ckpt,
        }
        print(f"[enc-val] {mode}:", json.dumps(results[mode]), flush=True)

    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    m, a = results["mean"], results["attention"]
    open(f".data/_ENCVAL_mean_corr{m['val_ante_corr']:.2f}_jd{m['joker_removal_delta_mean']:.3f}"
         f"__attn_corr{a['val_ante_corr']:.2f}_jd{a['joker_removal_delta_mean']:.3f}", "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
