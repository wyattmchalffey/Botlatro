"""Option C, increment 1: TD(lambda) relabel of the bootstrap captures.

The value head is NEAR-CONSTANT on shop states (std 0.074 -> can't guide buys)
because MC final-ante labels are too high-variance there. TD(lambda) fixes the
TARGET: instead of labeling a state with the far-off run outcome, blend in the
net's OWN estimate of downstream states (G_i = (1-lam) V(s_{i+1}) + lam G_{i+1},
terminal G = final_ante/9). A shop state then inherits value-variance from the
downstream BLIND states it leads to (where the net already discriminates), so
shop-state value should stop being flat.

Fitted value iteration: freeze V, relabel all states, train a FRESH net to fit,
swap V <- new net, repeat. Reuses the existing captures (replay once, cache the
encoded states), so this is cheap (no new data-gen, no rollouts).

GATE: does val shop-state std climb above 0.074 (the value that doomed the shop
A/B), while val ante_corr to final_ante holds/improves?

    PYTHONPATH=src python scripts/phase8_td_relabel.py \
        --captures .data/phase8-bootstrap-basic.jsonl --init-ckpt .data/phase8_value_v3_bootstrap.pt \
        --lam 0.5 --iters 4 --epochs 15 \
        --ckpt-out .data/phase8_value_td_v1.pt --metrics .data/phase8_td_relabel.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_NORM = 9.0  # final_ante (~1..9) -> [0,1] to match the sigmoid ante head


def _prep_runs(caps):
    """Replay each capture ONCE; cache per-run encoded states + shop mask + outcome."""
    from balatro_ai.api.state import GamePhase
    from balatro_ai.ml.dataset import replay_states
    from balatro_ai.ml.encoding import encode_state

    runs = []
    for c in caps:
        states, shop = [], []
        for st, _action in replay_states(c.seed, c.actions, stake=c.stake):
            states.append(encode_state(st))
            shop.append(st.phase == GamePhase.SHOP)
        if states:
            runs.append({"states": states, "shop": shop, "final_ante": c.final_ante})
    return runs


def _lambda_returns(run, vvals, lam):
    """Backward TD(lambda) return per state (gamma=1, terminal=final_ante/_NORM)."""
    n = len(run["states"])
    g = [0.0] * n
    g[n - 1] = run["final_ante"] / _NORM
    for i in range(n - 2, -1, -1):
        g[i] = (1.0 - lam) * vvals[i + 1] + lam * g[i + 1]
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=".data/phase8-bootstrap-basic.jsonl")
    ap.add_argument("--init-ckpt", default=".data/phase8_value_v3_bootstrap.pt")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-runs", type=int, default=0, help="0=all; cap for smoke tests")
    ap.add_argument("--ckpt-out", default=".data/phase8_value_td_v1.pt")
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    import torch
    from torch import nn

    from balatro_ai.ml.bootstrap import read_captures
    from balatro_ai.ml.model import ValueNet, collate_states
    from balatro_ai.ml.train import _pearson, load_checkpoint, save_checkpoint

    torch.manual_seed(0)
    caps = list(read_captures(args.captures))
    random.Random(0).shuffle(caps)
    if args.max_runs:
        caps = caps[:args.max_runs]
    n_val = max(1, int(len(caps) * args.val_frac))
    val_caps, train_caps = caps[:n_val], caps[n_val:]
    print(f"[td] runs train/val = {len(train_caps)}/{len(val_caps)}; replaying...", flush=True)
    train_runs = _prep_runs(train_caps)
    val_runs = _prep_runs(val_caps)
    n_tr = sum(len(r["states"]) for r in train_runs)
    n_va = sum(len(r["states"]) for r in val_runs)

    # Flatten val once for eval: states, final_ante per state, shop mask.
    val_states, val_fa, val_shop = [], [], []
    for r in val_runs:
        val_states.extend(r["states"])
        val_fa.extend([r["final_ante"]] * len(r["states"]))
        val_shop.extend(r["shop"])
    n_val_shop = sum(val_shop)
    print(f"[td] examples train/val = {n_tr}/{n_va}; val shop states = {n_val_shop}", flush=True)

    @torch.no_grad()
    def vvals_for(net, states):
        net.eval()
        return net.ante_value(collate_states(states)).tolist()

    V = load_checkpoint(args.init_ckpt)  # frozen value for round 0 (bootstrap-v3)
    history = []
    model = None
    for it in range(args.iters):
        # ---- relabel all train states with TD(lambda) targets under frozen V ----
        tr_states, tr_tgt = [], []
        for r in train_runs:
            g = _lambda_returns(r, vvals_for(V, r["states"]), args.lam)
            tr_states.extend(r["states"])
            tr_tgt.extend(g)

        # ---- train a FRESH net to fit the new targets ----
        model = ValueNet(dropout=args.dropout)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        mse = nn.MSELoss()
        idx = list(range(len(tr_states)))
        model.train()
        for ep in range(args.epochs):
            random.Random(1000 * it + ep).shuffle(idx)
            for s in range(0, len(idx), args.batch_size):
                chunk = idx[s:s + args.batch_size]
                batch = collate_states([tr_states[i] for i in chunk])
                tgt = torch.tensor([tr_tgt[i] for i in chunk], dtype=torch.float32)
                opt.zero_grad()
                loss = mse(model.ante_value(batch), tgt)
                loss.backward()
                opt.step()

        # ---- eval on held-out runs ----
        preds = vvals_for(model, val_states)
        corr = _pearson(preds, [float(a) for a in val_fa])
        shop_preds = [p for p, sh in zip(preds, val_shop) if sh]
        shop_std = statistics.pstdev(shop_preds) if len(shop_preds) > 1 else 0.0
        all_std = statistics.pstdev(preds) if len(preds) > 1 else 0.0
        tgt_std = statistics.pstdev(tr_tgt) if len(tr_tgt) > 1 else 0.0
        rec = {"iter": it, "lam": args.lam, "val_corr_finalante": round(corr, 4),
               "val_shop_std": round(shop_std, 4), "val_all_std": round(all_std, 4),
               "train_target_std": round(tgt_std, 4),
               "shop_mean": round(statistics.mean(shop_preds), 4) if shop_preds else None}
        history.append(rec)
        print(f"[td] {json.dumps(rec)}", flush=True)
        V = model  # fitted value iteration: swap in the new net

    save_checkpoint(model, args.ckpt_out)
    out = {"captures": args.captures, "init_ckpt": args.init_ckpt, "lam": args.lam,
           "iters": args.iters, "n_train": n_tr, "n_val": n_va, "n_val_shop": n_val_shop,
           "baseline_shop_std": 0.074, "history": history, "ckpt": args.ckpt_out}
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    best = max(history, key=lambda r: r["val_shop_std"])
    print(f"[td] DONE. best shop_std={best['val_shop_std']} (baseline 0.074) "
          f"at iter {best['iter']}, corr={best['val_corr_finalante']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
