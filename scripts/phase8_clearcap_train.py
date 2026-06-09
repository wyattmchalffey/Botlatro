"""Try the neural in a better way: a CLEAR-CAPACITY model, not a win-value model.

The win-value net (V = P(win whole run)) was blind to build construction (graft gate AUC 0.567)
because it was trained on a global, long-horizon, noisy target. This trains the SHARP, LOCAL,
near-deterministic target the out-test proved matters: per-blind (build, blind) -> cleared.

Label: replay each capture; at every blind start, encode the state (build + the wall it faces);
cleared = 1 unless it is the final blind of a LOSING run. Train a ValueNet's forward/win head as
a binary clear-capacity classifier (BCE) so it is a drop-in for value_buildgate.py (which reads
sigmoid(model.forward)). Eval: held-out clear AUC by ante; then run value_buildgate with --ckpt
pointed here for the apples-to-apples graft gate vs the win-net's 0.567.

    PYTHONPATH=src py -3.12 scripts/phase8_clearcap_train.py \
        --caps .data/onpolicy_solver_caps_384.jsonl --encoder attention --epochs 50 \
        --ckpt .data/clearcap_attn_v1.pt --metrics .data/clearcap_attn_v1.metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _collect(caps_path):
    from balatro_ai.api.actions import Action
    from balatro_ai.api.state import GamePhase
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    active = {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND, GamePhase.ROUND_EVAL}
    examples = []  # (seed, ante, encoded_state, cleared, kind)
    caps = [json.loads(l) for l in open(caps_path, encoding="utf-8") if l.strip()]
    for cap in caps:
        seed = cap["seed"]
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        prev = sim.state.phase
        evs = []  # (kind, ante, EncodedState)  kind in {"blind","shop"}
        for ad in cap.get("actions", ()):
            try:
                sim.step(Action.from_mapping(ad))
            except Exception:
                break
            ph = sim.state.phase
            if prev == GamePhase.BLIND_SELECT and ph in active:
                evs.append(["blind", int(sim.state.ante), encode_state(sim.state)])
            elif ph == GamePhase.SHOP:  # every post-action shop (leaf-like) state
                evs.append(["shop", int(sim.state.ante), encode_state(sim.state)])
            prev = ph
        blind_pos = [i for i, e in enumerate(evs) if e[0] == "blind"]
        if not blind_pos:
            continue
        nb = len(blind_pos)
        cleared_at = {}  # event index -> cleared label
        for j, i in enumerate(blind_pos):
            cleared_at[i] = 1 if (cap["won"] or j < nb - 1) else 0
        for i, (kind, ante, enc) in enumerate(evs):
            if kind == "blind":
                cl = cleared_at[i]
            else:  # shop: label by whether the NEXT blind faced is cleared
                nxt = next((p for p in blind_pos if p > i), None)
                if nxt is None:
                    continue
                cl = cleared_at[nxt]
            examples.append((seed, ante, enc, cl, kind))
    return examples


def _auc(model, items):
    import torch
    from balatro_ai.ml.model import collate_states
    if not items:
        return None
    batch = collate_states([ex[2] for ex in items])
    with torch.no_grad():
        p = torch.sigmoid(model.forward(batch)).tolist()
    pos = [p[i] for i, it in enumerate(items) if it[3] == 1]
    neg = [p[i] for i, it in enumerate(items) if it[3] == 0]
    if not pos or not neg:
        return None
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--encoder", default="attention", choices=("mean", "attention"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    import torch
    from torch import nn
    from balatro_ai.ml.model import ValueNet, collate_states
    from balatro_ai.ml.train import save_checkpoint

    ex = _collect(args.caps)
    pos = sum(1 for e in ex if e[3] == 1)
    print(f"[clearcap] {len(ex)} blind examples | cleared {pos} / not-cleared {len(ex)-pos}", flush=True)

    seeds = sorted({e[0] for e in ex})
    rng = random.Random(0)
    rng.shuffle(seeds)
    n_val = max(1, int(len(seeds) * args.val_frac))
    val_seeds = set(seeds[:n_val])
    train = [e for e in ex if e[0] not in val_seeds]
    val = [e for e in ex if e[0] in val_seeds]
    print(f"[clearcap] train {len(train)} / val {len(val)} (by seed)", flush=True)

    torch.manual_seed(0)
    model = ValueNet(encoder=args.encoder, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # class-balanced BCE (negatives are rare)
    pos_weight = torch.tensor([(len(train) - sum(e[3] for e in train)) / max(1, sum(e[3] for e in train))])
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    items = list(train)
    for epoch in range(args.epochs):
        rng.shuffle(items)
        model.train()
        for s in range(0, len(items), args.batch_size):
            chunk = items[s:s + args.batch_size]
            batch = collate_states([ex[2] for ex in chunk])
            y = torch.tensor([float(e[3]) for e in chunk])
            opt.zero_grad()
            loss = lossfn(model.forward(batch), y)
            loss.backward()
            opt.step()
    model.eval()
    val_auc = _auc(model, val)
    train_auc = _auc(model, train)
    by_ante = {}
    bucket = defaultdict(list)
    for e in val:
        bucket[e[1]].append(e)
    for a in sorted(bucket):
        by_ante[str(a)] = {"n": len(bucket[a]),
                           "n_neg": sum(1 for e in bucket[a] if e[3] == 0),
                           "auc": round(_auc(model, bucket[a]), 3) if _auc(model, bucket[a]) is not None else None}
    save_checkpoint(model, args.ckpt)
    metrics = {"encoder": args.encoder, "n_examples": len(ex), "n_cleared": pos,
               "train_auc": round(train_auc, 3) if train_auc else None,
               "val_auc": round(val_auc, 3) if val_auc else None,
               "val_auc_by_ante": by_ante, "ckpt": args.ckpt}
    print(json.dumps(metrics, indent=2), flush=True)
    json.dump(metrics, open(args.metrics, "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
