"""S0 step 2b/3: the predictive-foresight GATE.

Joins oracle best-basin labels with early-state features and asks the load-bearing
question: can antes-1..3 visible state PREDICT which build basin the oracle won with?
If yes, whole-run foresight is learnable and the engine has a signal to climb; if no,
the +6.2% oracle ceiling is real but not predictable at commit time -> bank ~22-25%.

Reports (k-fold CV, no sklearn -> torch multinomial logistic + numpy AUC):
  - 5-way top-1 vs majority-class baseline; macro one-vs-rest AUC.
  - HELPED-subset 4-way (seeds where an archetype strictly beat baseline -- the
    decision-relevant ones): can we pick WHICH archetype helps?
  - binary commit (any archetype helps vs stay baseline) AUC.
  - OFFLINE conservative-selector gap-closure: using the oracle's per-archetype
    results, what fraction of the (oracle - baseline) winrate gap would a
    confidence-gated commit have captured on held-out folds (allowing None).

    PYTHONPATH=src py -3.12 scripts/s0_foresight_classifier.py \
        --oracle .data/s0_oracle_white_200.json --features .data/s0_early_features_white_200.json \
        --ante 2 --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

ARCHS = ["flush", "scaling_joker", "high_card_mult", "pair_retrigger"]
CLASSES = ["baseline"] + ARCHS
FEATURE_KEYS = [
    "money", "n_jokers", "deck_size",
    "suit_frac_S", "suit_frac_H", "suit_frac_D", "suit_frac_C",
    "suit_max_single", "suit_max_smeared_pair", "deck_nonstone",
    "lvl_flush", "lvl_straight", "lvl_pair", "lvl_two_pair", "lvl_three_of_a_kind", "lvl_high_card",
    "owned_key_flush", "seen_key_flush", "owned_key_scaling_joker", "seen_key_scaling_joker",
    "owned_key_high_card_mult", "seen_key_high_card_mult", "owned_key_pair_retrigger", "seen_key_pair_retrigger",
]


def _result_key(v):  # (won, ante, score) sort key
    return (int(bool(v[0])), int(v[1]), int(v[2]))


def _softmax_logreg_cv(X, y, n_classes, folds, epochs=400, lr=0.1, l2=1e-3, seed=0):
    """Multinomial logistic regression with k-fold CV. Returns held-out probs per row."""
    import torch
    torch.manual_seed(seed)
    n, d = X.shape
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_id = np.zeros(n, dtype=int)
    for i, idx in enumerate(order):
        fold_id[idx] = i % folds
    oof = np.zeros((n, n_classes), dtype=float)
    for f in range(folds):
        tr = fold_id != f
        te = fold_id == f
        Xtr = torch.tensor(Xz[tr], dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.long)
        Xte = torch.tensor(Xz[te], dtype=torch.float32)
        W = torch.zeros((d, n_classes), requires_grad=True)
        b = torch.zeros(n_classes, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr, weight_decay=l2)
        lossf = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad()
            logits = Xtr @ W + b
            loss = lossf(logits, ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = torch.softmax(Xte @ W + b, dim=1).numpy()
        oof[te] = p
    return oof


def _ovr_auc(y_bin, score):
    """One-vs-rest AUC for a single binary label via rank statistic."""
    pos = score[y_bin == 1]
    neg = score[y_bin == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    rsum = ranks[: len(pos)].sum()
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--ante", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    oracle = json.load(open(args.oracle, encoding="utf-8"))
    feats = json.load(open(args.features, encoding="utf-8"))
    labels = oracle["labels"]
    # oracle file stores labels + aggregate; per-seed results (rows) needed for gap-closure.
    rows = oracle.get("rows")
    per_seed_results = {}
    if rows:
        per_seed_results = {r["seed"]: r["results"] for r in rows}

    seeds = [s for s in labels if s in feats and str(args.ante) in feats[s]]
    print(f"[gate] {len(seeds)} seeds with label + ante-{args.ante} features", flush=True)
    X = np.array([[float(feats[s][str(args.ante)].get(k, 0)) for k in FEATURE_KEYS] for s in seeds])
    lab = [labels[s] for s in seeds]
    y = np.array([CLASSES.index(l) for l in lab])

    # 5-way
    from collections import Counter
    cnt = Counter(lab)
    majority = max(cnt.values()) / len(lab)
    oof = _softmax_logreg_cv(X, y, len(CLASSES), args.folds)
    pred = oof.argmax(1)
    top1 = (pred == y).mean()
    aucs = {CLASSES[c]: round(_ovr_auc((y == c).astype(int), oof[:, c]), 3) for c in range(len(CLASSES))}
    print("\n=== 5-way (predict oracle best basin incl. baseline) ===")
    print(f"  class distribution: {dict(cnt)}")
    print(f"  majority-class acc: {majority:.3f}   CV top-1: {top1:.3f}   lift: {top1-majority:+.3f}")
    print(f"  one-vs-rest AUC: {aucs}")

    # HELPED subset: archetype strictly beats baseline
    helped_mask = np.zeros(len(seeds), dtype=bool)
    helped_arch = []
    if per_seed_results:
        for i, s in enumerate(seeds):
            res = per_seed_results[s]
            base = _result_key(res["baseline"])
            best_a, best_k = None, base
            for a in ARCHS:
                k = _result_key(res[a])
                if k > best_k:
                    best_k, best_a = k, a
            if best_a is not None:
                helped_mask[i] = True
            helped_arch.append(best_a)
        hidx = np.where(helped_mask)[0]
        print(f"\n=== HELPED subset (archetype strictly beats baseline): {len(hidx)}/{len(seeds)} seeds ===")
        if len(hidx) >= args.folds * 2:
            Xh = X[hidx]
            yh = np.array([ARCHS.index(helped_arch[i]) for i in hidx])
            ch = Counter(yh.tolist())
            majh = max(ch.values()) / len(yh)
            oofh = _softmax_logreg_cv(Xh, yh, len(ARCHS), args.folds)
            top1h = (oofh.argmax(1) == yh).mean()
            print(f"  which-archetype-helps class dist: {dict(Counter([ARCHS[v] for v in yh]))}")
            print(f"  majority acc: {majh:.3f}   CV top-1: {top1h:.3f}   lift: {top1h-majh:+.3f}")
        else:
            print("  too few helped seeds for a stable CV; report counts only.")

        # binary commit (any archetype helps)
        oofb = _softmax_logreg_cv(X, helped_mask.astype(int), 2, args.folds)
        auc_b = _ovr_auc(helped_mask.astype(int), oofb[:, 1])
        print(f"\n=== binary commit (any-archetype-helps): rate {helped_mask.mean():.3f}, AUC {auc_b:.3f} ===")

        # OFFLINE conservative-selector gap-closure (confidence-gated commit, allow None).
        # Uses the held-out 5-way probs: commit to the predicted archetype only when the
        # top class is an archetype (not baseline) AND its prob clears the threshold;
        # otherwise stay baseline. sel outcome = that archetype's REAL per-seed result.
        print("\n=== OFFLINE conservative-selector gap-closure (held-out, 5-way confidence-gated) ===")
        base_w = np.array([int(per_seed_results[s]["baseline"][0]) for s in seeds])
        oracle_w = np.array([int(per_seed_results[s][labels[s]][0]) for s in seeds])
        gap = int(oracle_w.sum() - base_w.sum())
        for thr in (0.0, 0.35, 0.45, 0.55, 0.65, 0.75):
            sel_w = base_w.copy()
            commits = 0
            wrong = 0
            for i, s in enumerate(seeds):
                p = oof[i]  # 5-way probs over CLASSES = [baseline, *ARCHS]
                c_idx = int(p.argmax())
                if c_idx != 0 and p[c_idx] >= thr:
                    arch = CLASSES[c_idx]
                    new = int(per_seed_results[s][arch][0])
                    if base_w[i] == 1 and new == 0:
                        wrong += 1
                    sel_w[i] = new
                    commits += 1
            captured = int(sel_w.sum() - base_w.sum())
            frac = captured / gap if gap else float("nan")
            print(f"  thr={thr:.2f}: commits={commits:3d} (wrong-commits-that-lost-a-win={wrong:2d})  "
                  f"sel_wins={int(sel_w.sum())}  base={int(base_w.sum())}  oracle={int(oracle_w.sum())}  "
                  f"captured/gap={captured}/{gap}={frac:+.1%}")
    else:
        print("\n(no per-seed 'rows' in oracle file -> gap-closure + helped analysis skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
