"""Gate V0: does the value head resolve win-probability on HELD-OUT states?

Pre-registered (PHASE_B_ARCHITECTURE.md): before any advantage-weighted
iteration, the value head must show per-decision advantage RESOLUTION — i.e.
V(state) must carry real, spread-out, calibrated signal about the eventual
outcome on data it did NOT train on. The improvement loop uses V(s) as the
credit-assignment baseline (advantage ≈ outcome − V(s)); if V is FLAT (the
project's documented 0-for-5 value failure: std ~0.05, AUC ~0.5), the
advantage collapses to the raw noisy win/loss label and iteration 1 cannot
learn. V0 is the go/no-go for advantage-weighting vs the fallback
(win-conditioned BC + dense fork-audit play labels).

PASS criteria (pre-registered):
  - held-out win-AUC >= 0.65 (meaningfully discriminative; the historical
    on-policy value net hit 0.708, the bar to match), AND
  - prediction spread std >= 0.10 (not flat), AND
  - calibration monotonic across deciles (higher V -> higher realized winrate).

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python scripts/phaseb_gate_v0.py \
        --heldout .data/phaseb_heldout_200.jsonl --ckpt .data/phaseb_policy_b0v2.pt
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    # Mann-Whitney U / (n_pos*n_neg) via rank sum.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-states", type=int, default=20000)
    args = ap.parse_args()

    import torch

    from balatro_ai.ml.dataset import RunCapture, replay_states
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.model import collate_states
    from balatro_ai.ml.policy_net import load_policy

    net = load_policy(args.ckpt)
    net.eval()

    rows = [json.loads(l) for l in open(args.heldout, encoding="utf-8") if l.strip()]
    print(f"[v0] held-out: {len(rows)} runs, winrate "
          f"{sum(1 for r in rows if r.get('won'))}/{len(rows)}", flush=True)

    # Light expansion: encode states + per-run outcome label (NO candidate
    # building — V0 only needs V(s) vs the outcome).
    states = []
    labels = []
    for r in rows:
        cap = RunCapture.from_json_dict(r)
        won = 1 if cap.won else 0
        for state, _action in replay_states(cap.seed, cap.actions, stake=cap.stake):
            states.append(encode_state(state))
            labels.append(won)
            if len(states) >= args.max_states:
                break
        if len(states) >= args.max_states:
            break
    print(f"[v0] {len(states)} held-out states", flush=True)

    # Score V(s) in batches.
    preds: list[float] = []
    with torch.no_grad():
        for i in range(0, len(states), 512):
            batch = collate_states(states[i:i + 512])
            logits = net.value_net(batch)
            preds.extend(torch.sigmoid(logits).tolist())

    auc = _auc(preds, labels)
    std = statistics.pstdev(preds)
    lo, hi = min(preds), max(preds)
    # Decile calibration: realized winrate per predicted-value decile.
    order = sorted(range(len(preds)), key=lambda i: preds[i])
    deciles = []
    for d in range(10):
        chunk = order[d * len(order) // 10:(d + 1) * len(order) // 10]
        if chunk:
            deciles.append(sum(labels[i] for i in chunk) / len(chunk))
    monotonic = all(deciles[i] <= deciles[i + 1] + 0.02 for i in range(len(deciles) - 1))

    print(f"[v0] held-out win-AUC = {auc:.4f}  (bar >= 0.65; historical 0.708)", flush=True)
    print(f"[v0] prediction spread: std={std:.4f} range=[{lo:.3f},{hi:.3f}]  (bar std >= 0.10)", flush=True)
    print(f"[v0] decile realized winrate: {[round(x, 3) for x in deciles]}", flush=True)
    print(f"[v0] calibration monotonic: {monotonic}", flush=True)
    passed = auc >= 0.65 and std >= 0.10 and monotonic
    print(f"[v0] VERDICT: {'PASS' if passed else 'FAIL'} — "
          f"{'value head resolves outcome; advantage-weighting is viable' if passed else 'value head too flat/weak; use win-conditioned-BC + dense play labels fallback'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
