"""Train the Phase 8 value model (pure numpy) on the value dataset.

Logistic regression by default; --mlp for a 1-hidden-layer net. Splits
train/val BY SEED (states from one run share a label -> seed split avoids
leakage). Reports AUC / accuracy / log-loss / calibration-by-ante and the
top standardized feature weights, then saves the model for the bot.

    PYTHONPATH=src python scripts/phase8_train.py [dataset.npz] [--mlp]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from balatro_ai.ml.features import FEATURE_NAMES  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / ".data" / "phase8_value_model.npz"

# Features whose MARGINAL direction confounds 1-step action guidance: they
# correlate with winning but move the wrong way (or double-count cost) when
# you take a build-up action. Buying a joker reduces open_joker_slots /
# raises n_jokers (both confounded weights) and spends money; buying a planet
# raises total_leveling. Zeroing them keeps the model's gradient causally
# sound for ranking buys, at a small cost to raw state-ranking AUC.
GUIDANCE_EXCLUDE = {
    "open_joker_slots", "n_jokers", "total_leveling",
    "money", "log_money", "n_consumables", "n_vouchers",
    "hands_remaining", "discards_remaining",
    # count/binary role features whose marginal sign is backwards: filling a
    # NEW role lowers n_missing_roles and flips has_X on, but both carry weights
    # that would PENALISE that buy. The continuous *_score features already
    # capture role progress with the correct positive sign, so drop these.
    "n_missing_roles",
    "has_chips", "has_mult", "has_xmult", "has_scaling", "has_economy",
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auc(y, p):
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return 0.5
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def train_logreg(X, y, iters=4000, lr=0.5, l2=1e-3):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    pos_w = (1 - y.mean()) / max(1e-6, y.mean())  # upweight rare positives
    sw = np.where(y == 1, pos_w, 1.0)
    sw /= sw.mean()
    for _ in range(iters):
        p = _sigmoid(X @ w + b)
        g = (p - y) * sw
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b


def train_mlp(X, y, hidden=16, iters=4000, lr=0.3, l2=1e-4, seed_arr=None):
    n, d = X.shape
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, 0.3, (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.3, hidden); b2 = 0.0
    pos_w = (1 - y.mean()) / max(1e-6, y.mean())
    sw = np.where(y == 1, pos_w, 1.0); sw /= sw.mean()
    for _ in range(iters):
        H = np.tanh(X @ W1 + b1)
        p = _sigmoid(H @ W2 + b2)
        g = (p - y) * sw / n
        gW2 = H.T @ g + l2 * W2
        gb2 = g.sum()
        dH = np.outer(g, W2) * (1 - H ** 2)
        gW1 = X.T @ dH + l2 * W1
        gb1 = dH.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return W1, b1, W2, b2


def main() -> int:
    ds = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ".data/phase8_value_dataset.npz"
    use_mlp = "--mlp" in sys.argv
    d = np.load(ds)
    X, y, ante, seed_idx = d["X"].copy(), d["y"], d["ante"], d["seed_idx"]
    print(f"dataset: {X.shape[0]} states, {X.shape[1]} feats, win rate {y.mean():.1%}", flush=True)

    # Guidance model (default): zero confounded columns so their weights stay
    # ~0 and the 1-step marginal ΔV reflects only causally-sound build power.
    # --raw keeps every feature (best pure state-ranking AUC, unsafe for buys).
    guidance = "--raw" not in sys.argv
    if guidance:
        excl = [i for i, n in enumerate(FEATURE_NAMES) if n in GUIDANCE_EXCLUDE]
        X[:, excl] = 0.0
        print(f"guidance mode: zeroed {len(excl)} confounded feats "
              f"({', '.join(FEATURE_NAMES[i] for i in excl)})", flush=True)

    # split by seed
    seeds = np.unique(seed_idx)
    rng = np.random.default_rng(42)
    rng.shuffle(seeds)
    n_val = max(1, int(0.2 * len(seeds)))
    val_seeds = set(seeds[:n_val].tolist())
    val_mask = np.array([s in val_seeds for s in seed_idx])
    tr, va = ~val_mask, val_mask

    mean, std = X[tr].mean(0), X[tr].std(0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    if use_mlp:
        W1, b1, W2, b2 = train_mlp(Xn[tr], y[tr])
        pv = _sigmoid(np.tanh(Xn[va] @ W1 + b1) @ W2 + b2)
        pt = _sigmoid(np.tanh(Xn[tr] @ W1 + b1) @ W2 + b2)
        kind = "mlp"
    else:
        w, b = train_logreg(Xn[tr], y[tr])
        pv = _sigmoid(Xn[va] @ w + b)
        pt = _sigmoid(Xn[tr] @ w + b)
        kind = "logreg"

    print(f"[{kind}] train: AUC={auc(y[tr],pt):.3f} logloss={logloss(y[tr],pt):.3f}", flush=True)
    print(f"[{kind}] VAL:   AUC={auc(y[va],pv):.3f} logloss={logloss(y[va],pv):.3f} (base rate {y[va].mean():.1%})", flush=True)
    # calibration by ante on val
    print("  val calibration by ante (pred_winprob vs actual):", flush=True)
    for a in sorted(set(ante[va].tolist())):
        m = va & (ante == a)
        if m.sum() >= 5:
            print(f"    ante {a}: n={int(m.sum())} pred={_sigmoid_pred(Xn[m], kind, locals()):.2f} actual={y[m].mean():.2f}", flush=True)
    if not use_mlp:
        order = np.argsort(-np.abs(w))
        print("  top features (|standardized weight|):", flush=True)
        for i in order[:10]:
            print(f"    {FEATURE_NAMES[i]:22} {w[i]:+.3f}", flush=True)

    if use_mlp:
        np.savez(OUT, kind="mlp", mean=mean, std=std, W1=W1, b1=b1, W2=W2, b2=b2)
    else:
        np.savez(OUT, kind="logreg", mean=mean, std=std, w=w, b=b)
    print(f"saved model -> {OUT}", flush=True)
    return 0


def _sigmoid_pred(Xn_subset, kind, ns):
    if kind == "mlp":
        return float(_sigmoid(np.tanh(Xn_subset @ ns["W1"] + ns["b1"]) @ ns["W2"] + ns["b2"]).mean())
    return float(_sigmoid(Xn_subset @ ns["w"] + ns["b"]).mean())


if __name__ == "__main__":
    raise SystemExit(main())
