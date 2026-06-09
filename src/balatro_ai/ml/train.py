"""Training harness for the value net (Stage 0.3).

A minimal, dependency-light torch loop over `TrainingExample`s
(`ml/dataset.py`): collate → forward `ValueNet` → BCE on the win label →
Adam step, with an eval split, checkpoint save/load (recording the encoding
version + architecture so it can be rebuilt), and an `overfit_check` used as the
Stage 0.3 gate ("the wiring can fit a tiny synthetic set").

Only the value (win-probability) head is trained here; ante/score regression
heads are a straightforward extension once the loop is proven. Imports torch, so
this module (and `model.py`) are the torch-gated part of the ml layer —
`encoding.py`/`dataset.py` stay importable without it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from balatro_ai.ml.dataset import TrainingExample
from balatro_ai.ml.encoding import ENCODING_VERSION
from balatro_ai.ml.model import ValueNet, collate_states


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 1e-2
    batch_size: int = 0      # 0 = full batch
    weight_decay: float = 0.0
    ante_loss_weight: float = 1.0
    dropout: float = 0.0
    seed: int = 0
    encoder: str = "mean"


@dataclass
class TrainResult:
    model: ValueNet
    history: list[float] = field(default_factory=list)  # per-epoch train loss
    final_train_loss: float = 0.0
    final_val_loss: float | None = None


def _labels(examples: Sequence[TrainingExample]) -> torch.Tensor:
    return torch.tensor([1.0 if ex.value.won else 0.0 for ex in examples], dtype=torch.float32)


def _states(examples: Sequence[TrainingExample]):
    return [ex.encoded_state for ex in examples]


def _ante_targets(examples: Sequence[TrainingExample]) -> torch.Tensor:
    """Final ante normalized to [0, 1] (ante 8 = 1.0) — the dense value target."""
    return torch.tensor(
        [min(1.0, max(0.0, ex.value.final_ante / 8.0)) for ex in examples],
        dtype=torch.float32,
    )


def _auc(scores: list[float], labels: list[float]) -> float:
    """Mann-Whitney AUC; nan if a class is absent."""
    pos = [s for s, y in zip(scores, labels) if y >= 0.5]
    neg = [s for s, y in zip(scores, labels) if y < 0.5]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def split_examples(
    examples: Sequence[TrainingExample],
    val_frac: float,
    *,
    seed: int = 0,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Deterministic train/val split by example (not by seed-run)."""
    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = int(len(items) * val_frac)
    return items[n_val:], items[:n_val]


def train(
    examples: Sequence[TrainingExample],
    config: TrainConfig | None = None,
    *,
    val_examples: Sequence[TrainingExample] | None = None,
) -> TrainResult:
    """Train a fresh `ValueNet` on `examples`. Deterministic given `config.seed`."""
    config = config or TrainConfig()
    if not examples:
        raise ValueError("train() needs at least one example")
    torch.manual_seed(config.seed)

    model = ValueNet(dropout=config.dropout, encoder=config.encoder)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    states = _states(examples)
    labels = _labels(examples)
    ante = _ante_targets(examples)
    n = len(states)
    bs = config.batch_size if config.batch_size and config.batch_size > 0 else n

    history: list[float] = []
    rng = random.Random(config.seed)
    model.train()
    for _ in range(config.epochs):
        order = list(range(n))
        rng.shuffle(order)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            batch = collate_states([states[i] for i in idx])
            opt.zero_grad()
            win_logit, ante_pred = model.heads(batch)
            loss = bce(win_logit, labels[idx]) + config.ante_loss_weight * mse(ante_pred, ante[idx])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        history.append(epoch_loss / max(1, n_batches))

    final_val = None
    if val_examples:
        final_val = evaluate(model, val_examples)[0]
    return TrainResult(
        model=model,
        history=history,
        final_train_loss=history[-1] if history else 0.0,
        final_val_loss=final_val,
    )


@torch.no_grad()
def evaluate(model: ValueNet, examples: Sequence[TrainingExample]) -> tuple[float, float]:
    """Return (BCE loss, win-prediction accuracy) over `examples`."""
    model.eval()
    states = _states(examples)
    labels = _labels(examples)
    batch = collate_states(states)
    logits = model(batch)
    loss = float(nn.BCEWithLogitsLoss()(logits, labels))
    preds = (torch.sigmoid(logits) >= 0.5).float()
    accuracy = float((preds == labels).float().mean())
    return loss, accuracy


@torch.no_grad()
def evaluate_full(model: ValueNet, examples: Sequence[TrainingExample]) -> dict:
    """Both heads: win loss/acc/AUC + ante MSE/correlation over `examples`."""
    model.eval()
    win = _labels(examples)
    ante = _ante_targets(examples)
    batch = collate_states(_states(examples))
    win_logit, ante_pred = model.heads(batch)
    win_prob = torch.sigmoid(win_logit)
    return {
        "n": len(examples),
        "win_loss": float(nn.BCEWithLogitsLoss()(win_logit, win)),
        "win_acc": float(((win_prob >= 0.5).float() == win).float().mean()),
        "win_auc": _auc(win_prob.tolist(), win.tolist()),
        "ante_mse": float(nn.MSELoss()(ante_pred, ante)),
        "ante_corr": _pearson(ante_pred.tolist(), ante.tolist()),
    }


def overfit_check(
    examples: Sequence[TrainingExample],
    config: TrainConfig | None = None,
    *,
    loss_threshold: float = 0.05,
) -> tuple[bool, float, float]:
    """Stage 0.3 gate: can the net fit a tiny set? Returns (ok, loss, accuracy)."""
    result = train(examples, config)
    loss, accuracy = evaluate(result.model, examples)
    return (loss < loss_threshold and accuracy >= 0.999), loss, accuracy


def save_checkpoint(model: ValueNet, path: str | Path) -> None:
    """Persist weights + the spec/hparams/encoding-version needed to rebuild."""
    torch.save(
        {
            "encoding_version": model.encoding_version,
            "spec": model.spec,
            "hparams": model.hparams,
            "state_dict": model.state_dict(),
        },
        str(path),
    )


def load_checkpoint(path: str | Path, *, strict_version: bool = True) -> ValueNet:
    """Rebuild a `ValueNet` from a checkpoint.

    Raises if the checkpoint's encoding version differs from the current
    `ENCODING_VERSION` (set `strict_version=False` to load anyway).
    """
    path = Path(path)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if strict_version and ckpt.get("encoding_version") != ENCODING_VERSION:
        raise ValueError(
            f"{path}: checkpoint encoding_version={ckpt.get('encoding_version')} "
            f"!= current ENCODING_VERSION={ENCODING_VERSION}; retrain/regenerate "
            "the checkpoint with the current encoder or pass a current-version ckpt"
        )
    model = ValueNet(ckpt["spec"], **ckpt.get("hparams", {}))
    # strict=False so checkpoints predating a newly-added head (e.g. clear_head)
    # still load — the missing head keeps its init weights and just isn't used.
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model
