"""Policy head training (Stage 2): imitate the teacher's actions.

Trains `ValueNet`'s policy heads on the `(state, action)` pairs already in the
bootstrap captures — no new data generation. Two targets:

- **action type** (14-way): cross-entropy to the chosen action's type.
- **per-card play pointer**: for play/discard steps, BCE over hand positions —
  1 where the chosen `card_indices` include that position. The trunk pools the
  hand, so card selection needs the per-card pointer (`ValueNet.policy`).

This is the AlphaZero-style search prior: once trained, it ranks candidate
actions so the beam can expand only the promising few (Stage 2.2 wires it in).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from balatro_ai.ml.dataset import TrainingExample
from balatro_ai.ml.model import (
    ACTION_TYPE_INDEX,
    PLAY_ACTION_TYPES,
    ValueNet,
    collate_states,
)


@dataclass
class PolicyConfig:
    epochs: int = 15
    lr: float = 1e-2
    batch_size: int = 512
    weight_decay: float = 1e-4
    dropout: float = 0.1
    card_loss_weight: float = 1.0
    seed: int = 0


def _type_targets(examples: Sequence[TrainingExample]) -> torch.Tensor:
    return torch.tensor(
        [ACTION_TYPE_INDEX.get(ex.action.get("type", ""), 0) for ex in examples],
        dtype=torch.long,
    )


def _card_targets(examples: Sequence[TrainingExample], height: int) -> torch.Tensor:
    target = torch.zeros(len(examples), height, dtype=torch.float32)
    for i, ex in enumerate(examples):
        for j in ex.action.get("card_indices", ()):
            if 0 <= j < height:
                target[i, j] = 1.0
    return target


def _is_play(examples: Sequence[TrainingExample]) -> torch.Tensor:
    return torch.tensor(
        [ex.action.get("type", "") in PLAY_ACTION_TYPES for ex in examples],
        dtype=torch.bool,
    )


def train_policy(
    examples: Sequence[TrainingExample],
    config: PolicyConfig | None = None,
) -> ValueNet:
    """Imitation-train the policy heads. Returns the trained `ValueNet`."""
    config = config or PolicyConfig()
    if not examples:
        raise ValueError("train_policy needs at least one example")
    torch.manual_seed(config.seed)
    model = ValueNet(dropout=config.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    n = len(examples)
    bs = config.batch_size if config.batch_size and config.batch_size > 0 else n
    rng = random.Random(config.seed)
    model.train()
    for _ in range(config.epochs):
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n, bs):
            chunk = [examples[i] for i in order[start:start + bs]]
            batch = collate_states([ex.encoded_state for ex in chunk])
            type_logits, card_logits = model.policy(batch)
            loss = ce(type_logits, _type_targets(chunk))
            # Per-card loss only on play/discard steps and valid hand positions.
            play_mask = _is_play(chunk).unsqueeze(1) & batch.card_mask
            if bool(play_mask.any()):
                card_target = _card_targets(chunk, batch.card_mask.shape[1])
                loss = loss + config.card_loss_weight * bce(
                    card_logits[play_mask], card_target[play_mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def eval_policy(
    model: ValueNet,
    examples: Sequence[TrainingExample],
    *,
    batch_size: int = 1024,
) -> dict:
    """Held-out accuracy: action-type top-1 (vs base rate) + per-card play acc."""
    model.eval()
    n = len(examples)
    type_correct = card_correct = card_total = play = subset_exact = 0
    type_counts: Counter = Counter()
    for start in range(0, n, batch_size):
        chunk = examples[start:start + batch_size]
        batch = collate_states([ex.encoded_state for ex in chunk])
        type_logits, card_logits = model.policy(batch)
        type_pred = type_logits.argmax(dim=-1)
        for i, ex in enumerate(chunk):
            t = ACTION_TYPE_INDEX.get(ex.action.get("type", ""), 0)
            type_counts[ex.action.get("type", "")] += 1
            if int(type_pred[i]) == t:
                type_correct += 1
            if ex.action.get("type", "") in PLAY_ACTION_TYPES:
                play += 1
                valid = batch.card_mask[i].nonzero().flatten().tolist()
                target = {j for j in ex.action.get("card_indices", ()) if j in valid}
                pred = {j for j in valid if float(card_logits[i, j]) > 0}
                for j in valid:
                    card_total += 1
                    if (j in target) == (j in pred):
                        card_correct += 1
                if pred == target:
                    subset_exact += 1
    base = max(type_counts.values()) / n if type_counts else 0.0
    return {
        "n": n,
        "type_acc": type_correct / n if n else 0.0,
        "type_base_rate": base,
        "n_play": play,
        "card_pos_acc": card_correct / card_total if card_total else 0.0,
        "subset_exact": subset_exact / play if play else 0.0,
    }
