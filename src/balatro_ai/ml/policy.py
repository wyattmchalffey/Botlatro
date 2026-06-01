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
from balatro_ai.ml.encoding import POKER_HANDS
from balatro_ai.ml.model import (
    ACTION_TYPE_INDEX,
    PLAY_ACTION_TYPES,
    ValueNet,
    collate_states,
)

_HAND_INDEX = {h: i for i, h in enumerate(POKER_HANDS)}


def _classify_hand(ranks: list[int], suits: list[int]) -> int:
    """Poker hand-type of a played subset, from encoded rank/suit indices.

    A standard classifier (ignores Four Fingers / Shortcut joker relaxations — a
    small approximation for the policy target). Matches `POKER_HANDS` order.
    """
    n = len(ranks)
    counts = sorted(Counter(ranks).values(), reverse=True)
    real = [s for s in suits if s < 4]
    is_flush = n >= 5 and len(real) == n and len(set(real)) == 1
    uniq = sorted(set(ranks))
    is_straight = n >= 5 and len(uniq) == 5 and (
        uniq[-1] - uniq[0] == 4 or set(uniq) == {0, 1, 2, 3, 12})
    if counts[0] == 5:
        return _HAND_INDEX["Flush Five" if is_flush else "Five of a Kind"]
    if counts[:2] == [3, 2]:
        return _HAND_INDEX["Flush House" if is_flush else "Full House"]
    if is_straight and is_flush:
        return _HAND_INDEX["Straight Flush"]
    if counts[0] == 4:
        return _HAND_INDEX["Four of a Kind"]
    if is_flush:
        return _HAND_INDEX["Flush"]
    if is_straight:
        return _HAND_INDEX["Straight"]
    if counts[0] == 3:
        return _HAND_INDEX["Three of a Kind"]
    if counts.count(2) >= 2:
        return _HAND_INDEX["Two Pair"]
    if counts[0] == 2:
        return _HAND_INDEX["Pair"]
    return _HAND_INDEX["High Card"]


def _play_hand_type(ex: TrainingExample) -> int | None:
    """Hand-type the chosen play forms, or None for non-play steps / no cards."""
    if ex.action.get("type", "") not in PLAY_ACTION_TYPES:
        return None
    hand = ex.encoded_state.hand
    cards = [hand[j] for j in ex.action.get("card_indices", ()) if 0 <= j < len(hand)]
    if not cards:
        return None
    return _classify_hand([c.rank_index for c in cards], [c.suit_index for c in cards])


@dataclass
class PolicyConfig:
    epochs: int = 15
    lr: float = 1e-2
    batch_size: int = 512
    weight_decay: float = 1e-4
    dropout: float = 0.1
    card_loss_weight: float = 1.0
    hand_type_loss_weight: float = 1.0
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
            ht_idx = [_play_hand_type(ex) for ex in chunk]
            ht_pos = [i for i, v in enumerate(ht_idx) if v is not None]
            if ht_pos:
                ht_logits = model.hand_type_logits(batch)[ht_pos]
                ht_target = torch.tensor([ht_idx[i] for i in ht_pos], dtype=torch.long)
                loss = loss + config.hand_type_loss_weight * ce(ht_logits, ht_target)
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
    ht_correct = ht_total = 0
    type_counts: Counter = Counter()
    ht_counts: Counter = Counter()
    for start in range(0, n, batch_size):
        chunk = examples[start:start + batch_size]
        batch = collate_states([ex.encoded_state for ex in chunk])
        type_logits, card_logits = model.policy(batch)
        type_pred = type_logits.argmax(dim=-1)
        ht_pred = model.hand_type_logits(batch).argmax(dim=-1)
        for i, ex in enumerate(chunk):
            t = ACTION_TYPE_INDEX.get(ex.action.get("type", ""), 0)
            type_counts[ex.action.get("type", "")] += 1
            if int(type_pred[i]) == t:
                type_correct += 1
            if ex.action.get("type", "") in PLAY_ACTION_TYPES:
                play += 1
                ht = _play_hand_type(ex)
                if ht is not None:
                    ht_total += 1
                    ht_counts[ht] += 1
                    if int(ht_pred[i]) == ht:
                        ht_correct += 1
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
    ht_base = max(ht_counts.values()) / ht_total if ht_counts else 0.0
    return {
        "n": n,
        "type_acc": type_correct / n if n else 0.0,
        "type_base_rate": base,
        "n_play": play,
        "card_pos_acc": card_correct / card_total if card_total else 0.0,
        "subset_exact": subset_exact / play if play else 0.0,
        "hand_type_acc": ht_correct / ht_total if ht_total else 0.0,
        "hand_type_base_rate": ht_base,
    }
