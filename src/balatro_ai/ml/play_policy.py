"""Candidate-subset play policy (Stage 2.2).

The fix for the failed per-card / hand-type play heads: don't predict the play
from the pooled state — **score the enumerated candidate plays directly.** Each
candidate (a subset of the hand) is encoded from its pooled card embeddings +
hand-type + size + global context (`ValueNet.play_candidate_scores`), and the
teacher's chosen play is trained to rank above sampled alternatives
(negative-sampling cross-entropy). At inference this is a learned replacement for
the beam's cheap play-ranker: score the candidates, keep the top-k.

Eval metric: top-1 accuracy — does the teacher's actual play score highest among
{chosen + N random subsets}? Random baseline = 1/(N+1). (This measures pruning
power — ranking the real play above junk subsets — not yet discrimination among
equally-plausible alternatives.)
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from balatro_ai.ml.dataset import TrainingExample
from balatro_ai.ml.model import ValueNet, collate_states
from balatro_ai.ml.policy import _classify_hand

_PLAY = "play_hand"


@dataclass
class PlayPolicyConfig:
    epochs: int = 15
    lr: float = 1e-2
    batch_size: int = 256
    weight_decay: float = 1e-4
    dropout: float = 0.1
    n_neg: int = 15
    seed: int = 0


def _play_examples(examples: Sequence[TrainingExample]) -> list[TrainingExample]:
    out = []
    for ex in examples:
        if ex.action.get("type") != _PLAY:
            continue
        hand = ex.encoded_state.hand
        if any(0 <= j < len(hand) for j in ex.action.get("card_indices", ())):
            out.append(ex)
    return out


def _candidate_tensors(chunk, height: int, n_neg: int, rng: random.Random):
    """Build [chosen + n_neg random subsets] per example as padded tensors."""
    b = len(chunk)
    c = n_neg + 1
    masks = torch.zeros(b, c, height)
    htypes = torch.zeros(b, c, dtype=torch.long)
    sizes = torch.zeros(b, c)
    for i, ex in enumerate(chunk):
        hand = ex.encoded_state.hand
        hsize = len(hand)
        chosen = tuple(sorted(j for j in ex.action.get("card_indices", ()) if 0 <= j < hsize))
        cands = [chosen]
        seen = {chosen}
        tries = 0
        while len(cands) < c and tries < n_neg * 20:
            tries += 1
            size = rng.randint(1, min(5, hsize))
            sub = tuple(sorted(rng.sample(range(hsize), size)))
            if sub not in seen:
                seen.add(sub)
                cands.append(sub)
        while len(cands) < c:
            cands.append(())  # degenerate negative (small hand)
        for k, sub in enumerate(cands):
            for j in sub:
                masks[i, k, j] = 1.0
            sizes[i, k] = len(sub)
            if sub:
                htypes[i, k] = _classify_hand(
                    [hand[j].rank_index for j in sub],
                    [hand[j].suit_index for j in sub])
    return masks, htypes, sizes


def train_play_policy(
    examples: Sequence[TrainingExample],
    config: PlayPolicyConfig | None = None,
) -> ValueNet:
    config = config or PlayPolicyConfig()
    plays = _play_examples(examples)
    if not plays:
        raise ValueError("train_play_policy needs play_hand examples")
    torch.manual_seed(config.seed)
    model = ValueNet(dropout=config.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ce = nn.CrossEntropyLoss()
    n = len(plays)
    bs = config.batch_size
    rng = random.Random(config.seed)
    model.train()
    for _ in range(config.epochs):
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n, bs):
            chunk = [plays[i] for i in order[start:start + bs]]
            batch = collate_states([ex.encoded_state for ex in chunk])
            masks, htypes, sizes = _candidate_tensors(
                chunk, batch.card_mask.shape[1], config.n_neg, rng)
            scores = model.play_candidate_scores(batch, masks, htypes, sizes)
            target = torch.zeros(len(chunk), dtype=torch.long)  # chosen at index 0
            opt.zero_grad()
            ce(scores, target).backward()
            opt.step()
    return model


@torch.no_grad()
def eval_play_policy(
    model: ValueNet,
    examples: Sequence[TrainingExample],
    *,
    n_neg: int = 31,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    """top-1: does the teacher's play score highest vs `n_neg` random subsets?"""
    model.eval()
    plays = _play_examples(examples)
    rng = random.Random(seed)
    correct = 0
    n = len(plays)
    for start in range(0, n, batch_size):
        chunk = plays[start:start + batch_size]
        batch = collate_states([ex.encoded_state for ex in chunk])
        masks, htypes, sizes = _candidate_tensors(
            chunk, batch.card_mask.shape[1], n_neg, rng)
        scores = model.play_candidate_scores(batch, masks, htypes, sizes)
        correct += int((scores.argmax(dim=-1) == 0).sum())
    return {
        "n_play": n,
        "top1_acc": correct / n if n else 0.0,
        "random_baseline": 1.0 / (n_neg + 1),
    }
