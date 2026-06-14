"""Decision-shaped policy net (Phase B, component 4).

The chassis-replacement architecture: ONE network scores the legal candidate
actions of EVERY decision and picks one (softmax argmax), with a value head
for diagnostics/baselines. Reuses the `ValueNet` set-encoder trunk for state
context; adds a generic candidate-scoring head over the schema-v2
`CandidateToken` features (action-type embedding + structural features + the
heuristic's fused play-score).

Iteration 0 is behavior cloning: train the policy to reproduce the action the
(recipe-mixture) policy took, over ALL trajectories (diversity preserved; no
outcome weighting until iteration 1+). Gate B0 is a plumbing check — does this
net, deployed, reproduce the mixture's winrate?

Anti-shortcut harness (review demand): the fused play-score feature can be
dropped at train/eval time, and `evaluate` reports top-1 WITH and WITHOUT it —
if the net only works with the feature present, it learned argmax-of-heuristic
and starved the trunk (the Stage-2.3 lossy-compression failure in disguise).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from balatro_ai.ml.dataset import _ACTION_TYPE_ORDER, CandidateToken, TrainingExample
from balatro_ai.ml.model import Batch, ValueNet, collate_states

_N_ACTION_TYPES = len(_ACTION_TYPE_ORDER)
_CAND_NNUM = 6  # n_cards, amount, has_target, play_score, has_play_score, heuristic_choice
_HEURISTIC_COLS = (3, 4, 5)  # play_score, has_play_score, heuristic_choice (ablation drops these)


@dataclass
class PolicyConfig:
    epochs: int = 20
    lr: float = 3e-3
    batch_size: int = 256
    weight_decay: float = 1e-4
    dropout: float = 0.1
    value_weight: float = 1.0
    d_type: int = 8
    d_hidden: int = 64
    n_neg: int = 31      # training: score chosen + n_neg sampled negatives, not all
    eval_sample: int = 2000  # eval top-1 on this many examples (full candidate sets)
    seed: int = 0


@dataclass
class CandidateBatch:
    """A padded minibatch of (state, candidate-set, label, outcome)."""

    state: Batch
    cand_type: torch.Tensor   # [B, C] long — action-type index per candidate
    cand_num: torch.Tensor    # [B, C, _CAND_NNUM] float
    cand_mask: torch.Tensor   # [B, C] bool — True = real candidate
    chosen: torch.Tensor      # [B] long — index of the taken candidate
    won: torch.Tensor         # [B] float — run outcome (value target)


def _labelled(examples: Sequence[TrainingExample]) -> list[TrainingExample]:
    """Keep only examples with a real candidate set AND a found chosen index."""
    return [e for e in examples if e.candidates and 0 <= e.chosen_index < len(e.candidates)]


def collate_candidates(examples: Sequence[TrainingExample]) -> CandidateBatch:
    state = collate_states([e.encoded_state for e in examples])
    max_c = max(len(e.candidates) for e in examples)
    # Bulk construction via Python lists + a single torch.tensor per field —
    # element-wise tensor assignment was the training bottleneck (~140
    # candidates x examples of tiny tensor allocs per batch per epoch).
    pad_num = [0.0] * _CAND_NNUM
    type_rows: list[list[int]] = []
    num_rows: list[list[list[float]]] = []
    mask_rows: list[list[bool]] = []
    chosen: list[int] = []
    won: list[float] = []
    for ex in examples:
        cs = ex.candidates
        pad = max_c - len(cs)
        type_rows.append([c.action_type_index for c in cs] + [0] * pad)
        num_rows.append(
            [[c.n_cards, c.amount, c.has_target, c.play_score, c.has_play_score,
              c.heuristic_choice] for c in cs]
            + [pad_num] * pad
        )
        mask_rows.append([True] * len(cs) + [False] * pad)
        chosen.append(ex.chosen_index)
        won.append(1.0 if ex.value.won else 0.0)
    return CandidateBatch(
        state,
        torch.tensor(type_rows, dtype=torch.long),
        torch.tensor(num_rows, dtype=torch.float),
        torch.tensor(mask_rows, dtype=torch.bool),
        torch.tensor(chosen, dtype=torch.long),
        torch.tensor(won, dtype=torch.float),
    )


def collate_candidates_sampled(
    examples: Sequence[TrainingExample], n_neg: int, rng: random.Random
) -> CandidateBatch:
    """Training collate with negative sampling: each example keeps the CHOSEN
    candidate (at index 0) + up to `n_neg` random negatives. Cuts the candidate
    axis from ~440 (play decisions) to ~32 — the dominant training cost — with
    no inference-side change (the bot still scores all candidates). Chosen is at
    index 0, so the CE target is uniformly 0; the scorer is position-free so this
    introduces no bias."""
    sub: list[TrainingExample] = []
    for ex in examples:
        cs = ex.candidates
        others = [i for i in range(len(cs)) if i != ex.chosen_index]
        rng.shuffle(others)
        keep = [ex.chosen_index] + others[:n_neg]
        sub.append(
            TrainingExample(
                step=ex.step, phase=ex.phase, encoded_state=ex.encoded_state,
                action=ex.action, value=ex.value, steps_to_end=ex.steps_to_end,
                candidates=tuple(cs[i] for i in keep), chosen_index=0,
            )
        )
    return collate_candidates(sub)


class DecisionPolicyNet(nn.Module):
    """ValueNet trunk (state context + win head) + a generic candidate head."""

    def __init__(self, config: PolicyConfig, *, spec: dict | None = None) -> None:
        super().__init__()
        self.value_net = ValueNet(spec=spec, dropout=config.dropout)
        d_trunk = self.value_net.hparams["d_trunk"]
        self.type_emb = nn.Embedding(_N_ACTION_TYPES, config.d_type)
        self.cand_mlp = nn.Sequential(
            nn.Linear(d_trunk + config.d_type + _CAND_NNUM, config.d_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, 1),
        )

    def candidate_logits(
        self, cb: CandidateBatch, *, drop_heuristic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (candidate_logits [B, C] with masked = -inf, win_logit [B])."""
        trunk = self.value_net._trunk(cb.state)            # [B, d_trunk]
        win_logit = self.value_net.win_head(trunk).squeeze(-1)
        b, c = cb.cand_type.shape
        type_e = self.type_emb(cb.cand_type)               # [B, C, d_type]
        num = cb.cand_num
        if drop_heuristic:
            # Anti-shortcut ablation: zero ALL heuristic hints (play_score +
            # has_play_score + heuristic_choice) to test whether the trunk has
            # signal independent of the heuristic, or is a pure copy.
            num = num.clone()
            for col in _HEURISTIC_COLS:
                num[..., col] = 0.0
        ctx = trunk.unsqueeze(1).expand(b, c, -1)          # [B, C, d_trunk]
        feats = torch.cat([ctx, type_e, num], dim=-1)
        logits = self.cand_mlp(feats).squeeze(-1)          # [B, C]
        logits = logits.masked_fill(~cb.cand_mask, float("-inf"))
        return logits, win_logit


def train_decision_policy(
    examples: Sequence[TrainingExample],
    config: PolicyConfig | None = None,
) -> tuple[DecisionPolicyNet, dict]:
    config = config or PolicyConfig()
    data = _labelled(examples)
    if not data:
        raise ValueError("train_decision_policy needs labelled candidate examples")
    torch.manual_seed(config.seed)
    net = DecisionPolicyNet(config)
    opt = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = random.Random(config.seed)
    n, bs = len(data), config.batch_size
    net.train()
    last_loss = 0.0
    for _ in range(config.epochs):
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n, bs):
            chunk = [data[i] for i in order[start:start + bs]]
            cb = collate_candidates_sampled(chunk, config.n_neg, rng)
            logits, win_logit = net.candidate_logits(cb)
            policy_loss = F.cross_entropy(logits, cb.chosen)
            value_loss = F.binary_cross_entropy_with_logits(win_logit, cb.won)
            loss = policy_loss + config.value_weight * value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
    metrics = evaluate(net, data, sample=config.eval_sample, seed=config.seed)
    metrics["final_loss"] = round(last_loss, 4)
    metrics["n_examples"] = n
    return net, metrics


@torch.no_grad()
def evaluate(
    net: DecisionPolicyNet,
    examples: Sequence[TrainingExample],
    *,
    sample: int | None = None,
    seed: int = 0,
) -> dict:
    """Top-1 BC accuracy (does the net rank the taken action first?) WITH and
    WITHOUT the fused play-score feature — the anti-shortcut ablation. Scores the
    FULL candidate set (unlike sampled training); capped to `sample` examples to
    keep the [N, ~440] tensor bounded."""
    data = _labelled(examples)
    if not data:
        return {"top1": 0.0, "top1_no_heuristic": 0.0, "value_auc": 0.0}
    if sample is not None and len(data) > sample:
        data = random.Random(seed).sample(data, sample)
    net.eval()
    cb = collate_candidates(data)
    logits, win_logit = net.candidate_logits(cb)
    top1 = (logits.argmax(dim=1) == cb.chosen).float().mean().item()
    logits_abl, _ = net.candidate_logits(cb, drop_heuristic=True)
    top1_abl = (logits_abl.argmax(dim=1) == cb.chosen).float().mean().item()
    # chance baseline: 1 / mean candidate count
    n_cands = cb.cand_mask.sum(dim=1).float().mean().item()
    return {
        "top1": round(top1, 4),
        "top1_no_heuristic": round(top1_abl, 4),
        "chance": round(1.0 / max(1.0, n_cands), 4),
        "value_auc": round(_auc(torch.sigmoid(win_logit), cb.won), 4),
        "mean_candidates": round(n_cands, 1),
    }


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    # Mann-Whitney U / (n_pos * n_neg).
    wins = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
    return float((wins + 0.5 * ties) / (pos.numel() * neg.numel()))


# --------------------------------------------------------------------------- #
# Checkpoint I/O + single-state inference (the deployed-bot seam).
# --------------------------------------------------------------------------- #

def save_policy(net: DecisionPolicyNet, path: str, *, config: PolicyConfig) -> None:
    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": vars(config),
            "spec": net.value_net.spec,
        },
        path,
    )


def load_policy(path: str) -> DecisionPolicyNet:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**{k: v for k, v in blob["config"].items() if k in PolicyConfig.__annotations__})
    net = DecisionPolicyNet(cfg, spec=blob.get("spec"))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net


@torch.no_grad()
def best_candidate_index(net: DecisionPolicyNet, encoded_state, candidates) -> int:
    """Argmax candidate for one state — the deployed bot's scoring call.
    `candidates` is a tuple of CandidateToken (parallel to legal_actions)."""
    if not candidates:
        return -1
    from balatro_ai.ml.dataset import TrainingExample, ValueTarget

    ex = TrainingExample(
        step=0, phase="", encoded_state=encoded_state, action={}, value=ValueTarget(False, 0, 0),
        steps_to_end=0, candidates=tuple(candidates), chosen_index=0,
    )
    cb = collate_candidates([ex])
    logits, _ = net.candidate_logits(cb)
    return int(logits[0].argmax().item())
