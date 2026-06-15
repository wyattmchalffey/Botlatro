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
# Candidate numeric feature columns (order MUST match collate_candidates' rows
# and CandidateToken). v2 added hand_strength/has_hand_type/is_discard (cols
# 6-8, objective facts about the action) and draw_odds/has_draw_odds (cols 9-10,
# the heuristic's estimate).
_CAND_NNUM = 11
# Anti-shortcut ablation drops the heuristic's OPINION features (its score, its
# pick, its draw-odds estimate) but KEEPS objective action facts (hand_strength,
# is_discard) — so the ablation tests "without the heuristic's opinions, can the
# trunk still rank?", not "without knowing what the action is".
_HEURISTIC_COLS = (3, 4, 5, 9, 10)


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

    def to(self, device) -> "CandidateBatch":
        return CandidateBatch(
            self.state.to(device), self.cand_type.to(device), self.cand_num.to(device),
            self.cand_mask.to(device), self.chosen.to(device), self.won.to(device),
        )


def _resolve_device():
    """Training device: BALATRO_DEVICE override, else CUDA if present, else CPU.
    GPU training is ~4x cheaper+faster for this tiny net at scale; CPU is the
    default so local runs and the bench are unaffected."""
    import os

    name = os.environ.get("BALATRO_DEVICE", "").strip()
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _net_device(net: nn.Module):
    return next(net.parameters()).device


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
              c.heuristic_choice, c.hand_strength, c.has_hand_type, c.is_discard,
              c.draw_odds, c.has_draw_odds] for c in cs]
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
    *,
    val_examples: Sequence[TrainingExample] | None = None,
    example_weights: Sequence[float] | None = None,
) -> tuple[DecisionPolicyNet, dict]:
    """BC-train the decision policy. If `val_examples` (a HELD-OUT run set) is
    given, track held-out value-AUC each epoch and KEEP THE BEST checkpoint —
    the value-head salvage for V0 (the B0 head overfit: train 0.99 / held 0.63
    from training all epochs with no early-stop). Held-out val avoids the
    in-dataset leakage of splitting examples that share a run's outcome label.

    `example_weights` (aligned with `_labelled(examples)` order) weights each
    example's POLICY loss — the iteration-1 improvement operator: advantage
    weights upweight decisions from winning trajectories, so the policy tilts
    toward what won. Uniform weights == plain BC (iteration 0)."""
    config = config or PolicyConfig()
    data = _labelled(examples)
    if not data:
        raise ValueError("train_decision_policy needs labelled candidate examples")
    weights = list(example_weights) if example_weights is not None else None
    if weights is not None and len(weights) != len(data):
        raise ValueError(f"example_weights ({len(weights)}) must align with labelled data ({len(data)})")
    torch.manual_seed(config.seed)
    device = _resolve_device()
    net = DecisionPolicyNet(config).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = random.Random(config.seed)
    n, bs = len(data), config.batch_size
    val_data = _labelled(val_examples) if val_examples else None
    best_val_auc, best_state, best_epoch = -1.0, None, -1
    last_loss = 0.0
    for epoch in range(config.epochs):
        net.train()
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n, bs):
            idxs = order[start:start + bs]
            chunk = [data[i] for i in idxs]
            cb = collate_candidates_sampled(chunk, config.n_neg, rng).to(device)
            logits, win_logit = net.candidate_logits(cb)
            if weights is None:
                policy_loss = F.cross_entropy(logits, cb.chosen)
            else:
                ce = F.cross_entropy(logits, cb.chosen, reduction="none")
                w = torch.tensor([weights[i] for i in idxs], dtype=torch.float, device=device)
                policy_loss = (ce * w).sum() / w.sum().clamp_min(1e-6)
            value_loss = F.binary_cross_entropy_with_logits(win_logit, cb.won)
            loss = policy_loss + config.value_weight * value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
        if val_data is not None:
            va = _held_out_value_auc(net, val_data, sample=config.eval_sample, seed=config.seed)
            if va > best_val_auc:
                import copy
                best_val_auc, best_epoch = va, epoch
                best_state = copy.deepcopy(net.state_dict())
    if best_state is not None:
        net.load_state_dict(best_state)  # restore best-held-out-value checkpoint
    metrics = evaluate(net, data, sample=config.eval_sample, seed=config.seed)
    metrics["final_loss"] = round(last_loss, 4)
    metrics["n_examples"] = n
    if val_data is not None:
        metrics["best_val_value_auc"] = round(best_val_auc, 4)
        metrics["best_epoch"] = best_epoch
    return net, metrics


@torch.no_grad()
def compute_advantage_weights(
    examples: Sequence[TrainingExample],
    baseline: DecisionPolicyNet,
    *,
    beta: float = 2.0,
    w_max: float = 5.0,
    batch: int = 512,
    per_policy_blend: float = 0.0,
) -> list[float]:
    """Per-example AWR weights = clip(exp(beta * (R - base)), 0, w_max), where
    R = run outcome (won) and `base` is the advantage baseline. With
    per_policy_blend=0, base = V(s), the FROZEN value head (the salvaged V0
    head) — the standard state-conditioned baseline.

    With per_policy_blend>0, base blends V(s) with the GENERATING POLICY's mean
    winrate: base = (1-b)*V(s) + b*mean_winrate(source). This is the v2 fix for
    'naive winner-upweighting deletes diversity' — a low-winrate recipe's wins
    are upweighted relative to ITS OWN baseline, not the global one, so the
    high-variance recipes keep contributing their above-average lines instead
    of being flattened by a neural-dominated global baseline.

    Aligned with `_labelled(examples)` order (the trainer's data order)."""
    data = _labelled(examples)
    policy_mean: dict[str, float] = {}
    if per_policy_blend > 0.0:
        from collections import defaultdict

        wins: dict[str, int] = defaultdict(int)
        tot: dict[str, int] = defaultdict(int)
        for ex in data:
            tot[ex.source] += 1
            wins[ex.source] += 1 if ex.value.won else 0
        policy_mean = {k: wins[k] / max(1, tot[k]) for k in tot}
    baseline.eval()
    weights: list[float] = []
    import math
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        cb = collate_candidates(chunk)
        _, win_logit = baseline.candidate_logits(cb)
        v = torch.sigmoid(win_logit).tolist()
        for ex, vs in zip(chunk, v):
            r = 1.0 if ex.value.won else 0.0
            base = vs
            if per_policy_blend > 0.0:
                pm = policy_mean.get(ex.source, vs)
                base = (1.0 - per_policy_blend) * vs + per_policy_blend * pm
            weights.append(min(w_max, math.exp(beta * (r - base))))
    return weights


# --------------------------------------------------------------------------- #
# Out-of-core (sharded) training — for 50k-scale data that won't fit in RAM.
# Mirrors train_decision_policy / compute_advantage_weights exactly, but streams
# one shard at a time (dataset.ShardedExampleStore) instead of a full in-RAM list.
# --------------------------------------------------------------------------- #

def _shard_weight_path(weight_dir: str, idx: int) -> str:
    import os
    return os.path.join(weight_dir, f"w{idx:05d}.pkl")


@torch.no_grad()
def compute_advantage_weights_sharded(
    store,
    baseline: "DecisionPolicyNet",
    weight_dir: str,
    *,
    beta: float = 2.0,
    w_max: float = 5.0,
    per_policy_blend: float = 0.0,
    batch: int = 512,
) -> dict:
    """Per-shard AWR weight files (w{idx}.pkl, each aligned with _labelled(shard)).
    Two streaming passes: (1) global per-policy mean winrate over all shards,
    (2) write weights per shard. Never holds more than one shard in RAM."""
    import os
    import pickle
    import math
    from collections import defaultdict

    os.makedirs(weight_dir, exist_ok=True)
    policy_mean: dict[str, float] = {}
    if per_policy_blend > 0.0:
        wins: dict[str, int] = defaultdict(int)
        tot: dict[str, int] = defaultdict(int)
        for _, examples in store.iter_shards():
            for ex in _labelled(examples):
                tot[ex.source] += 1
                wins[ex.source] += 1 if ex.value.won else 0
        policy_mean = {k: wins[k] / max(1, tot[k]) for k in tot}
    baseline.eval()
    n_written = 0
    for idx, examples in store.iter_shards():
        data = _labelled(examples)
        weights: list[float] = []
        for i in range(0, len(data), batch):
            chunk = data[i:i + batch]
            cb = collate_candidates(chunk)
            _, win_logit = baseline.candidate_logits(cb)
            v = torch.sigmoid(win_logit).tolist()
            for ex, vs in zip(chunk, v):
                r = 1.0 if ex.value.won else 0.0
                base = vs
                if per_policy_blend > 0.0:
                    pm = policy_mean.get(ex.source, vs)
                    base = (1.0 - per_policy_blend) * vs + per_policy_blend * pm
                weights.append(min(w_max, math.exp(beta * (r - base))))
        with open(_shard_weight_path(weight_dir, idx), "wb") as fh:
            pickle.dump(weights, fh, protocol=pickle.HIGHEST_PROTOCOL)
        n_written += len(weights)
    return {"weight_dir": weight_dir, "n_weights": n_written, "policy_mean": policy_mean}


def train_decision_policy_sharded(
    store,
    config: PolicyConfig | None = None,
    *,
    val_examples: Sequence[TrainingExample] | None = None,
    weight_dir: str | None = None,
    extra_examples: Sequence[TrainingExample] | None = None,
    extra_weight: float = 1.0,
) -> tuple[DecisionPolicyNet, dict]:
    """Streaming BC/AWR training over a ShardedExampleStore. Same loss/neg-
    sampling/value-head/best-checkpoint logic as train_decision_policy, but
    one shard resident at a time. `weight_dir` supplies per-shard AWR weights
    (compute_advantage_weights_sharded). `extra_examples` (e.g. the small
    fork-audit label set, in RAM) is appended each epoch at a flat `extra_weight`
    — the out-of-core analog of concatenating mixture+fork in the in-RAM path."""
    import os
    import pickle

    config = config or PolicyConfig()
    torch.manual_seed(config.seed)
    device = _resolve_device()
    net = DecisionPolicyNet(config).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = random.Random(config.seed)
    bs = config.batch_size
    val_data = _labelled(val_examples) if val_examples else None
    extra = _labelled(extra_examples) if extra_examples else []
    best_val_auc, best_state, best_epoch = -1.0, None, -1
    last_loss = 0.0

    def _run_chunk(chunk, weights):
        nonlocal last_loss
        cb = collate_candidates_sampled(chunk, config.n_neg, rng).to(device)
        logits, win_logit = net.candidate_logits(cb)
        if weights is None:
            policy_loss = F.cross_entropy(logits, cb.chosen)
        else:
            ce = F.cross_entropy(logits, cb.chosen, reduction="none")
            w = torch.tensor(weights, dtype=torch.float, device=device)
            policy_loss = (ce * w).sum() / w.sum().clamp_min(1e-6)
        value_loss = F.binary_cross_entropy_with_logits(win_logit, cb.won)
        loss = policy_loss + config.value_weight * value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())

    for epoch in range(config.epochs):
        net.train()
        for idx, examples in store.iter_shards(shuffle=True, rng=rng):
            data = _labelled(examples)
            shard_w = None
            if weight_dir is not None:
                wp = _shard_weight_path(weight_dir, idx)
                if os.path.exists(wp):
                    with open(wp, "rb") as fh:
                        shard_w = pickle.load(fh)
            # append the flat-weighted extra set into this shard's pool
            pool = data + extra
            pool_w = None
            if shard_w is not None:
                pool_w = list(shard_w) + [extra_weight] * len(extra)
            order = list(range(len(pool)))
            rng.shuffle(order)
            for start in range(0, len(pool), bs):
                idxs = order[start:start + bs]
                chunk = [pool[i] for i in idxs]
                w = [pool_w[i] for i in idxs] if pool_w is not None else None
                _run_chunk(chunk, w)
        if val_data is not None:
            va = _held_out_value_auc(net, val_data, sample=config.eval_sample, seed=config.seed)
            if va > best_val_auc:
                import copy
                best_val_auc, best_epoch = va, epoch
                best_state = copy.deepcopy(net.state_dict())
    if best_state is not None:
        net.load_state_dict(best_state)
    # metrics: top1 on one representative shard (bounded RAM), value-auc on val
    sample_shard = next(iter(store.iter_shards()), (None, []))[1]
    metrics = evaluate(net, _labelled(sample_shard), sample=config.eval_sample, seed=config.seed)
    metrics["final_loss"] = round(last_loss, 4)
    metrics["n_examples"] = store.n_examples
    if val_data is not None:
        metrics["best_val_value_auc"] = round(best_val_auc, 4)
        metrics["best_epoch"] = best_epoch
    return net, metrics


@torch.no_grad()
def _held_out_value_auc(
    net: DecisionPolicyNet, val_data: Sequence[TrainingExample], *, sample: int, seed: int
) -> float:
    data = list(val_data)
    if len(data) > sample:
        data = random.Random(seed).sample(data, sample)
    net.eval()
    cb = collate_candidates(data).to(_net_device(net))
    _, win_logit = net.candidate_logits(cb)
    return _auc(torch.sigmoid(win_logit), cb.won)


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
    cb = collate_candidates(data).to(_net_device(net))
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
def candidate_logit_vector(net: DecisionPolicyNet, encoded_state, candidates) -> torch.Tensor:
    """The full [C] candidate-logit vector for one state (masked entries already
    dropped — every candidate here is real). The scoring primitive behind both
    the deployed argmax bot and the on-policy exploration sampler (which needs
    the whole distribution to measure top-2 margin and temperature-sample)."""
    from balatro_ai.ml.dataset import TrainingExample, ValueTarget

    ex = TrainingExample(
        step=0, phase="", encoded_state=encoded_state, action={}, value=ValueTarget(False, 0, 0),
        steps_to_end=0, candidates=tuple(candidates), chosen_index=0,
    )
    cb = collate_candidates([ex])
    logits, _ = net.candidate_logits(cb)
    return logits[0]


@torch.no_grad()
def best_candidate_index(net: DecisionPolicyNet, encoded_state, candidates) -> int:
    """Argmax candidate for one state — the deployed bot's scoring call.
    `candidates` is a tuple of CandidateToken (parallel to legal_actions)."""
    if not candidates:
        return -1
    return int(candidate_logit_vector(net, encoded_state, candidates).argmax().item())
