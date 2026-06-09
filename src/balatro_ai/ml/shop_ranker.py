"""Neural shop candidate ranker.

This is the first trainable model for the candidate-ranker path: given one
encoded shop state and a padded list of candidate actions, score each candidate
and train from CRN rollout values. The default loss is soft cross-entropy over
rollout means so near-tied actions do not get treated like hard mistakes.
"""

from __future__ import annotations

import json
from math import isfinite
import random
from copy import deepcopy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from balatro_ai.api.actions import ActionType
from balatro_ai.ml.shop_candidate_dataset import paired_rollout_advantage
from balatro_ai.ml.encoding import (
    CardToken,
    EncodedState,
    ENCODING_VERSION,
    JokerToken,
    ShopToken,
    encoding_spec,
)
from balatro_ai.ml.model import Batch, ValueNet, collate_states


ACTION_TYPES = tuple(action.value for action in ActionType)
ACTION_TYPE_INDEX = {name: index for index, name in enumerate(ACTION_TYPES)}
TARGET_KINDS = ("", "card", "pack", "voucher", "joker", "consumable", "blind", "tag")
TARGET_KIND_INDEX = {name: index for index, name in enumerate(TARGET_KINDS)}


@dataclass(frozen=True, slots=True)
class ShopRankerExample:
    state: EncodedState
    action_keys: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]
    shop_token_indices: tuple[int, ...]
    mean_values: tuple[float, ...]
    heuristic_mask: tuple[float, ...]
    baseline_index: int | None
    baseline_value: float | None
    advantages: tuple[float, ...]
    advantage_lcbs: tuple[float, ...]
    advantage_ucbs: tuple[float, ...]
    advantage_positive_rates: tuple[float, ...]
    advantage_rollout_counts: tuple[int, ...]
    target_index: int
    seed: str = ""
    state_index: int = 0
    source_bot: str = ""
    ante: int = 0
    best_margin: float = 0.0
    split_half_agreement: bool = False
    actions_within_0_05: int = 0
    actions_within_0_10: int = 0
    best_runnerup_lcb: float = float("nan")
    best_runnerup_positive_rate: float = float("nan")
    best_vs_baseline_lcb: float = float("nan")
    best_vs_baseline_positive_rate: float = float("nan")
    high_conf_override_candidates: int = 0
    high_conf_practical_override_candidates: int = 0


@dataclass
class ShopRankerBatch:
    states: Batch
    action_type: torch.Tensor       # [B, C] long
    target_kind: torch.Tensor       # [B, C] long
    numeric: torch.Tensor           # [B, C, 3] amount,index,is_heuristic
    shop_token_index: torch.Tensor  # [B, C] long; -1 = no linked offer
    candidate_mask: torch.Tensor    # [B, C] bool
    target: torch.Tensor            # [B] long
    mean_values: torch.Tensor       # [B, C] float; 0 on padding
    baseline_index: torch.Tensor    # [B] long; -1 = baseline not in candidates
    advantages: torch.Tensor        # [B, C] float; candidate value - baseline value
    advantage_lcbs: torch.Tensor    # [B, C] float; paired candidate-baseline lower bounds
    advantage_ucbs: torch.Tensor    # [B, C] float; paired candidate-baseline upper bounds
    advantage_positive_rates: torch.Tensor  # [B, C] float
    advantage_rollout_counts: torch.Tensor  # [B, C] long


@dataclass
class ShopRankerTrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 16
    weight_decay: float = 0.0
    seed: int = 0
    d_id: int = 16
    d_small: int = 4
    d_token: int = 32
    d_trunk: int = 64
    dropout: float = 0.0
    encoder: str = "mean"
    loss: str = "soft"
    target_temperature: float = 0.10
    acceptable_margin: float = 0.05
    pairwise_margin: float = 0.10
    val_every: int = 5


@dataclass
class ShopRankerTrainResult:
    model: "ShopCandidateRanker"
    history: list[float]
    final_train: dict[str, float]
    final_val: dict[str, float] | None = None
    best_epoch: int | None = None
    best_val: dict[str, float] | None = None


class ShopCandidateRanker(nn.Module):
    """Score candidate shop actions from a shared state encoder trunk."""

    def __init__(
        self,
        *,
        spec: dict | None = None,
        d_id: int = 16,
        d_small: int = 4,
        d_token: int = 32,
        d_trunk: int = 64,
        dropout: float = 0.0,
        encoder: str = "mean",
    ) -> None:
        super().__init__()
        spec = spec or encoding_spec()
        self.encoding_version = spec["version"]
        self.spec = spec
        self.hparams = {
            "d_id": d_id,
            "d_small": d_small,
            "d_token": d_token,
            "d_trunk": d_trunk,
            "dropout": dropout,
            "encoder": encoder,
        }
        self.state_net = ValueNet(
            spec,
            d_id=d_id,
            d_small=d_small,
            d_token=d_token,
            d_trunk=d_trunk,
            dropout=dropout,
            encoder=encoder,
        )
        self.action_type_emb = nn.Embedding(len(ACTION_TYPES), d_small)
        self.target_kind_emb = nn.Embedding(len(TARGET_KINDS), d_small)
        self.scorer = nn.Sequential(
            nn.Linear(d_trunk + d_token + d_small * 2 + 3, d_trunk),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_trunk, 1),
        )

    def forward(self, batch: ShopRankerBatch) -> torch.Tensor:
        ctx = self.state_net._trunk(batch.states)  # [B, d_trunk]
        shop_tokens = self.state_net._shop_tokens(batch.states)  # [B, S, d_token]
        b, c = batch.action_type.shape

        indices = batch.shop_token_index
        valid_shop = indices >= 0
        max_index = max(0, shop_tokens.shape[1] - 1)
        safe_indices = indices.clamp(min=0, max=max_index)
        gathered = torch.gather(
            shop_tokens,
            1,
            safe_indices.unsqueeze(-1).expand(-1, -1, shop_tokens.shape[-1]),
        )
        gathered = gathered * valid_shop.unsqueeze(-1).to(gathered.dtype)

        features = torch.cat(
            [
                ctx.unsqueeze(1).expand(-1, c, -1),
                gathered,
                self.action_type_emb(batch.action_type),
                self.target_kind_emb(batch.target_kind),
                batch.numeric,
            ],
            dim=-1,
        )
        logits = self.scorer(features).squeeze(-1)
        return logits.masked_fill(~batch.candidate_mask, -1e9)


def encoded_state_from_json(data: dict[str, Any]) -> EncodedState:
    """Rebuild an EncodedState from JSON/asdict output."""

    return EncodedState(
        version=int(data["version"]),
        scalars=tuple(float(x) for x in data["scalars"]),
        phase_onehot=tuple(float(x) for x in data["phase_onehot"]),
        blind_type_onehot=tuple(float(x) for x in data["blind_type_onehot"]),
        boss_index=int(data["boss_index"]),
        hand_levels=tuple(float(x) for x in data["hand_levels"]),
        deck_counts=tuple(float(x) for x in data["deck_counts"]),
        jokers=tuple(JokerToken(**_coerce_token(token)) for token in data.get("jokers", ())),
        hand=tuple(CardToken(**_coerce_token(token)) for token in data.get("hand", ())),
        shop=tuple(ShopToken(**_coerce_token(token)) for token in data.get("shop", ())),
        consumables=tuple(int(x) for x in data.get("consumables", ())),
        vouchers=tuple(int(x) for x in data.get("vouchers", ())),
    )


def example_from_record(
    record: dict[str, Any],
    *,
    allowed_action_types: frozenset[ActionType] | None = None,
    allowed_action_kinds: frozenset[str] | None = None,
    keep_heuristic_action: bool = False,
    confidence_z: float = 1.0,
    practical_margin: float = 0.10,
) -> ShopRankerExample | None:
    candidates = tuple(dict(candidate) for candidate in record.get("candidates", ()))
    if allowed_action_types or allowed_action_kinds:
        candidates = tuple(
            candidate
            for candidate in candidates
            if (
                (not allowed_action_types or _action_type_from_candidate(candidate) in allowed_action_types)
                and _candidate_kind_allowed(candidate, allowed_action_kinds)
            )
            or (keep_heuristic_action and bool(candidate.get("is_heuristic_action")))
        )
    if len(candidates) < 2:
        return None
    action_keys = tuple(str(candidate["action_key"]) for candidate in candidates)
    mean_values = tuple(float(candidate.get("mean_value", 0.0)) for candidate in candidates)
    best_index = max(range(len(candidates)), key=lambda index: mean_values[index])
    best_key = action_keys[best_index]
    baseline_index = next(
        (index for index, candidate in enumerate(candidates) if candidate.get("is_heuristic_action")),
        None,
    )
    baseline_value = float(mean_values[baseline_index]) if baseline_index is not None else None
    advantages = (
        tuple(float(value - baseline_value) for value in mean_values)
        if baseline_value is not None
        else tuple(0.0 for _ in mean_values)
    )
    ordered_values = sorted(mean_values, reverse=True)
    best_value = ordered_values[0]
    best_margin = best_value - ordered_values[1] if len(ordered_values) >= 2 else 0.0
    first_half_best = _candidate_key_for_max(candidates, "first_half_mean")
    second_half_best = _candidate_key_for_max(candidates, "second_half_mean")
    confidence = _confidence_features(
        candidates,
        best_index=best_index,
        baseline_index=baseline_index,
        z=confidence_z,
        practical_margin=practical_margin,
    )
    baseline_confidence = _baseline_advantage_features(candidates, baseline_index=baseline_index, z=confidence_z)
    return ShopRankerExample(
        state=encoded_state_from_json(dict(record["encoded_state"])),
        action_keys=action_keys,
        actions=tuple(dict(candidate["action"]) for candidate in candidates),
        shop_token_indices=tuple(int(candidate.get("shop_token_index", -1)) for candidate in candidates),
        mean_values=mean_values,
        heuristic_mask=tuple(1.0 if candidate.get("is_heuristic_action") else 0.0 for candidate in candidates),
        baseline_index=baseline_index,
        baseline_value=baseline_value,
        advantages=advantages,
        advantage_lcbs=baseline_confidence["lcbs"],
        advantage_ucbs=baseline_confidence["ucbs"],
        advantage_positive_rates=baseline_confidence["positive_rates"],
        advantage_rollout_counts=baseline_confidence["rollout_counts"],
        target_index=best_index,
        seed=str(record.get("seed", "")),
        state_index=int(record.get("state_index", 0) or 0),
        source_bot=str(record.get("source_bot", "")),
        ante=int(record.get("ante", 0) or 0),
        best_margin=float(best_margin),
        split_half_agreement=bool(first_half_best is not None and first_half_best == second_half_best),
        actions_within_0_05=sum(1 for value in mean_values if best_value - value <= 0.05),
        actions_within_0_10=sum(1 for value in mean_values if best_value - value <= 0.10),
        best_runnerup_lcb=confidence["best_runnerup_lcb"],
        best_runnerup_positive_rate=confidence["best_runnerup_positive_rate"],
        best_vs_baseline_lcb=confidence["best_vs_baseline_lcb"],
        best_vs_baseline_positive_rate=confidence["best_vs_baseline_positive_rate"],
        high_conf_override_candidates=int(confidence["high_conf_override_candidates"]),
        high_conf_practical_override_candidates=int(confidence["high_conf_practical_override_candidates"]),
    )


def examples_from_jsonl(
    path: str | Path,
    *,
    allowed_action_types: frozenset[ActionType] | None = None,
    allowed_action_kinds: frozenset[str] | None = None,
    keep_heuristic_action: bool = False,
    confidence_z: float = 1.0,
    practical_margin: float = 0.10,
) -> list[ShopRankerExample]:
    examples: list[ShopRankerExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = example_from_record(
                json.loads(line),
                allowed_action_types=allowed_action_types,
                allowed_action_kinds=allowed_action_kinds,
                keep_heuristic_action=keep_heuristic_action,
                confidence_z=confidence_z,
                practical_margin=practical_margin,
            )
            if example is not None:
                examples.append(example)
    return examples


def examples_from_jsonl_paths(
    paths: Iterable[str | Path],
    *,
    dedupe: bool = True,
    allowed_action_types: frozenset[ActionType] | None = None,
    allowed_action_kinds: frozenset[str] | None = None,
    keep_heuristic_action: bool = False,
    confidence_z: float = 1.0,
    practical_margin: float = 0.10,
) -> list[ShopRankerExample]:
    if dedupe:
        records_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        ordered_keys: list[tuple[str, str, int]] = []
        undeduped_records: list[dict[str, Any]] = []
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = (
                        str(record.get("source_bot", "")),
                        str(record.get("seed", "")),
                        int(record.get("state_index", 0) or 0),
                    )
                    if not key[1]:
                        undeduped_records.append(record)
                        continue
                    if key not in records_by_key:
                        records_by_key[key] = record
                        ordered_keys.append(key)
                    else:
                        _merge_record_candidates(records_by_key[key], record)
        examples: list[ShopRankerExample] = []
        for record in [*(records_by_key[key] for key in ordered_keys), *undeduped_records]:
            example = example_from_record(
                record,
                allowed_action_types=allowed_action_types,
                allowed_action_kinds=allowed_action_kinds,
                keep_heuristic_action=keep_heuristic_action,
                confidence_z=confidence_z,
                practical_margin=practical_margin,
            )
            if example is not None:
                examples.append(example)
        return examples

    examples: list[ShopRankerExample] = []
    for path in paths:
        for example in examples_from_jsonl(
            path,
            allowed_action_types=allowed_action_types,
            allowed_action_kinds=allowed_action_kinds,
            keep_heuristic_action=keep_heuristic_action,
            confidence_z=confidence_z,
            practical_margin=practical_margin,
        ):
            examples.append(example)
    return examples


def _merge_record_candidates(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge focused candidate labels for the same state without duplicating keys.

    When the same candidate appears twice, keep the version with more paired
    rollout samples. This lets narrow r8 confirmations replace cheaper r4
    labels when callers pass both records into the loader.
    """

    target_candidates = target.setdefault("candidates", [])
    if not isinstance(target_candidates, list):
        return
    existing_indices = {
        str(candidate.get("action_key", "")): index
        for index, candidate in enumerate(target_candidates)
        if isinstance(candidate, dict)
    }
    source_candidates = source.get("candidates", ())
    if not isinstance(source_candidates, list):
        return
    for candidate in source_candidates:
        if not isinstance(candidate, dict):
            continue
        key = str(candidate.get("action_key", ""))
        if key in existing_indices:
            index = existing_indices[key]
            existing = target_candidates[index]
            if (
                isinstance(existing, dict)
                and _candidate_rollout_count(candidate) > _candidate_rollout_count(existing)
            ):
                target_candidates[index] = dict(candidate)
            continue
        has_heuristic = any(
            bool(item.get("is_heuristic_action")) for item in target_candidates if isinstance(item, dict)
        )
        if bool(candidate.get("is_heuristic_action")) and has_heuristic:
            continue
        target_candidates.append(dict(candidate))
        existing_indices[key] = len(target_candidates) - 1


def _candidate_rollout_count(candidate: dict[str, Any]) -> int:
    values = candidate.get("rollout_values", ())
    return len(values) if isinstance(values, (list, tuple)) else 0


def parse_action_kinds_csv(raw: str) -> frozenset[str]:
    return frozenset(item.strip().lower() for item in raw.split(",") if item.strip())


def _candidate_kind_allowed(candidate: dict[str, Any], allowed_action_kinds: frozenset[str] | None) -> bool:
    if not allowed_action_kinds:
        return True
    action = candidate.get("action", {})
    metadata = action.get("metadata", {}) if isinstance(action, dict) else {}
    kind = str(metadata.get("kind") or "").strip().lower() if isinstance(metadata, dict) else ""
    if kind:
        return kind in allowed_action_kinds
    return _action_type_from_candidate(candidate) == ActionType.END_SHOP


def _confidence_features(
    candidates: Sequence[dict[str, Any]],
    *,
    best_index: int,
    baseline_index: int | None,
    z: float,
    practical_margin: float,
) -> dict[str, float]:
    runnerup_index = _runnerup_index(candidates, best_index=best_index)
    best_runnerup_lcb = float("nan")
    best_runnerup_positive_rate = float("nan")
    if runnerup_index is not None:
        stats = paired_rollout_advantage(candidates[best_index], candidates[runnerup_index], z=z)
        if stats is not None:
            best_runnerup_lcb = stats.lower_bound
            best_runnerup_positive_rate = stats.positive_rate

    best_vs_baseline_lcb = float("nan")
    best_vs_baseline_positive_rate = float("nan")
    high_conf = 0
    high_conf_practical = 0
    if baseline_index is not None:
        stats = paired_rollout_advantage(candidates[best_index], candidates[baseline_index], z=z)
        if stats is not None:
            best_vs_baseline_lcb = stats.lower_bound
            best_vs_baseline_positive_rate = stats.positive_rate
        baseline_key = str(candidates[baseline_index].get("action_key", ""))
        for candidate in candidates:
            if str(candidate.get("action_key", "")) == baseline_key:
                continue
            advantage = paired_rollout_advantage(candidate, candidates[baseline_index], z=z)
            if advantage is None:
                continue
            if advantage.lower_bound > 0.0:
                high_conf += 1
            if advantage.lower_bound > practical_margin:
                high_conf_practical += 1

    return {
        "best_runnerup_lcb": float(best_runnerup_lcb),
        "best_runnerup_positive_rate": float(best_runnerup_positive_rate),
        "best_vs_baseline_lcb": float(best_vs_baseline_lcb),
        "best_vs_baseline_positive_rate": float(best_vs_baseline_positive_rate),
        "high_conf_override_candidates": float(high_conf),
        "high_conf_practical_override_candidates": float(high_conf_practical),
    }


def _baseline_advantage_features(
    candidates: Sequence[dict[str, Any]],
    *,
    baseline_index: int | None,
    z: float,
) -> dict[str, tuple[float, ...] | tuple[int, ...]]:
    """Per-candidate paired confidence for candidate-minus-baseline advantages."""

    if baseline_index is None:
        zeros = tuple(0.0 for _ in candidates)
        return {
            "lcbs": zeros,
            "ucbs": zeros,
            "positive_rates": zeros,
            "rollout_counts": tuple(0 for _ in candidates),
        }
    baseline = candidates[baseline_index]
    lcbs: list[float] = []
    ucbs: list[float] = []
    positive_rates: list[float] = []
    rollout_counts: list[int] = []
    for candidate in candidates:
        stats = paired_rollout_advantage(candidate, baseline, z=z)
        if stats is None:
            lcbs.append(0.0)
            ucbs.append(0.0)
            positive_rates.append(0.0)
            rollout_counts.append(0)
            continue
        lcbs.append(float(stats.lower_bound))
        ucbs.append(float(stats.upper_bound))
        positive_rates.append(float(stats.positive_rate))
        rollout_counts.append(int(stats.n))
    return {
        "lcbs": tuple(lcbs),
        "ucbs": tuple(ucbs),
        "positive_rates": tuple(positive_rates),
        "rollout_counts": tuple(rollout_counts),
    }


def _runnerup_index(candidates: Sequence[dict[str, Any]], *, best_index: int) -> int | None:
    indices = [index for index in range(len(candidates)) if index != best_index]
    if not indices:
        return None
    return max(indices, key=lambda index: _safe_float(candidates[index].get("mean_value"), default=float("-inf")))


def collate_shop_ranker_examples(examples: Sequence[ShopRankerExample]) -> ShopRankerBatch:
    if not examples:
        raise ValueError("need at least one shop ranker example")
    states = collate_states([example.state for example in examples])
    b = len(examples)
    max_candidates = max(len(example.actions) for example in examples)
    action_type = torch.zeros(b, max_candidates, dtype=torch.long)
    target_kind = torch.zeros(b, max_candidates, dtype=torch.long)
    numeric = torch.zeros(b, max_candidates, 3, dtype=torch.float32)
    shop_token_index = torch.full((b, max_candidates), -1, dtype=torch.long)
    candidate_mask = torch.zeros(b, max_candidates, dtype=torch.bool)
    mean_values = torch.zeros(b, max_candidates, dtype=torch.float32)
    baseline_index = torch.tensor(
        [-1 if example.baseline_index is None else int(example.baseline_index) for example in examples],
        dtype=torch.long,
    )
    advantages = torch.zeros(b, max_candidates, dtype=torch.float32)
    advantage_lcbs = torch.zeros(b, max_candidates, dtype=torch.float32)
    advantage_ucbs = torch.zeros(b, max_candidates, dtype=torch.float32)
    advantage_positive_rates = torch.zeros(b, max_candidates, dtype=torch.float32)
    advantage_rollout_counts = torch.zeros(b, max_candidates, dtype=torch.long)
    target = torch.tensor([example.target_index for example in examples], dtype=torch.long)

    for row, example in enumerate(examples):
        for col, action in enumerate(example.actions):
            action_type[row, col] = ACTION_TYPE_INDEX.get(str(action.get("type", "")), 0)
            meta = action.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            target_kind[row, col] = TARGET_KIND_INDEX.get(str(meta.get("kind", action.get("target_id", ""))).lower(), 0)
            amount = _safe_float(action.get("amount"), default=-1.0)
            meta_index = _safe_float(meta.get("index"), default=-1.0)
            numeric[row, col] = torch.tensor(
                [
                    max(-1.0, min(2.0, amount / 10.0)),
                    max(-1.0, min(2.0, meta_index / 10.0)),
                    float(example.heuristic_mask[col]),
                ],
                dtype=torch.float32,
            )
            shop_token_index[row, col] = int(example.shop_token_indices[col])
            mean_values[row, col] = float(example.mean_values[col])
            advantages[row, col] = float(example.advantages[col])
            advantage_lcbs[row, col] = float(example.advantage_lcbs[col])
            advantage_ucbs[row, col] = float(example.advantage_ucbs[col])
            advantage_positive_rates[row, col] = float(example.advantage_positive_rates[col])
            advantage_rollout_counts[row, col] = int(example.advantage_rollout_counts[col])
            candidate_mask[row, col] = True
    return ShopRankerBatch(
        states=states,
        action_type=action_type,
        target_kind=target_kind,
        numeric=numeric,
        shop_token_index=shop_token_index,
        candidate_mask=candidate_mask,
        target=target,
        mean_values=mean_values,
        baseline_index=baseline_index,
        advantages=advantages,
        advantage_lcbs=advantage_lcbs,
        advantage_ucbs=advantage_ucbs,
        advantage_positive_rates=advantage_positive_rates,
        advantage_rollout_counts=advantage_rollout_counts,
    )


def train_shop_ranker(
    examples: Sequence[ShopRankerExample],
    config: ShopRankerTrainConfig | None = None,
    *,
    val_examples: Sequence[ShopRankerExample] | None = None,
) -> ShopRankerTrainResult:
    config = config or ShopRankerTrainConfig()
    if not examples:
        raise ValueError("train_shop_ranker() needs at least one example")
    torch.manual_seed(config.seed)
    model = ShopCandidateRanker(
        d_id=config.d_id,
        d_small=config.d_small,
        d_token=config.d_token,
        d_trunk=config.d_trunk,
        dropout=config.dropout,
        encoder=config.encoder,
    )
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    items = list(examples)
    rng = random.Random(config.seed)
    batch_size = max(1, int(config.batch_size))
    val_items = list(val_examples or [])
    val_every = max(1, int(config.val_every))
    best_epoch: int | None = None
    best_val: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[float] = []
    for epoch in range(config.epochs):
        rng.shuffle(items)
        epoch_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, len(items), batch_size):
            batch_examples = items[start:start + batch_size]
            batch = collate_shop_ranker_examples(batch_examples)
            opt.zero_grad()
            logits = model(batch)
            if config.loss == "hard":
                loss = nn.CrossEntropyLoss()(logits, batch.target)
            elif config.loss == "soft":
                loss = _soft_value_cross_entropy(logits, batch, temperature=config.target_temperature)
            elif config.loss == "acceptable":
                loss = _acceptable_set_cross_entropy(
                    logits,
                    batch,
                    margin=config.acceptable_margin,
                )
            elif config.loss == "pairwise":
                loss = _pairwise_preference_loss(
                    logits,
                    batch,
                    margin=config.pairwise_margin,
                )
            elif config.loss == "advantage_mse":
                loss = _advantage_mse_loss(logits, batch)
            elif config.loss == "advantage_tie_mse":
                loss = _advantage_tie_mse_loss(
                    logits,
                    batch,
                    margin=config.acceptable_margin,
                )
            elif config.loss == "confidence_advantage_tie_mse":
                loss = _confidence_advantage_tie_mse_loss(
                    logits,
                    batch,
                    margin=config.acceptable_margin,
                )
            else:
                raise ValueError(f"unknown shop ranker loss: {config.loss!r}")
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            batches += 1
        history.append(epoch_loss / max(1, batches))
        if val_items and ((epoch + 1) % val_every == 0 or epoch + 1 == config.epochs):
            val_metrics = evaluate_shop_ranker(model, val_items, target_temperature=config.target_temperature)
            if _uses_advantage_selection(config.loss):
                _add_prefixed_metrics(
                    val_metrics,
                    "advantage_override",
                    evaluate_advantage_override(model, val_items, threshold=0.0),
                )
            selection_key = _advantage_val_selection_key if _uses_advantage_selection(config.loss) else _val_selection_key
            if best_val is None or selection_key(val_metrics) < selection_key(best_val):
                best_epoch = epoch + 1
                best_val = val_metrics
                best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return ShopRankerTrainResult(
        model=model,
        history=history,
        final_train=evaluate_shop_ranker(model, examples, target_temperature=config.target_temperature),
        final_val=(
            evaluate_shop_ranker(model, val_items, target_temperature=config.target_temperature)
            if val_items
            else None
        ),
        best_epoch=best_epoch,
        best_val=best_val,
    )


@torch.no_grad()
def evaluate_shop_ranker(
    model: ShopCandidateRanker,
    examples: Sequence[ShopRankerExample],
    *,
    target_temperature: float = 0.10,
) -> dict[str, float]:
    if not examples:
        return {"n": 0, "loss": float("nan"), "top1_acc": float("nan"), "mean_regret": float("nan")}
    model.eval()
    batch = collate_shop_ranker_examples(examples)
    logits = model(batch)
    loss = float(nn.CrossEntropyLoss()(logits, batch.target))
    soft_loss = float(_soft_value_cross_entropy(logits, batch, temperature=target_temperature))
    pred = logits.argmax(dim=1)
    acc = float((pred == batch.target).float().mean())
    regrets = []
    near_005 = []
    near_010 = []
    pred_ranks = []
    acceptable_025 = []
    for row, p in enumerate(pred.tolist()):
        best_value = batch.mean_values[row, batch.target[row]].item()
        pred_value = batch.mean_values[row, p].item()
        regret = max(0.0, best_value - pred_value)
        regrets.append(regret)
        acceptable_025.append(regret <= 0.25)
        near_005.append(regret <= 0.05)
        near_010.append(regret <= 0.10)
        valid_values = batch.mean_values[row][batch.candidate_mask[row]].tolist()
        pred_ranks.append(1 + sum(1 for value in valid_values if value > pred_value + 1e-6))
    return {
        "n": len(examples),
        "loss": loss,
        "soft_loss": soft_loss,
        "top1_acc": acc,
        "mean_regret": float(sum(regrets) / max(1, len(regrets))),
        "near_best_acc_0_05": float(sum(near_005) / max(1, len(near_005))),
        "near_best_acc_0_10": float(sum(near_010) / max(1, len(near_010))),
        "acceptable_acc_0_25": float(sum(acceptable_025) / max(1, len(acceptable_025))),
        "mean_pred_rank_by_value": float(sum(pred_ranks) / max(1, len(pred_ranks))),
    }


@torch.no_grad()
def evaluate_advantage_override(
    model: ShopCandidateRanker,
    examples: Sequence[ShopRankerExample],
    *,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Evaluate a baseline-aware override policy from model scores.

    The model may be trained as a ranker or as an advantage regressor. At
    deployment time we compare the model score for its favorite candidate with
    the score for the baseline/heuristic candidate and only override when that
    predicted margin clears ``threshold``.
    """

    if not examples:
        return _empty_advantage_metrics(0, threshold=threshold)
    model.eval()
    batch = collate_shop_ranker_examples(examples)
    logits = model(batch)
    threshold = float(threshold)
    covered = 0
    overrides = 0
    helpful = 0
    harmful = 0
    neutral = 0
    realized_advantages: list[float] = []
    override_advantages: list[float] = []
    chosen_regrets: list[float] = []
    baseline_regrets: list[float] = []
    oracle_positive = 0
    predicted_margins: list[float] = []
    for row in range(len(examples)):
        baseline_index = int(batch.baseline_index[row].item())
        if baseline_index < 0:
            continue
        covered += 1
        valid_logits = logits[row].masked_fill(~batch.candidate_mask[row], -1e9)
        pred_index = int(torch.argmax(valid_logits).item())
        predicted_margin = float(logits[row, pred_index] - logits[row, baseline_index])
        should_override = pred_index != baseline_index and predicted_margin >= threshold
        chosen_index = pred_index if should_override else baseline_index
        best_index = int(batch.target[row].item())
        best_value = float(batch.mean_values[row, best_index].item())
        baseline_value = float(batch.mean_values[row, baseline_index].item())
        chosen_value = float(batch.mean_values[row, chosen_index].item())
        if best_value > baseline_value + 1e-6:
            oracle_positive += 1
        baseline_regrets.append(max(0.0, best_value - baseline_value))
        chosen_regrets.append(max(0.0, best_value - chosen_value))
        realized_advantage = chosen_value - baseline_value
        realized_advantages.append(realized_advantage)
        if should_override:
            overrides += 1
            predicted_margins.append(predicted_margin)
            override_advantages.append(realized_advantage)
            if realized_advantage > 1e-6:
                helpful += 1
            elif realized_advantage < -1e-6:
                harmful += 1
            else:
                neutral += 1
    if covered == 0:
        return _empty_advantage_metrics(len(examples), threshold=threshold)
    mean_baseline_regret = float(sum(baseline_regrets) / covered)
    mean_chosen_regret = float(sum(chosen_regrets) / covered)
    mean_lift = float(sum(realized_advantages) / covered)
    return {
        "n": len(examples),
        "covered_n": covered,
        "coverage_rate": float(covered / max(1, len(examples))),
        "threshold": threshold,
        "override_rate": float(overrides / covered),
        "helpful_override_rate": float(helpful / max(1, overrides)),
        "harmful_override_rate": float(harmful / max(1, overrides)),
        "neutral_override_rate": float(neutral / max(1, overrides)),
        "helpful_covered_rate": float(helpful / covered),
        "harmful_covered_rate": float(harmful / covered),
        "neutral_covered_rate": float(neutral / covered),
        "mean_lift_vs_baseline": mean_lift,
        "mean_override_lift": float(sum(override_advantages) / max(1, overrides)),
        "mean_chosen_regret": mean_chosen_regret,
        "mean_baseline_regret": mean_baseline_regret,
        "mean_regret_delta_vs_baseline": mean_chosen_regret - mean_baseline_regret,
        "oracle_positive_advantage_rate": float(oracle_positive / covered),
        "mean_predicted_override_margin": float(sum(predicted_margins) / max(1, overrides)),
    }


def evaluate_advantage_overrides(
    model: ShopCandidateRanker,
    examples: Sequence[ShopRankerExample],
    *,
    thresholds: Sequence[float] = (0.0, 0.05, 0.10),
) -> dict[str, dict[str, float]]:
    return {
        _threshold_key(threshold): evaluate_advantage_override(model, examples, threshold=float(threshold))
        for threshold in thresholds
    }


def evaluate_heuristic_baseline(examples: Sequence[ShopRankerExample]) -> dict[str, float]:
    if not examples:
        return {
            "n": 0,
            "covered_n": 0,
            "coverage_rate": float("nan"),
            "top1_acc": float("nan"),
            "mean_regret": float("nan"),
        }
    covered = 0
    top1 = 0
    regrets = []
    near_005 = []
    near_010 = []
    acceptable_025 = []
    pred_ranks = []
    for example in examples:
        heuristic_index = next(
            (index for index, value in enumerate(example.heuristic_mask) if value > 0.5),
            None,
        )
        if heuristic_index is None:
            continue
        covered += 1
        if heuristic_index == example.target_index:
            top1 += 1
        best_value = float(example.mean_values[example.target_index])
        pred_value = float(example.mean_values[heuristic_index])
        regret = max(0.0, best_value - pred_value)
        regrets.append(regret)
        near_005.append(regret <= 0.05)
        near_010.append(regret <= 0.10)
        acceptable_025.append(regret <= 0.25)
        pred_ranks.append(1 + sum(1 for value in example.mean_values if value > pred_value + 1e-6))
    return {
        "n": len(examples),
        "covered_n": covered,
        "coverage_rate": float(covered / max(1, len(examples))),
        "top1_acc": float(top1 / max(1, covered)),
        "mean_regret": float(sum(regrets) / max(1, len(regrets))),
        "near_best_acc_0_05": float(sum(near_005) / max(1, len(near_005))),
        "near_best_acc_0_10": float(sum(near_010) / max(1, len(near_010))),
        "acceptable_acc_0_25": float(sum(acceptable_025) / max(1, len(acceptable_025))),
        "mean_pred_rank_by_value": float(sum(pred_ranks) / max(1, len(pred_ranks))),
    }


def label_quality_summary(examples: Sequence[ShopRankerExample]) -> dict[str, float]:
    if not examples:
        return {
            "n": 0,
            "mean_best_margin": float("nan"),
            "split_half_agreement_rate": float("nan"),
            "mean_actions_within_0_05": float("nan"),
            "mean_actions_within_0_10": float("nan"),
            "mean_best_runnerup_lcb": float("nan"),
            "best_runnerup_lcb_positive_rate": float("nan"),
            "mean_best_vs_baseline_lcb": float("nan"),
            "best_vs_baseline_lcb_positive_rate": float("nan"),
            "mean_high_conf_override_candidates": float("nan"),
            "mean_high_conf_practical_override_candidates": float("nan"),
        }
    return {
        "n": len(examples),
        "mean_best_margin": float(sum(example.best_margin for example in examples) / len(examples)),
        "split_half_agreement_rate": float(sum(example.split_half_agreement for example in examples) / len(examples)),
        "mean_actions_within_0_05": float(sum(example.actions_within_0_05 for example in examples) / len(examples)),
        "mean_actions_within_0_10": float(sum(example.actions_within_0_10 for example in examples) / len(examples)),
        "mean_best_runnerup_lcb": _finite_mean(example.best_runnerup_lcb for example in examples),
        "best_runnerup_lcb_positive_rate": _finite_positive_rate(example.best_runnerup_lcb for example in examples),
        "mean_best_vs_baseline_lcb": _finite_mean(example.best_vs_baseline_lcb for example in examples),
        "best_vs_baseline_lcb_positive_rate": _finite_positive_rate(
            example.best_vs_baseline_lcb for example in examples
        ),
        "mean_high_conf_override_candidates": float(
            sum(example.high_conf_override_candidates for example in examples) / len(examples)
        ),
        "mean_high_conf_practical_override_candidates": float(
            sum(example.high_conf_practical_override_candidates for example in examples) / len(examples)
        ),
    }


def confidence_advantage_label_summary(
    examples: Sequence[ShopRankerExample],
    *,
    margin: float = 0.0,
) -> dict[str, float]:
    """Summarize candidate-vs-baseline confidence labels.

    This excludes the baseline action itself. A candidate is positive when its
    paired lower bound clears ``margin``, negative when its upper bound is below
    ``-margin``, and ambiguous otherwise.
    """

    margin = max(0.0, float(margin))
    covered_records = 0
    candidates = 0
    positives = 0
    negatives = 0
    ambiguous = 0
    records_with_positive = 0
    records_with_negative = 0
    records_with_ambiguous = 0
    for example in examples:
        if example.baseline_index is None:
            continue
        covered_records += 1
        record_positive = False
        record_negative = False
        record_ambiguous = False
        for index, (lcb, ucb, count) in enumerate(
            zip(example.advantage_lcbs, example.advantage_ucbs, example.advantage_rollout_counts)
        ):
            if index == example.baseline_index or int(count) < 2:
                continue
            candidates += 1
            if float(lcb) > margin:
                positives += 1
                record_positive = True
            elif float(ucb) < -margin:
                negatives += 1
                record_negative = True
            else:
                ambiguous += 1
                record_ambiguous = True
        records_with_positive += int(record_positive)
        records_with_negative += int(record_negative)
        records_with_ambiguous += int(record_ambiguous)
    return {
        "n": len(examples),
        "covered_records": covered_records,
        "margin": float(margin),
        "candidate_labels": candidates,
        "positive_labels": positives,
        "negative_labels": negatives,
        "ambiguous_labels": ambiguous,
        "positive_label_rate": float(positives / candidates) if candidates else float("nan"),
        "negative_label_rate": float(negatives / candidates) if candidates else float("nan"),
        "ambiguous_label_rate": float(ambiguous / candidates) if candidates else float("nan"),
        "records_with_positive_rate": float(records_with_positive / covered_records) if covered_records else float("nan"),
        "records_with_negative_rate": float(records_with_negative / covered_records) if covered_records else float("nan"),
        "records_with_ambiguous_rate": (
            float(records_with_ambiguous / covered_records) if covered_records else float("nan")
        ),
    }


def filter_examples_by_label_quality(
    examples: Sequence[ShopRankerExample],
    *,
    min_best_margin: float = 0.0,
    require_split_half_agreement: bool = False,
    max_actions_within_0_05: int | None = None,
    max_actions_within_0_10: int | None = None,
    min_best_runnerup_lcb: float | None = None,
    min_best_vs_baseline_lcb: float | None = None,
    min_high_conf_override_candidates: int | None = None,
    min_high_conf_practical_override_candidates: int | None = None,
) -> list[ShopRankerExample]:
    out: list[ShopRankerExample] = []
    for example in examples:
        if example.best_margin + 1e-12 < min_best_margin:
            continue
        if require_split_half_agreement and not example.split_half_agreement:
            continue
        if max_actions_within_0_05 is not None and example.actions_within_0_05 > max_actions_within_0_05:
            continue
        if max_actions_within_0_10 is not None and example.actions_within_0_10 > max_actions_within_0_10:
            continue
        if min_best_runnerup_lcb is not None and not _finite_at_least(
            example.best_runnerup_lcb,
            min_best_runnerup_lcb,
        ):
            continue
        if min_best_vs_baseline_lcb is not None and not _finite_at_least(
            example.best_vs_baseline_lcb,
            min_best_vs_baseline_lcb,
        ):
            continue
        if (
            min_high_conf_override_candidates is not None
            and example.high_conf_override_candidates < min_high_conf_override_candidates
        ):
            continue
        if (
            min_high_conf_practical_override_candidates is not None
            and example.high_conf_practical_override_candidates < min_high_conf_practical_override_candidates
        ):
            continue
        out.append(example)
    return out


def _finite_mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if isfinite(float(value))]
    return float(sum(items) / len(items)) if items else float("nan")


def _finite_positive_rate(values: Iterable[float]) -> float:
    items = [float(value) for value in values if isfinite(float(value))]
    return float(sum(value > 0.0 for value in items) / len(items)) if items else float("nan")


def _finite_at_least(value: float, threshold: float) -> bool:
    value = float(value)
    return isfinite(value) and value + 1e-12 >= float(threshold)


def _val_selection_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("mean_regret", float("inf"))),
        -float(metrics.get("near_best_acc_0_05", 0.0)),
        -float(metrics.get("near_best_acc_0_10", 0.0)),
        float(metrics.get("loss", float("inf"))),
    )


def _advantage_val_selection_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("advantage_override_mean_chosen_regret", float("inf"))),
        float(metrics.get("advantage_override_mean_regret_delta_vs_baseline", float("inf"))),
        -float(metrics.get("advantage_override_mean_lift_vs_baseline", 0.0)),
        float(metrics.get("loss", float("inf"))),
    )


def _add_prefixed_metrics(target: dict[str, float], prefix: str, source: dict[str, float]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[f"{prefix}_{key}"] = float(value)


def _uses_advantage_selection(loss: str) -> bool:
    return loss in {"advantage_mse", "advantage_tie_mse", "confidence_advantage_tie_mse"}


def _empty_advantage_metrics(n: int, *, threshold: float) -> dict[str, float]:
    return {
        "n": n,
        "covered_n": 0,
        "coverage_rate": float("nan"),
        "threshold": float(threshold),
        "override_rate": float("nan"),
        "helpful_override_rate": float("nan"),
        "harmful_override_rate": float("nan"),
        "neutral_override_rate": float("nan"),
        "helpful_covered_rate": 0.0,
        "harmful_covered_rate": 0.0,
        "neutral_covered_rate": 0.0,
        "mean_lift_vs_baseline": float("nan"),
        "mean_override_lift": float("nan"),
        "mean_chosen_regret": float("nan"),
        "mean_baseline_regret": float("nan"),
        "mean_regret_delta_vs_baseline": float("nan"),
        "oracle_positive_advantage_rate": float("nan"),
        "mean_predicted_override_margin": float("nan"),
    }


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.3f}"


def _candidate_key_for_max(candidates: Sequence[dict[str, Any]], field: str) -> str | None:
    if not candidates:
        return None
    return str(max(candidates, key=lambda candidate: _safe_float(candidate.get(field), default=float("-inf"))).get("action_key", ""))


def parse_action_types_csv(raw: str) -> frozenset[ActionType]:
    if not raw.strip():
        return frozenset()
    by_value = {action_type.value: action_type for action_type in ActionType}
    by_name = {action_type.name.lower(): action_type for action_type in ActionType}
    out: set[ActionType] = set()
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        action_type = by_value.get(key) or by_name.get(key)
        if action_type is not None:
            out.add(action_type)
    return frozenset(out)


def _action_type_from_candidate(candidate: dict[str, Any]) -> ActionType | None:
    action = candidate.get("action", {})
    if not isinstance(action, dict):
        return None
    raw = str(action.get("type", "")).lower()
    if not raw:
        return None
    by_value = {action_type.value: action_type for action_type in ActionType}
    by_name = {action_type.name.lower(): action_type for action_type in ActionType}
    return by_value.get(raw) or by_name.get(raw)


def split_examples(
    examples: Sequence[ShopRankerExample],
    val_frac: float,
    *,
    seed: int = 0,
    split_by: str = "random",
) -> tuple[list[ShopRankerExample], list[ShopRankerExample]]:
    items = list(examples)
    rng = random.Random(seed)
    if split_by == "seed":
        return _split_examples_by_group(items, val_frac, rng=rng)
    if split_by != "random":
        raise ValueError(f"unknown split_by: {split_by!r}")
    rng.shuffle(items)
    n_val = int(len(items) * val_frac)
    return items[n_val:], items[:n_val]


def _split_examples_by_group(
    examples: Sequence[ShopRankerExample],
    val_frac: float,
    *,
    rng: random.Random,
) -> tuple[list[ShopRankerExample], list[ShopRankerExample]]:
    if val_frac <= 0 or len(examples) <= 1:
        return list(examples), []
    groups: dict[str, list[ShopRankerExample]] = {}
    for index, example in enumerate(examples):
        key = example.seed or f"__row_{index}"
        groups.setdefault(key, []).append(example)
    keys = list(groups)
    rng.shuffle(keys)
    if len(keys) <= 1:
        return list(examples), []
    n_val_groups = max(1, int(len(keys) * val_frac))
    n_val_groups = min(n_val_groups, len(keys) - 1)
    val_keys = set(keys[:n_val_groups])
    train: list[ShopRankerExample] = []
    val: list[ShopRankerExample] = []
    for key, group in groups.items():
        (val if key in val_keys else train).extend(group)
    return train, val


def save_checkpoint(model: ShopCandidateRanker, path: str | Path) -> None:
    torch.save(
        {
            "encoding_version": model.encoding_version,
            "spec": model.spec,
            "hparams": model.hparams,
            "state_dict": model.state_dict(),
        },
        str(path),
    )


def load_checkpoint(path: str | Path, *, strict_version: bool = True) -> ShopCandidateRanker:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if strict_version and ckpt.get("encoding_version") != ENCODING_VERSION:
        raise ValueError(
            f"{path}: checkpoint encoding_version={ckpt.get('encoding_version')} "
            f"!= current ENCODING_VERSION={ENCODING_VERSION}; retrain with the current encoder"
        )
    model = ShopCandidateRanker(spec=ckpt["spec"], **ckpt.get("hparams", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _soft_value_cross_entropy(
    logits: torch.Tensor,
    batch: ShopRankerBatch,
    *,
    temperature: float,
) -> torch.Tensor:
    temperature = max(1e-6, float(temperature))
    centered_values = batch.mean_values - batch.mean_values.masked_fill(~batch.candidate_mask, -1e9).amax(
        dim=1,
        keepdim=True,
    )
    target_logits = (centered_values / temperature).masked_fill(~batch.candidate_mask, -1e9)
    targets = torch.softmax(target_logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def _acceptable_set_cross_entropy(
    logits: torch.Tensor,
    batch: ShopRankerBatch,
    *,
    margin: float,
) -> torch.Tensor:
    """Uniformly reward any candidate close enough to the sampled best value."""

    margin = max(0.0, float(margin))
    valid_values = batch.mean_values.masked_fill(~batch.candidate_mask, -1e9)
    best_values = valid_values.amax(dim=1, keepdim=True)
    acceptable = batch.candidate_mask & ((best_values - batch.mean_values) <= margin)
    counts = acceptable.sum(dim=1, keepdim=True).clamp(min=1).to(logits.dtype)
    targets = acceptable.to(logits.dtype) / counts
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def _pairwise_preference_loss(
    logits: torch.Tensor,
    batch: ShopRankerBatch,
    *,
    margin: float,
) -> torch.Tensor:
    """Train only on candidate pairs whose rollout gap clears ``margin``."""

    margin = max(0.0, float(margin))
    value_diff = batch.mean_values.unsqueeze(2) - batch.mean_values.unsqueeze(1)
    clear = value_diff > margin
    valid_pairs = batch.candidate_mask.unsqueeze(2) & batch.candidate_mask.unsqueeze(1) & clear
    if not bool(valid_pairs.any()):
        return logits.sum() * 0.0
    logit_diff = logits.unsqueeze(2) - logits.unsqueeze(1)
    return nn.functional.softplus(-logit_diff[valid_pairs]).mean()


def _advantage_mse_loss(logits: torch.Tensor, batch: ShopRankerBatch) -> torch.Tensor:
    baseline_present = batch.baseline_index >= 0
    valid = batch.candidate_mask & baseline_present.unsqueeze(1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return nn.MSELoss()(logits[valid], batch.advantages[valid])


def _advantage_tie_mse_loss(
    logits: torch.Tensor,
    batch: ShopRankerBatch,
    *,
    margin: float,
) -> torch.Tensor:
    margin = max(0.0, float(margin))
    baseline_present = batch.baseline_index >= 0
    valid = batch.candidate_mask & baseline_present.unsqueeze(1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    targets = torch.where(
        batch.advantages.abs() <= margin,
        torch.zeros_like(batch.advantages),
        batch.advantages,
    )
    return nn.MSELoss()(logits[valid], targets[valid])


def _confidence_advantage_tie_mse_loss(
    logits: torch.Tensor,
    batch: ShopRankerBatch,
    *,
    margin: float,
) -> torch.Tensor:
    """Regress clear candidate-baseline advantages and collapse uncertain gaps to ties."""

    margin = max(0.0, float(margin))
    baseline_present = batch.baseline_index >= 0
    valid = batch.candidate_mask & baseline_present.unsqueeze(1) & (batch.advantage_rollout_counts >= 2)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    clear = (batch.advantage_lcbs > margin) | (batch.advantage_ucbs < -margin)
    targets = torch.where(clear, batch.advantages, torch.zeros_like(batch.advantages))
    return nn.MSELoss()(logits[valid], targets[valid])


def _coerce_token(token: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(token).items()}


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
