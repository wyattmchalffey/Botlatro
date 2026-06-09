"""Neural shop-ranker bot wrappers."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.base import Bot


_RANKER_CACHE: tuple[Path, object] | None = None


@dataclass(slots=True)
class RankerGuidedShopBot:
    """Use a trained shop candidate ranker for shop states, fallback elsewhere."""

    wrapped: Bot
    seed: int | None = None
    name: str = "shop_ranker_guided_bot"
    _ranker_actions_in_shop: int = field(default=0, init=False, repr=False)
    _ranker_actions_in_run: int = field(default=0, init=False, repr=False)
    _was_in_shop: bool = field(default=False, init=False, repr=False)

    def choose_action(self, state: GameState) -> Action:
        ckpt = os.environ.get("BALATRO_SHOP_RANKER_CKPT", "").strip()
        if state.phase == GamePhase.BOOSTER_OPENED:
            return self.wrapped.choose_action(state)
        if state.phase != GamePhase.SHOP:
            self._ranker_actions_in_shop = 0
            self._was_in_shop = False
            return self.wrapped.choose_action(state)
        if not self._was_in_shop:
            self._ranker_actions_in_shop = 0
            self._was_in_shop = True
        if not ckpt:
            return self.wrapped.choose_action(state)
        min_ante = _env_int("BALATRO_SHOP_RANKER_MIN_ANTE", default=0)
        max_ante = _env_int("BALATRO_SHOP_RANKER_MAX_ANTE", default=99)
        if int(state.ante) < min_ante or int(state.ante) > max_ante:
            return self.wrapped.choose_action(state)
        max_ranker_actions = _env_int("BALATRO_SHOP_RANKER_MAX_ACTIONS_PER_SHOP", default=99)
        if self._ranker_actions_in_shop >= max_ranker_actions:
            return self.wrapped.choose_action(state)
        max_ranker_actions_run = _env_int("BALATRO_SHOP_RANKER_MAX_ACTIONS_PER_RUN", default=99)
        if self._ranker_actions_in_run >= max_ranker_actions_run:
            return self.wrapped.choose_action(state)

        baseline_action_getter = None
        if _env_bool("BALATRO_SHOP_RANKER_COMPARE_BASELINE", default=False):
            baseline_action_getter = lambda: _probe_wrapped_action(self.wrapped, state)
        action = _ranker_shop_action(
            state,
            ckpt=Path(ckpt),
            baseline_action_getter=baseline_action_getter,
        )
        if action is not None:
            self._ranker_actions_in_shop += 1
            self._ranker_actions_in_run += 1
            return action
        return self.wrapped.choose_action(state)


def _ranker_shop_action(
    state: GameState,
    *,
    ckpt: Path,
    baseline_action_getter: Callable[[], Action | None] | None = None,
) -> Action | None:
    from balatro_ai.ml.shop_candidate_dataset import action_key, candidate_shop_actions

    max_actions = _env_int("BALATRO_SHOP_RANKER_MAX_ACTIONS", default=12)
    base_candidates = candidate_shop_actions(state, max_actions=max_actions)
    allowed_types = _env_action_types("BALATRO_SHOP_RANKER_ACTION_TYPES")
    allowed_kinds = _env_csv_set("BALATRO_SHOP_RANKER_ACTION_KINDS")
    override_candidates = tuple(
        action for action in base_candidates
        if not allowed_types or action.action_type in allowed_types
        if _action_kind_allowed(action, allowed_kinds)
    )
    if not override_candidates:
        return None

    baseline_action = baseline_action_getter() if baseline_action_getter is not None else None
    baseline_key = action_key(baseline_action) if baseline_action is not None else None
    candidates = override_candidates
    if baseline_action is not None:
        candidates = _append_missing_action(candidates, baseline_action)
    if len(candidates) < 2:
        return None
    candidate_keys = tuple(action_key(action) for action in candidates)
    baseline_index = candidate_keys.index(baseline_key) if baseline_key in candidate_keys else None
    override_indices = tuple(
        index for index, action in enumerate(candidates)
        if not allowed_types or action.action_type in allowed_types
        if _action_kind_allowed(action, allowed_kinds)
    )
    if not override_indices:
        return None

    model = _load_ranker_model(ckpt)
    example = _ranker_example(state, candidates, baseline_index=baseline_index)
    if example is None:
        return None

    from balatro_ai.ml.shop_ranker import collate_shop_ranker_examples
    import torch

    batch = collate_shop_ranker_examples([example])
    model.eval()
    with torch.no_grad():
        logits = model(batch)[0]
    if logits.numel() == 0:
        return None
    best = max(override_indices, key=lambda index: float(logits[index]))
    order = [int(index) for index in torch.argsort(logits, descending=True).tolist()]
    second = next((index for index in order if index != best), best)
    margin = float(logits[best] - logits[second]) if best != second else 0.0
    min_margin = _env_float("BALATRO_SHOP_RANKER_MIN_MARGIN", default=0.0)
    if margin < min_margin:
        return None
    baseline_margin: float | None = None
    baseline_score: float | None = None
    if baseline_action_getter is not None:
        if baseline_index is None:
            return None
        if baseline_index == best:
            return None
        baseline_score = float(logits[baseline_index])
        baseline_margin = float(logits[best] - logits[baseline_index])
        min_baseline_margin = _env_float("BALATRO_SHOP_RANKER_MIN_BASELINE_MARGIN", default=0.0)
        if baseline_margin < min_baseline_margin:
            return None
    return _annotate_ranker_action(
        candidates[best],
        score=float(logits[best]),
        margin=margin,
        ckpt=ckpt,
        baseline_score=baseline_score,
        baseline_margin=baseline_margin,
        baseline_key=baseline_key,
    )


def _ranker_example(
    state: GameState,
    candidates: tuple[Action, ...],
    *,
    baseline_index: int | None = None,
):
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.shop_candidate_dataset import _shop_token_index, action_key
    from balatro_ai.ml.shop_ranker import ShopRankerExample

    if not candidates:
        return None
    return ShopRankerExample(
        state=encode_state(state),
        action_keys=tuple(action_key(action) for action in candidates),
        actions=tuple(action.to_json() for action in candidates),
        shop_token_indices=tuple(_shop_token_index(state, action) for action in candidates),
        mean_values=tuple(0.0 for _ in candidates),
        heuristic_mask=tuple(1.0 if index == baseline_index else 0.0 for index in range(len(candidates))),
        baseline_index=baseline_index,
        baseline_value=0.0 if baseline_index is not None else None,
        advantages=tuple(0.0 for _ in candidates),
        advantage_lcbs=tuple(0.0 for _ in candidates),
        advantage_ucbs=tuple(0.0 for _ in candidates),
        advantage_positive_rates=tuple(0.0 for _ in candidates),
        advantage_rollout_counts=tuple(0 for _ in candidates),
        target_index=0,
    )


def _append_missing_action(actions: tuple[Action, ...], action: Action) -> tuple[Action, ...]:
    from balatro_ai.ml.shop_candidate_dataset import action_key

    key = action_key(action)
    if any(action_key(existing) == key for existing in actions):
        return actions
    return (*actions, action)


def _load_ranker_model(path: Path):
    global _RANKER_CACHE

    resolved = path.resolve()
    if _RANKER_CACHE is None or _RANKER_CACHE[0] != resolved:
        from balatro_ai.ml.shop_ranker import load_checkpoint

        _RANKER_CACHE = (resolved, load_checkpoint(resolved))
    return _RANKER_CACHE[1]


def _annotate_ranker_action(
    action: Action,
    *,
    score: float,
    margin: float,
    ckpt: Path,
    baseline_score: float | None = None,
    baseline_margin: float | None = None,
    baseline_key: str | None = None,
) -> Action:
    payload: dict[str, Any] = {
        "score": round(score, 6),
        "margin": round(margin, 6),
        "ckpt": ckpt.name,
    }
    if baseline_score is not None:
        payload["baseline_score"] = round(baseline_score, 6)
    if baseline_margin is not None:
        payload["baseline_margin"] = round(baseline_margin, 6)
    if baseline_key is not None:
        payload["baseline_key"] = baseline_key
    metadata: dict[str, Any] = {
        **action.metadata,
        "reason": "shop_ranker",
        "shop_ranker": payload,
    }
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata=metadata,
    )


def _probe_wrapped_action(wrapped: Bot, state: GameState) -> Action | None:
    try:
        probe = copy.deepcopy(wrapped)
        action = probe.choose_action(state)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError, RuntimeError):
        return None
    return action


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, *, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_action_types(name: str) -> frozenset[ActionType]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    out: set[ActionType] = set()
    by_value = {action_type.value: action_type for action_type in ActionType}
    by_name = {action_type.name.lower(): action_type for action_type in ActionType}
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        action_type = by_value.get(key) or by_name.get(key)
        if action_type is not None:
            out.add(action_type)
    return frozenset(out)


def _env_csv_set(name: str) -> frozenset[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    return frozenset(item.strip().lower() for item in raw.split(",") if item.strip())


def _action_kind(action: Action) -> str:
    kind = action.metadata.get("kind") if isinstance(action.metadata, dict) else None
    return str(kind or "").strip().lower()


def _action_kind_allowed(action: Action, allowed_kinds: frozenset[str]) -> bool:
    if not allowed_kinds:
        return True
    kind = _action_kind(action)
    if kind:
        return kind in allowed_kinds
    return action.action_type == ActionType.END_SHOP
