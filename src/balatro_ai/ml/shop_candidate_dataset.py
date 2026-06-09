"""Shop candidate-ranking data for the neural policy.

The old value-head path asks ``V(state)`` to infer long-horizon shop quality from
noisy whole-run labels. This module instead creates action-relative records:
for one shop state, enumerate plausible legal actions, roll out every candidate
under common random numbers, and store a ranked candidate set. A neural ranker
can train on ``score(state, candidate_action)`` without being capped to a single
teacher action.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from math import sqrt
from statistics import mean
from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, with_derived_legal_actions
from balatro_ai.ml.encoding import ENCODING_VERSION, encode_state


SHOP_CANDIDATE_ACTION_TYPES = frozenset(
    {
        ActionType.BUY,
        ActionType.OPEN_PACK,
        ActionType.SELL,
        ActionType.REROLL,
        ActionType.END_SHOP,
    }
)
LABEL_VALUE_VERSION = 3
_HORIZON_QUALITY_SCALE = 240.0
_HORIZON_QUALITY_MAX = 0.95
_HORIZON_RESOURCE_SCALE = 480.0


@dataclass(frozen=True, slots=True)
class ShopCandidateLabel:
    action_key: str
    action: dict[str, Any]
    shop_token_index: int
    rollout_values: tuple[float, ...]
    mean_value: float
    first_half_mean: float
    second_half_mean: float
    rank: int
    is_heuristic_action: bool


@dataclass(frozen=True, slots=True)
class ShopCandidateRecord:
    seed: str
    state_index: int
    source_bot: str
    label_value_version: int
    phase: str
    ante: int
    money: int
    encoding_version: int
    encoded_state: dict[str, Any]
    state_snapshot: dict[str, Any]
    heuristic_action_key: str | None
    best_action_key: str
    candidates: tuple[ShopCandidateLabel, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedRolloutAdvantage:
    """Paired CRN advantage stats for one candidate against another."""

    n: int
    mean: float
    sem: float
    lower_bound: float
    upper_bound: float
    positive_rate: float


def action_key(action: Action) -> str:
    """Stable candidate identity, including shop metadata when present."""

    meta = action.metadata or {}
    kind = meta.get("kind", "")
    index = meta.get("index", "")
    return f"{action.stable_key}|kind={kind}|index={index}"


def paired_rollout_advantage(
    candidate: ShopCandidateLabel | dict[str, Any],
    baseline: ShopCandidateLabel | dict[str, Any],
    *,
    z: float = 1.0,
) -> PairedRolloutAdvantage | None:
    """Return paired rollout advantage stats for ``candidate - baseline``.

    Candidate records are labeled with common random numbers in the same order,
    so paired differences remove much of the seed noise that makes early shop
    labels look like single-action truth when they are really near-ties.
    """

    candidate_values = _candidate_rollout_values(candidate)
    baseline_values = _candidate_rollout_values(baseline)
    n = min(len(candidate_values), len(baseline_values))
    if n < 2:
        return None
    diffs = [candidate_values[index] - baseline_values[index] for index in range(n)]
    avg = float(mean(diffs))
    if n > 1:
        variance = sum((value - avg) ** 2 for value in diffs) / (n - 1)
        sem = float(sqrt(max(0.0, variance) / n))
    else:
        sem = 0.0
    z = max(0.0, float(z))
    return PairedRolloutAdvantage(
        n=n,
        mean=avg,
        sem=sem,
        lower_bound=avg - z * sem,
        upper_bound=avg + z * sem,
        positive_rate=float(sum(value > 0.0 for value in diffs) / n),
    )


def rollout_confidence_summary(
    records: Sequence[ShopCandidateRecord | dict[str, Any]],
    *,
    z: float = 1.0,
    practical_margin: float = 0.10,
) -> dict[str, float]:
    """Summarize whether rollout labels support one-best or override training.

    The mean winner can be a noisy sample artifact. These metrics use paired CRN
    differences to separate clear preferences from ambiguous early-game choices.
    """

    z = max(0.0, float(z))
    practical_margin = max(0.0, float(practical_margin))
    best_runnerup: list[PairedRolloutAdvantage] = []
    heuristic_best: list[PairedRolloutAdvantage] = []
    high_conf_override_counts: list[int] = []
    high_conf_practical_counts: list[int] = []

    candidate_record_count = 0
    heuristic_covered = 0
    for record in records:
        candidates = _record_candidates(record)
        if len(candidates) < 2:
            continue
        candidate_record_count += 1
        ordered = sorted(candidates, key=lambda candidate: _candidate_float(candidate, "mean_value"), reverse=True)
        best = ordered[0]
        runnerup = ordered[1]
        stats = paired_rollout_advantage(best, runnerup, z=z)
        if stats is not None:
            best_runnerup.append(stats)

        heuristic = next((candidate for candidate in candidates if _candidate_bool(candidate, "is_heuristic_action")), None)
        if heuristic is None:
            continue
        heuristic_covered += 1
        stats = paired_rollout_advantage(best, heuristic, z=z)
        if stats is not None:
            heuristic_best.append(stats)

        clear_count = 0
        practical_count = 0
        heuristic_key = _candidate_str(heuristic, "action_key")
        for candidate in candidates:
            if _candidate_str(candidate, "action_key") == heuristic_key:
                continue
            advantage = paired_rollout_advantage(candidate, heuristic, z=z)
            if advantage is None:
                continue
            if advantage.lower_bound > 0.0:
                clear_count += 1
            if advantage.lower_bound > practical_margin:
                practical_count += 1
        high_conf_override_counts.append(clear_count)
        high_conf_practical_counts.append(practical_count)

    return {
        "rollout_confidence_z": z,
        "rollout_practical_margin": practical_margin,
        "rollout_confidence_candidate_records": candidate_record_count,
        "best_runnerup_covered_n": len(best_runnerup),
        "mean_best_runnerup_delta": _mean_or_nan(stat.mean for stat in best_runnerup),
        "mean_best_runnerup_sem": _mean_or_nan(stat.sem for stat in best_runnerup),
        "mean_best_runnerup_lcb": _mean_or_nan(stat.lower_bound for stat in best_runnerup),
        "mean_best_runnerup_positive_sample_rate": _mean_or_nan(stat.positive_rate for stat in best_runnerup),
        "best_runnerup_high_conf_rate": _mean_bool(stat.lower_bound > 0.0 for stat in best_runnerup),
        "best_runnerup_practical_high_conf_rate": _mean_bool(
            stat.lower_bound > practical_margin for stat in best_runnerup
        ),
        "best_runnerup_ambiguous_rate": _mean_bool(
            stat.lower_bound <= 0.0 <= stat.upper_bound for stat in best_runnerup
        ),
        "heuristic_confidence_covered_n": heuristic_covered,
        "best_vs_heuristic_covered_n": len(heuristic_best),
        "mean_best_advantage_vs_heuristic": _mean_or_nan(stat.mean for stat in heuristic_best),
        "mean_best_sem_vs_heuristic": _mean_or_nan(stat.sem for stat in heuristic_best),
        "mean_best_lcb_advantage_vs_heuristic": _mean_or_nan(stat.lower_bound for stat in heuristic_best),
        "mean_best_positive_sample_rate_vs_heuristic": _mean_or_nan(
            stat.positive_rate for stat in heuristic_best
        ),
        "oracle_positive_vs_heuristic_rate": _mean_bool(stat.mean > 1e-9 for stat in heuristic_best),
        "oracle_practical_positive_vs_heuristic_rate": _mean_bool(
            stat.mean > practical_margin for stat in heuristic_best
        ),
        "best_high_conf_beats_heuristic_rate": _mean_bool(stat.lower_bound > 0.0 for stat in heuristic_best),
        "best_practical_high_conf_beats_heuristic_rate": _mean_bool(
            stat.lower_bound > practical_margin for stat in heuristic_best
        ),
        "best_ambiguous_vs_heuristic_rate": _mean_bool(
            stat.mean > 1e-9 and stat.lower_bound <= 0.0 for stat in heuristic_best
        ),
        "mean_high_conf_override_candidates_per_state": _mean_or_nan(high_conf_override_counts),
        "mean_high_conf_practical_override_candidates_per_state": _mean_or_nan(high_conf_practical_counts),
        "any_high_conf_override_candidate_rate": _mean_bool(count > 0 for count in high_conf_override_counts),
        "any_high_conf_practical_override_candidate_rate": _mean_bool(
            count > 0 for count in high_conf_practical_counts
        ),
    }


def candidate_shop_actions(
    state: GameState,
    *,
    max_actions: int = 12,
    max_sells: int = 2,
    action_types: Collection[ActionType] | None = None,
    priority: str = "legal",
) -> tuple[Action, ...]:
    """Plausible shop candidates for action-relative neural labels."""

    if state.phase != GamePhase.SHOP:
        return ()
    derived = with_derived_legal_actions(state)
    allowed = SHOP_CANDIDATE_ACTION_TYPES if action_types is None else frozenset(action_types)
    actions = [a for a in derived.legal_actions if a.action_type in allowed]
    non_sell = [a for a in actions if a.action_type != ActionType.SELL]
    sells = [a for a in actions if a.action_type == ActionType.SELL][:max_sells]
    ordered = (*non_sell, *sells)
    if priority == "deep_advantage":
        ordered = tuple(sorted(enumerate(ordered), key=lambda item: (_deep_advantage_priority(item[1]), item[0])))
        ordered = tuple(action for _, action in ordered)
    elif priority != "legal":
        raise ValueError(f"unknown shop candidate priority: {priority!r}")
    deduped: list[Action] = []
    seen: set[str] = set()
    for action in ordered:
        key = action_key(action)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
        if len(deduped) >= max_actions:
            break
    return tuple(deduped)


def label_shop_state(
    state: GameState,
    *,
    seed: str,
    state_index: int,
    crn_seeds: Sequence[int],
    rollout_bot: str,
    heuristic_bot: str | None = None,
    source_bot: str = "",
    max_antes: int,
    max_steps: int = 300,
    max_actions: int = 12,
    candidate_action_types: Collection[ActionType] | None = None,
    candidate_priority: str = "legal",
    include_heuristic_action: bool = False,
) -> ShopCandidateRecord | None:
    """Create one detailed ranker record for a shop state."""

    actions = candidate_shop_actions(
        state,
        max_actions=max_actions,
        action_types=candidate_action_types,
        priority=candidate_priority,
    )
    heuristic_action = _heuristic_action(state, heuristic_bot or rollout_bot)
    heuristic_action_key = action_key(heuristic_action) if heuristic_action is not None else None
    if include_heuristic_action and heuristic_action is not None:
        actions = _append_missing_action(actions, heuristic_action)
    if len(actions) < 2:
        return None
    if len(crn_seeds) < 2 or len(crn_seeds) % 2:
        raise ValueError("crn_seeds must have an even length >= 2")

    raw: list[tuple[Action, tuple[float, ...]]] = []
    for action in actions:
        values: list[float] = []
        ok = True
        for crn_seed in crn_seeds:
            value = rollout_value_after_action(
                state,
                action,
                seed=int(crn_seed),
                rollout_bot=rollout_bot,
                max_antes=max_antes,
                max_steps=max_steps,
            )
            if value is None:
                ok = False
                break
            values.append(value)
        if ok:
            raw.append((action, tuple(values)))
    if len(raw) < 2:
        return None

    half = len(crn_seeds) // 2
    ranked = sorted(raw, key=lambda item: mean(item[1]), reverse=True)
    labels: list[ShopCandidateLabel] = []
    for rank, (action, values) in enumerate(ranked, start=1):
        key = action_key(action)
        labels.append(
            ShopCandidateLabel(
                action_key=key,
                action=action.to_json(),
                shop_token_index=_shop_token_index(state, action),
                rollout_values=values,
                mean_value=float(mean(values)),
                first_half_mean=float(mean(values[:half])),
                second_half_mean=float(mean(values[half:])),
                rank=rank,
                is_heuristic_action=key == heuristic_action_key,
            )
        )

    return ShopCandidateRecord(
        seed=seed,
        state_index=state_index,
        source_bot=source_bot,
        label_value_version=LABEL_VALUE_VERSION,
        phase=state.phase.value,
        ante=int(state.ante),
        money=int(state.money),
        encoding_version=ENCODING_VERSION,
        encoded_state=asdict(encode_state(state)),
        state_snapshot=state_snapshot(state),
        heuristic_action_key=heuristic_action_key,
        best_action_key=labels[0].action_key,
        candidates=tuple(labels),
    )


def state_snapshot(state: GameState) -> dict[str, Any]:
    """Return a JSON-friendly state mapping that ``GameState.from_mapping`` can reload."""

    modifiers = _jsonable_mapping(state.modifiers)
    return {
        "phase": state.phase.value,
        "stake": state.stake.value,
        "seed": state.seed,
        "ante": int(state.ante),
        "blind": state.blind,
        "required_score": int(state.required_score),
        "current_score": int(state.current_score),
        "hands_remaining": int(state.hands_remaining),
        "discards_remaining": int(state.discards_remaining),
        "money": int(state.money),
        "deck_size": int(state.deck_size),
        "hand": [_card_snapshot(card) for card in state.hand],
        "known_deck": [_card_snapshot(card) for card in state.known_deck],
        "jokers": [_joker_snapshot(joker) for joker in state.jokers],
        "consumables": list(state.consumables),
        "owned_vouchers": list(state.vouchers),
        "shop": _jsonable_sequence(modifiers.get("shop_cards", state.shop)),
        "vouchers": _jsonable_sequence(modifiers.get("voucher_cards", ())),
        "packs": _jsonable_sequence(modifiers.get("booster_packs", ())),
        "pack": _jsonable_sequence(modifiers.get("pack_cards", state.pack)),
        "hand_levels": dict(state.hand_levels or {}),
        "modifiers": modifiers,
        "legal_actions": [action.to_json() for action in state.legal_actions],
        "run_over": bool(state.run_over),
        "won": bool(state.won),
    }


def _record_candidates(record: ShopCandidateRecord | dict[str, Any]) -> tuple[ShopCandidateLabel | dict[str, Any], ...]:
    if isinstance(record, ShopCandidateRecord):
        return tuple(record.candidates)
    candidates = record.get("candidates", ())
    if not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes | bytearray):
        return ()
    return tuple(candidate for candidate in candidates if isinstance(candidate, dict))


def _candidate_rollout_values(candidate: ShopCandidateLabel | dict[str, Any]) -> tuple[float, ...]:
    raw = candidate.rollout_values if isinstance(candidate, ShopCandidateLabel) else candidate.get("rollout_values", ())
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        return ()
    values: list[float] = []
    for value in raw:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return ()
    return tuple(values)


def _candidate_float(candidate: ShopCandidateLabel | dict[str, Any], field: str) -> float:
    raw = getattr(candidate, field) if isinstance(candidate, ShopCandidateLabel) else candidate.get(field)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("-inf")


def _candidate_bool(candidate: ShopCandidateLabel | dict[str, Any], field: str) -> bool:
    raw = getattr(candidate, field) if isinstance(candidate, ShopCandidateLabel) else candidate.get(field)
    return bool(raw)


def _candidate_str(candidate: ShopCandidateLabel | dict[str, Any], field: str) -> str:
    raw = getattr(candidate, field) if isinstance(candidate, ShopCandidateLabel) else candidate.get(field, "")
    return str(raw)


def _mean_or_nan(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(mean(items)) if items else float("nan")


def _mean_bool(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    return float(sum(items) / len(items)) if items else float("nan")


def _card_snapshot(card: Card) -> dict[str, Any]:
    return {
        "rank": card.rank,
        "suit": card.suit,
        "enhancement": card.enhancement,
        "seal": card.seal,
        "edition": card.edition,
        "debuffed": bool(card.debuffed),
        "metadata": _jsonable_mapping(card.metadata),
    }


def _joker_snapshot(joker: Joker) -> dict[str, Any]:
    return {
        "name": joker.name,
        "edition": joker.edition,
        "sell_value": joker.sell_value,
        "metadata": _jsonable_mapping(joker.metadata),
    }


def _jsonable_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _jsonable(raw) for key, raw in value.items()}


def _jsonable_sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return []


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Card):
        return _card_snapshot(value)
    if isinstance(value, Joker):
        return _joker_snapshot(value)
    if isinstance(value, Action):
        return value.to_json()
    if isinstance(value, dict):
        return {str(key): _jsonable(raw) for key, raw in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)


def collect_shop_states(
    seeds: Iterable[str],
    *,
    bot_name: str,
    cap: int,
    per_seed: int,
    stake: str = "white",
    max_steps: int = 2500,
    min_ante: int = 1,
    max_ante: int | None = None,
    balance_antes: bool = False,
    candidate_action_types: Collection[ActionType] | None = None,
    candidate_priority: str = "legal",
) -> list[tuple[str, int, GameState]]:
    """Collect real shop choice states from a policy trajectory."""

    from dataclasses import replace

    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    out: list[tuple[str, int, GameState]] = []
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for seed in seeds:
            sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake=stake)
            sim.state = SeedGame(seed, stake=stake).initial_state()
            bot = create_bot(bot_name, seed=0)
            taken = 0
            taken_by_ante: Counter[int] = Counter()
            for step in range(max_steps):
                state = sim.state
                if state.run_over or state.phase == GamePhase.RUN_OVER:
                    break
                candidates = candidate_shop_actions(
                    state,
                    action_types=candidate_action_types,
                    priority=candidate_priority,
                )
                ante = int(state.ante)
                in_ante_range = ante >= min_ante and (max_ante is None or ante <= max_ante)
                has_build_action = any(
                    action.action_type in {ActionType.BUY, ActionType.OPEN_PACK}
                    for action in candidates
                )
                if (
                    in_ante_range
                    and len(candidates) >= 2
                    and has_build_action
                    and _capture_quota_available(
                        ante=ante,
                        taken=taken,
                        taken_by_ante=taken_by_ante,
                        per_seed=per_seed,
                        max_ante=max_ante,
                        balance_antes=balance_antes,
                    )
                ):
                    out.append((seed, step, state))
                    taken += 1
                    taken_by_ante[ante] += 1
                    if len(out) >= cap:
                        return out
                    if _seed_capture_quota_full(
                        taken=taken,
                        taken_by_ante=taken_by_ante,
                        per_seed=per_seed,
                        min_ante=min_ante,
                        max_ante=max_ante,
                        balance_antes=balance_antes,
                    ):
                        break
                action = bot.choose_action(state)
                if action is None or action.action_type == ActionType.NO_OP:
                    break
                sim.step(action)
    return out


def _capture_quota_available(
    *,
    ante: int,
    taken: int,
    taken_by_ante: Counter[int],
    per_seed: int,
    max_ante: int | None,
    balance_antes: bool,
) -> bool:
    if balance_antes and max_ante is not None:
        return taken_by_ante[ante] < per_seed
    return taken < per_seed


def _seed_capture_quota_full(
    *,
    taken: int,
    taken_by_ante: Counter[int],
    per_seed: int,
    min_ante: int,
    max_ante: int | None,
    balance_antes: bool,
) -> bool:
    if balance_antes and max_ante is not None:
        return all(taken_by_ante[ante] >= per_seed for ante in range(min_ante, max_ante + 1))
    return taken >= per_seed


def rollout_value_after_action(
    state: GameState,
    action: Action,
    *,
    seed: int,
    rollout_bot: str,
    max_antes: int,
    max_steps: int = 300,
) -> float | None:
    """Apply one candidate, then run a bounded continuation policy."""

    from dataclasses import replace

    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    sim = LocalBalatroSimulator(seed=seed, stake="white")
    sim.state = state
    try:
        sim.step(action)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        return None
    bot = create_bot(rollout_bot, seed=seed)
    start_ante = state.ante
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(max_steps):
            current = sim.state
            if current.won:
                return _rollout_terminal_value(current, root_state=state)
            if current.run_over or current.phase == GamePhase.RUN_OVER:
                break
            if current.ante - start_ante >= max_antes:
                return _rollout_terminal_value(current, root_state=state)
            next_action = bot.choose_action(current)
            if next_action is None or next_action.action_type == ActionType.NO_OP:
                break
            try:
                sim.step(next_action)
            except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                break
    final = sim.state
    return _rollout_terminal_value(final, root_state=state)


def _rollout_terminal_value(state: GameState, *, root_state: GameState | None = None) -> float:
    """Outcome value with a bounded quality bonus for same-horizon survivors."""

    if state.won:
        return 9.0 + _horizon_quality_bonus(state, root_state=root_state)

    base = float(state.ante)
    if state.run_over or state.phase == GamePhase.RUN_OVER:
        return base + _score_fraction(state)
    if state.phase in {GamePhase.SHOP, GamePhase.BOOSTER_OPENED, GamePhase.BLIND_SELECT}:
        return base + _horizon_quality_bonus(state, root_state=root_state)
    return base + _score_fraction(state)


def _horizon_quality_bonus(state: GameState, *, root_state: GameState | None = None) -> float:
    """Small tie-breaker for candidates that all survive to the rollout horizon.

    The shop leaf value already includes scoring/build quality and some money
    terms, but it can saturate same-horizon survivors around clear strength.
    Add an explicit bounded resource term so a safely clearing line with a
    healthier bankroll remains distinguishable from a slightly louder score.
    """

    try:
        from balatro_ai.search.shop_search import shop_leaf_value

        raw = float(shop_leaf_value(state, root_state=root_state))
    except (ImportError, TypeError, ValueError, AttributeError):
        raw = _fallback_horizon_quality(state)
    strategic_bonus = max(0.0, raw / _HORIZON_QUALITY_SCALE)
    resource_bonus = max(0.0, _fallback_horizon_quality(state) / _HORIZON_RESOURCE_SCALE)
    return min(_HORIZON_QUALITY_MAX, strategic_bonus + resource_bonus)


def _fallback_horizon_quality(state: GameState) -> float:
    money_value = min(40.0, max(0.0, float(state.money))) * 1.4
    joker_value = min(5.0, float(len(state.jokers))) * 8.0
    consumable_value = min(2.0, float(len(state.consumables))) * 4.0
    voucher_value = min(6.0, float(len(state.vouchers))) * 3.0
    levels = state.hand_levels or {}
    leveling_value = min(20.0, float(sum(max(0, int(level) - 1) for level in levels.values()))) * 1.5
    return money_value + joker_value + consumable_value + voucher_value + leveling_value


def _heuristic_action_key(state: GameState, bot_name: str) -> str | None:
    action = _heuristic_action(state, bot_name)
    return action_key(action) if action is not None else None


def _heuristic_action(state: GameState, bot_name: str) -> Action | None:
    from balatro_ai.bots.registry import create_bot

    try:
        action = create_bot(bot_name, seed=0).choose_action(state)
    except Exception:
        return None
    return action


def _append_missing_action(actions: tuple[Action, ...], action: Action) -> tuple[Action, ...]:
    key = action_key(action)
    if any(action_key(existing) == key for existing in actions):
        return actions
    return (*actions, action)


def _deep_advantage_priority(action: Action) -> int:
    if action.action_type == ActionType.END_SHOP:
        return 0
    if action.action_type == ActionType.OPEN_PACK:
        return 1
    if action.action_type == ActionType.BUY:
        return 2
    if action.action_type == ActionType.REROLL:
        return 3
    if action.action_type == ActionType.SELL:
        return 4
    return 5


def _score_fraction(state: GameState) -> float:
    required = max(0, int(state.required_score or 0))
    if required <= 0:
        return 0.0
    return min(1.0, max(0.0, float(state.current_score) / required))


def _shop_token_index(state: GameState, action: Action) -> int:
    """Index into encode_state(state).shop for actions tied to a shop offer."""

    meta = action.metadata or {}
    kind = str(meta.get("kind", action.target_id or "")).lower()
    try:
        index = int(meta.get("index", action.amount if action.amount is not None else -1))
    except (TypeError, ValueError):
        return -1
    if index < 0:
        return -1

    shop_cards = tuple(item for item in state.modifiers.get("shop_cards", ()) if isinstance(item, dict))
    booster_packs = tuple(item for item in state.modifiers.get("booster_packs", ()) if isinstance(item, dict))
    voucher_cards = tuple(item for item in state.modifiers.get("voucher_cards", ()) if isinstance(item, dict))
    if kind == "card":
        return index if index < len(shop_cards) else -1
    if kind == "pack":
        return len(shop_cards) + index if index < len(booster_packs) else -1
    if kind == "voucher":
        return len(shop_cards) + len(booster_packs) + index if index < len(voucher_cards) else -1
    return -1
