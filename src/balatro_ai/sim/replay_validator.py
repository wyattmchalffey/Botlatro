"""Validate the local Python simulator against detailed bridge replays."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Iterable

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, with_derived_legal_actions
from balatro_ai.search import replay_diff
from balatro_ai.search.forward_sim import simulate_choose_pack_card
from balatro_ai.sim.local_runner import (
    _cleared_blind_state_for_progression,
    _jokers_after_skip,
    _with_blind_selection_surface,
    _with_cash_out_shop_surface,
    _with_next_blind_surface,
)

LOCAL_SUPPORTED_ACTIONS = replay_diff.SUPPORTED_ACTIONS | {
    ActionType.SKIP_BLIND,
    ActionType.REARRANGE,
    ActionType.NO_OP,
}


@dataclass(frozen=True, slots=True)
class LocalFieldMismatch:
    field: str
    simulated: object
    actual: object

    def to_text(self) -> str:
        return f"{self.field}: simulated={_short_value(self.simulated)} actual={_short_value(self.actual)}"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "simulated": _json_safe(self.simulated),
            "actual": _json_safe(self.actual),
        }


@dataclass(frozen=True, slots=True)
class LocalReplayDivergence:
    path: Path
    row_number: int
    seed: str
    stage: str
    action_type: ActionType
    mismatch_details: tuple[LocalFieldMismatch, ...]
    known_gap_reason: str | None = None

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(detail.field for detail in self.mismatch_details)

    def to_text(self) -> str:
        fields = ", ".join(self.fields) if self.fields else "exact"
        details = "; ".join(detail.to_text() for detail in self.mismatch_details)
        reason = f" known_gap={self.known_gap_reason}" if self.known_gap_reason else ""
        return (
            f"{self.path.name}:{self.row_number} seed={self.seed} stage={self.stage} "
            f"action={self.action_type.value} mismatches={fields} details=[{details}]{reason}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "file": str(self.path),
            "row_number": self.row_number,
            "seed": self.seed,
            "stage": self.stage,
            "action_type": self.action_type.value,
            "mismatches": list(self.fields),
            "mismatch_details": [detail.to_json_dict() for detail in self.mismatch_details],
            "known_gap_reason": self.known_gap_reason,
        }


@dataclass(frozen=True, slots=True)
class LocalSimReplayValidationSummary:
    files_scanned: int
    malformed_rows: int = 0
    skipped_rows: int = 0
    transitions_checked: int = 0
    divergences: tuple[LocalReplayDivergence, ...] = ()
    first_divergences: tuple[LocalReplayDivergence, ...] = ()
    action_counts: Counter[str] = field(default_factory=Counter)
    exact_transition_counts: Counter[str] = field(default_factory=Counter)
    stage_field_mismatches: Counter[str] = field(default_factory=Counter)
    action_field_mismatches: Counter[str] = field(default_factory=Counter)
    known_gap_counts: Counter[str] = field(default_factory=Counter)

    @property
    def exact_transitions(self) -> int:
        return sum(self.exact_transition_counts.values())

    @property
    def transition_exact_rate(self) -> float:
        return self.exact_transitions / self.transitions_checked if self.transitions_checked else 0.0

    @property
    def comparable_divergences(self) -> int:
        return sum(1 for divergence in self.divergences if divergence.known_gap_reason is None)

    def to_text(self, *, example_limit: int = 10) -> str:
        lines = [
            "Local simulator replay validation",
            f"Files scanned: {self.files_scanned}",
            f"Malformed rows: {self.malformed_rows}",
            f"Skipped rows: {self.skipped_rows}",
            f"Transitions checked: {self.transitions_checked}",
            f"Post-transition exact: {self.exact_transitions}/{self.transitions_checked} ({self.transition_exact_rate:.1%})",
            f"Comparable divergences: {self.comparable_divergences}",
        ]
        for action_type in sorted(self.action_counts):
            total = self.action_counts[action_type]
            exact = self.exact_transition_counts[action_type]
            rate = exact / total if total else 0.0
            lines.append(f"{action_type}: post exact {exact}/{total} ({rate:.1%})")
        lines.append(f"Stage field mismatches: {_format_counter(self.stage_field_mismatches)}")
        lines.append(f"Action field mismatches: {_format_counter(self.action_field_mismatches)}")
        lines.append(f"Known gap reasons: {_format_counter(self.known_gap_counts)}")

        examples = self.divergences[:example_limit]
        if examples:
            lines.extend(("", "Divergence examples:"))
            for divergence in examples:
                lines.append(f"- {divergence.to_text()}")

        if self.first_divergences:
            lines.extend(("", "First divergence per file:"))
            for divergence in self.first_divergences[:example_limit]:
                lines.append(f"- {divergence.to_text()}")
        return "\n".join(lines)

    def to_json_dict(self, *, example_limit: int = 20) -> dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "malformed_rows": self.malformed_rows,
            "skipped_rows": self.skipped_rows,
            "transitions_checked": self.transitions_checked,
            "exact_transitions": self.exact_transitions,
            "transition_exact_rate": self.transition_exact_rate,
            "comparable_divergences": self.comparable_divergences,
            "action_counts": _counter_to_json_dict(self.action_counts),
            "exact_transition_counts": _counter_to_json_dict(self.exact_transition_counts),
            "stage_field_mismatches": _counter_to_json_dict(self.stage_field_mismatches),
            "action_field_mismatches": _counter_to_json_dict(self.action_field_mismatches),
            "known_gap_counts": _counter_to_json_dict(self.known_gap_counts),
            "examples": [item.to_json_dict() for item in self.divergences[:example_limit]],
            "first_divergences": [item.to_json_dict() for item in self.first_divergences[:example_limit]],
        }


def validate_local_sim_replays(
    paths: Iterable[Path],
    *,
    resync_on_divergence: bool = True,
) -> LocalSimReplayValidationSummary:
    """Replay recorded actions through the local simulator and compare state drift."""

    replay_paths = replay_diff._expand_paths(paths)
    malformed_rows = 0
    skipped_rows = 0
    transitions_checked = 0
    divergences: list[LocalReplayDivergence] = []
    first_divergences: list[LocalReplayDivergence] = []
    action_counts: Counter[str] = Counter()
    exact_transition_counts: Counter[str] = Counter()

    for path in replay_paths:
        rows, bad_rows = replay_diff._load_rows(path)
        malformed_rows += bad_rows
        file_result = _validate_rows(path, rows, resync_on_divergence=resync_on_divergence)
        skipped_rows += file_result.skipped_rows
        transitions_checked += file_result.transitions_checked
        divergences.extend(file_result.divergences)
        if file_result.first_divergence is not None:
            first_divergences.append(file_result.first_divergence)
        action_counts.update(file_result.action_counts)
        exact_transition_counts.update(file_result.exact_transition_counts)

    stage_field_mismatches: Counter[str] = Counter(
        f"{divergence.stage}.{field}" for divergence in divergences for field in divergence.fields
    )
    action_field_mismatches: Counter[str] = Counter(
        f"{divergence.action_type.value}.{field}" for divergence in divergences for field in divergence.fields
    )
    known_gap_counts: Counter[str] = Counter(
        divergence.known_gap_reason for divergence in divergences if divergence.known_gap_reason
    )
    return LocalSimReplayValidationSummary(
        files_scanned=len(replay_paths),
        malformed_rows=malformed_rows,
        skipped_rows=skipped_rows,
        transitions_checked=transitions_checked,
        divergences=tuple(divergences),
        first_divergences=tuple(first_divergences),
        action_counts=action_counts,
        exact_transition_counts=exact_transition_counts,
        stage_field_mismatches=stage_field_mismatches,
        action_field_mismatches=action_field_mismatches,
        known_gap_counts=known_gap_counts,
    )


@dataclass(frozen=True, slots=True)
class _FileValidationResult:
    skipped_rows: int
    transitions_checked: int
    divergences: tuple[LocalReplayDivergence, ...]
    first_divergence: LocalReplayDivergence | None
    action_counts: Counter[str]
    exact_transition_counts: Counter[str]


def _validate_rows(
    path: Path,
    rows: tuple[dict[str, object], ...],
    *,
    resync_on_divergence: bool,
) -> _FileValidationResult:
    sim_state: GameState | None = None
    skipped_rows = 0
    transitions_checked = 0
    divergences: list[LocalReplayDivergence] = []
    first_divergence: LocalReplayDivergence | None = None
    action_counts: Counter[str] = Counter()
    exact_transition_counts: Counter[str] = Counter()
    tracker = _ReplayTracker()

    for index, row in enumerate(rows):
        if _is_independent_bridge_smoke_pre_row(row):
            sim_state = None
            tracker = _ReplayTracker()
        action = replay_diff._action_from_row(row)
        tracker.observe_row_start(row)
        if action is None or action.action_type not in LOCAL_SUPPORTED_ACTIONS:
            tracker.observe_row_end(row, action)
            continue

        observed_pre = _prepared_state_from_row(row, tracker)
        observed_post = replay_diff._post_state_after(rows, index)
        if observed_post is not None:
            observed_post = _with_tracker_active_blind(observed_post, tracker)
        if observed_pre is None or observed_post is None:
            skipped_rows += 1
            tracker.observe_row_end(row, action)
            continue

        if sim_state is None:
            sim_state = observed_pre
        else:
            pre_details = _local_state_mismatch_details(
                sim_state,
                observed_pre,
                stage="pre_state",
                action_type=action.action_type,
            )
            if pre_details:
                divergence = _divergence(
                    path,
                    index,
                    row,
                    action,
                    stage="pre_state",
                    details=pre_details,
                    state=observed_pre,
                )
                divergences.append(divergence)
                first_divergence = first_divergence or divergence
                if resync_on_divergence:
                    sim_state = observed_pre

        action_counts[action.action_type.value] += 1
        transitions_checked += 1
        simulation_pre_state = _with_observed_validation_modifiers(sim_state, observed_pre)
        try:
            simulated_post = _simulate_replay_step(simulation_pre_state, observed_post, action)
        except (IndexError, ValueError, RuntimeError) as exc:
            details = (LocalFieldMismatch("transition_error", type(exc).__name__, str(exc)),)
            divergence = _divergence(
                path,
                index,
                row,
                action,
                stage="post_transition",
                details=details,
                state=sim_state,
            )
            divergences.append(divergence)
            first_divergence = first_divergence or divergence
            skipped_rows += 1
            if resync_on_divergence:
                sim_state = observed_post
            tracker.observe_row_end(row, action)
            continue

        post_details = _local_state_mismatch_details(
            simulated_post,
            observed_post,
            stage="post_transition",
            action_type=action.action_type,
        )
        if post_details:
            known_gap_reason = _summary_win_flag_gap(rows, index, observed_post, post_details)
            divergence = _divergence(
                path,
                index,
                row,
                action,
                stage="post_transition",
                details=post_details,
                state=sim_state,
                known_gap_reason=known_gap_reason,
            )
            divergences.append(divergence)
            first_divergence = first_divergence or divergence
            sim_state = (
                _carry_hidden_validation_modifiers(observed_post, simulated_post)
                if resync_on_divergence
                else simulated_post
            )
        else:
            exact_transition_counts[action.action_type.value] += 1
            sim_state = simulated_post

        tracker.observe_row_end(row, action)

    return _FileValidationResult(
        skipped_rows=skipped_rows,
        transitions_checked=transitions_checked,
        divergences=tuple(divergences),
        first_divergence=first_divergence,
        action_counts=action_counts,
        exact_transition_counts=exact_transition_counts,
    )


def _is_independent_bridge_smoke_pre_row(row: dict[str, object]) -> bool:
    return row.get("record_type") == "bridge_joker_smoke" and row.get("stage") == "pre"


@dataclass(slots=True)
class _ReplayTracker:
    current_blind_key: tuple[int, str, int] | None = None
    played_hand_types: list[str] = field(default_factory=list)
    played_hand_counts: Counter[str] = field(default_factory=Counter)
    current_joker_names: set[str] = field(default_factory=set)
    ice_cream_hands_played: int = 0
    popcorn_rounds_elapsed: int = 0
    square_four_card_hands: int = 0
    ramen_discarded_cards: int = 0
    green_current_mult: int = 0
    loyalty_hands_played: int = 0

    def observe_row_start(self, row: dict[str, object]) -> None:
        row_joker_names = replay_diff._joker_names_from_row(row)
        if "Ice Cream" in row_joker_names and "Ice Cream" not in self.current_joker_names:
            self.ice_cream_hands_played = 0
        if "Popcorn" in row_joker_names and "Popcorn" not in self.current_joker_names:
            self.popcorn_rounds_elapsed = 0
        if "Square Joker" in row_joker_names and "Square Joker" not in self.current_joker_names:
            self.square_four_card_hands = 0
        if "Ramen" in row_joker_names and "Ramen" not in self.current_joker_names:
            self.ramen_discarded_cards = 0
        if "Green Joker" in row_joker_names and "Green Joker" not in self.current_joker_names:
            self.green_current_mult = 0
        if "Loyalty Card" in row_joker_names and "Loyalty Card" not in self.current_joker_names:
            self.loyalty_hands_played = 0
        self.current_joker_names = row_joker_names

        state_key = replay_diff._blind_key_from_detail(row.get("state_detail"))
        if state_key is not None and state_key != self.current_blind_key:
            self.current_blind_key = state_key
            self.played_hand_types = []

    def observe_row_end(self, row: dict[str, object], action: Action | None) -> None:
        if action is None:
            return
        if action.action_type == ActionType.PLAY_HAND:
            hand_type = replay_diff._played_hand_type_from_row(row)
            if hand_type:
                self.played_hand_types.append(hand_type)
                self.played_hand_counts[hand_type] += 1
            if "Ice Cream" in self.current_joker_names:
                self.ice_cream_hands_played += 1
            if "Square Joker" in self.current_joker_names and len(action.card_indices) == 4:
                self.square_four_card_hands += 1
            if "Green Joker" in self.current_joker_names:
                if _row_blind_name(row) == "The Hook":
                    self.green_current_mult = max(1, self.green_current_mult)
                else:
                    self.green_current_mult += 1
            if "Loyalty Card" in self.current_joker_names:
                self.loyalty_hands_played += 1
        elif action.action_type == ActionType.DISCARD:
            if "Ramen" in self.current_joker_names:
                self.ramen_discarded_cards += len(action.card_indices)
            if "Green Joker" in self.current_joker_names:
                self.green_current_mult = max(0, self.green_current_mult - 1)
        elif action.action_type == ActionType.CASH_OUT:
            self.played_hand_types = []
            self.current_blind_key = None
            if "Popcorn" in self.current_joker_names:
                self.popcorn_rounds_elapsed += 1
        elif action.action_type in {ActionType.SELECT_BLIND, ActionType.SKIP_BLIND}:
            self.played_hand_types = []


def _prepared_state_from_row(row: dict[str, object], tracker: _ReplayTracker) -> GameState | None:
    state = replay_diff._state_from_detail(row.get("state_detail"), audit=replay_diff._score_audit_from_row(row))
    action = replay_diff._action_from_row(row)
    if state is None or action is None:
        return state
    if action.action_type == ActionType.CASH_OUT and tracker.current_blind_key is not None:
        state = replay_diff._with_active_blind_key(state, tracker.current_blind_key)
    state = replay_diff._with_reconstructed_play_history(
        state,
        tracker.played_hand_types,
        tracker.played_hand_counts,
    )
    return replay_diff._with_inferred_visible_joker_values(
        state,
        action=action,
        ice_cream_hands_played=tracker.ice_cream_hands_played,
        popcorn_rounds_elapsed=tracker.popcorn_rounds_elapsed,
        square_four_card_hands=tracker.square_four_card_hands,
        ramen_discarded_cards=tracker.ramen_discarded_cards,
        green_current_mult=tracker.green_current_mult,
        loyalty_hands_played=tracker.loyalty_hands_played,
    )


def _row_blind_name(row: dict[str, object]) -> str:
    detail = row.get("state_detail")
    if isinstance(detail, dict):
        return str(detail.get("blind") or "")
    return ""


def _with_tracker_active_blind(state: GameState, tracker: _ReplayTracker) -> GameState:
    if state.phase == GamePhase.ROUND_EVAL and tracker.current_blind_key is not None:
        return replay_diff._with_active_blind_key(state, tracker.current_blind_key)
    return state


def _carry_hidden_validation_modifiers(actual: GameState, simulated: GameState) -> GameState:
    hidden_keys = ("cleared_blind", "played_pile", "discard_pile", "destroyed_cards")
    additions = {key: simulated.modifiers[key] for key in hidden_keys if key in simulated.modifiers}
    if not additions:
        return actual
    return replace(actual, modifiers={**actual.modifiers, **additions})


def _with_observed_validation_modifiers(simulated: GameState, observed: GameState) -> GameState:
    oracle_keys = (
        "round",
        "cash_out_money_delta",
        "current_round_dollars",
        "round_eval_dollars",
        "round_dollars",
        "reroll_cost",
        "current_reroll_cost",
        "blind_reward",
        "blind_score",
        "blind_on_deck",
        "blind_states",
        "money_per_hand",
        "money_per_discard",
        "no_extra_hand_money",
        "no_interest",
        "no_blind_reward",
        "interest_amount",
        "interest_cap",
        "shop_cards",
        "voucher_cards",
        "booster_packs",
        "pack_cards",
    )
    additions = {key: observed.modifiers[key] for key in oracle_keys if key in observed.modifiers}
    modifiers = {**simulated.modifiers, **additions} if additions else simulated.modifiers
    return replace(
        simulated,
        deck_size=observed.deck_size,
        hand=observed.hand,
        known_deck=observed.known_deck,
        jokers=observed.jokers,
        hand_levels=dict(observed.hand_levels),
        modifiers=modifiers,
    )


def _summary_win_flag_gap(
    rows: tuple[dict[str, object], ...],
    row_index: int,
    observed_post: GameState,
    details: tuple[LocalFieldMismatch, ...],
) -> str | None:
    if row_index + 1 >= len(rows):
        return None
    next_row = rows[row_index + 1]
    if next_row.get("record_type") != "run_summary" or next_row.get("won") is not True:
        return None
    if not any(detail.field == "won" for detail in details):
        return None
    if observed_post.required_score <= 0 or observed_post.current_score >= observed_post.required_score:
        return None
    return "Run summary win flag conflicts with final score below required score"


def _simulate_replay_step(pre_state: GameState, post_state: GameState, action: Action) -> GameState:
    if action.action_type == ActionType.CASH_OUT:
        simulated = replay_diff._simulate_transition(pre_state, post_state, action)
        simulated = _with_cash_out_shop_surface(
            simulated,
            cleared_state=pre_state,
            boss_name=_boss_name_for_surface(pre_state, post_state),
        )
    elif action.action_type == ActionType.END_SHOP:
        simulated = replay_diff._simulate_transition(pre_state, post_state, action)
        cleared_state = _cleared_blind_state_for_progression(pre_state)
        simulated = _with_next_blind_surface(
            simulated,
            cleared_state=cleared_state,
            phase=GamePhase.BLIND_SELECT,
            boss_name=_boss_name_for_surface(cleared_state, post_state),
        )
    elif action.action_type in replay_diff.SUPPORTED_ACTIONS:
        simulated = replay_diff._simulate_transition(pre_state, post_state, action)
    elif action.action_type == ActionType.SKIP_BLIND:
        simulated = _simulate_skip_blind(pre_state, post_state)
    elif action.action_type == ActionType.REARRANGE:
        simulated = _simulate_rearrange(pre_state, action)
    elif action.action_type == ActionType.NO_OP:
        simulated = _simulate_no_op(pre_state)
    else:
        raise ValueError(f"Unsupported local replay action: {action.action_type.value}")
    return with_derived_legal_actions(simulated) if not simulated.run_over else simulated


def _simulate_skip_blind(pre_state: GameState, post_state: GameState) -> GameState:
    blind_kind = _blind_kind(pre_state)
    if blind_kind == "BOSS":
        raise ValueError("Cannot skip boss blind")
    next_kind = "BIG" if blind_kind == "SMALL" else "BOSS"
    boss_name = _boss_name_for_surface(pre_state, post_state)
    skipped = replace(pre_state, jokers=_jokers_after_skip(pre_state.jokers))
    return _with_blind_selection_surface(
        skipped,
        ante=pre_state.ante,
        blind_kind=next_kind,
        boss_name=boss_name,
    )


def _simulate_rearrange(pre_state: GameState, action: Action) -> GameState:
    order = action.card_indices
    if sorted(order) != list(range(len(pre_state.jokers))):
        raise ValueError(f"Invalid joker rearrange order: {order}")
    return replace(pre_state, jokers=tuple(pre_state.jokers[index] for index in order))


def _simulate_no_op(pre_state: GameState) -> GameState:
    if pre_state.phase == GamePhase.BOOSTER_OPENED and not pre_state.pack:
        modifiers = dict(pre_state.modifiers)
        modifiers.pop("pack_choices_remaining", None)
        return replace(pre_state, phase=GamePhase.SHOP, modifiers=modifiers)
    return pre_state


def _boss_name_for_surface(pre_state: GameState, post_state: GameState) -> str:
    if post_state.blind not in {"Small Blind", "Big Blind"}:
        return post_state.blind
    for key in ("upcoming_boss", "next_boss", "boss_blind", "boss"):
        value = pre_state.modifiers.get(key)
        name = _blind_name_from_mapping(value)
        if name:
            return name
    blinds = pre_state.modifiers.get("blinds")
    if isinstance(blinds, dict):
        boss = blinds.get("boss") or blinds.get("Boss") or blinds.get("boss_blind")
        name = _blind_name_from_mapping(boss)
        if name:
            return name
    return "The Club"


def _blind_name_from_mapping(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        name = str(value.get("name", value.get("label", "")))
        return name or None
    return None


def _blind_kind(state: GameState) -> str:
    current_blind = state.modifiers.get("current_blind")
    if isinstance(current_blind, dict):
        kind = str(current_blind.get("type", current_blind.get("kind", ""))).upper()
        if kind in {"SMALL", "BIG", "BOSS"}:
            return kind
    if state.blind == "Small Blind":
        return "SMALL"
    if state.blind == "Big Blind":
        return "BIG"
    return "BOSS"


def _local_state_mismatch_details(
    simulated: GameState,
    actual: GameState,
    *,
    stage: str,
    action_type: ActionType,
) -> tuple[LocalFieldMismatch, ...]:
    mismatches: list[LocalFieldMismatch] = []
    for field in ("phase", "ante", "run_over", "won"):
        simulated_value = getattr(simulated, field)
        actual_value = getattr(actual, field)
        if isinstance(simulated_value, GamePhase):
            simulated_value = simulated_value.value
        if isinstance(actual_value, GamePhase):
            actual_value = actual_value.value
        if simulated_value != actual_value:
            mismatches.append(LocalFieldMismatch(field, simulated_value, actual_value))
    if _compare_blind_surface(actual):
        for field in ("blind", "required_score"):
            if getattr(simulated, field) != getattr(actual, field):
                mismatches.append(LocalFieldMismatch(field, getattr(simulated, field), getattr(actual, field)))
    for field in ("current_score", "money", "hands_remaining", "discards_remaining"):
        if getattr(simulated, field) != getattr(actual, field):
            mismatches.append(LocalFieldMismatch(field, getattr(simulated, field), getattr(actual, field)))

    if _compare_visible_hand(stage, action_type, actual):
        if simulated.deck_size != actual.deck_size:
            mismatches.append(LocalFieldMismatch("deck_size", simulated.deck_size, actual.deck_size))
        _append_card_counter_mismatch(mismatches, "hand_multiset", simulated.hand, actual.hand)
    if actual.known_deck:
        _append_card_counter_mismatch(mismatches, "known_deck_multiset", simulated.known_deck, actual.known_deck)

    _append_text_counter_mismatch(
        mismatches,
        "joker_multiset",
        _joker_counter(simulated.jokers),
        _joker_counter(actual.jokers),
    )
    _append_text_counter_mismatch(
        mismatches,
        "consumable_multiset",
        Counter(simulated.consumables),
        Counter(actual.consumables),
    )
    _append_text_counter_mismatch(
        mismatches,
        "voucher_multiset",
        Counter(simulated.vouchers),
        Counter(actual.vouchers),
    )
    _append_text_counter_mismatch(
        mismatches,
        "shop_surface",
        _shop_surface_counter(simulated),
        _shop_surface_counter(actual),
    )
    _append_text_counter_mismatch(
        mismatches,
        "pack_surface",
        _pack_surface_counter(simulated),
        _pack_surface_counter(actual),
    )
    return tuple(mismatches)


def _compare_blind_surface(actual: GameState) -> bool:
    return actual.phase != GamePhase.ROUND_EVAL


def _compare_visible_hand(stage: str, action_type: ActionType, actual: GameState) -> bool:
    if stage == "pre_state":
        return bool(actual.hand)
    if action_type == ActionType.REARRANGE:
        return False
    if action_type == ActionType.PLAY_HAND:
        return actual.phase in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND}
    return bool(actual.hand)


def _append_card_counter_mismatch(
    mismatches: list[LocalFieldMismatch],
    field: str,
    simulated: tuple[Card, ...],
    actual: tuple[Card, ...],
) -> None:
    simulated_counter = Counter(_card_key(card) for card in simulated)
    actual_counter = Counter(_card_key(card) for card in actual)
    if simulated_counter == actual_counter:
        return
    delta = _counter_delta(simulated_counter, actual_counter, key_text=_card_key_text)
    mismatches.append(LocalFieldMismatch(field, delta["simulated_only"], delta["actual_only"]))


def _append_text_counter_mismatch(
    mismatches: list[LocalFieldMismatch],
    field: str,
    simulated: Counter[str],
    actual: Counter[str],
) -> None:
    if simulated == actual:
        return
    delta = _counter_delta(simulated, actual, key_text=str)
    mismatches.append(LocalFieldMismatch(field, delta["simulated_only"], delta["actual_only"]))


def _counter_delta(simulated: Counter, actual: Counter, *, key_text) -> dict[str, dict[str, int]]:
    simulated_only: dict[str, int] = {}
    actual_only: dict[str, int] = {}
    for key in sorted(set(simulated) | set(actual), key=lambda item: key_text(item)):
        difference = simulated[key] - actual[key]
        if difference > 0:
            simulated_only[key_text(key)] = difference
        elif difference < 0:
            actual_only[key_text(key)] = abs(difference)
    return {"simulated_only": simulated_only, "actual_only": actual_only}


def _joker_counter(jokers: tuple[Joker, ...]) -> Counter[str]:
    return Counter(_joker_key(joker) for joker in jokers)


def _joker_key(joker: Joker) -> str:
    parts = [joker.name]
    if joker.edition:
        parts.append(str(joker.edition))
    if joker.sell_value is not None:
        parts.append(f"${joker.sell_value}")
    return "/".join(parts)


def _shop_surface_counter(state: GameState) -> Counter[str]:
    labels: list[str] = []
    for prefix, key in (("card", "shop_cards"), ("voucher", "voucher_cards"), ("pack", "booster_packs")):
        labels.extend(f"{prefix}:{_item_label(item)}" for item in replay_diff._modifier_items(state.modifiers, key))
    return Counter(labels)


def _pack_surface_counter(state: GameState) -> Counter[str]:
    return Counter(_item_label(item) for item in replay_diff._modifier_items(state.modifiers, "pack_cards"))


def _item_label(item: object) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    return str(item.get("label", item.get("name", item.get("key", "unknown"))))


def _card_key(card: Card) -> tuple[object, ...]:
    return (card.rank, card.suit, card.enhancement, card.seal, card.edition, card.debuffed)


def _card_key_text(key: tuple[object, ...]) -> str:
    rank, suit, enhancement, seal, edition, debuffed = key
    parts = [f"{rank}{suit}"]
    if enhancement:
        parts.append(str(enhancement))
    if seal:
        parts.append(str(seal))
    if edition:
        parts.append(str(edition))
    if debuffed:
        parts.append("debuffed")
    return "/".join(parts)


def _divergence(
    path: Path,
    row_index: int,
    row: dict[str, object],
    action: Action,
    *,
    stage: str,
    details: tuple[LocalFieldMismatch, ...],
    state: GameState,
    known_gap_reason: str | None = None,
) -> LocalReplayDivergence:
    return LocalReplayDivergence(
        path=path,
        row_number=row_index + 1,
        seed=str(row.get("seed", path.stem)),
        stage=stage,
        action_type=action.action_type,
        mismatch_details=details,
        known_gap_reason=known_gap_reason
        or replay_diff._known_gap_reason(state, action=action, mismatch_details=details),
    )


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "{}"
    return "{" + ", ".join(f"{key}: {counter[key]}" for key in sorted(counter)) + "}"


def _counter_to_json_dict(counter: Counter) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def _short_value(value: object) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{" + ", ".join(f"{key}: {value[key]}" for key in sorted(value)) + "}"
    return repr(value)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the local simulator against detailed replay JSONL files.")
    parser.add_argument("paths", nargs="*", type=Path, help="Replay JSONL file(s) or directories.")
    parser.add_argument("--replay-dir", type=Path, help="Replay directory to scan recursively.")
    parser.add_argument("--no-resync", action="store_true", help="Keep carrying drift after a divergence.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary.")
    parser.add_argument("--examples", type=int, default=10, help="Maximum text examples to print.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = list(args.paths)
    if args.replay_dir is not None:
        paths.append(args.replay_dir)
    if not paths:
        raise SystemExit("Provide at least one replay path or --replay-dir.")
    summary = validate_local_sim_replays(paths, resync_on_divergence=not args.no_resync)
    if args.json:
        print(json.dumps(summary.to_json_dict(example_limit=args.examples), indent=2, sort_keys=True))
    else:
        print(summary.to_text(example_limit=args.examples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
