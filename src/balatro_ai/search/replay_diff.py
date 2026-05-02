"""Replay-diff validation for deterministic forward simulators."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Iterable

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, Stake
from balatro_ai.search.forward_sim import (
    simulate_buy,
    simulate_cash_out,
    simulate_choose_pack_card,
    simulate_discard,
    simulate_end_shop,
    simulate_open_pack,
    simulate_play,
    simulate_reroll,
    simulate_select_blind,
    simulate_sell,
)

SUPPORTED_ACTIONS = frozenset(
    {
        ActionType.PLAY_HAND,
        ActionType.DISCARD,
        ActionType.CASH_OUT,
        ActionType.BUY,
        ActionType.SELL,
        ActionType.REROLL,
        ActionType.OPEN_PACK,
        ActionType.CHOOSE_PACK_CARD,
        ActionType.END_SHOP,
        ActionType.SELECT_BLIND,
    }
)
KNOWN_GAP_JOKERS = {
    "Bloodstone": "Bloodstone has probabilistic XMult triggers",
    "Business Card": "Business Card has probabilistic money triggers",
    "Ceremonial Dagger": "Ceremonial Dagger current mult is dynamic",
    "Certificate": "Certificate-created cards can differ in older replay hand/deck snapshots",
    "Misprint": "Misprint has random mult",
    "Obelisk": "Obelisk depends on prior hand-type history and reset timing",
    "Ramen": "Ramen fractional XMult can differ by Lua rounding",
    "Red Card": "Red Card increments when booster packs are skipped; skip timing is not fully modeled yet",
    "Space Joker": "Space Joker can randomly upgrade hand level before scoring",
}
KNOWN_GAP_BLINDS = {
    "Cerulean Bell": "Cerulean Bell forced selection can make scored cards differ from requested action cards",
    "The Hook": "The Hook discards random held cards during play transitions",
    "The Pillar": "The Pillar debuffs previously played cards not always represented in card state",
}
COMPARE_FIELDS = (
    "phase",
    "run_over",
    "current_score",
    "money",
    "hands_remaining",
    "discards_remaining",
    "deck_size",
    "hand_multiset",
    "joker_multiset",
    "consumable_multiset",
    "voucher_multiset",
    "shop_surface",
    "pack_surface",
)


@dataclass(frozen=True, slots=True)
class FieldMismatch:
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
class TransitionDiff:
    path: Path
    row_number: int
    seed: str
    action_type: ActionType
    mismatches: tuple[str, ...]
    mismatch_details: tuple[FieldMismatch, ...] = ()
    known_gap_reason: str | None = None

    @property
    def exact_match(self) -> bool:
        return not self.mismatches

    def to_text(self) -> str:
        fields = ", ".join(self.mismatches) if self.mismatches else "exact"
        details = "; ".join(detail.to_text() for detail in self.mismatch_details)
        detail_text = f" details=[{details}]" if details else ""
        reason = f" known_gap={self.known_gap_reason}" if self.known_gap_reason else ""
        return (
            f"{self.path.name}:{self.row_number} seed={self.seed} action={self.action_type.value} "
            f"mismatches={fields}{detail_text}{reason}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "file": str(self.path),
            "row_number": self.row_number,
            "seed": self.seed,
            "action_type": self.action_type.value,
            "mismatches": list(self.mismatches),
            "mismatch_details": [detail.to_json_dict() for detail in self.mismatch_details],
            "exact_match": self.exact_match,
            "known_gap_reason": self.known_gap_reason,
        }


@dataclass(frozen=True, slots=True)
class ReplayDiffSummary:
    files_scanned: int
    malformed_rows: int = 0
    skipped_rows: int = 0
    diffs: tuple[TransitionDiff, ...] = ()
    transition_counts: Counter[str] = field(default_factory=Counter)
    exact_counts: Counter[str] = field(default_factory=Counter)
    field_mismatches: Counter[str] = field(default_factory=Counter)
    action_field_mismatches: Counter[str] = field(default_factory=Counter)
    mismatch_groups: Counter[str] = field(default_factory=Counter)
    known_gap_counts: Counter[str] = field(default_factory=Counter)

    @property
    def compared_transitions(self) -> int:
        return len(self.diffs)

    @property
    def exact_matches(self) -> int:
        return sum(1 for diff in self.diffs if diff.exact_match)

    @property
    def known_gap_mismatches(self) -> int:
        return sum(1 for diff in self.diffs if not diff.exact_match and diff.known_gap_reason)

    @property
    def comparable_transitions(self) -> int:
        return sum(1 for diff in self.diffs if diff.exact_match or not diff.known_gap_reason)

    @property
    def comparable_exact_matches(self) -> int:
        return sum(1 for diff in self.diffs if diff.exact_match and not diff.known_gap_reason)

    def to_text(self, *, example_limit: int = 10) -> str:
        comparable_rate = (
            self.comparable_exact_matches / self.comparable_transitions if self.comparable_transitions else 0.0
        )
        lines = [
            "Replay transition diff",
            f"Files scanned: {self.files_scanned}",
            f"Malformed rows: {self.malformed_rows}",
            f"Skipped rows: {self.skipped_rows}",
            f"Compared transitions: {self.compared_transitions}",
            (
                f"Comparable exact: {self.comparable_exact_matches}/{self.comparable_transitions} "
                f"({comparable_rate:.1%})"
            ),
            f"Known-gap mismatches: {self.known_gap_mismatches}",
        ]
        for action_type in sorted(self.transition_counts):
            total = self.transition_counts[action_type]
            exact = self.exact_counts[action_type]
            rate = exact / total if total else 0.0
            lines.append(f"{action_type}: exact {exact}/{total} ({rate:.1%})")
        lines.append(f"Field mismatches: {_format_counter(self.field_mismatches)}")
        lines.append(f"Action field mismatches: {_format_counter(self.action_field_mismatches)}")
        lines.append(f"Mismatch groups: {_format_counter(self.mismatch_groups)}")
        lines.append(f"Known gap reasons: {_format_counter(self.known_gap_counts)}")

        examples = tuple(diff for diff in self.diffs if not diff.exact_match)[:example_limit]
        if examples:
            lines.extend(("", "Mismatch examples:"))
            for diff in examples:
                lines.append(f"- {diff.to_text()}")
        return "\n".join(lines)

    def to_json_dict(self, *, example_limit: int = 20) -> dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "malformed_rows": self.malformed_rows,
            "skipped_rows": self.skipped_rows,
            "compared_transitions": self.compared_transitions,
            "exact_matches": self.exact_matches,
            "comparable_transitions": self.comparable_transitions,
            "comparable_exact_matches": self.comparable_exact_matches,
            "known_gap_mismatches": self.known_gap_mismatches,
            "transition_counts": _counter_to_json_dict(self.transition_counts),
            "exact_counts": _counter_to_json_dict(self.exact_counts),
            "field_mismatches": _counter_to_json_dict(self.field_mismatches),
            "action_field_mismatches": _counter_to_json_dict(self.action_field_mismatches),
            "mismatch_groups": _counter_to_json_dict(self.mismatch_groups),
            "known_gap_counts": _counter_to_json_dict(self.known_gap_counts),
            "examples": [
                diff.to_json_dict()
                for diff in tuple(diff for diff in self.diffs if not diff.exact_match)[:example_limit]
            ],
        }


def diff_replays(paths: Iterable[Path]) -> ReplayDiffSummary:
    replay_paths = _expand_paths(paths)
    diffs: list[TransitionDiff] = []
    malformed_rows = 0
    skipped_rows = 0
    for path in replay_paths:
        rows, bad_rows = _load_rows(path)
        malformed_rows += bad_rows
        file_diffs, file_skipped = _diff_rows(path, rows)
        diffs.extend(file_diffs)
        skipped_rows += file_skipped

    transition_counts: Counter[str] = Counter(diff.action_type.value for diff in diffs)
    exact_counts: Counter[str] = Counter(diff.action_type.value for diff in diffs if diff.exact_match)
    field_mismatches: Counter[str] = Counter(field for diff in diffs for field in diff.mismatches)
    action_field_mismatches: Counter[str] = Counter(
        f"{diff.action_type.value}.{field}" for diff in diffs for field in diff.mismatches
    )
    mismatch_groups: Counter[str] = Counter(
        f"{diff.action_type.value}:{'+'.join(diff.mismatches)}" for diff in diffs if diff.mismatches
    )
    known_gap_counts: Counter[str] = Counter(
        diff.known_gap_reason for diff in diffs if diff.known_gap_reason and not diff.exact_match
    )
    return ReplayDiffSummary(
        files_scanned=len(replay_paths),
        malformed_rows=malformed_rows,
        skipped_rows=skipped_rows,
        diffs=tuple(diffs),
        transition_counts=transition_counts,
        exact_counts=exact_counts,
        field_mismatches=field_mismatches,
        action_field_mismatches=action_field_mismatches,
        mismatch_groups=mismatch_groups,
        known_gap_counts=known_gap_counts,
    )


def _diff_rows(path: Path, rows: tuple[dict[str, object], ...]) -> tuple[list[TransitionDiff], int]:
    diffs: list[TransitionDiff] = []
    skipped = 0
    current_blind_key: tuple[int, str, int] | None = None
    played_hand_types: list[str] = []
    played_hand_counts: Counter[str] = Counter()
    current_joker_names: set[str] = set()
    ice_cream_hands_played = 0
    popcorn_rounds_elapsed = 0
    square_four_card_hands = 0
    ramen_discarded_cards = 0
    green_current_mult = 0
    loyalty_hands_played = 0
    for index, row in enumerate(rows):
        action = _action_from_row(row)
        row_joker_names = _joker_names_from_row(row)
        if "Ice Cream" in row_joker_names and "Ice Cream" not in current_joker_names:
            ice_cream_hands_played = 0
        if "Popcorn" in row_joker_names and "Popcorn" not in current_joker_names:
            popcorn_rounds_elapsed = 0
        if "Square Joker" in row_joker_names and "Square Joker" not in current_joker_names:
            square_four_card_hands = 0
        if "Ramen" in row_joker_names and "Ramen" not in current_joker_names:
            ramen_discarded_cards = 0
        if "Green Joker" in row_joker_names and "Green Joker" not in current_joker_names:
            green_current_mult = 0
        if "Loyalty Card" in row_joker_names and "Loyalty Card" not in current_joker_names:
            loyalty_hands_played = 0
        current_joker_names = row_joker_names
        state_key = _blind_key_from_detail(row.get("state_detail"))
        if state_key is not None and state_key != current_blind_key:
            current_blind_key = state_key
            played_hand_types = []
        if action is None or action.action_type not in SUPPORTED_ACTIONS:
            if action is not None and action.action_type in {ActionType.SELECT_BLIND, ActionType.SKIP_BLIND}:
                played_hand_types = []
                current_blind_key = None
            continue
        pre_state = _state_from_detail(row.get("state_detail"), audit=_score_audit_from_row(row))
        post_state = _post_state_after(rows, index)
        if pre_state is None or post_state is None:
            skipped += 1
            continue
        if action.action_type == ActionType.CASH_OUT and current_blind_key is not None:
            pre_state = _with_active_blind_key(pre_state, current_blind_key)
        pre_state = _with_reconstructed_play_history(pre_state, played_hand_types, played_hand_counts)
        pre_state = _with_inferred_visible_joker_values(
            pre_state,
            action=action,
            ice_cream_hands_played=ice_cream_hands_played,
            popcorn_rounds_elapsed=popcorn_rounds_elapsed,
            square_four_card_hands=square_four_card_hands,
            ramen_discarded_cards=ramen_discarded_cards,
            green_current_mult=green_current_mult,
            loyalty_hands_played=loyalty_hands_played,
        )
        try:
            simulated = _simulate_transition(pre_state, post_state, action)
        except (IndexError, ValueError):
            skipped += 1
            continue
        mismatch_details = _state_mismatch_details(simulated, post_state, action.action_type)
        mismatches = tuple(detail.field for detail in mismatch_details)
        known_gap_reason = _known_gap_reason(pre_state) if mismatches else None
        diffs.append(
            TransitionDiff(
                path=path,
                row_number=index + 1,
                seed=str(row.get("seed", path.stem)),
                action_type=action.action_type,
                mismatches=mismatches,
                mismatch_details=mismatch_details,
                known_gap_reason=known_gap_reason,
            )
        )
        if action.action_type == ActionType.PLAY_HAND:
            hand_type = _played_hand_type_from_row(row)
            if hand_type:
                played_hand_types.append(hand_type)
                played_hand_counts[hand_type] += 1
            if "Ice Cream" in row_joker_names:
                ice_cream_hands_played += 1
            if "Square Joker" in row_joker_names and len(action.card_indices) == 4:
                square_four_card_hands += 1
            if "Green Joker" in row_joker_names:
                green_current_mult += 1
            if "Loyalty Card" in row_joker_names:
                loyalty_hands_played += 1
        elif action.action_type == ActionType.DISCARD:
            if "Ramen" in row_joker_names:
                ramen_discarded_cards += len(action.card_indices)
            if "Green Joker" in row_joker_names:
                green_current_mult = max(0, green_current_mult - 1)
        elif action.action_type == ActionType.CASH_OUT:
            played_hand_types = []
            current_blind_key = None
            if "Popcorn" in row_joker_names:
                popcorn_rounds_elapsed += 1
    return diffs, skipped


def _simulate_transition(pre_state: GameState, post_state: GameState, action: Action) -> GameState:
    if action.action_type == ActionType.PLAY_HAND:
        held_cards = tuple(card for index, card in enumerate(pre_state.hand) if index not in set(action.card_indices))
        drawn_cards = _drawn_cards_from_post_state(held_cards, post_state.hand)
        return simulate_play(
            pre_state,
            action,
            drawn_cards,
            created_consumables=_added_labels(pre_state.consumables, post_state.consumables),
        )
    if action.action_type == ActionType.DISCARD:
        held_cards = tuple(card for index, card in enumerate(pre_state.hand) if index not in set(action.card_indices))
        drawn_cards = _drawn_cards_from_post_state(held_cards, post_state.hand)
        return simulate_discard(pre_state, action, drawn_cards)
    if action.action_type == ActionType.CASH_OUT:
        return simulate_cash_out(
            pre_state,
            next_shop_cards=_modifier_items(post_state.modifiers, "shop_cards"),
            next_voucher_cards=_modifier_items(post_state.modifiers, "voucher_cards"),
            next_booster_packs=_modifier_items(post_state.modifiers, "booster_packs"),
            next_to_do_targets=_to_do_list_cash_out_targets(post_state),
            removed_jokers=_removed_joker_names(pre_state.jokers, post_state.jokers),
        )
    if action.action_type == ActionType.SELECT_BLIND:
        certificate_cards = _select_blind_created_hand_cards(pre_state, post_state)
        drawn_cards = post_state.hand[: len(post_state.hand) - len(certificate_cards)]
        return simulate_select_blind(
            pre_state,
            drawn_cards=drawn_cards,
            created_hand_cards=certificate_cards,
            created_deck_cards=_select_blind_created_deck_cards(pre_state, post_state),
            created_consumables=_added_labels(pre_state.consumables, post_state.consumables),
            created_jokers=_added_joker_items(pre_state, post_state),
        )
    if action.action_type == ActionType.BUY:
        return simulate_buy(pre_state, action)
    if action.action_type == ActionType.SELL:
        return simulate_sell(pre_state, action)
    if action.action_type == ActionType.REROLL:
        return simulate_reroll(
            _with_inferred_reroll_cost(pre_state, post_state),
            action,
            new_shop_cards=_modifier_items(post_state.modifiers, "shop_cards"),
        )
    if action.action_type == ActionType.OPEN_PACK:
        return simulate_open_pack(
            pre_state,
            action,
            pack_contents=_modifier_items(post_state.modifiers, "pack_cards"),
            created_consumables=_added_labels(pre_state.consumables, post_state.consumables),
        )
    if action.action_type == ActionType.CHOOSE_PACK_CARD:
        remaining_pack_cards = (
            _modifier_items(post_state.modifiers, "pack_cards")
            if post_state.phase == GamePhase.BOOSTER_OPENED
            else None
        )
        return simulate_choose_pack_card(pre_state, action, remaining_pack_cards=remaining_pack_cards)
    if action.action_type == ActionType.END_SHOP:
        return simulate_end_shop(
            pre_state,
            created_consumables=_added_labels(pre_state.consumables, post_state.consumables),
        )
    raise ValueError(f"Unsupported action: {action.action_type.value}")


def _with_inferred_reroll_cost(pre_state: GameState, post_state: GameState) -> GameState:
    if any(key in pre_state.modifiers for key in ("reroll_cost", "current_reroll_cost", "free_rerolls")):
        return pre_state
    money_spent = pre_state.money - post_state.money
    if money_spent > 0:
        return replace(pre_state, modifiers={**pre_state.modifiers, "reroll_cost": money_spent})
    if money_spent == 0:
        return replace(pre_state, modifiers={**pre_state.modifiers, "free_rerolls": 1})
    return pre_state


def _modifier_items(modifiers: dict[str, object], key: str) -> tuple[object, ...]:
    raw = modifiers.get(key, ())
    if isinstance(raw, dict):
        raw = raw.get("cards", ())
    if isinstance(raw, list | tuple):
        return tuple(raw)
    return ()


def _added_labels(before: tuple[str, ...], after: tuple[str, ...]) -> tuple[str, ...]:
    remaining = Counter(before)
    added: list[str] = []
    for item in after:
        if remaining[item] > 0:
            remaining[item] -= 1
        else:
            added.append(item)
    return tuple(added)


def _select_blind_created_hand_cards(pre_state: GameState, post_state: GameState) -> tuple[Card, ...]:
    count = _first_draw_effective_joker_count(pre_state.jokers, "Certificate")
    if count <= 0:
        return ()
    if count > len(post_state.hand):
        raise ValueError(f"Certificate expected {count} created hand cards, got {len(post_state.hand)} total hand cards")
    return post_state.hand[-count:]


def _select_blind_created_deck_cards(pre_state: GameState, post_state: GameState) -> tuple[Card, ...]:
    count = _first_draw_effective_joker_count(pre_state.jokers, "Marble Joker")
    if count <= 0:
        return ()
    observed_added = _added_cards(pre_state.known_deck, post_state.known_deck)
    if len(observed_added) >= count:
        return observed_added[:count]
    return tuple(Card("2", "S") for _ in range(count))


def _added_cards(before: tuple[Card, ...], after: tuple[Card, ...]) -> tuple[Card, ...]:
    remaining = _card_counter(before)
    added: list[Card] = []
    for card in after:
        key = _card_key(card)
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            added.append(card)
    return tuple(added)


def _added_joker_items(pre_state: GameState, post_state: GameState) -> tuple[object, ...]:
    remaining = _joker_counter(pre_state.jokers)
    post_details = _modifier_items(post_state.modifiers, "joker_cards")
    items: list[object] = []
    if len(post_details) == len(post_state.jokers):
        for joker, detail in zip(post_state.jokers, post_details, strict=False):
            key = _joker_key(joker)
            if remaining[key] > 0:
                remaining[key] -= 1
            else:
                items.append(detail)
        return tuple(items)

    for joker in post_state.jokers:
        key = _joker_key(joker)
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            items.append(_joker_to_item(joker))
    return tuple(items)


def _removed_joker_names(before: tuple[Joker, ...], after: tuple[Joker, ...]) -> tuple[str, ...]:
    remaining = _joker_counter(after)
    removed: list[str] = []
    for joker in before:
        key = _joker_key(joker)
        if remaining[key] > 0:
            remaining[key] -= 1
        elif joker.name in {"Gros Michel", "Cavendish"}:
            removed.append(joker.name)
    return tuple(removed)


def _joker_to_item(joker: Joker) -> dict[str, object]:
    item: dict[str, object] = {"name": joker.name}
    if joker.edition is not None:
        item["edition"] = joker.edition
    if joker.sell_value is not None:
        item["sell_value"] = joker.sell_value
    if joker.metadata:
        item["metadata"] = dict(joker.metadata)
    return item


def _first_draw_effective_joker_count(jokers: tuple[Joker, ...], name: str) -> int:
    return sum(1 for joker in _active_effective_jokers(jokers) if joker.name == name)


def _active_effective_jokers(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    sources = _effective_ability_joker_indices(jokers)
    effective = tuple(jokers[index] for index in sources)
    return tuple(
        ability
        for physical, ability in zip(jokers, effective, strict=False)
        if not _joker_is_disabled(physical) and not _joker_is_disabled(ability)
    )


def _effective_ability_joker_indices(jokers: tuple[Joker, ...]) -> tuple[int, ...]:
    def resolve(index: int, seen: frozenset[int] = frozenset()) -> int:
        joker = jokers[index]
        if index in seen:
            return index
        if joker.name == "Blueprint" and index + 1 < len(jokers):
            return resolve(index + 1, seen | {index})
        if joker.name == "Brainstorm" and index != 0:
            return resolve(0, seen | {index})
        return index

    return tuple(resolve(index) for index in range(len(jokers)))


def _joker_is_disabled(joker: Joker) -> bool:
    state = joker.metadata.get("state")
    if isinstance(state, dict) and state.get("debuff"):
        return True
    return "all abilities are disabled" in _joker_effect_text(joker).lower()


def _joker_effect_text(joker: Joker) -> str:
    value = joker.metadata.get("value")
    if isinstance(value, dict):
        return str(value.get("effect", ""))
    return str(joker.metadata.get("effect", ""))


def _to_do_list_cash_out_targets(state: GameState) -> tuple[str, ...]:
    targets: list[str] = []
    for joker in state.jokers:
        if joker.name == "To Do List":
            targets.append(_to_do_list_current_target(joker))
    return tuple(targets)


def _to_do_list_current_target(joker: Joker) -> str:
    for source in _metadata_sources(joker.metadata):
        for key in ("target_hand", "to_do_poker_hand", "poker_hand", "hand_type"):
            value = source.get(key)
            if value:
                return str(value)
        value = source.get("value")
        if isinstance(value, dict):
            effect = value.get("effect")
            if isinstance(effect, str):
                target = _to_do_target_from_effect(effect)
                if target:
                    return target
    return "High Card"


def _to_do_target_from_effect(effect: str) -> str | None:
    lowered = effect.lower()
    hand_names = (
        "Flush Five",
        "Flush House",
        "Five of a Kind",
        "Straight Flush",
        "Four of a Kind",
        "Full House",
        "Three of a Kind",
        "Two Pair",
        "Straight",
        "Flush",
        "Pair",
        "High Card",
    )
    for name in hand_names:
        if name.lower() in lowered:
            return name
    return None


def _state_mismatches(simulated: GameState, actual: GameState, action_type: ActionType) -> tuple[str, ...]:
    return tuple(detail.field for detail in _state_mismatch_details(simulated, actual, action_type))


def _state_mismatch_details(
    simulated: GameState,
    actual: GameState,
    action_type: ActionType,
) -> tuple[FieldMismatch, ...]:
    mismatches: list[FieldMismatch] = []
    if simulated.phase != actual.phase:
        mismatches.append(FieldMismatch("phase", simulated.phase.value, actual.phase.value))
    if simulated.run_over != actual.run_over:
        mismatches.append(FieldMismatch("run_over", simulated.run_over, actual.run_over))
    if simulated.current_score != actual.current_score:
        mismatches.append(FieldMismatch("current_score", simulated.current_score, actual.current_score))
    if simulated.money != actual.money:
        mismatches.append(FieldMismatch("money", simulated.money, actual.money))
    if simulated.hands_remaining != actual.hands_remaining:
        mismatches.append(FieldMismatch("hands_remaining", simulated.hands_remaining, actual.hands_remaining))
    if simulated.discards_remaining != actual.discards_remaining:
        mismatches.append(FieldMismatch("discards_remaining", simulated.discards_remaining, actual.discards_remaining))
    if _should_compare_visible_hand(action_type, actual):
        if simulated.deck_size != actual.deck_size:
            mismatches.append(FieldMismatch("deck_size", simulated.deck_size, actual.deck_size))
        simulated_hand = _card_counter(simulated.hand)
        actual_hand = _card_counter(actual.hand)
        if simulated_hand != actual_hand:
            delta = _counter_delta(simulated_hand, actual_hand)
            mismatches.append(
                FieldMismatch(
                    "hand_multiset",
                    delta["simulated_only"],
                    delta["actual_only"],
                )
            )
    if _should_compare_inventory(action_type):
        simulated_jokers = _joker_counter(simulated.jokers)
        actual_jokers = _joker_counter(actual.jokers)
        if simulated_jokers != actual_jokers:
            delta = _counter_delta_text(simulated_jokers, actual_jokers)
            mismatches.append(FieldMismatch("joker_multiset", delta["simulated_only"], delta["actual_only"]))

        simulated_consumables = _text_counter(simulated.consumables)
        actual_consumables = _text_counter(actual.consumables)
        if simulated_consumables != actual_consumables:
            delta = _counter_delta_text(simulated_consumables, actual_consumables)
            mismatches.append(FieldMismatch("consumable_multiset", delta["simulated_only"], delta["actual_only"]))

        simulated_vouchers = _text_counter(simulated.vouchers)
        actual_vouchers = _text_counter(actual.vouchers)
        if simulated_vouchers != actual_vouchers:
            delta = _counter_delta_text(simulated_vouchers, actual_vouchers)
            mismatches.append(FieldMismatch("voucher_multiset", delta["simulated_only"], delta["actual_only"]))

        simulated_shop = _shop_surface_counter(simulated)
        actual_shop = _shop_surface_counter(actual)
        if simulated_shop != actual_shop:
            delta = _counter_delta_text(simulated_shop, actual_shop)
            mismatches.append(FieldMismatch("shop_surface", delta["simulated_only"], delta["actual_only"]))

        simulated_pack = _pack_surface_counter(simulated)
        actual_pack = _pack_surface_counter(actual)
        if simulated_pack != actual_pack:
            delta = _counter_delta_text(simulated_pack, actual_pack)
            mismatches.append(FieldMismatch("pack_surface", delta["simulated_only"], delta["actual_only"]))
    return tuple(mismatches)


def _should_compare_visible_hand(action_type: ActionType, actual: GameState) -> bool:
    if action_type != ActionType.PLAY_HAND:
        return True
    if actual.blind == "The Hook":
        return False
    return actual.phase in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND}


def _should_compare_inventory(action_type: ActionType) -> bool:
    return action_type in {
        ActionType.CASH_OUT,
        ActionType.BUY,
        ActionType.SELL,
        ActionType.REROLL,
        ActionType.OPEN_PACK,
        ActionType.CHOOSE_PACK_CARD,
        ActionType.END_SHOP,
        ActionType.SELECT_BLIND,
    }


def _known_gap_reason(state: GameState) -> str | None:
    if state.blind in KNOWN_GAP_BLINDS:
        return KNOWN_GAP_BLINDS[state.blind]
    for joker in state.jokers:
        reason = KNOWN_GAP_JOKERS.get(joker.name)
        if reason:
            return reason
    return None


def _joker_names_from_row(row: dict[str, object]) -> set[str]:
    detail = row.get("state_detail")
    if not isinstance(detail, dict):
        return set()
    return {
        str(joker.get("name"))
        for joker in _list_of_dicts(detail.get("jokers"))
        if joker.get("name") is not None
    }


def _post_state_after(rows: tuple[dict[str, object], ...], row_index: int) -> GameState | None:
    if row_index + 1 >= len(rows):
        return None
    next_row = rows[row_index + 1]
    if next_row.get("record_type") == "run_summary":
        return _state_from_detail(next_row.get("final_state_detail"), summary_row=next_row)
    return _state_from_detail(next_row.get("state_detail"))


def _with_active_blind_key(state: GameState, blind_key: tuple[int, str, int]) -> GameState:
    ante, blind, required_score = blind_key
    return replace(state, ante=ante, blind=blind, required_score=required_score)


def _with_reconstructed_play_history(
    state: GameState,
    played_hand_types: list[str],
    played_hand_counts: Counter[str],
) -> GameState:
    if not played_hand_types and not played_hand_counts:
        return state
    hands = state.modifiers.get("hands")
    if isinstance(hands, dict) and hands:
        return state
    reconstructed: dict[str, dict[str, int]] = {}
    for hand_type, played in played_hand_counts.items():
        reconstructed[hand_type] = {"played_this_round": 0, "played": int(played), "order": 0}
    for order, hand_type in enumerate(played_hand_types, start=1):
        entry = reconstructed.setdefault(hand_type, {"played_this_round": 0, "played": 0, "order": order})
        entry["played_this_round"] += 1
        entry["played"] = max(entry["played"], int(played_hand_counts.get(hand_type, 0)))
        if not entry.get("order"):
            entry["order"] = order
    return GameState(
        phase=state.phase,
        stake=state.stake,
        seed=state.seed,
        ante=state.ante,
        blind=state.blind,
        required_score=state.required_score,
        current_score=state.current_score,
        hands_remaining=state.hands_remaining,
        discards_remaining=state.discards_remaining,
        money=state.money,
        deck_size=state.deck_size,
        hand=state.hand,
        known_deck=state.known_deck,
        jokers=state.jokers,
        consumables=state.consumables,
        vouchers=state.vouchers,
        shop=state.shop,
        pack=state.pack,
        hand_levels=state.hand_levels,
        modifiers={**state.modifiers, "hands": reconstructed},
        legal_actions=state.legal_actions,
        run_over=state.run_over,
        won=state.won,
    )


def _with_inferred_visible_joker_values(
    state: GameState,
    *,
    action: Action,
    ice_cream_hands_played: int,
    popcorn_rounds_elapsed: int,
    square_four_card_hands: int,
    ramen_discarded_cards: int,
    green_current_mult: int,
    loyalty_hands_played: int,
) -> GameState:
    if not any(
        joker.name in {"Ice Cream", "Popcorn", "Square Joker", "Ramen", "Green Joker", "Loyalty Card"}
        for joker in state.jokers
    ):
        return state

    jokers: list[Joker] = []
    for joker in state.jokers:
        if joker.name == "Ice Cream" and not _joker_has_current_value(joker, "chips"):
            jokers.append(_with_joker_metadata(joker, {"current_chips": max(0, 100 - (5 * ice_cream_hands_played))}))
        elif joker.name == "Popcorn" and not _joker_has_current_value(joker, "mult"):
            jokers.append(_with_joker_metadata(joker, {"current_mult": max(0, 20 - (4 * popcorn_rounds_elapsed))}))
        elif joker.name == "Square Joker" and not _joker_has_current_value(joker, "chips"):
            jokers.append(_with_joker_metadata(joker, {"current_chips": 4 * square_four_card_hands}))
        elif joker.name == "Ramen" and not _joker_has_current_value(joker, "xmult"):
            jokers.append(_with_joker_metadata(joker, {"current_xmult": _ramen_xmult_after_discards(ramen_discarded_cards)}))
        elif joker.name == "Green Joker" and not _joker_has_current_value(joker, "mult"):
            jokers.append(_with_joker_metadata(joker, {"current_mult": max(0, green_current_mult)}))
        elif joker.name == "Loyalty Card" and not _joker_has_countdown_value(joker):
            remaining = (5 - loyalty_hands_played) % 6
            jokers.append(_with_joker_metadata(joker, {"current_remaining": remaining}))
        else:
            jokers.append(joker)
    return GameState(
        phase=state.phase,
        stake=state.stake,
        seed=state.seed,
        ante=state.ante,
        blind=state.blind,
        required_score=state.required_score,
        current_score=state.current_score,
        hands_remaining=state.hands_remaining,
        discards_remaining=state.discards_remaining,
        money=state.money,
        deck_size=state.deck_size,
        hand=state.hand,
        known_deck=state.known_deck,
        jokers=tuple(jokers),
        consumables=state.consumables,
        vouchers=state.vouchers,
        shop=state.shop,
        pack=state.pack,
        hand_levels=state.hand_levels,
        modifiers=state.modifiers,
        legal_actions=state.legal_actions,
        run_over=state.run_over,
        won=state.won,
    )


def _with_joker_metadata(joker: Joker, updates: dict[str, object]) -> Joker:
    return Joker(
        name=joker.name,
        edition=joker.edition,
        sell_value=joker.sell_value,
        metadata={**joker.metadata, **updates},
    )


def _ramen_xmult_after_discards(discarded_cards: int) -> float:
    xmult = 2.0
    for _ in range(max(0, discarded_cards)):
        xmult -= 0.01
    return max(1.0, xmult)


def _joker_has_current_value(joker: Joker, suffix: str) -> bool:
    keys = {
        "chips": ("current_chips", "chips"),
        "mult": ("current_mult", "mult"),
    }.get(suffix, (f"current_{suffix}", suffix))
    for source in _metadata_sources(joker.metadata):
        if any(key in source for key in keys):
            return True
        effect = source.get("effect")
        if isinstance(effect, str) and "currently" in effect.lower():
            return True
    value = joker.metadata.get("value")
    if isinstance(value, dict):
        effect = value.get("effect")
        if isinstance(effect, str) and "currently" in effect.lower():
            return True
    return False


def _joker_has_countdown_value(joker: Joker) -> bool:
    for source in _metadata_sources(joker.metadata):
        if any(key in source for key in ("current_remaining", "remaining", "hands_remaining", "hands_left")):
            return True
        effect = source.get("effect")
        if isinstance(effect, str) and (
            "remaining" in effect.lower() or "ready" in effect.lower() or "active" in effect.lower()
        ):
            return True
    value = joker.metadata.get("value")
    if isinstance(value, dict):
        effect = value.get("effect")
        if isinstance(effect, str) and (
            "remaining" in effect.lower() or "ready" in effect.lower() or "active" in effect.lower()
        ):
            return True
    return False


def _metadata_sources(metadata: dict[str, object]) -> tuple[dict[str, object], ...]:
    sources = [metadata]
    for key in ("ability", "config", "extra"):
        value = metadata.get(key)
        if isinstance(value, dict):
            sources.append(value)
            nested_extra = value.get("extra")
            if isinstance(nested_extra, dict):
                sources.append(nested_extra)
    return tuple(sources)


def _state_from_detail(
    detail: object,
    *,
    summary_row: dict[str, object] | None = None,
    audit: dict[str, object] | None = None,
) -> GameState | None:
    if not isinstance(detail, dict):
        return None
    final = summary_row or {}
    audit = audit or {}
    phase = _phase_from_value(detail.get("phase"))
    hand_details = audit.get("hand_before_details")
    if not isinstance(hand_details, list):
        hand_details = detail.get("hand")
    joker_details = audit.get("joker_details")
    if not isinstance(joker_details, list):
        joker_details = detail.get("jokers")
    hand_levels = audit.get("hand_levels")
    if not isinstance(hand_levels, dict):
        hand_levels = detail.get("hand_levels")
    shop_cards = _list_of_dicts(detail.get("shop"))
    voucher_cards = _list_of_dicts(detail.get("voucher_shop"))
    booster_packs = _list_of_dicts(detail.get("booster_packs"))
    pack_cards = _list_of_dicts(detail.get("pack"))
    modifiers = {
        **_mapping(detail.get("modifiers")),
        "hands": _mapping(detail.get("hands")),
        "joker_cards": tuple(_list_of_dicts(joker_details)),
        "shop_cards": shop_cards,
        "voucher_cards": voucher_cards,
        "booster_packs": booster_packs,
        "pack_cards": pack_cards,
    }
    for source_key, modifier_key in (
        ("played_pile", "played_pile"),
        ("discard_pile", "discard_pile"),
        ("destroyed_cards", "destroyed_cards"),
    ):
        cards = _cards_from_detail_value(detail.get(source_key))
        if cards:
            modifiers[modifier_key] = cards
    current_blind = detail.get("current_blind")
    if isinstance(current_blind, dict):
        modifiers["current_blind"] = current_blind
    return GameState(
        phase=phase,
        stake=Stake.UNKNOWN,
        seed=_optional_int(final.get("seed")),
        ante=_int_value(detail.get("ante")),
        blind=str(detail.get("blind", "")),
        required_score=_int_value(detail.get("required_score")),
        current_score=_int_value(detail.get("current_score")),
        hands_remaining=_int_value(detail.get("hands_remaining")),
        discards_remaining=_int_value(detail.get("discards_remaining")),
        money=_int_value(detail.get("money", final.get("final_money"))),
        deck_size=_int_value(detail.get("deck_size")),
        hand=tuple(_card_from_detail(card) for card in _list_of_dicts(hand_details)),
        known_deck=tuple(_card_from_detail(card) for card in _list_of_dicts(detail.get("known_deck"))),
        jokers=tuple(_joker_from_detail(joker) for joker in _list_of_dicts(joker_details)),
        consumables=_item_labels_from_value(detail.get("consumables")),
        vouchers=_item_labels_from_value(detail.get("owned_vouchers", detail.get("redeemed_vouchers", ()))),
        shop=tuple(_item_label(item) for item in shop_cards),
        pack=tuple(_item_label(item) for item in pack_cards),
        hand_levels={str(key): _int_value(value) for key, value in _mapping(hand_levels).items()},
        modifiers=modifiers,
        run_over=bool(final.get("record_type") == "run_summary" and final.get("won") is not True)
        or phase == GamePhase.RUN_OVER,
        won=bool(final.get("won", False)),
    )


def _action_from_row(row: dict[str, object]) -> Action | None:
    action = row.get("chosen_action")
    if not isinstance(action, dict):
        return None
    try:
        return Action.from_mapping(action)
    except (KeyError, ValueError, TypeError):
        return None


def _score_audit_from_row(row: dict[str, object]) -> dict[str, object] | None:
    extra = row.get("extra")
    if not isinstance(extra, dict):
        return None
    audit = extra.get("score_audit")
    return audit if isinstance(audit, dict) else None


def _played_hand_type_from_row(row: dict[str, object]) -> str | None:
    audit = _score_audit_from_row(row)
    if not isinstance(audit, dict):
        return None
    hand_type = audit.get("hand_type")
    return str(hand_type) if hand_type else None


def _blind_key_from_detail(detail: object) -> tuple[int, str, int] | None:
    if not isinstance(detail, dict):
        return None
    phase = _phase_from_value(detail.get("phase"))
    if phase not in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND}:
        return None
    return (
        _int_value(detail.get("ante")),
        str(detail.get("blind", "")),
        _int_value(detail.get("required_score")),
    )


def _drawn_cards_from_post_state(held_cards: tuple[Card, ...], actual_hand: tuple[Card, ...]) -> tuple[Card, ...]:
    remaining_held = _card_counter(held_cards)
    drawn: list[Card] = []
    for card in actual_hand:
        key = _card_key(card)
        if remaining_held[key] > 0:
            remaining_held[key] -= 1
        else:
            drawn.append(card)
    return tuple(drawn)


def _card_counter(cards: tuple[Card, ...]) -> Counter[tuple[object, ...]]:
    return Counter(_card_key(card) for card in cards)


def _joker_counter(jokers: tuple[Joker, ...]) -> Counter[str]:
    return Counter(_joker_key(joker) for joker in jokers)


def _joker_key(joker: Joker) -> str:
    parts = [joker.name]
    if joker.edition:
        parts.append(str(joker.edition))
    if joker.sell_value is not None:
        parts.append(f"${joker.sell_value}")
    return "/".join(parts)


def _text_counter(items: tuple[str, ...]) -> Counter[str]:
    return Counter(str(item) for item in items)


def _shop_surface_counter(state: GameState) -> Counter[str]:
    labels: list[str] = []
    for prefix, key in (
        ("card", "shop_cards"),
        ("voucher", "voucher_cards"),
        ("pack", "booster_packs"),
    ):
        labels.extend(f"{prefix}:{_item_label(item)}" for item in _modifier_items(state.modifiers, key))
    return Counter(labels)


def _pack_surface_counter(state: GameState) -> Counter[str]:
    return Counter(_item_label(item) for item in _modifier_items(state.modifiers, "pack_cards"))


def _card_key(card: Card) -> tuple[object, ...]:
    return (card.rank, card.suit, card.enhancement, card.seal, card.edition, card.debuffed)


def _counter_delta_text(simulated: Counter[str], actual: Counter[str]) -> dict[str, dict[str, int]]:
    simulated_only: dict[str, int] = {}
    actual_only: dict[str, int] = {}
    for key in sorted(set(simulated) | set(actual)):
        difference = simulated[key] - actual[key]
        if difference > 0:
            simulated_only[str(key)] = difference
        elif difference < 0:
            actual_only[str(key)] = abs(difference)
    return {"simulated_only": simulated_only, "actual_only": actual_only}


def _counter_delta(
    simulated: Counter[tuple[object, ...]],
    actual: Counter[tuple[object, ...]],
) -> dict[str, dict[str, int]]:
    simulated_only: dict[str, int] = {}
    actual_only: dict[str, int] = {}
    for key in sorted(set(simulated) | set(actual), key=_card_key_text):
        difference = simulated[key] - actual[key]
        if difference > 0:
            simulated_only[_card_key_text(key)] = difference
        elif difference < 0:
            actual_only[_card_key_text(key)] = abs(difference)
    return {"simulated_only": simulated_only, "actual_only": actual_only}


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


def _phase_from_value(value: object) -> GamePhase:
    try:
        return GamePhase(str(value))
    except ValueError:
        return GamePhase.UNKNOWN


def _card_from_detail(detail: dict[str, object]) -> Card:
    rank = detail.get("rank")
    suit = detail.get("suit")
    if rank is None or suit is None:
        name = str(detail.get("name", ""))
        rank = name[:-1]
        suit = name[-1:]
    return Card(
        rank=str(rank),
        suit=str(suit),
        enhancement=_optional_str(detail.get("enhancement")),
        seal=_optional_str(detail.get("seal")),
        edition=_optional_str(detail.get("edition")),
        debuffed=bool(detail.get("debuffed", False)),
        metadata=_mapping(detail.get("metadata")),
    )


def _joker_from_detail(detail: dict[str, object]) -> Joker:
    metadata = dict(_mapping(detail.get("metadata")))
    for key in ("set", "rarity", "cost"):
        if key in detail and detail[key] is not None:
            metadata[key] = detail[key]
    return Joker(
        name=str(detail.get("name", "Unknown Joker")),
        edition=_optional_str(detail.get("edition")),
        sell_value=_optional_int(detail.get("sell_value")),
        metadata=metadata,
    )


def _load_rows(path: Path) -> tuple[tuple[dict[str, object], ...], int]:
    rows: list[dict[str, object]] = []
    bad_rows = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_rows += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    return tuple(rows), bad_rows


def _expand_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.jsonl")))
        elif path.exists():
            expanded.append(path)
    return tuple(expanded)


def _list_of_dicts(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _cards_from_detail_value(value: object) -> tuple[Card, ...]:
    return tuple(_card_from_detail(item) for item in _list_of_dicts(value))


def _item_labels_from_value(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(_item_label(item) for item in value if isinstance(item, str | dict))


def _item_label(item: object) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    return str(item.get("label", item.get("name", item.get("key", "unknown"))))


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "{}"
    return "{" + ", ".join(f"{key}: {counter[key]}" for key in sorted(counter)) + "}"


def _short_value(value: object) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ", ".join(f"{key}: {value[key]}" for key in sorted(value))
        return "{" + items + "}"
    return repr(value)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _counter_to_json_dict(counter: Counter) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diff replay transitions against Phase 7 forward simulators.")
    parser.add_argument("paths", nargs="*", type=Path, help="Replay JSONL file(s) or directories.")
    parser.add_argument("--replay-dir", type=Path, help="Replay directory to scan recursively.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary.")
    parser.add_argument("--examples", type=int, default=10, help="Maximum text mismatch examples to print.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = list(args.paths)
    if args.replay_dir is not None:
        paths.append(args.replay_dir)
    if not paths:
        raise SystemExit("Provide at least one replay path or --replay-dir.")
    summary = diff_replays(paths)
    if args.json:
        print(json.dumps(summary.to_json_dict(example_limit=args.examples), indent=2, sort_keys=True))
    else:
        print(summary.to_text(example_limit=args.examples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
