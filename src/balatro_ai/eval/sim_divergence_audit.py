"""Diagnose where forward_sim diverges from real game state evolution.

Where balatrobench_score_audit only checks per-hand scoring, this audit feeds
the full prior state into simulate_play and compares the predicted post-state
to the actual post-state, field by field. The point is to attribute drift to
specific output fields (money, joker counters, deck composition, modifiers)
rather than lumping everything into a single score-error number.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.search.forward_sim import (
    simulate_discard,
    simulate_end_shop,
    simulate_play,
    simulate_reroll,
    simulate_sell,
)


STOCHASTIC_JOKERS = frozenset(
    {
        "Misprint",
        "Bloodstone",
        "Space Joker",
        "Business Card",
        "Reserved Parking",
        "Lucky Cat",
        "Gros Michel",
        "Cavendish",
        "Ramen",
        "Ice Cream",
        "Selzer",
        "Seltzer",
        "Hit the Road",
        "Hologram",
        "Madness",
        "Ancient Joker",
        "Castle",
        "Constellation",
        "Square Joker",
        "Wee Joker",
        "Riff-raff",
        "Cartomancer",
    }
)
STOCHASTIC_BLINDS = frozenset(
    {
        "The Hook",
        "The Mouth",
        "Cerulean Bell",
        "Crimson Heart",
        "Amber Acorn",
        "Verdant Leaf",
        "Violet Vessel",
    }
)
STOCHASTIC_ENHANCEMENTS = frozenset({"LUCKY", "GLASS"})
COMPARED_TOP_FIELDS = (
    "current_score",
    "money",
    "hands_remaining",
    "discards_remaining",
    "deck_size",
    "phase",
    "run_over",
)


@dataclass(frozen=True, slots=True)
class FieldDivergence:
    field: str
    predicted: Any
    actual: Any

    def short(self) -> str:
        return f"{self.field}: pred={self.predicted!r} actual={self.actual!r}"


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    run_path: Path
    transition_index: int
    ante: int
    blind: str
    action_type: str
    hand_type: str
    selected_cards: tuple[str, ...]
    jokers: tuple[str, ...]
    divergences: tuple[FieldDivergence, ...]
    skip_reason: str | None = None

    @property
    def has_divergence(self) -> bool:
        return bool(self.divergences)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_path": str(self.run_path),
            "transition_index": self.transition_index,
            "ante": self.ante,
            "blind": self.blind,
            "action_type": self.action_type,
            "hand_type": self.hand_type,
            "selected_cards": list(self.selected_cards),
            "jokers": list(self.jokers),
            "divergences": [
                {"field": d.field, "predicted": _json_safe(d.predicted), "actual": _json_safe(d.actual)}
                for d in self.divergences
            ],
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True, slots=True)
class SimDivergenceAudit:
    records: tuple[TransitionRecord, ...]
    runs_scanned: int
    skips: Counter[str]
    errors: tuple[str, ...] = ()

    @property
    def compared(self) -> tuple[TransitionRecord, ...]:
        return tuple(r for r in self.records if r.skip_reason is None)

    @property
    def diverged(self) -> tuple[TransitionRecord, ...]:
        return tuple(r for r in self.compared if r.has_divergence)

    @property
    def exact(self) -> int:
        return sum(1 for r in self.compared if not r.has_divergence)

    def field_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for record in self.diverged:
            for div in record.divergences:
                counts[div.field] += 1
        return counts

    def joker_divergence_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for record in self.diverged:
            for name in record.jokers:
                counts[name] += 1
        return counts

    def action_type_counts(self) -> Counter[str]:
        return Counter(r.action_type for r in self.compared)

    def action_type_divergence_counts(self) -> Counter[str]:
        return Counter(r.action_type for r in self.diverged)

    def to_text(self, *, worst_count: int = 12) -> str:
        compared = len(self.compared)
        diverged = len(self.diverged)
        lines = [
            "Forward-sim transition divergence audit",
            f"Runs scanned: {self.runs_scanned}",
            f"Transitions seen: {len(self.records)}",
            f"Compared: {compared}",
            f"Exact: {self.exact} ({self._pct(self.exact, compared)})",
            f"Diverged: {diverged} ({self._pct(diverged, compared)})",
            f"Errors: {len(self.errors)}",
        ]
        if self.skips:
            lines.append("Skips: " + json.dumps(dict(sorted(self.skips.items())), sort_keys=True))
        action_counts = self.action_type_counts()
        action_diverge = self.action_type_divergence_counts()
        if action_counts:
            lines.append("")
            lines.append("Per-action-type accuracy:")
            for action_name, count in action_counts.most_common():
                bad = action_diverge.get(action_name, 0)
                ok = count - bad
                lines.append(f"- {action_name}: {ok}/{count} exact ({self._pct(ok, count)}), {bad} diverged")
        field_counts = self.field_counts()
        if field_counts:
            lines.append("")
            lines.append("Diverged-record counts by field:")
            for field_name, count in field_counts.most_common():
                lines.append(f"- {field_name}: {count}")
        joker_counts = self.joker_divergence_counts()
        if joker_counts:
            lines.append("")
            lines.append("Top jokers present in diverged records:")
            for name, count in joker_counts.most_common(20):
                lines.append(f"- {name}: {count}")
        if self.diverged:
            lines.append("")
            lines.append(f"Sample diverged transitions (up to {worst_count}):")
            sample = sorted(
                self.diverged,
                key=lambda r: (-len(r.divergences), r.ante),
            )[:worst_count]
            for r in sample:
                lines.append(
                    f"- action={r.action_type} ante={r.ante} blind={r.blind or '-'} hand={r.hand_type} "
                    f"cards={' '.join(r.selected_cards)} jokers={list(r.jokers)} "
                    f"divergences=[{'; '.join(d.short() for d in r.divergences)}] "
                    f"file={r.run_path.name}:{r.transition_index}"
                )
        return "\n".join(lines)

    @staticmethod
    def _pct(part: int, whole: int) -> str:
        if whole <= 0:
            return "0.0%"
        return f"{100.0 * part / whole:.1f}%"


def audit_runs(paths: Iterable[Path]) -> SimDivergenceAudit:
    run_paths = tuple(_expand_run_paths(paths))
    records: list[TransitionRecord] = []
    skips: Counter[str] = Counter()
    errors: list[str] = []
    for run_path in run_paths:
        try:
            rows = _load_jsonl(run_path / "gamestates.jsonl")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{run_path}: {exc!r}")
            continue
        for index in range(1, len(rows)):
            record = _record_from_transition(run_path, index, rows[index - 1], rows[index], skips, errors)
            if record is not None:
                records.append(record)
    return SimDivergenceAudit(
        records=tuple(records),
        runs_scanned=len(run_paths),
        skips=skips,
        errors=tuple(errors),
    )


def _record_from_transition(
    run_path: Path,
    index: int,
    before: dict,
    after: dict,
    skips: Counter[str],
    errors: list[str],
) -> TransitionRecord | None:
    try:
        state = GameState.from_mapping(before)
        next_state = GameState.from_mapping(after)
    except (TypeError, ValueError) as exc:
        errors.append(f"{run_path}:{index + 1}: state parse failed: {exc!r}")
        return None

    if next_state.ante != state.ante or after.get("round_num") != before.get("round_num"):
        skips["round_or_ante_changed"] += 1
        return None

    # Hand-play / discard transitions: both states in the hand-play phases.
    if state.phase in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND} and next_state.phase in {
        GamePhase.SELECTING_HAND,
        GamePhase.PLAYING_BLIND,
    }:
        if next_state.current_score > state.current_score:
            return _audit_play_transition(run_path, index, state, next_state, skips, errors)
        if next_state.discards_remaining < state.discards_remaining:
            return _audit_discard_transition(run_path, index, state, next_state, skips, errors)
        skips["non_score_non_discard_transition"] += 1
        return None

    if state.phase == GamePhase.SHOP:
        if next_state.phase == GamePhase.SHOP:
            return _audit_shop_in_shop_transition(run_path, index, state, next_state, skips, errors)
        if next_state.phase == GamePhase.BLIND_SELECT:
            return _audit_end_shop_transition(run_path, index, state, next_state, skips, errors)
        skips[f"shop_to_{next_state.phase.value}"] += 1
        return None

    skips[f"pre_phase_{state.phase.value}"] += 1
    return None


def _audit_play_transition(
    run_path: Path,
    index: int,
    state: GameState,
    next_state: GameState,
    skips: Counter[str],
    errors: list[str],
) -> TransitionRecord | None:
    selected_cards, held_cards = _infer_selected_cards(state.hand, next_state.hand)
    if not 1 <= len(selected_cards) <= 5:
        skips[f"selected_count_{len(selected_cards)}"] += 1
        return None

    if any(j.name in STOCHASTIC_JOKERS for j in state.jokers):
        skips["stochastic_joker_present"] += 1
        return None
    if state.blind in STOCHASTIC_BLINDS:
        skips["stochastic_blind_present"] += 1
        return None
    if any(_card_enhancement(card) in STOCHASTIC_ENHANCEMENTS for card in selected_cards):
        skips["stochastic_enhancement_present"] += 1
        return None
    if any(_card_enhancement(card) in STOCHASTIC_ENHANCEMENTS for card in held_cards):
        skips["stochastic_enhancement_present"] += 1
        return None

    drawn_cards = _infer_drawn_cards(state.hand, next_state.hand, selected_cards)
    action = Action(action_type=ActionType.PLAY_HAND, card_indices=_indices_for_cards(state.hand, selected_cards))

    try:
        predicted = simulate_play(state, action, drawn_cards=drawn_cards)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        errors.append(f"{run_path}:{index + 1}: simulate_play failed: {exc!r}")
        skips["sim_error"] += 1
        return _sim_error_record(
            run_path, index, state, "play_hand", selected_cards, "sim_error"
        )

    divergences = _compare_states(predicted, next_state)
    return TransitionRecord(
        run_path=run_path,
        transition_index=index + 1,
        ante=state.ante,
        blind=state.blind,
        action_type="play_hand",
        hand_type=_infer_hand_type_label(predicted),
        selected_cards=tuple(card.short_name for card in selected_cards),
        jokers=tuple(j.name for j in state.jokers),
        divergences=divergences,
    )


def _audit_discard_transition(
    run_path: Path,
    index: int,
    state: GameState,
    next_state: GameState,
    skips: Counter[str],
    errors: list[str],
) -> TransitionRecord | None:
    selected_cards, held_cards = _infer_selected_cards(state.hand, next_state.hand)
    if not 1 <= len(selected_cards) <= 5:
        skips[f"discard_selected_count_{len(selected_cards)}"] += 1
        return None

    # Discard money/destruction effects are deterministic except for hook
    # discards. The Hook is already in STOCHASTIC_BLINDS — but discards under
    # other boss blinds can still trigger probabilistic joker behavior, so we
    # apply the same stochastic-joker gate as plays.
    if any(j.name in STOCHASTIC_JOKERS for j in state.jokers):
        skips["stochastic_joker_present"] += 1
        return None
    if state.blind in STOCHASTIC_BLINDS:
        skips["stochastic_blind_present"] += 1
        return None
    if any(_card_enhancement(card) in STOCHASTIC_ENHANCEMENTS for card in selected_cards):
        skips["stochastic_enhancement_present"] += 1
        return None

    drawn_cards = _infer_drawn_cards(state.hand, next_state.hand, selected_cards)
    action = Action(action_type=ActionType.DISCARD, card_indices=_indices_for_cards(state.hand, selected_cards))

    try:
        predicted = simulate_discard(state, action, drawn_cards=drawn_cards)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        errors.append(f"{run_path}:{index + 1}: simulate_discard failed: {exc!r}")
        skips["sim_error"] += 1
        return _sim_error_record(
            run_path, index, state, "discard", selected_cards, "sim_error"
        )

    divergences = _compare_states(predicted, next_state)
    return TransitionRecord(
        run_path=run_path,
        transition_index=index + 1,
        ante=state.ante,
        blind=state.blind,
        action_type="discard",
        hand_type="-",
        selected_cards=tuple(card.short_name for card in selected_cards),
        jokers=tuple(j.name for j in state.jokers),
        divergences=divergences,
    )


def _audit_shop_in_shop_transition(
    run_path: Path,
    index: int,
    state: GameState,
    next_state: GameState,
    skips: Counter[str],
    errors: list[str],
) -> TransitionRecord | None:
    # Classify what happened inside the shop. We support a deterministic
    # subset: SELL, REROLL, and BUY of items that don't trigger stochastic
    # creation. Everything else is skipped — auditing those requires either
    # RNG matching or extensive injection of created items.
    pre_jokers = tuple(j.name for j in state.jokers)
    post_jokers = tuple(j.name for j in next_state.jokers)
    money_delta = next_state.money - state.money

    pre_shop = _shop_item_labels(state)
    post_shop = _shop_item_labels(next_state)

    # SELL: joker removed, money increased.
    if money_delta > 0 and len(post_jokers) < len(pre_jokers):
        return _audit_sell_transition(run_path, index, state, next_state, skips, errors)

    # REROLL: shop contents fully changed, money decreased, joker set unchanged.
    if (
        money_delta < 0
        and pre_jokers == post_jokers
        and pre_shop != post_shop
        and _is_full_reroll(pre_shop, post_shop)
    ):
        return _audit_reroll_transition(run_path, index, state, next_state, skips, errors)

    # Everything else (BUY, USE_CONSUMABLE, OPEN_PACK, REARRANGE) — too much
    # injected state to audit cleanly at v2.
    skips["shop_action_unsupported"] += 1
    return None


def _audit_sell_transition(
    run_path: Path,
    index: int,
    state: GameState,
    next_state: GameState,
    skips: Counter[str],
    errors: list[str],
) -> TransitionRecord | None:
    if any(j.name in STOCHASTIC_JOKERS for j in state.jokers):
        skips["stochastic_joker_present"] += 1
        return None

    sold_index = _find_sold_joker_index(state.jokers, next_state.jokers)
    if sold_index is None:
        skips["sell_target_unclear"] += 1
        return None

    action = Action(
        action_type=ActionType.SELL,
        amount=sold_index,
        metadata={"target": "joker"},
    )
    try:
        predicted = simulate_sell(state, action)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        errors.append(f"{run_path}:{index + 1}: simulate_sell failed: {exc!r}")
        skips["sim_error"] += 1
        return _sim_error_record(run_path, index, state, "sell", (), "sim_error")

    divergences = _compare_states(predicted, next_state)
    sold_name = state.jokers[sold_index].name if 0 <= sold_index < len(state.jokers) else "?"
    return TransitionRecord(
        run_path=run_path,
        transition_index=index + 1,
        ante=state.ante,
        blind=state.blind,
        action_type="sell",
        hand_type=sold_name,
        selected_cards=(),
        jokers=tuple(j.name for j in state.jokers),
        divergences=divergences,
    )


def _audit_reroll_transition(
    run_path: Path,
    index: int,
    state: GameState,
    next_state: GameState,
    skips: Counter[str],
    errors: list[str],
) -> TransitionRecord | None:
    # Inject the actual shop roll from the post-state so we can isolate sim
    # bugs from RNG mismatch.
    next_shop_items = _modifier_items(next_state.modifiers, "shop_cards")
    action = Action(action_type=ActionType.REROLL)
    try:
        predicted = simulate_reroll(state, action, next_shop_items)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        errors.append(f"{run_path}:{index + 1}: simulate_reroll failed: {exc!r}")
        skips["sim_error"] += 1
        return _sim_error_record(run_path, index, state, "reroll", (), "sim_error")

    divergences = _compare_states(predicted, next_state)
    return TransitionRecord(
        run_path=run_path,
        transition_index=index + 1,
        ante=state.ante,
        blind=state.blind,
        action_type="reroll",
        hand_type="-",
        selected_cards=(),
        jokers=tuple(j.name for j in state.jokers),
        divergences=divergences,
    )


def _audit_end_shop_transition(
    run_path: Path,
    index: int,
    state: GameState,
    next_state: GameState,
    skips: Counter[str],
    errors: list[str],
) -> TransitionRecord | None:
    # End-shop may create consumables (e.g., Cartomancer) or change deck
    # composition (Marble Joker). For v2 we only audit the simple path where
    # no creation happened.
    if len(next_state.consumables) != len(state.consumables):
        skips["end_shop_consumable_created"] += 1
        return None
    if any(j.name in {"Cartomancer", "Marble Joker", "Riff-raff", "DNA"} for j in state.jokers):
        skips["end_shop_creation_joker"] += 1
        return None

    try:
        predicted = simulate_end_shop(state)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        errors.append(f"{run_path}:{index + 1}: simulate_end_shop failed: {exc!r}")
        skips["sim_error"] += 1
        return _sim_error_record(run_path, index, state, "end_shop", (), "sim_error")

    divergences = _compare_states(predicted, next_state)
    return TransitionRecord(
        run_path=run_path,
        transition_index=index + 1,
        ante=state.ante,
        blind=state.blind,
        action_type="end_shop",
        hand_type="-",
        selected_cards=(),
        jokers=tuple(j.name for j in state.jokers),
        divergences=divergences,
    )


def _sim_error_record(
    run_path: Path,
    index: int,
    state: GameState,
    action_type: str,
    selected_cards: tuple[Card, ...],
    reason: str,
) -> TransitionRecord:
    return TransitionRecord(
        run_path=run_path,
        transition_index=index + 1,
        ante=state.ante,
        blind=state.blind,
        action_type=action_type,
        hand_type="?",
        selected_cards=tuple(card.short_name for card in selected_cards),
        jokers=tuple(j.name for j in state.jokers),
        divergences=(),
        skip_reason=reason,
    )


def _shop_item_labels(state: GameState) -> tuple[str, ...]:
    items = _modifier_items(state.modifiers, "shop_cards")
    return tuple(_item_label(item) for item in items)


def _item_label(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("label") or item.get("name") or item.get("key") or "")
    return str(item)


def _modifier_items(modifiers: dict[str, Any], key: str) -> tuple[Any, ...]:
    raw = modifiers.get(key)
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return ()


def _is_full_reroll(pre_shop: tuple[str, ...], post_shop: tuple[str, ...]) -> bool:
    if not pre_shop or not post_shop:
        return False
    if len(pre_shop) != len(post_shop):
        return False
    return all(a != b for a, b in zip(pre_shop, post_shop))


def _find_sold_joker_index(
    pre: tuple[Joker, ...], post: tuple[Joker, ...]
) -> int | None:
    if len(post) >= len(pre):
        return None
    post_remaining = list(post)
    for index, joker in enumerate(pre):
        signature = (joker.name, joker.edition)
        match_index = next(
            (j for j, candidate in enumerate(post_remaining) if (candidate.name, candidate.edition) == signature),
            None,
        )
        if match_index is None:
            return index
        post_remaining.pop(match_index)
    return None


def _compare_states(predicted: GameState, actual: GameState) -> tuple[FieldDivergence, ...]:
    divergences: list[FieldDivergence] = []
    for field_name in COMPARED_TOP_FIELDS:
        pred_value = getattr(predicted, field_name)
        actual_value = getattr(actual, field_name)
        if pred_value != actual_value:
            divergences.append(FieldDivergence(field=field_name, predicted=pred_value, actual=actual_value))

    pred_identity = _joker_identity_signatures(predicted.jokers)
    actual_identity = _joker_identity_signatures(actual.jokers)
    if pred_identity != actual_identity:
        divergences.append(
            FieldDivergence(field="joker_set", predicted=pred_identity, actual=actual_identity)
        )
    else:
        pred_jokers = _joker_signatures(predicted.jokers)
        actual_jokers = _joker_signatures(actual.jokers)
        if pred_jokers != actual_jokers:
            divergences.append(FieldDivergence(field="joker_meta", predicted=pred_jokers, actual=actual_jokers))

    if predicted.hand_levels != actual.hand_levels:
        diff = _hand_level_diff(predicted.hand_levels, actual.hand_levels)
        if diff:
            divergences.append(FieldDivergence(field="hand_levels", predicted=diff[0], actual=diff[1]))

    pred_hand = _card_multiset(predicted.hand)
    actual_hand = _card_multiset(actual.hand)
    if pred_hand != actual_hand:
        divergences.append(FieldDivergence(field="hand_composition", predicted=pred_hand, actual=actual_hand))

    return tuple(divergences)


# Joker metadata keys that the sim tracks internally for scoring but that
# Balatro never exposes in its state dump. Comparing them produces false
# positives; the per-hand scoring audit already validates they are applied
# correctly. We compare only the keys the game actually publishes.
_SIM_INTERNAL_JOKER_META_KEYS = frozenset(
    {
        "current_mult",
        "current_chips",
        "current_int",
        "current_xmult",
        "current_xmult_visible",
        "remaining",
        "cards_remaining",
        "next_hands",
        "leading_plus",
        "current_plus",
        "text",
        "lower_text",
        "ready",
        "active",
        "earn_dollars",
        "target_suit",
        "discarded_suit",
        "target_rank",
        "discarded_rank",
    }
)


def _joker_signatures(jokers: tuple[Joker, ...]) -> tuple[tuple[str, str | None, dict[str, Any]], ...]:
    signatures: list[tuple[str, str | None, dict[str, Any]]] = []
    for joker in jokers:
        meta = {
            k: v
            for k, v in joker.metadata.items()
            if isinstance(v, (int, float, str, bool)) and k not in _SIM_INTERNAL_JOKER_META_KEYS
        }
        signatures.append((joker.name, joker.edition, meta))
    return tuple(sorted(signatures, key=lambda item: (item[0], item[1] or "")))


def _joker_identity_signatures(jokers: tuple[Joker, ...]) -> tuple[tuple[str, str | None], ...]:
    return tuple(sorted((joker.name, joker.edition) for joker in jokers))


def _hand_level_diff(predicted: dict[str, int], actual: dict[str, int]) -> tuple[dict[str, int], dict[str, int]] | None:
    keys = set(predicted) | set(actual)
    pred_diff: dict[str, int] = {}
    actual_diff: dict[str, int] = {}
    for key in keys:
        p = int(predicted.get(key, 0) or 0)
        a = int(actual.get(key, 0) or 0)
        if p != a:
            pred_diff[key] = p
            actual_diff[key] = a
    if not pred_diff:
        return None
    return pred_diff, actual_diff


def _card_multiset(cards: tuple[Card, ...]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((card.rank, card.suit, card.enhancement or "") for card in cards))


def _infer_selected_cards(
    before: tuple[Card, ...], after: tuple[Card, ...]
) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    after_ids = Counter(_card_id(card) for card in after if _card_id(card) is not None)
    selected: list[Card] = []
    held: list[Card] = []
    for card in before:
        card_id = _card_id(card)
        if card_id is not None and after_ids[card_id] > 0:
            after_ids[card_id] -= 1
            held.append(card)
        else:
            selected.append(card)
    return tuple(selected), tuple(held)


def _infer_drawn_cards(
    before: tuple[Card, ...], after: tuple[Card, ...], selected: tuple[Card, ...]
) -> tuple[Card, ...]:
    before_ids = Counter(_card_id(card) for card in before if _card_id(card) is not None)
    drawn: list[Card] = []
    for card in after:
        card_id = _card_id(card)
        if card_id is not None and before_ids[card_id] > 0:
            before_ids[card_id] -= 1
        else:
            drawn.append(card)
    return tuple(drawn)


def _indices_for_cards(hand: tuple[Card, ...], selected: tuple[Card, ...]) -> tuple[int, ...]:
    selected_ids = Counter(_card_id(card) for card in selected if _card_id(card) is not None)
    indices: list[int] = []
    for index, card in enumerate(hand):
        cid = _card_id(card)
        if cid is not None and selected_ids[cid] > 0:
            indices.append(index)
            selected_ids[cid] -= 1
    return tuple(indices)


def _card_id(card: Card) -> object:
    return card.metadata.get("id")


def _card_enhancement(card: Card) -> str:
    return str(card.enhancement or "").upper()


def _infer_hand_type_label(state: GameState) -> str:
    last = state.modifiers.get("last_played_hand")
    if isinstance(last, str):
        return last
    return "?"


def _load_jsonl(path: Path) -> tuple[dict, ...]:
    with path.open("r", encoding="utf-8") as file:
        return tuple(json.loads(line) for line in file if line.strip())


def _expand_run_paths(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            continue
        candidates = (
            (resolved,)
            if (resolved / "gamestates.jsonl").exists()
            else tuple(parent for parent in resolved.rglob("gamestates.jsonl"))
        )
        for candidate in candidates:
            run_path = candidate.parent if candidate.name == "gamestates.jsonl" else candidate
            if run_path not in seen:
                seen.add(run_path)
                yield run_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit forward_sim post-play state evolution against BalatroBench gamestates."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Run directories or parents containing gamestates.jsonl.")
    parser.add_argument("--worst", type=int, default=12, help="Number of diverged samples to print.")
    parser.add_argument("--jsonl-out", type=Path, help="Optional path for per-record JSONL output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_runs(args.paths)
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_out.write_text(
            "\n".join(json.dumps(record.to_json()) for record in audit.records)
            + ("\n" if audit.records else ""),
            encoding="utf-8",
        )
    print(audit.to_text(worst_count=max(0, args.worst)))
    return 0 if not audit.diverged else 1


if __name__ == "__main__":
    raise SystemExit(main())
