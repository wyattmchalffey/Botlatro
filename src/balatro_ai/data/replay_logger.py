"""JSONL replay logging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from balatro_ai.api.actions import Action
from balatro_ai.api.state import GameState


@dataclass(slots=True)
class ReplayLogger:
    path: Path

    def log_step(
        self,
        *,
        state: GameState,
        legal_actions: tuple[Action, ...],
        chosen_action: Action,
        reward: float,
        outcome: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "state": state.debug_summary,
            "state_detail": _state_detail(state),
            "legal_actions": [action.to_json() for action in legal_actions],
            "chosen_action": chosen_action.to_json(),
            "chosen_item": _chosen_item_detail(state, chosen_action),
            "reward": reward,
            "score": state.current_score,
            "money": state.money,
            "outcome": outcome,
            "seed": state.seed,
            "extra": extra or {},
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def log_summary(
        self,
        *,
        bot_version: str,
        seed: int,
        stake: str,
        won: bool,
        ante_reached: int,
        final_score: int,
        final_money: int,
        runtime_seconds: float,
        death_reason: str | None = None,
        final_state: GameState | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        outcome = "win" if won else "loss"
        if death_reason and death_reason.startswith("error:"):
            outcome = "error"
        record = {
            "record_type": "run_summary",
            "bot_version": bot_version,
            "seed": seed,
            "stake": stake,
            "won": won,
            "outcome": outcome,
            "ante": ante_reached,
            "final_score": final_score,
            "final_money": final_money,
            "runtime_seconds": runtime_seconds,
            "death_reason": death_reason,
        }
        if final_state is not None:
            record["final_state"] = final_state.debug_summary
            record["final_state_detail"] = _state_detail(final_state)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _state_detail(state: GameState) -> dict[str, object]:
    return {
        "phase": state.phase.value,
        "ante": state.ante,
        "blind": state.blind,
        "required_score": state.required_score,
        "current_blind": state.modifiers.get("current_blind"),
        "current_score": state.current_score,
        "hands_remaining": state.hands_remaining,
        "discards_remaining": state.discards_remaining,
        "money": state.money,
        "deck_size": state.deck_size,
        "hand": [_card_detail(card) for card in state.hand],
        "known_deck": [_card_detail(card) for card in state.known_deck],
        "hand_levels": dict(state.hand_levels),
        "hands": dict(state.modifiers.get("hands", {})) if isinstance(state.modifiers.get("hands"), dict) else {},
        "jokers": [_joker_detail(joker) for joker in state.jokers],
        "consumables": list(state.consumables),
        "owned_vouchers": list(state.vouchers),
        "vouchers": list(state.vouchers),
        "shop": [_raw_card_detail(item) for item in state.modifiers.get("shop_cards", ())],
        "voucher_shop": [_raw_card_detail(item) for item in state.modifiers.get("voucher_cards", ())],
        "booster_packs": [_raw_card_detail(item) for item in state.modifiers.get("booster_packs", ())],
        "pack": [_raw_card_detail(item) for item in state.modifiers.get("pack_cards", ())],
    }


def _chosen_item_detail(state: GameState, action: Action) -> dict[str, object] | None:
    kind = str(action.metadata.get("kind") or action.target_id or "")
    index = action.metadata.get("index", action.amount)
    if kind == "skip" or index is None:
        return None
    try:
        item_index = int(index)
    except (TypeError, ValueError):
        return None

    if action.action_type.value == "sell":
        return _indexed_detail(tuple(_joker_detail(joker) for joker in state.jokers), item_index)
    if action.action_type.value == "choose_pack_card":
        return _indexed_detail(tuple(_raw_card_detail(item) for item in state.modifiers.get("pack_cards", ())), item_index)
    if action.action_type.value == "open_pack":
        return _indexed_detail(
            tuple(_raw_card_detail(item) for item in state.modifiers.get("booster_packs", ())),
            item_index,
        )
    if action.action_type.value != "buy":
        return None
    if kind == "voucher":
        return _indexed_detail(
            tuple(_raw_card_detail(item) for item in state.modifiers.get("voucher_cards", ())),
            item_index,
        )
    return _indexed_detail(tuple(_raw_card_detail(item) for item in state.modifiers.get("shop_cards", ())), item_index)


def _indexed_detail(items: tuple[dict[str, object], ...], index: int) -> dict[str, object] | None:
    if 0 <= index < len(items):
        return items[index]
    return None


def _joker_detail(joker) -> dict[str, object]:
    return {
        "name": joker.name,
        "edition": joker.edition,
        "sell_value": joker.sell_value,
        "set": joker.metadata.get("set"),
        "rarity": joker.metadata.get("rarity"),
        "cost": joker.metadata.get("cost"),
        "metadata": dict(joker.metadata),
    }


def _card_detail(card) -> dict[str, object]:
    return {
        "rank": card.rank,
        "suit": card.suit,
        "name": card.short_name,
        "enhancement": card.enhancement,
        "seal": card.seal,
        "edition": card.edition,
        "debuffed": card.debuffed,
    }


def _raw_card_detail(item: object) -> dict[str, object]:
    if isinstance(item, str):
        return {"name": item}
    if not isinstance(item, dict):
        return {"name": str(item)}
    cost = item.get("cost") if isinstance(item.get("cost"), dict) else {}
    modifier = item.get("modifier") if isinstance(item.get("modifier"), dict) else {}
    return {
        "name": str(item.get("label", item.get("name", item.get("key", "unknown")))),
        "key": item.get("key"),
        "set": item.get("set"),
        "cost": cost,
        "edition": item.get("edition", modifier.get("edition")),
        "rarity": item.get("rarity"),
        "rarity_name": item.get("rarity_name"),
    }
