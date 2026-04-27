"""Action models used by bots and environment wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ActionType(StrEnum):
    """High-level Balatro action categories.

    The local API bridge may expose more specific action names. These categories
    are intentionally broad so the Python side can stay stable while the bridge
    adapter translates to exact RPC payloads.
    """

    PLAY_HAND = "play_hand"
    DISCARD = "discard"
    SELECT_BLIND = "select_blind"
    SKIP_BLIND = "skip_blind"
    BUY = "buy"
    SELL = "sell"
    USE_CONSUMABLE = "use_consumable"
    OPEN_PACK = "open_pack"
    CHOOSE_PACK_CARD = "choose_pack_card"
    REROLL = "reroll"
    END_SHOP = "end_shop"
    CASH_OUT = "cash_out"
    NO_OP = "no_op"


@dataclass(frozen=True, slots=True)
class Action:
    """A legal action candidate for the current state."""

    action_type: ActionType
    card_indices: tuple[int, ...] = ()
    target_id: str | None = None
    amount: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stable_key(self) -> str:
        """Return a deterministic key suitable for masks and replay logs."""

        cards = ",".join(str(index) for index in self.card_indices)
        target = self.target_id or ""
        amount = "" if self.amount is None else str(self.amount)
        return f"{self.action_type.value}|{cards}|{target}|{amount}"

    def to_json(self) -> dict[str, Any]:
        """Serialize the action into a JSON-friendly payload."""

        payload: dict[str, Any] = {
            "type": self.action_type.value,
            "card_indices": list(self.card_indices),
        }
        if self.target_id is not None:
            payload["target_id"] = self.target_id
        if self.amount is not None:
            payload["amount"] = self.amount
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Action":
        """Build an action from a bridge or replay mapping."""

        return cls(
            action_type=ActionType(data["type"]),
            card_indices=tuple(int(index) for index in data.get("card_indices", ())),
            target_id=data.get("target_id"),
            amount=data.get("amount"),
            metadata=dict(data.get("metadata", {})),
        )


def actions_from_mappings(items: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[Action, ...]:
    """Convert JSON-style action payloads into immutable actions."""

    return tuple(Action.from_mapping(item) for item in items)

