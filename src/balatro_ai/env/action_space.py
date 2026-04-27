"""Action-space utilities."""

from __future__ import annotations

from dataclasses import dataclass

from balatro_ai.api.actions import Action
from balatro_ai.api.state import GameState


@dataclass(frozen=True, slots=True)
class ActionCatalog:
    """Stable catalog used to turn legal actions into masks."""

    actions: tuple[Action, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(action.stable_key for action in self.actions)

    def legal_action_mask(self, state: GameState) -> tuple[bool, ...]:
        legal_keys = {action.stable_key for action in state.legal_actions}
        return tuple(key in legal_keys for key in self.keys)

    def index_of(self, action: Action) -> int:
        return self.keys.index(action.stable_key)

    @classmethod
    def from_legal_actions(cls, actions: tuple[Action, ...]) -> "ActionCatalog":
        return cls(actions=tuple(actions))

