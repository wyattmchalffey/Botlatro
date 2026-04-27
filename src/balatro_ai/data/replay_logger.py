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
            "legal_actions": [action.to_json() for action in legal_actions],
            "chosen_action": chosen_action.to_json(),
            "reward": reward,
            "score": state.current_score,
            "money": state.money,
            "outcome": outcome,
            "seed": state.seed,
            "extra": extra or {},
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

