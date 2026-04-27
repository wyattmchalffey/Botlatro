"""Client interfaces for talking to a local Balatro bridge."""

from __future__ import annotations

import json
from http.client import HTTPException
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from balatro_ai.api.actions import Action
from balatro_ai.api.state import GameState


class BalatroBridgeError(RuntimeError):
    """Raised when the JSON-RPC bridge returns an error payload."""

    def __init__(self, error: dict[str, Any]) -> None:
        self.error = error
        data = error.get("data") if isinstance(error.get("data"), dict) else {}
        self.code = error.get("code")
        self.name = data.get("name")
        self.message = str(error.get("message", "Balatro bridge error"))
        super().__init__(f"Balatro bridge error: {error}")


class BalatroClient(Protocol):
    """Minimal client contract expected by `BalatroEnv`."""

    def start_run(self, seed: int | None = None, stake: str | None = None) -> GameState:
        """Start a new run and return the initial state."""

    def get_state(self) -> GameState:
        """Return the current game state."""

    def send_action(self, action: Action) -> GameState:
        """Apply an action and return the resulting state."""


@dataclass(slots=True)
class JsonRpcBalatroClient:
    """Small JSON-RPC client for a local Balatro API bridge.

    The method names are configurable because local Balatro bridges may differ.
    Once the exact BalatroBot payloads are confirmed, this adapter should be the
    only layer that needs bridge-specific translation.
    """

    endpoint: str = "http://127.0.0.1:12346"
    deck: str = "RED"
    start_method: str = "start"
    state_method: str = "gamestate"
    timeout_seconds: float = 10.0

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, HTTPException, OSError, TimeoutError) as exc:
            raise ConnectionError(f"Could not reach Balatro bridge at {self.endpoint}") from exc

        if "error" in response_data and response_data["error"]:
            raise BalatroBridgeError(response_data["error"])
        return response_data.get("result")

    def start_run(self, seed: int | None = None, stake: str | None = None) -> GameState:
        self.call("menu")
        params: dict[str, Any] = {"deck": self.deck.upper(), "stake": (stake or "white").upper()}
        if seed is not None:
            params["seed"] = str(seed)
        return GameState.from_mapping(self.call(self.start_method, params))

    def get_state(self) -> GameState:
        return GameState.from_mapping(self.call(self.state_method))

    def send_action(self, action: Action) -> GameState:
        method, params = self._action_to_rpc(action)
        return GameState.from_mapping(self.call(method, params))

    def _action_to_rpc(self, action: Action) -> tuple[str, dict[str, Any] | None]:
        if action.action_type.value == "play_hand":
            return "play", {"cards": list(action.card_indices)}
        if action.action_type.value == "discard":
            return "discard", {"cards": list(action.card_indices)}
        if action.action_type.value == "select_blind":
            return "select", None
        if action.action_type.value == "skip_blind":
            return "skip", None
        if action.action_type.value == "reroll":
            return "reroll", None
        if action.action_type.value == "end_shop":
            return "next_round", None
        if action.action_type.value == "cash_out":
            return "cash_out", None
        if action.action_type.value == "choose_pack_card":
            return "pack", _indexed_params(action, default_kind="card")
        if action.action_type.value == "open_pack":
            return "buy", _indexed_params(action, default_kind="pack")
        if action.action_type.value == "buy":
            return "buy", _indexed_params(action, default_kind="card")
        if action.action_type.value == "sell":
            return "sell", _indexed_params(action, default_kind="joker")
        if action.action_type.value == "use_consumable":
            params = _indexed_params(action, default_kind="consumable")
            if action.card_indices:
                params["cards"] = list(action.card_indices)
            return "use", params
        if action.action_type.value == "no_op":
            return self.state_method, None
        raise ValueError(f"Unsupported action for JSON-RPC bridge: {action.action_type.value}")


def _indexed_params(action: Action, default_kind: str) -> dict[str, Any]:
    kind = str(action.metadata.get("kind") or action.target_id or default_kind)
    if kind == "skip":
        return {"skip": True}
    index = action.metadata.get("index", action.amount)
    if index is None:
        raise ValueError(f"Action {action.stable_key} needs an index for {kind}")
    return {kind: int(index)}
