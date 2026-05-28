"""Capture initial-shop state from the bridge.

After ``start_run``, the game is in BLIND_SELECT. To reach the first SHOP we
need to: select the small blind, play through it, and cash out. That sequence
deterministically lands the game in SHOP with the first shop pool generated.
The captured state's ``shop`` field has the joker/planet/tarot/etc. cards
that the pool generation produced.

Usage:
    python -m balatro_ai.rng.capture_shop --seed AAAAAAA
    python -m balatro_ai.rng.capture_shop --all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from balatro_ai.api.client import BalatroBridgeError, JsonRpcBalatroClient
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.rng.capture import CANONICAL_SEEDS, DEFAULT_FIXTURE_DIR


def shop_fixture_path(seed: str, *, stake: str = "white", deck: str = "RED", root: Path = DEFAULT_FIXTURE_DIR) -> Path:
    return root / f"shop_seed_{seed}_{deck.lower()}_{stake.lower()}.json"


def _state_phase(state: dict[str, Any]) -> str:
    return str(state.get("state", "")).upper()


def _call_with_retries(client: JsonRpcBalatroClient, method: str, params: dict | None, *, retries: int = 3) -> dict:
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            return client.call(method, params)
        except (BalatroBridgeError, ConnectionError) as exc:
            last_exc = exc
            time.sleep(0.5)
    raise RuntimeError(f"{method} failed after {retries} retries: {last_exc}")


def advance_to_first_shop(client: JsonRpcBalatroClient, seed: str, *, stake: str = "white", deck: str = "RED") -> dict[str, Any]:
    """Start the run, play the small blind via basic_strategy_bot, return SHOP state."""

    client.deck = deck
    client.call("menu")
    state_dict = _call_with_retries(
        client,
        "start",
        {"deck": deck.upper(), "stake": stake.upper(), "seed": seed},
    )

    bot = BasicStrategyBot(seed=0)
    safety = 0
    while _state_phase(state_dict) != "SHOP":
        safety += 1
        if safety > 60:
            raise RuntimeError(f"Stuck advancing to SHOP at phase {_state_phase(state_dict)}")
        if _state_phase(state_dict) == "GAME_OVER":
            raise RuntimeError("Bot died before reaching SHOP")
        state = GameState.from_mapping(state_dict)
        if state.phase == GamePhase.UNKNOWN:
            # Bridge sometimes returns transient states; small wait + re-poll.
            time.sleep(0.05)
            state_dict = _call_with_retries(client, "gamestate", None)
            continue
        action = bot.choose_action(state)
        try:
            state_dict = client.send_action_dict(action) if hasattr(client, "send_action_dict") else None
        except Exception:
            state_dict = None
        if state_dict is None:
            # Fall back to manual translation: replicate JsonRpcBalatroClient._action_to_rpc.
            method, params = _action_to_rpc(action)
            state_dict = _call_with_retries(client, method, params)
    return state_dict


def _action_to_rpc(action) -> tuple[str, dict | None]:
    """Mirror of JsonRpcBalatroClient._action_to_rpc but returning the raw dict."""

    from balatro_ai.api.actions import ActionType
    if action.action_type == ActionType.PLAY_HAND:
        return "play", {"cards": list(action.card_indices)}
    if action.action_type == ActionType.DISCARD:
        return "discard", {"cards": list(action.card_indices)}
    if action.action_type == ActionType.SELECT_BLIND:
        return "select", None
    if action.action_type == ActionType.SKIP_BLIND:
        return "skip", None
    if action.action_type == ActionType.CASH_OUT:
        return "cash_out", None
    if action.action_type == ActionType.END_SHOP:
        return "next_round", None
    if action.action_type == ActionType.NO_OP:
        return "gamestate", None
    raise RuntimeError(f"Unhandled action type during shop capture: {action.action_type}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture first-shop state from a Balatro run.")
    parser.add_argument("--seed", help="Run seed string. Required unless --all is given.")
    parser.add_argument("--all", action="store_true", help="Capture all canonical seeds.")
    parser.add_argument("--stake", default="white")
    parser.add_argument("--deck", default="RED")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.all and not args.seed:
        print("Specify --seed SEED or --all.")
        return 2

    client = JsonRpcBalatroClient(endpoint=args.endpoint, timeout_seconds=args.timeout)
    seeds = CANONICAL_SEEDS if args.all else (args.seed,)
    failures: list[str] = []
    for seed in seeds:
        try:
            state = advance_to_first_shop(client, seed, stake=args.stake, deck=args.deck)
        except (BalatroBridgeError, ConnectionError, RuntimeError) as exc:
            print(f"FAIL  seed={seed}: {exc}")
            failures.append(seed)
            continue
        path = shop_fixture_path(seed, stake=args.stake, deck=args.deck, root=args.out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        shop_payload = state.get("shop", {})
        shop_cards = shop_payload.get("cards", []) if isinstance(shop_payload, dict) else []
        print(f"OK    seed={seed} -> {path} (shop={len(shop_cards)} cards)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
