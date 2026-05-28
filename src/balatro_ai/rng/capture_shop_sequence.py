"""Capture a no-purchase sequence of shop surfaces from a live bridge.

The first-shop fixture verifies the run-opening surface. This module captures
multiple shops in one run while deliberately ending shops without buying
anything. Blinds are cleared through the dev ``scenario`` endpoint by setting
the current score just below the blind requirement, then playing up to five
cards. That keeps the capture focused on shop RNG advancement rather than bot
skill.

Usage:
    python -m balatro_ai.rng.capture_shop_sequence --seed AAAAAAA --shops 4
    python -m balatro_ai.rng.capture_shop_sequence --all --shops 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from balatro_ai.api.client import BalatroBridgeError, JsonRpcBalatroClient
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.rng.capture import CANONICAL_SEEDS, DEFAULT_FIXTURE_DIR
from balatro_ai.rng.capture_shop import _call_with_retries
from balatro_ai.rng.capture_surfaces import _edition_rate_for_vouchers, extract_area_cards


def shop_sequence_fixture_path(
    seed: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    vouchers: Iterable[str] = (),
    root: Path = DEFAULT_FIXTURE_DIR,
) -> Path:
    suffix = _voucher_suffix(vouchers)
    voucher_part = "_with_" + suffix if suffix else ""
    return root / f"shop_sequence_seed_{seed}{voucher_part}_{deck.lower()}_{stake.lower()}.json"


def capture_shop_sequence(
    client: JsonRpcBalatroClient,
    seed: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    shops: int = 4,
    used_vouchers: Iterable[str] = (),
) -> dict[str, Any]:
    """Capture ``shops`` shop states from one seed without purchases."""

    if shops <= 0:
        raise ValueError("shops must be positive")

    client.deck = deck
    voucher_keys = tuple(dict.fromkeys(str(key) for key in used_vouchers))
    _bridge_state(client, "menu", None)
    state = _bridge_state(client, "start", {"deck": deck.upper(), "stake": stake.upper(), "seed": seed})

    captured: list[dict[str, Any]] = []
    while len(captured) < shops:
        state = advance_to_next_shop(client, state, used_vouchers=voucher_keys)
        captured.append(
            {
                "shop_index": len(captured),
                "ante": int(state.get("ante_num", state.get("ante", 1))),
                "round_num": int(state.get("round_num", 0) or 0),
                "blind_on_deck": _blind_on_deck(state),
                "state": state,
            }
        )
        if len(captured) < shops:
            state = _settled_raw_state(client, _bridge_state(client, "next_round", None))

    return {
        "record_type": "rng_shop_sequence",
        "seed": seed,
        "stake": stake.lower(),
        "deck": deck.upper(),
        "vouchers": list(voucher_keys),
        "edition_rate": _edition_rate_for_vouchers(voucher_keys),
        "voucher_effective_shop_index": 0,
        "shops": captured,
    }


def advance_to_next_shop(
    client: JsonRpcBalatroClient,
    state: dict[str, Any],
    *,
    used_vouchers: Iterable[str] = (),
) -> dict[str, Any]:
    """Advance from any active run state to the next SHOP state."""

    state = _settled_raw_state(client, state)
    for _ in range(80):
        phase = _state_phase(state)
        if phase == "SHOP":
            return state
        if phase == "GAME_OVER":
            raise RuntimeError("Run ended before the requested shop was reached")
        if phase == "BLIND_SELECT":
            state = _settled_raw_state(client, _bridge_state(client, "select", None))
            continue
        if phase in {"SELECTING_HAND", "PLAYING_BLIND"}:
            state = _clear_current_blind(client, state, used_vouchers=used_vouchers)
            continue
        if phase == "ROUND_EVAL":
            state = _settled_raw_state(client, _bridge_state(client, "cash_out", None))
            continue
        state = _settled_raw_state(client, _bridge_state(client, "gamestate", None))
    raise RuntimeError(f"Stuck advancing to shop at phase {_state_phase(state)}")


def save_fixture(fixture: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")


def _clear_current_blind(
    client: JsonRpcBalatroClient,
    state: dict[str, Any],
    *,
    used_vouchers: Iterable[str] = (),
) -> dict[str, Any]:
    parsed = GameState.from_mapping(state)
    required = max(1, parsed.required_score)
    scenario_params: dict[str, Any] = {
        "chips": required - 1,
        "hands": max(1, parsed.hands_remaining),
        "discards": max(0, parsed.discards_remaining),
    }
    voucher_keys = tuple(dict.fromkeys(str(key) for key in used_vouchers))
    if voucher_keys:
        scenario_params["used_vouchers"] = [{"key": key} for key in voucher_keys]
    scenario_state = _settled_raw_state(
        client,
        _bridge_state(client, "scenario", scenario_params),
    )
    hand_count = len(extract_area_cards(scenario_state, "hand"))
    if hand_count <= 0:
        raise RuntimeError("Cannot clear blind because the bridge exposed no hand cards")
    card_indexes = list(range(min(5, hand_count)))
    return _settled_raw_state(client, _bridge_state(client, "play", {"cards": card_indexes}))


def _bridge_state(client: JsonRpcBalatroClient, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    state = _call_with_retries(client, method, params)
    if not isinstance(state, dict):
        raise RuntimeError(f"Expected {method} to return a state dict, got {type(state).__name__}")
    return state


def _settled_raw_state(
    client: JsonRpcBalatroClient,
    state: dict[str, Any] | None = None,
    *,
    polls: int = 8,
    delay_seconds: float = 0.05,
) -> dict[str, Any]:
    latest = state if state is not None else _bridge_state(client, "gamestate", None)
    latest_signature = _state_signature(latest)
    stable_polls = 0
    for _ in range(polls):
        time.sleep(delay_seconds)
        current = _bridge_state(client, "gamestate", None)
        current_signature = _state_signature(current)
        if current_signature == latest_signature:
            stable_polls += 1
            if stable_polls >= 2:
                return current
        else:
            latest = current
            latest_signature = current_signature
            stable_polls = 0
    return latest


def _state_signature(state: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _state_phase(state),
        state.get("ante_num", state.get("ante")),
        state.get("money"),
        state.get("round_num"),
        _area_keys(state, "hand"),
        _area_keys(state, "shop"),
        _area_keys(state, "packs"),
        _area_keys(state, "vouchers"),
    )


def _area_keys(state: Mapping[str, Any], area_name: str) -> tuple[str, ...]:
    return tuple(str(card.get("key", "")) for card in extract_area_cards(state, area_name))


def _state_phase(state: Mapping[str, Any]) -> str:
    raw = str(state.get("state", state.get("phase", ""))).upper()
    if raw == GamePhase.SHOP.value.upper():
        return "SHOP"
    return raw


def _blind_on_deck(state: Mapping[str, Any]) -> str:
    round_payload = state.get("round")
    if isinstance(round_payload, Mapping):
        return str(round_payload.get("blind_on_deck", ""))
    return ""


def _voucher_suffix(vouchers: Iterable[str]) -> str:
    return "_".join(_slug(voucher) for voucher in sorted(str(voucher) for voucher in vouchers))


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.lower())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture no-purchase shop sequence RNG fixtures.")
    parser.add_argument("--seed", help="Run seed string. Required unless --all is given.")
    parser.add_argument("--all", action="store_true", help="Capture all canonical seeds.")
    parser.add_argument("--shops", type=int, default=4, help="Number of shop states to capture per seed.")
    parser.add_argument("--stake", default="white")
    parser.add_argument("--deck", default="RED")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument(
        "--used-voucher",
        action="append",
        default=[],
        help="Mark this voucher key as already owned before shop generation; repeatable.",
    )
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
            fixture = capture_shop_sequence(
                client,
                seed,
                stake=args.stake,
                deck=args.deck,
                shops=args.shops,
                used_vouchers=tuple(args.used_voucher),
            )
        except (BalatroBridgeError, ConnectionError, RuntimeError, ValueError) as exc:
            print(f"FAIL  seed={seed}: {exc}")
            failures.append(seed)
            continue
        path = shop_sequence_fixture_path(
            seed,
            stake=args.stake,
            deck=args.deck,
            vouchers=tuple(args.used_voucher),
            root=args.out_dir,
        )
        save_fixture(fixture, path)
        print(f"OK    seed={seed} -> {path} ({len(fixture['shops'])} shops)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
