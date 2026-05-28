"""Capture per-card Spectral RNG helper fixtures from a live bridge.

Familiar, Grim, and Incantation destroy one random hand card, then create a
small batch of enhanced playing cards using card-specific rank/suit RNG and
the shared ``spe_card`` enhancement stream. These fixtures validate that helper
surface directly without depending on a natural Spectral Pack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from balatro_ai.api.client import BalatroBridgeError, JsonRpcBalatroClient
from balatro_ai.rng.capture import DEFAULT_FIXTURE_DIR
from balatro_ai.rng.capture_shop import _call_with_retries
from balatro_ai.rng.capture_shop_sequence import _settled_raw_state
from balatro_ai.rng.capture_surfaces import extract_area_cards


SPECTRAL_HELPER_KEYS = ("c_familiar", "c_grim", "c_incantation")

BASE_HAND = (
    {"key": "H_2"},
    {"key": "D_3"},
    {"key": "C_4"},
    {"key": "S_5"},
    {"key": "H_6"},
    {"key": "D_7"},
    {"key": "C_8"},
    {"key": "S_9"},
)


def spectral_helper_fixture_path(
    seed: str,
    spectral_key: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    root: Path = DEFAULT_FIXTURE_DIR,
) -> Path:
    return root / f"spectral_seed_{seed}_{spectral_key}_{deck.lower()}_{stake.lower()}.json"


def capture_spectral_helper(
    client: JsonRpcBalatroClient,
    seed: str,
    spectral_key: str,
    *,
    stake: str = "white",
    deck: str = "RED",
) -> dict[str, Any]:
    if spectral_key not in SPECTRAL_HELPER_KEYS:
        raise ValueError(f"Unsupported spectral helper key: {spectral_key}")

    client.deck = deck
    _bridge_state(client, "menu", None)
    state = _bridge_state(client, "start", {"deck": deck.upper(), "stake": stake.upper(), "seed": seed})
    state = _settled_raw_state(client, state)
    if _state_phase(state) == "BLIND_SELECT":
        state = _settled_raw_state(client, _bridge_state(client, "select", None))
    if _state_phase(state) != "SELECTING_HAND":
        raise RuntimeError(f"Expected SELECTING_HAND after selecting blind, got {_state_phase(state)}")

    before_state = _settled_raw_state(
        client,
        _bridge_state(
            client,
            "scenario",
            {
                "clear_hand": True,
                "clear_consumables": True,
                "hand": list(BASE_HAND),
                "consumables": [{"key": spectral_key}],
            },
        ),
        polls=12,
    )
    after_state = _settled_raw_state(client, _bridge_state(client, "use", {"consumable": 0}), polls=16)
    if not _created_enhanced_cards(after_state):
        raise RuntimeError(f"Using {spectral_key} did not expose any created enhanced cards")

    return {
        "record_type": "rng_spectral_helper",
        "seed": seed,
        "stake": stake.lower(),
        "deck": deck.upper(),
        "spectral_key": spectral_key,
        "before_state": before_state,
        "after_state": after_state,
    }


def save_fixture(fixture: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")


def _bridge_state(client: JsonRpcBalatroClient, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    state = _call_with_retries(client, method, params)
    if not isinstance(state, dict):
        raise RuntimeError(f"Expected {method} to return a state dict, got {type(state).__name__}")
    return state


def _created_enhanced_cards(state: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(card for card in extract_area_cards(state, "hand") if _is_enhanced(card))


def _is_enhanced(card: Mapping[str, Any]) -> bool:
    if str(card.get("set", "")).upper() == "ENHANCED":
        return True
    modifier = card.get("modifier")
    return isinstance(modifier, Mapping) and bool(modifier.get("enhancement"))


def _state_phase(state: Mapping[str, Any]) -> str:
    return str(state.get("state", state.get("phase", ""))).upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture Spectral created-card RNG fixtures.")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--spectral-key", choices=SPECTRAL_HELPER_KEYS, action="append")
    parser.add_argument("--all-helpers", action="store_true")
    parser.add_argument("--stake", default="white")
    parser.add_argument("--deck", default="RED")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spectral_keys = SPECTRAL_HELPER_KEYS if args.all_helpers else tuple(args.spectral_key or ())
    if not spectral_keys:
        print("Specify --spectral-key KEY or --all-helpers.")
        return 2

    client = JsonRpcBalatroClient(endpoint=args.endpoint, timeout_seconds=args.timeout)
    failures: list[str] = []
    for spectral_key in spectral_keys:
        try:
            fixture = capture_spectral_helper(
                client,
                args.seed,
                spectral_key,
                stake=args.stake,
                deck=args.deck,
            )
        except (BalatroBridgeError, ConnectionError, RuntimeError, ValueError) as exc:
            print(f"FAIL  seed={args.seed} spectral={spectral_key}: {exc}")
            failures.append(spectral_key)
            continue
        path = spectral_helper_fixture_path(
            args.seed,
            spectral_key,
            stake=args.stake,
            deck=args.deck,
            root=args.out_dir,
        )
        save_fixture(fixture, path)
        print(f"OK    seed={args.seed} spectral={spectral_key} -> {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
