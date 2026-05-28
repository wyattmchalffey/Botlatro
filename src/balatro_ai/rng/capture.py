"""Capture ground-truth RNG-affected state from a running Balatro bridge.

The Python ``BalatroRNG`` implementation in this package is a port of Balatro's
documented Lua primitives, but verifying that it actually matches the live
game requires real game data. This module connects to the local bridge,
starts a fresh run with a known seed, and saves the initial observable state
to a JSON fixture under `.data/rng-validation/`.

Once captured, fixtures are checked by `validate.py` (and the tests in
`tests/test_rng_against_bridge.py`) without needing the bridge to be running
again — capture once, iterate on the math offline.

Usage:
    # With Balatro + BalatroBot running on the default endpoint:
    python -m balatro_ai.rng.capture --seed AAAAAAA --stake white

    # To re-capture all canonical seeds:
    python -m balatro_ai.rng.capture --all
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from balatro_ai.api.client import BalatroBridgeError, JsonRpcBalatroClient


# Canonical seeds we want fixtures for. Add more here as the validation
# surface grows (different decks/stakes will need their own captures).
CANONICAL_SEEDS = (
    "AAAAAAA",
    "BBBBBBB",
    "CCCCCCC",
    "1234567",
)

DEFAULT_FIXTURE_DIR = Path(".data/rng-validation")


@dataclass(slots=True)
class CaptureRequest:
    seed: str
    stake: str = "white"
    deck: str = "RED"


def capture_initial_state(
    client: JsonRpcBalatroClient,
    request: CaptureRequest,
) -> dict[str, Any]:
    """Start a fresh run with ``request.seed`` and return the raw state dict.

    Returns the JSON payload the bridge sent back (not a parsed GameState) so
    we preserve every field — including ones our GameState dataclass drops on
    the floor — for fixture inspection.
    """

    client.deck = request.deck
    client.call("menu")
    params = {
        "deck": request.deck.upper(),
        "stake": request.stake.upper(),
        "seed": request.seed,
    }
    raw = client.call(client.start_method, params)
    if not isinstance(raw, dict):
        raise BalatroBridgeError({"message": f"Expected dict state, got {type(raw).__name__}"})
    return raw


def fixture_path(seed: str, *, stake: str = "white", deck: str = "RED", root: Path = DEFAULT_FIXTURE_DIR) -> Path:
    return root / f"seed_{seed}_{deck.lower()}_{stake.lower()}.json"


def save_fixture(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture initial Balatro state for RNG validation.")
    parser.add_argument("--seed", help="Run seed string (e.g., AAAAAAA). Required unless --all is given.")
    parser.add_argument("--stake", default="white", help="Stake (white/red/.../gold). Default: white.")
    parser.add_argument("--deck", default="RED", help="Deck (RED/BLUE/.../ABANDONED). Default: RED.")
    parser.add_argument("--all", action="store_true", help="Capture all canonical seeds.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346", help="Balatro bridge endpoint.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-call timeout in seconds.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIXTURE_DIR, help="Where to write fixtures.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.all and not args.seed:
        print("Specify --seed SEED or --all.", flush=True)
        return 2

    client = JsonRpcBalatroClient(endpoint=args.endpoint, timeout_seconds=args.timeout)

    seeds: tuple[str, ...] = CANONICAL_SEEDS if args.all else (args.seed,)
    failures: list[str] = []
    for seed in seeds:
        request = CaptureRequest(seed=seed, stake=args.stake, deck=args.deck)
        try:
            state = capture_initial_state(client, request)
        except (BalatroBridgeError, ConnectionError) as exc:
            print(f"FAIL  seed={seed}: {exc}", flush=True)
            failures.append(seed)
            continue
        path = fixture_path(seed, stake=args.stake, deck=args.deck, root=args.out_dir)
        save_fixture(state, path)
        hand_payload = state.get("hand", {})
        hand_cards = hand_payload.get("cards", []) if isinstance(hand_payload, dict) else []
        deck_payload = state.get("deck", {})
        deck_cards = deck_payload.get("cards", []) if isinstance(deck_payload, dict) else []
        print(
            f"OK    seed={seed} -> {path} (hand={len(hand_cards)} deck_visible={len(deck_cards)})",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
