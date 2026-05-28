"""Capture a sequence of shop rerolls from a live bridge.

Rerolling the shop re-rolls the rerollable card slots (not the voucher or
booster packs), advancing the same per-ante shop-card RNG streams. To validate
per-reroll RNG advancement we need ground truth: the initial shop, then the
shop after each reroll, captured without buying anything.

The flow reaches the first shop, injects money via the dev ``scenario``
endpoint so rerolls are affordable, then calls ``reroll`` ``--rerolls`` times,
capturing the shop card slots after each.

Usage:
    python -m balatro_ai.rng.capture_reroll --seed AAAAAAA --rerolls 4
    python -m balatro_ai.rng.capture_reroll --all --rerolls 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from balatro_ai.api.client import BalatroBridgeError, JsonRpcBalatroClient
from balatro_ai.rng.capture import CANONICAL_SEEDS, DEFAULT_FIXTURE_DIR
from balatro_ai.rng.capture_shop import advance_to_first_shop
from balatro_ai.rng.capture_surfaces import (
    _bridge_state,
    _settled_raw_state,
    _state_phase,
    extract_area_cards,
    save_fixture,
)


def reroll_fixture_path(
    seed: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    root: Path = DEFAULT_FIXTURE_DIR,
) -> Path:
    return root / f"shop_reroll_seed_{seed}_{deck.lower()}_{stake.lower()}.json"


def capture_shop_rerolls(
    client: JsonRpcBalatroClient,
    seed: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    rerolls: int = 4,
    money: int = 1000,
) -> dict[str, Any]:
    """Capture the initial shop plus ``rerolls`` rerolls without purchases."""

    if rerolls < 0:
        raise ValueError("rerolls must be non-negative")

    shop_state = advance_to_first_shop(client, seed, stake=stake, deck=deck)
    if _state_phase(shop_state) != "SHOP":
        raise RuntimeError(f"Expected SHOP after advance, got {_state_phase(shop_state)}")

    captured: list[dict[str, Any]] = [{"reroll_index": 0, "state": shop_state}]

    # Fund rerolls. Money-only scenario does not regenerate the shop, so the
    # reroll_index=0 slots above stay pristine.
    funded = _settled_raw_state(client, _bridge_state(client, "scenario", {"money": money}))
    if _shop_keys(funded) != _shop_keys(shop_state):
        raise RuntimeError("scenario money injection unexpectedly changed the shop")

    state = funded
    for index in range(1, rerolls + 1):
        before = _shop_keys(state)
        state = _settled_raw_state(client, _bridge_state(client, "reroll", None))
        if _state_phase(state) != "SHOP":
            raise RuntimeError(f"Reroll {index} left SHOP phase: {_state_phase(state)}")
        if _shop_keys(state) == before:
            # A reroll that produced an identical slot signature is suspicious
            # (the bridge may not have settled). Re-poll once more to be sure.
            state = _settled_raw_state(client, state, polls=12)
        captured.append({"reroll_index": index, "state": state})

    return {
        "record_type": "rng_shop_reroll",
        "seed": seed,
        "stake": stake.lower(),
        "deck": deck.upper(),
        "ante": int(shop_state.get("ante_num", shop_state.get("ante", 1))),
        "money": money,
        "rerolls": captured,
    }


def _shop_keys(state: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(card.get("key", "")) for card in extract_area_cards(state, "shop"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture shop reroll RNG fixtures.")
    parser.add_argument("--seed", help="Run seed string. Required unless --all is given.")
    parser.add_argument("--all", action="store_true", help="Capture all canonical seeds.")
    parser.add_argument("--rerolls", type=int, default=4, help="Number of rerolls to capture per seed.")
    parser.add_argument("--money", type=int, default=1000, help="Scenario money to set before rerolling.")
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
            fixture = capture_shop_rerolls(
                client,
                seed,
                stake=args.stake,
                deck=args.deck,
                rerolls=args.rerolls,
                money=args.money,
            )
        except (BalatroBridgeError, ConnectionError, RuntimeError, ValueError) as exc:
            print(f"FAIL  seed={seed}: {exc}")
            failures.append(seed)
            continue
        path = reroll_fixture_path(seed, stake=args.stake, deck=args.deck, root=args.out_dir)
        save_fixture(fixture, path)
        print(f"OK    seed={seed} -> {path} ({len(fixture['rerolls'])} shop states)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
