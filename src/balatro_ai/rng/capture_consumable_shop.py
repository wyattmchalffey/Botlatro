"""Capture shop state across a consumable use, from a live bridge.

Tests whether USE_CONSUMABLE desyncs the shop-card RNG stream. For each
consumable key it reaches the first shop, records the initial shop cards,
injects + uses the consumable, then rerolls and records the resulting shop
cards. The reroll is one more ``predict_shop_cards`` roll on the persistent
rng, so if the consumable consumed no shop-stream RNG the post-use reroll
equals the no-use second roll (validated offline by ``validate_consumable_shop``).

Usage:
    python -m balatro_ai.rng.capture_consumable_shop --seed AAAAAAA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from balatro_ai.api.client import BalatroBridgeError, JsonRpcBalatroClient
from balatro_ai.rng.capture import CANONICAL_SEEDS, DEFAULT_FIXTURE_DIR
from balatro_ai.rng.capture_shop import advance_to_first_shop
from balatro_ai.rng.capture_surfaces import _bridge_state, _settled_raw_state, extract_area_cards, save_fixture


# Consumables usable in the shop with no hand-card selection, spanning the
# effect families that consume RNG: joker creation, consumable creation,
# legendary joker, and a no-RNG planet control.
DEFAULT_CONSUMABLE_KEYS = ("c_judgement", "c_emperor", "c_high_priestess", "c_soul", "c_mercury")


def consumable_shop_fixture_path(
    seed: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    root: Path = DEFAULT_FIXTURE_DIR,
) -> Path:
    return root / f"consumable_shop_seed_{seed}_{deck.lower()}_{stake.lower()}.json"


def capture_consumable_shop(
    client: JsonRpcBalatroClient,
    seed: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    consumable_keys: Iterable[str] = DEFAULT_CONSUMABLE_KEYS,
    money: int = 1000,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    ante = 1
    for key in consumable_keys:
        shop_state = advance_to_first_shop(client, seed, stake=stake, deck=deck)
        ante = int(shop_state.get("ante_num", shop_state.get("ante", 1)))
        shop0 = _shop_keys(shop_state)
        _settled_raw_state(
            client,
            _bridge_state(client, "scenario", {"money": money, "consumables": [{"key": key}]}),
            polls=12,
        )
        used = _settled_raw_state(client, _bridge_state(client, "use", {"consumable": 0}), polls=16)
        rerolled = _settled_raw_state(client, _bridge_state(client, "reroll", None))
        trials.append(
            {
                "consumable_key": key,
                "after_use_phase": str(used.get("state", used.get("phase", ""))).upper(),
                "shop0": list(shop0),
                "post_use_reroll": list(_shop_keys(rerolled)),
            }
        )
    return {
        "record_type": "rng_consumable_shop",
        "seed": seed,
        "stake": stake.lower(),
        "deck": deck.upper(),
        "ante": ante,
        "trials": trials,
    }


def _shop_keys(state: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(card.get("key", "")) for card in extract_area_cards(state, "shop"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture consumable-use shop RNG fixtures.")
    parser.add_argument("--seed", help="Run seed string. Required unless --all is given.")
    parser.add_argument("--all", action="store_true", help="Capture all canonical seeds.")
    parser.add_argument("--money", type=int, default=1000)
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
            fixture = capture_consumable_shop(client, seed, stake=args.stake, deck=args.deck, money=args.money)
        except (BalatroBridgeError, ConnectionError, RuntimeError, ValueError) as exc:
            print(f"FAIL  seed={seed}: {exc}")
            failures.append(seed)
            continue
        path = consumable_shop_fixture_path(seed, stake=args.stake, deck=args.deck, root=args.out_dir)
        save_fixture(fixture, path)
        print(f"OK    seed={seed} -> {path} ({len(fixture['trials'])} consumables)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
