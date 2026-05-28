"""Capture mid-hand probability-roll outcomes from a live bridge.

Each played hand consumes Balatro's per-key pseudorandom streams for the
stochastic scoring effects (lucky_mult, lucky_money, glass, bloodstone,
business, parking, space, misprint). The rolled values aren't exposed in the
gamestate, so we set up a controlled single-effect hand, play it repeatedly,
and read the observable signal (score / money / hand level) to recover the
hit/miss (or value) sequence. ``validate_play_rng`` then checks that sequence
against the offline ``pseudorandom_float``/``pseudorandom_int`` predictors.

Usage:
    python -m balatro_ai.rng.capture_play_rng --seed AAAAAAA --hands 8
    python -m balatro_ai.rng.capture_play_rng --all --hands 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from balatro_ai.api.client import BalatroBridgeError, JsonRpcBalatroClient
from balatro_ai.rng.capture import CANONICAL_SEEDS, DEFAULT_FIXTURE_DIR
from balatro_ai.rng.capture_shop_sequence import _bridge_state, _settled_raw_state
from balatro_ai.rng.capture_surfaces import extract_area_cards, save_fixture


def _phase(state: Mapping[str, Any]) -> str:
    return str(state.get("state", state.get("phase", ""))).upper()


def _round_chips(state: Mapping[str, Any]) -> int:
    return int((state.get("round") or {}).get("chips", 0) or 0)


def _money(state: Mapping[str, Any]) -> int:
    return int(state.get("money", 0) or 0)


def _high_card_level(state: Mapping[str, Any]) -> int:
    hands = state.get("hands")
    hc = hands.get("High Card") if isinstance(hands, Mapping) else None
    return int(hc.get("level", 1)) if isinstance(hc, Mapping) else 1


def _consumable_count(state: Mapping[str, Any]) -> int:
    return len(extract_area_cards(state, "consumables"))


# Each effect: a controlled single-effect setup + how to read the per-hand
# outcome. ``kind`` selects the offline predictor in validate_play_rng.
# Observables return either a bool (hit) or an int (misprint mult).
EFFECT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "misprint", "kind": "int", "odds_lo": 0, "odds_hi": 23,
        "jokers": [{"key": "j_misprint"}], "hand": [{"key": "S_5"}], "play": [0],
        "observe": lambda before, after: round(_round_chips(after) / 10 - 1),
    },
    {
        "key": "space", "kind": "float", "odds": 4,
        "jokers": [{"key": "j_space"}], "hand": [{"key": "S_5"}], "play": [0],
        "observe": lambda before, after: _high_card_level(after) > _high_card_level(before),
    },
    {
        "key": "lucky_mult", "kind": "float", "odds": 5,
        "jokers": [], "hand": [{"key": "S_5", "enhancement": "LUCKY"}], "play": [0],
        "observe": lambda before, after: _round_chips(after) >= 200,
    },
    {
        "key": "lucky_money", "kind": "float", "odds": 15,
        "jokers": [], "hand": [{"key": "S_5", "enhancement": "LUCKY"}], "play": [0],
        "observe": lambda before, after: (_money(after) - _money(before)) >= 20,
    },
    {
        "key": "bloodstone", "kind": "float", "odds": 2,
        "jokers": [{"key": "j_bloodstone"}], "hand": [{"key": "H_5"}], "play": [0],
        "observe": lambda before, after: _round_chips(after) > 10,
    },
    {
        "key": "business", "kind": "float", "odds": 2,
        "jokers": [{"key": "j_business"}], "hand": [{"key": "S_K"}], "play": [0],
        "observe": lambda before, after: (_money(after) - _money(before)) >= 2,
    },
    {
        "key": "parking", "kind": "float", "odds": 2,
        "jokers": [{"key": "j_reserved_parking"}], "hand": [{"key": "S_5"}, {"key": "H_K"}], "play": [0],
        "observe": lambda before, after: (_money(after) - _money(before)) >= 1,
    },
    {
        "key": "8ball", "kind": "float", "odds": 4,
        "jokers": [{"key": "j_8_ball"}], "hand": [{"key": "S_8"}], "play": [0],
        "scenario_extra": {"clear_consumables": True},
        "observe": lambda before, after: _consumable_count(after) > _consumable_count(before),
    },
)


def capture_play_rng(
    client: JsonRpcBalatroClient,
    seed: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    hands: int = 8,
) -> dict[str, Any]:
    effects: list[dict[str, Any]] = []
    for spec in EFFECT_SPECS:
        observed = _capture_effect(client, seed, spec, stake=stake, deck=deck, hands=hands)
        entry = {"key": spec["key"], "kind": spec["kind"], "observed": observed}
        if spec["kind"] == "float":
            entry["odds"] = spec["odds"]
        else:
            entry["odds_lo"], entry["odds_hi"] = spec["odds_lo"], spec["odds_hi"]
        effects.append(entry)
    return {
        "record_type": "rng_play_probabilities",
        "seed": seed,
        "stake": stake.lower(),
        "deck": deck.upper(),
        "effects": effects,
    }


def _capture_effect(
    client: JsonRpcBalatroClient,
    seed: str,
    spec: Mapping[str, Any],
    *,
    stake: str,
    deck: str,
    hands: int,
) -> list[Any]:
    observe: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] = spec["observe"]
    client.deck = deck
    _bridge_state(client, "menu", None)
    state = _settled_raw_state(
        client, _bridge_state(client, "start", {"deck": deck.upper(), "stake": stake.upper(), "seed": seed})
    )
    if _phase(state) == "BLIND_SELECT":
        state = _settled_raw_state(client, _bridge_state(client, "select", None))

    observed: list[Any] = []
    for _ in range(hands):
        if _phase(state) != "SELECTING_HAND":
            break
        params = {
            "chips": 0, "money": 50, "hands": 50, "discards": 0,
            "clear_jokers": True, "jokers": list(spec["jokers"]),
            "clear_hand": True, "hand": list(spec["hand"]),
        }
        params.update(spec.get("scenario_extra", {}))
        # The controlled deck depletes after a handful of hands and the run
        # ends; that's expected — keep the samples gathered so far rather
        # than failing the whole capture.
        try:
            before = _settled_raw_state(client, _bridge_state(client, "scenario", params), polls=12)
            after = _settled_raw_state(client, _bridge_state(client, "play", {"cards": list(spec["play"])}), polls=16)
        except (BalatroBridgeError, RuntimeError):
            break
        value = observe(before, after)
        observed.append(int(value) if isinstance(value, bool) else value)
        state = after
    return observed


def play_rng_fixture_path(
    seed: str, *, stake: str = "white", deck: str = "RED", root: Path = DEFAULT_FIXTURE_DIR
) -> Path:
    return root / f"play_rng_seed_{seed}_{deck.lower()}_{stake.lower()}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture mid-hand probability-roll fixtures.")
    parser.add_argument("--seed", help="Run seed string. Required unless --all is given.")
    parser.add_argument("--all", action="store_true", help="Capture all canonical seeds.")
    parser.add_argument("--hands", type=int, default=8, help="Hands to sample per effect.")
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
            fixture = capture_play_rng(client, seed, stake=args.stake, deck=args.deck, hands=args.hands)
        except (BalatroBridgeError, ConnectionError, RuntimeError, ValueError) as exc:
            print(f"FAIL  seed={seed}: {exc}")
            failures.append(seed)
            continue
        path = play_rng_fixture_path(seed, stake=args.stake, deck=args.deck, root=args.out_dir)
        save_fixture(fixture, path)
        n = min((len(e["observed"]) for e in fixture["effects"]), default=0)
        print(f"OK    seed={seed} -> {path} ({len(fixture['effects'])} effects, >={n} hands each)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
