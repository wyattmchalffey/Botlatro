"""Capture higher-level RNG surfaces from a running Balatro bridge.

The lower-level capture scripts validate the initial deck and first shop. This
module extends that workflow to opened booster packs, so pack-content
predictions can be checked offline after one bridge capture pass.

Usage:
    python -m balatro_ai.rng.capture_surfaces --seed AAAAAAA
    python -m balatro_ai.rng.capture_surfaces --all --all-pack-kinds
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from balatro_ai.api.client import BalatroBridgeError, JsonRpcBalatroClient
from balatro_ai.rng.capture import CANONICAL_SEEDS, DEFAULT_FIXTURE_DIR
from balatro_ai.rng.capture_shop import _call_with_retries, advance_to_first_shop
from balatro_ai.rng.surfaces import SHOP_POOL_DATA_PATH


DEFAULT_PACK_KIND_KEYS = (
    "p_buffoon_normal_1",
    "p_arcana_normal_1",
    "p_celestial_normal_1",
    "p_standard_normal_1",
    "p_spectral_normal_1",
)


def pack_fixture_path(
    seed: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    pack_index: int | None = None,
    pack_key: str | None = None,
    forced: bool = False,
    vouchers: Iterable[str] = (),
    played_hands: Mapping[str, int] | None = None,
    root: Path = DEFAULT_FIXTURE_DIR,
) -> Path:
    """Return the fixture path for an opened-pack capture."""

    if pack_key and forced:
        surface = "forced_" + _slug(pack_key)
    elif pack_key and pack_index is not None:
        surface = "pack" + str(pack_index) + "_" + _slug(pack_key)
    elif pack_key:
        surface = _slug(pack_key)
    else:
        surface = "pack" + str(0 if pack_index is None else pack_index)
    voucher_suffix = _voucher_suffix(vouchers)
    if voucher_suffix:
        surface += "_with_" + voucher_suffix
    played_suffix = _played_hand_suffix(played_hands or {})
    if played_suffix:
        surface += "_played_" + played_suffix
    return root / f"pack_seed_{seed}_{surface}_{deck.lower()}_{stake.lower()}.json"


def capture_opened_pack(
    client: JsonRpcBalatroClient,
    seed: str,
    *,
    stake: str = "white",
    deck: str = "RED",
    pack_index: int = 0,
    forced_pack_key: str | None = None,
    used_vouchers: Iterable[str] = (),
    played_hands: Mapping[str, int] | None = None,
    money: int | None = 100,
) -> dict[str, Any]:
    """Open a first-shop booster pack and return a fixture payload.

    If ``forced_pack_key`` is supplied, the dev ``scenario`` endpoint replaces
    the visible shop boosters with exactly that pack before opening it. This
    lets one seed validate all pack-content key families without waiting for
    those packs to appear naturally.
    """

    original_shop_state = advance_to_first_shop(client, seed, stake=stake, deck=deck)
    shop_state = original_shop_state
    forced = forced_pack_key is not None
    voucher_keys = tuple(dict.fromkeys(str(key) for key in used_vouchers))
    played_hand_counts = dict(played_hands or {})

    if forced_pack_key is not None or voucher_keys or played_hand_counts:
        params: dict[str, Any] = {}
        if forced_pack_key is not None:
            params.update(
                {
                    "clear_shop": True,
                    "booster_packs": [_booster_item(forced_pack_key)],
                }
            )
        if voucher_keys:
            params["used_vouchers"] = [{"key": key} for key in voucher_keys]
        if played_hand_counts:
            params["played_hands"] = [
                {"name": name, "played": count}
                for name, count in sorted(played_hand_counts.items())
            ]
        if money is not None:
            params["money"] = money
        shop_state = _settled_raw_state(client, _bridge_state(client, "scenario", params))
        if forced_pack_key is not None:
            pack_index = 0
    elif money is not None:
        # The scenario endpoint is dev-only. If it is not installed, natural
        # first-shop captures can still proceed when the run has enough money.
        try:
            shop_state = _settled_raw_state(client, _bridge_state(client, "scenario", {"money": money}))
        except (BalatroBridgeError, RuntimeError):
            shop_state = _settled_raw_state(client, shop_state)

    boosters = extract_area_cards(shop_state, "packs")
    if not boosters:
        raise RuntimeError("No booster packs visible in shop state")
    if pack_index < 0 or pack_index >= len(boosters):
        raise RuntimeError(f"Pack index {pack_index} out of range for {len(boosters)} visible pack(s)")

    opened_pack_key = str(boosters[pack_index].get("key") or forced_pack_key or "")
    opened_state = _bridge_state(client, "buy", {"pack": pack_index})
    opened_state = _settled_raw_state(client, opened_state, polls=12)
    if _state_phase(opened_state) not in {"SMODS_BOOSTER_OPENED", "BOOSTER_OPENED"}:
        raise RuntimeError(f"Opening pack did not reach booster state; got {_state_phase(opened_state)}")
    if not extract_area_cards(opened_state, "pack"):
        raise RuntimeError("Opened booster state did not expose pack cards")

    return {
        "record_type": "rng_surface_pack",
        "seed": seed,
        "stake": stake.lower(),
        "deck": deck.upper(),
        "ante": int(opened_state.get("ante_num", opened_state.get("ante", 1))),
        "pack_index": pack_index,
        "pack_key": opened_pack_key,
        "forced_pack": forced,
        "vouchers": list(voucher_keys),
        "played_hands": played_hand_counts,
        "edition_rate": _edition_rate_for_vouchers(voucher_keys),
        "original_shop_state": original_shop_state,
        "shop_state": shop_state,
        "opened_state": opened_state,
    }


def save_fixture(fixture: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")


def extract_area_cards(state: Mapping[str, Any], area_name: str) -> tuple[dict[str, Any], ...]:
    """Extract bridge card dictionaries from an area such as ``pack``."""

    area = state.get(area_name)
    if isinstance(area, Mapping):
        cards = area.get("cards", ())
    elif isinstance(area, list):
        cards = area
    else:
        cards = ()
    return tuple(card for card in cards if isinstance(card, dict))


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
        _area_keys(state, "shop"),
        _area_keys(state, "packs"),
        _area_keys(state, "pack"),
        _area_keys(state, "jokers"),
        _area_keys(state, "consumables"),
    )


def _area_keys(state: Mapping[str, Any], area_name: str) -> tuple[str, ...]:
    return tuple(str(card.get("key", "")) for card in extract_area_cards(state, area_name))


def _state_phase(state: Mapping[str, Any]) -> str:
    return str(state.get("state", state.get("phase", ""))).upper()


def _booster_item(pack_key: str) -> dict[str, object]:
    record = _booster_record(pack_key)
    name = str(record.get("name", pack_key))
    return {
        "key": pack_key,
        "name": name,
        "label": name,
        "set": "BOOSTER",
        "cost": {"buy": int(record.get("cost", 4)), "sell": 2},
    }


def _booster_record(pack_key: str) -> Mapping[str, Any]:
    for record in _booster_records():
        if str(record.get("key")) == pack_key:
            return record
    raise RuntimeError(f"Unknown booster pack key: {pack_key}")


@lru_cache(maxsize=1)
def _booster_records() -> tuple[Mapping[str, Any], ...]:
    data = json.loads(SHOP_POOL_DATA_PATH.read_text(encoding="utf-8"))
    return tuple(data.get("boosters", ()))


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.lower())


def _voucher_suffix(vouchers: Iterable[str]) -> str:
    return "_".join(_slug(voucher) for voucher in sorted(str(voucher) for voucher in vouchers))


def _played_hand_suffix(played_hands: Mapping[str, int]) -> str:
    return "_".join(_slug(name) for name, count in sorted(played_hands.items()) if int(count) > 0)


def _edition_rate_for_vouchers(vouchers: Iterable[str]) -> float:
    voucher_set = {str(voucher) for voucher in vouchers}
    if "v_glow_up" in voucher_set or "Glow Up" in voucher_set:
        return 4.0
    if "v_hone" in voucher_set or "Hone" in voucher_set:
        return 2.0
    return 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture opened booster-pack RNG surfaces.")
    parser.add_argument("--seed", help="Run seed string. Required unless --all is given.")
    parser.add_argument("--all", action="store_true", help="Capture all canonical seeds.")
    parser.add_argument("--stake", default="white")
    parser.add_argument("--deck", default="RED")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--pack-index", type=int, default=0, help="Visible shop pack index for natural captures.")
    parser.add_argument("--pack-key", action="append", help="Force this pack key via the scenario endpoint; repeatable.")
    parser.add_argument(
        "--used-voucher",
        action="append",
        default=[],
        help="Mark this voucher key as already owned through the scenario endpoint; repeatable.",
    )
    parser.add_argument(
        "--played-hand",
        action="append",
        default=[],
        metavar="NAME=COUNT",
        help="Set a poker hand's played count through the scenario endpoint; repeatable.",
    )
    parser.add_argument(
        "--all-pack-kinds",
        action="store_true",
        help="Force one normal pack for each Buffoon/Arcana/Celestial/Standard/Spectral kind.",
    )
    parser.add_argument("--money", type=int, default=100, help="Scenario money to set before opening packs.")
    parser.add_argument("--no-scenario-money", action="store_true", help="Do not call scenario just to add money.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.all and not args.seed:
        print("Specify --seed SEED or --all.")
        return 2

    seeds = CANONICAL_SEEDS if args.all else (args.seed,)
    pack_keys: tuple[str | None, ...]
    if args.all_pack_kinds:
        pack_keys = DEFAULT_PACK_KIND_KEYS
    elif args.pack_key:
        pack_keys = tuple(args.pack_key)
    else:
        pack_keys = (None,)
    money = None if args.no_scenario_money else args.money
    played_hands = _parse_played_hands(args.played_hand)

    client = JsonRpcBalatroClient(endpoint=args.endpoint, timeout_seconds=args.timeout)
    failures: list[str] = []
    for seed in seeds:
        for pack_key in pack_keys:
            try:
                fixture = capture_opened_pack(
                    client,
                    seed,
                    stake=args.stake,
                    deck=args.deck,
                    pack_index=args.pack_index,
                    forced_pack_key=pack_key,
                    used_vouchers=tuple(args.used_voucher),
                    played_hands=played_hands,
                    money=money,
                )
            except (BalatroBridgeError, ConnectionError, RuntimeError) as exc:
                label = pack_key or f"pack{args.pack_index}"
                print(f"FAIL  seed={seed} pack={label}: {exc}")
                failures.append(f"{seed}:{label}")
                continue

            actual_key = str(fixture.get("pack_key", pack_key or ""))
            path = pack_fixture_path(
                seed,
                stake=args.stake,
                deck=args.deck,
                pack_index=int(fixture.get("pack_index", args.pack_index)),
                pack_key=actual_key,
                forced=pack_key is not None,
                vouchers=tuple(args.used_voucher),
                played_hands=played_hands,
                root=args.out_dir,
            )
            save_fixture(fixture, path)
            cards = extract_area_cards(fixture["opened_state"], "pack")
            print(f"OK    seed={seed} pack={actual_key} -> {path} ({len(cards)} cards)")
    return 1 if failures else 0


def _parse_played_hands(values: Iterable[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --played-hand value {value!r}; expected NAME=COUNT")
        name, raw_count = value.rsplit("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"Invalid --played-hand value {value!r}; hand name is empty")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise SystemExit(f"Invalid --played-hand count in {value!r}") from exc
        parsed[name] = count
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
