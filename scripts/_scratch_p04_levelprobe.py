"""Scratch (P04 round 5): replay a seed on the bridge and at a target step
read the live save.jkr's G.GAME.hands LEVELS + joker config counters, then
compare against the sim's state at that step. Localizes hand-level / joker
counter divergences (Blue-seal planet, Constellation, Green Joker, etc).

Usage: PYTHONPATH=src python scripts/_scratch_p04_levelprobe.py 0000014 117
"""
from __future__ import annotations

import os
import re
import sys
import zlib

os.environ["BALATRO_SEED_FAITHFUL"] = "1"

from balatro_ai.api.client import JsonRpcBalatroClient  # noqa: E402
from balatro_ai.api.state import GameState, with_derived_legal_actions  # noqa: E402
from balatro_ai.rng.capture_shop_sequence import _settled_raw_state, _bridge_state  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sim_bridge_replay_check import run_sim, _summary  # noqa: E402

SAVE_PATH = r"C:/Users/Wyatt/AppData/Roaming/Balatro/1/save.jkr"
HAND_NAMES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush",
    "Full House", "Four of a Kind", "Straight Flush", "Five of a Kind",
    "Flush House", "Flush Five",
]


def read_save():
    try:
        raw = open(SAVE_PATH, "rb").read()
        return zlib.decompress(raw, -15).decode("latin-1")
    except Exception as exc:  # noqa: BLE001
        return f"<save read failed: {exc}>"


def hand_levels(out: str) -> dict[str, int]:
    res = {}
    for nm in HAND_NAMES:
        for m in re.finditer(r'\["' + re.escape(nm) + r'"\]=\{', out):
            blk = out[m.end():m.end() + 400]
            lm = re.search(r'\["level"\]=([0-9]+)', blk)
            if lm:
                res[nm] = int(lm.group(1))
                break
    return res


def streams(out: str, keys) -> dict[str, str]:
    i = out.find('["pseudorandom"]')
    table = out[i:i + 6000] if i >= 0 else ""
    res = {}
    for key in keys:
        m = re.search(r'\["' + re.escape(key) + r'"\]=([0-9.]+)', table)
        res[key] = m.group(1) if m else "<absent>"
    return res


def joker_mults(out: str) -> dict[str, list]:
    """Pull config.mult / config.x_mult / config.extra for owned jokers."""
    res = {}
    for jname, fld in [("Green Joker", "mult"), ("Constellation", "Xmult"),
                       ("Ride the Bus", "mult"), ("Red Card", "mult")]:
        i = out.find(jname)
        if i < 0:
            continue
        ctx = out[max(0, i - 600):i + 200]
        m = re.search(r'\["mult"\]=([0-9.]+)', ctx)
        x = re.search(r'\["x_mult"\]=([0-9.]+)', ctx)
        res[jname] = (m.group(1) if m else None, x.group(1) if x else None)
    return res


def main():
    seed = sys.argv[1]
    target = int(sys.argv[2])
    keys = sys.argv[3:] or []
    _, records = run_sim(seed)
    print(f"sim {len(records)} steps; replaying to step {target}")

    client = JsonRpcBalatroClient(endpoint="http://127.0.0.1:12346", timeout_seconds=120.0)
    _bridge_state(client, "menu", None)
    raw = _settled_raw_state(client, _bridge_state(client, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}))
    state = with_derived_legal_actions(GameState.from_mapping(raw))

    for i, (action, sim_pre) in enumerate(records):
        if i == target:
            out = read_save()
            print(f"\n=== AT STEP {target} (pre-action {action.action_type.value}{tuple(action.card_indices)}) ===")
            print(f"bridge_summary={_summary(state)}")
            print(f"sim_summary   ={sim_pre}")
            bl = hand_levels(out)
            print(f"BRIDGE hand_levels={bl}")
            print(f"BRIDGE joker_mults={joker_mults(out)}")
            if keys:
                print(f"BRIDGE streams={streams(out, keys)}")
            lhp = re.search(r'\["last_hand_played"\]="([^"]+)"', out)
            print(f"BRIDGE last_hand_played={lhp.group(1) if lhp else '?'}")
            break
        method, params = client._action_to_rpc(action)
        raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=14)
        state = with_derived_legal_actions(GameState.from_mapping(raw))
        if state.run_over:
            print("bridge run_over before target")
            break


if __name__ == "__main__":
    main()
