"""Scratch: ONE instrumented bridge replay dumping SHOP/PACK contents per step.

P04 root-cause aid for shop-content RNG divergence (seed 0000015 class).
Replays the sim's recorded actions on the live bridge; at every step prints
both sides' shop slots, voucher, packs, and opened-pack contents.

Usage: PYTHONPATH=src python scripts/_scratch_p04_shopdump.py 0000015 13
"""
from __future__ import annotations

import os
import sys

os.environ["BALATRO_SEED_FAITHFUL"] = "1"

from balatro_ai.api.client import JsonRpcBalatroClient  # noqa: E402
from balatro_ai.api.state import GameState, with_derived_legal_actions  # noqa: E402
from balatro_ai.rng.capture_shop_sequence import _settled_raw_state, _bridge_state  # noqa: E402
from balatro_ai.api.actions import ActionType  # noqa: E402
from balatro_ai.bots.registry import create_bot  # noqa: E402
from balatro_ai.sim.local_runner import LocalBalatroSimulator  # noqa: E402
from balatro_ai.solver.seed_game import SeedGame  # noqa: E402
from balatro_ai.solver.trajectory import _stable_seed_int  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sim_bridge_replay_check import _summary  # noqa: E402


def _names(cards) -> list:
    out = []
    for c in cards or ():
        if isinstance(c, dict):
            label = c.get("label") or c.get("name") or c.get("key")
            key = c.get("key", "")
            cost = c.get("cost")
            if isinstance(cost, dict):
                cost = cost.get("buy")
            edition = c.get("edition")
            if isinstance(edition, dict):
                edition = edition.get("type") or edition.get("key")
            s = f"{label}[{key}]${cost}"
            if edition and edition not in ("base",):
                s += f"({edition})"
            out.append(s)
        else:
            out.append(str(c))
    return out


def _shop_dump(state: GameState) -> dict:
    m = state.modifiers
    return {
        "slots": _names(m.get("shop_cards", ())),
        "voucher": _names(m.get("voucher_cards", ())),
        "packs": _names(m.get("booster_packs", ())),
        "pack_cards": _names(m.get("pack_cards", ())),
    }


def run_sim_with_shops(seed: str, max_steps: int):
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("basic_strategy_bot", seed=0)
    records = []
    for _ in range(max_steps):
        state = sim.state
        if state.run_over:
            break
        action = bot.choose_action(state)
        if action.action_type == ActionType.NO_OP:
            break
        records.append((action, _summary(state), _shop_dump(state)))
        sim.step(action)
    return records


def main():
    seed = sys.argv[1]
    max_replay = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    records = run_sim_with_shops(seed, max_replay + 1)
    print(f"sim recorded {len(records)} steps; replaying on bridge")

    client = JsonRpcBalatroClient(endpoint="http://127.0.0.1:12346", timeout_seconds=120.0)
    _bridge_state(client, "menu", None)
    raw = _settled_raw_state(client, _bridge_state(client, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}))
    state = with_derived_legal_actions(GameState.from_mapping(raw))

    for i, (action, sim_pre, sim_shop) in enumerate(records):
        bridge_pre = _summary(state)
        bridge_shop = _shop_dump(state)
        marks = [k for k in ("phase", "score", "money", "hand") if bridge_pre[k] != sim_pre[k]]
        shop_marks = [k for k in ("slots", "voucher", "packs", "pack_cards") if bridge_shop[k] != sim_shop[k]]
        print(f"\n[{i:02d}] ACTION={action.action_type.value}{tuple(action.card_indices)} meta_idx={action.metadata.get('index') if action.metadata else None}"
              f"  STATE_DIFF={marks or 'none'}  SHOP_DIFF={shop_marks or 'none'}")
        print(f"  sim_pre   = {sim_pre}")
        print(f"  bridge_pre= {bridge_pre}")
        print(f"  sim_shop   = {sim_shop}")
        print(f"  bridge_shop= {bridge_shop}")
        method, params = client._action_to_rpc(action)
        raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=14)
        state = with_derived_legal_actions(GameState.from_mapping(raw))
        if state.run_over:
            print("bridge run_over")
            break
    print(f"\nFINAL bridge_post={_summary(state)}")
    print(f"FINAL bridge_shop={_shop_dump(state)}")


if __name__ == "__main__":
    main()
