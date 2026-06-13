"""Sim->bridge replay-equivalence check (action replay).

Runs the bot through the seed-faithful local sim to produce a trajectory,
then DRIVES that exact action sequence on the BalatroBot bridge for the same
seed string, comparing per-step state (ante / score / money / hand). If the
sim is seed-faithful, the recorded line reproduces on the live game.

We replay the sim's actions rather than re-deciding live: the sim bot plans
with perfect knowledge of the seed-faithful draw order, so re-deciding from
the bridge's (unordered) deck view would diverge. The user's goal is "solve
the seed offline, reproduce it on the real game" = action replay.

Usage (bridge up; BALATRO_SEED_FAITHFUL forced on):
    PYTHONPATH=src python scripts/sim_bridge_replay_check.py AAAAAAA
"""

from __future__ import annotations

import os
import sys

os.environ["BALATRO_SEED_FAITHFUL"] = "1"

from balatro_ai.api.actions import Action, ActionType  # noqa: E402
from balatro_ai.api.client import JsonRpcBalatroClient  # noqa: E402
from balatro_ai.api.state import GameState, with_derived_legal_actions  # noqa: E402
from balatro_ai.bots.registry import create_bot  # noqa: E402
from balatro_ai.rng.capture_shop_sequence import _settled_raw_state, _bridge_state  # noqa: E402
from balatro_ai.sim.local_runner import LocalBalatroSimulator  # noqa: E402
from balatro_ai.solver.seed_game import SeedGame  # noqa: E402
from balatro_ai.solver.trajectory import _stable_seed_int  # noqa: E402


def _hand_multiset(state: GameState):
    return sorted(f"{c.suit}_{('T' if c.rank == '10' else c.rank)}" for c in state.hand)


def _summary(state: GameState) -> dict:
    return {
        "phase": state.phase.value,
        "ante": state.ante,
        "score": state.current_score,
        "money": state.money,
        "hand": _hand_multiset(state),
    }


def run_sim(seed: str, max_steps: int = 3000):
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
        records.append((action, _summary(state)))
        sim.step(action)
    return sim.state, records


def replay_on_bridge(seed: str, records, endpoint: str = "http://127.0.0.1:12346"):
    client = JsonRpcBalatroClient(endpoint=endpoint, timeout_seconds=120.0)
    _bridge_state(client, "menu", None)
    raw = _settled_raw_state(client, _bridge_state(client, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}))
    state = with_derived_legal_actions(GameState.from_mapping(raw))
    first_diff = None
    bridge_final = state
    for i, (action, sim_pre) in enumerate(records):
        bridge_pre = _summary(state)
        # Compare PRE-action state before applying. `ante` is a counter that
        # increments at slightly different points (bridge at boss-defeat /
        # round_eval, sim at cash_out), so only compare it OUTSIDE round_eval
        # where both have settled; score/money/hand are the substantive state.
        compare_keys = ("score", "money", "hand")
        if sim_pre["phase"] != "round_eval" and bridge_pre["phase"] != "round_eval":
            compare_keys = ("ante",) + compare_keys
        diff_key = next((key for key in compare_keys if bridge_pre[key] != sim_pre[key]), None)
        if diff_key == "money" and first_diff is None:
            # Money reads can race the bridge's cash-out dollar ticker (the
            # easing can pause longer than the settle window — a PROVEN audit
            # artifact, see p04 round 3 seed 0000015). Before declaring a
            # divergence on money alone, give the ticker a long re-settle and
            # re-compare once.
            raw = _settled_raw_state(client, None, polls=40, delay_seconds=0.1)
            state = with_derived_legal_actions(GameState.from_mapping(raw))
            bridge_pre = _summary(state)
            diff_key = next((key for key in compare_keys if bridge_pre[key] != sim_pre[key]), None)
        if diff_key is not None and first_diff is None:
            first_diff = (i, diff_key, sim_pre, bridge_pre, action.action_type.value, tuple(action.card_indices))
        try:
            method, params = client._action_to_rpc(action)
            # Settle the bridge after each action so animated state (esp.
            # round-eval money crediting) is fully applied before the next read.
            raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=14)
        except Exception as exc:  # noqa: BLE001
            if first_diff is None:
                first_diff = (i, f"APPLY_ERROR:{type(exc).__name__}", sim_pre, bridge_pre, action.action_type.value, tuple(action.card_indices))
            break
        state = with_derived_legal_actions(GameState.from_mapping(raw))
        bridge_final = state
        if state.run_over:
            break
    return bridge_final, first_diff, len(records)


def main() -> int:
    seed = sys.argv[1] if len(sys.argv) > 1 else "AAAAAAA"
    sim_final, records = run_sim(seed)
    print(f"SIM    seed={seed} won={sim_final.won} ante={sim_final.ante} steps={len(records)}")

    bridge_final, first_diff, n = replay_on_bridge(seed, records)
    print(f"BRIDGE seed={seed} won={bridge_final.won} ante={bridge_final.ante} (replayed {n} actions)")
    if first_diff is None:
        print("PER-STEP STATE MATCH: True (sim line reproduces on bridge)")
    else:
        i, key, sim_pre, bridge_pre, atype, idx = first_diff
        print(f"FIRST STATE DIFF at step {i} field={key} action={atype}{idx}")
        print(f"  sim_pre   ={sim_pre}")
        print(f"  bridge_pre={bridge_pre}")
    print(f"OUTCOME: sim(won={sim_final.won},ante={sim_final.ante}) bridge(won={bridge_final.won},ante={bridge_final.ante})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
