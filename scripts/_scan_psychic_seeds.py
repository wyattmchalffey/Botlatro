"""Scratch: find seeds whose runs encounter The Psychic (for parity probes)."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor


def scan(seed: str) -> tuple[str, bool]:
    from dataclasses import replace

    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("basic_strategy_bot", seed=0)
    blinds: set[str] = set()
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(4000):
            st = sim.state
            if st.run_over:
                break
            if st.blind:
                blinds.add(st.blind)
            action = bot.choose_action(st)
            if action.action_type.value == "no_op":
                break
            sim.step(action)
    return seed, "The Psychic" in blinds


def main() -> int:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 36
    seeds = [f"{i:07d}" for i in range(start, start + n)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        for seed, hit in ex.map(scan, seeds):
            if hit:
                print("PSYCHIC seed:", seed, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
