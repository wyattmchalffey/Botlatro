"""Scratch: verify the Psychic rust_best_play_scores lift against pure Python.

At every SELECTING_HAND decision under The Psychic, compute the best play with
the Rust bestplay path enabled vs disabled (pure-Python ground truth, unchanged
by the lift). Chosen cards + hand type + score must match exactly.
"""

from __future__ import annotations

from dataclasses import replace

from balatro_ai.api.state import with_derived_legal_actions
from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
from balatro_ai.bots.registry import create_bot
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int
import balatro_ai.rules.hand_evaluator as he


def main() -> int:
    checked = mismatches = 0
    for seed in ["0000003", "0000010", "0000016", "0000018", "0000019"]:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
        bot = create_bot("solver_shop_basic_play_bot", seed=0)
        with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
            for _ in range(4000):
                st = sim.state
                if st.run_over:
                    break
                if st.phase.value == "selecting_hand" and st.blind == "The Psychic" and st.hand:
                    kwargs = dict(
                        debuffed_suits=he.debuffed_suits_for_blind(st.blind),
                        blind_name=st.blind,
                        jokers=st.jokers,
                        discards_remaining=st.discards_remaining,
                        hands_remaining=st.hands_remaining,
                        deck_size=st.deck_size,
                        money=st.money,
                    )
                    saved = he._RUST_BESTPLAY_ENABLED
                    try:
                        he._RUST_BESTPLAY_ENABLED = True
                        rust_eval = he.best_play_from_hand(st.hand, st.hand_levels, **kwargs)
                        he._RUST_BESTPLAY_ENABLED = False
                        py_eval = he.best_play_from_hand(st.hand, st.hand_levels, **kwargs)
                    finally:
                        he._RUST_BESTPLAY_ENABLED = saved
                    checked += 1
                    r = (rust_eval.cards, rust_eval.hand_type, rust_eval.score) if rust_eval else None
                    p = (py_eval.cards, py_eval.hand_type, py_eval.score) if py_eval else None
                    if r != p:
                        mismatches += 1
                        if mismatches <= 5:
                            print(f"MISMATCH seed={seed}: rust={r} python={p}")
                action = bot.choose_action(st)
                if action.action_type.value == "no_op":
                    break
                sim.step(action)
    print(f"psychic best-play parity: {checked} decisions checked, {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
