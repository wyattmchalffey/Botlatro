"""Non-confounded play-quality audit.

At each PLAY decision, enumerate every legal 1-5 card hand from the current
hand, score each with the engine (evaluate_played_cards), and compare the
bot's chosen play to the best available. Reports avg score fraction
(bot/max), % of plays that are the max, and the same restricted to
"must-clear" plays (last hand, or where only the max would clear). If the
bot systematically underplays its clearing hands, that's a causal lever;
if it already plays optimally, the gap is build power (-> Phase 8).

    PYTHONPATH=src python scripts/play_quality_audit.py [n]
"""

from __future__ import annotations

import sys
import statistics
from itertools import combinations

from balatro_ai.api.actions import ActionType
from balatro_ai.api.state import GamePhase, with_derived_legal_actions
from balatro_ai.bots.registry import create_bot
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int


def _score(state, indices) -> float:
    cards = tuple(state.hand[i] for i in indices)
    held = tuple(c for j, c in enumerate(state.hand) if j not in set(indices))
    blind_name = state.blind or ""
    try:
        ev = evaluate_played_cards(
            cards, state.hand_levels,
            debuffed_suits=debuffed_suits_for_blind(blind_name),
            blind_name=blind_name, jokers=state.jokers,
            discards_remaining=state.discards_remaining, hands_remaining=state.hands_remaining,
            held_cards=held, deck_size=state.deck_size, money=state.money,
        )
        return float(ev.score)
    except Exception:
        return 0.0


def _best_score(state) -> tuple[float, tuple[int, ...]]:
    n = len(state.hand)
    best, best_idx = -1.0, ()
    for k in range(1, min(5, n) + 1):
        for combo in combinations(range(n), k):
            s = _score(state, combo)
            if s > best:
                best, best_idx = s, combo
    return best, best_idx


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    # The decisive metric: last-hand plays where the bot's hand FAILS to clear
    # but the best available hand WOULD clear -> a play-selection loss.
    recoverable = 0          # max clears, bot doesn't, on a last hand
    last_fail = 0            # last-hand plays that didn't clear (bot)
    both_fail = 0            # neither bot nor max clears (build-power loss)
    examples = []
    for i in range(1, n + 1):
        seed = f"{i:07d}"
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
        bot = create_bot("basic_strategy_bot", seed=0)
        for _ in range(4000):
            st = sim.state
            if st.run_over:
                break
            a = bot.choose_action(st)
            if a.action_type == ActionType.NO_OP:
                break
            if (a.action_type == ActionType.PLAY_HAND and st.hand
                    and st.phase in (GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND)
                    and st.hands_remaining <= 1 and st.required_score > 0):
                chosen = _score(st, tuple(a.card_indices))
                best, best_idx = _best_score(st)
                need = st.required_score - st.current_score
                bot_clears = chosen >= need
                max_clears = best >= need
                if not bot_clears:
                    last_fail += 1
                    if max_clears:
                        recoverable += 1
                        if len(examples) < 8:
                            examples.append(
                                f"seed{seed} ante{st.ante} {st.blind}: need={need:.0f} bot={chosen:.0f} max={best:.0f}"
                            )
                    else:
                        both_fail += 1
            sim.step(a)
    print(f"last-hand failures: {last_fail}", flush=True)
    print(f"  RECOVERABLE (max would clear, bot underplayed): {recoverable}", flush=True)
    print(f"  build-power (neither clears): {both_fail}", flush=True)
    for ex in examples:
        print("   ", ex, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
