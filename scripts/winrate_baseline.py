"""Measure winrate + diagnose losses on the deck-faithful sim.

Runs a bot across a seed set, records the final/death state for each run
(ante, blind, score vs required, money, jokers, terminated reason), and
prints a winrate summary + ante histogram + death-reason breakdown + the
score gap on the failed blind. Use to target build/macro-strategy work.

    BALATRO_SEED_FAITHFUL=1 PYTHONPATH=src python scripts/winrate_baseline.py [bot] [n]
"""

from __future__ import annotations

import os
import sys
from collections import Counter

os.environ.setdefault("BALATRO_SEED_FAITHFUL", "1")

from balatro_ai.api.state import with_derived_legal_actions  # noqa: E402
from balatro_ai.bots.registry import create_bot  # noqa: E402
from balatro_ai.sim.local_runner import LocalBalatroSimulator  # noqa: E402
from balatro_ai.solver.seed_game import SeedGame  # noqa: E402
from balatro_ai.solver.trajectory import _stable_seed_int  # noqa: E402


def run_seed(bot_name: str, seed: str, max_steps: int = 4000) -> dict:
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot(bot_name, seed=0)
    steps = 0
    last_blind = ""
    for _ in range(max_steps):
        st = sim.state
        if st.run_over:
            break
        if st.blind:
            last_blind = st.blind
        a = bot.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
        steps += 1
    s = sim.state
    return {
        "seed": seed,
        "won": bool(s.won),
        "ante": s.ante,
        "blind": last_blind,
        "score": s.current_score,
        "required": s.required_score,
        "money": s.money,
        "n_jokers": len(s.jokers),
        "jokers": [j.name for j in s.jokers],
        "steps": steps,
    }


def main() -> int:
    bot_name = sys.argv[1] if len(sys.argv) > 1 else "basic_strategy_bot"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    seeds = ["AAAAAAA", "BBBBBBB", "CCCCCCC", "1234567"] + [f"{i:07d}" for i in range(1, n + 1)]

    results = []
    for seed in seeds:
        r = run_seed(bot_name, seed)
        results.append(r)
        tag = "WIN " if r["won"] else "loss"
        gap = "" if r["won"] else f" gap={r['required'] - r['score']} (got {r['score']}/{r['required']})"
        print(f"{tag} {seed} ante{r['ante']} {r['blind']}{gap} money={r['money']} jk={r['n_jokers']}")

    wins = sum(r["won"] for r in results)
    total = len(results)
    print(f"\n=== {bot_name}: winrate {wins}/{total} ({wins/total:.1%}) ===")
    ante_hist = Counter(r["ante"] for r in results)
    print("ante reached:", dict(sorted(ante_hist.items())))
    loss_ante = Counter(r["ante"] for r in results if not r["won"])
    print("death ante:", dict(sorted(loss_ante.items())))
    loss_blind = Counter(r["blind"] for r in results if not r["won"])
    print("death blind:", dict(loss_blind))
    # score-gap distribution on losses (how close were they)
    losses = [r for r in results if not r["won"]]
    if losses:
        ratios = sorted(r["score"] / max(1, r["required"]) for r in losses)
        import statistics
        print(f"loss score/required: median={statistics.median(ratios):.2f} min={ratios[0]:.2f} max={ratios[-1]:.2f}")
    # joker frequency among winners vs losers
    win_jk = Counter(j for r in results if r["won"] for j in r["jokers"])
    print("top jokers in WINS:", win_jk.most_common(8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
