"""Analyze WHERE and HOW the solver dies — build-too-weak vs play/marginal.

Most games die ~ante 4-5; that's where the bulk of winrate is lost. For each
losing seed, replay and capture the final (death) blind: ante, blind name, the
score/target ratio reached, and hands/discards left. Aggregate tells us whether
deaths are BLOWOUTS (ratio << 1 -> build far too weak, a scaling problem) or
CLOSE (ratio ~ 0.8-1.0 -> marginal build or play, a tuning problem).

    PYTHONPATH=src python scripts/solver_death_analysis.py [n_seeds] [jobs]
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


def run_task(seed: str) -> dict:
    import balatro_ai.search.hand_search as hs
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0)
    # track the last in-blind state we saw (the death blind when we lose)
    last_blind = {}
    for _ in range(6000):
        st = sim.state
        if st.run_over:
            break
        if str(st.phase.value) in ("selecting_hand", "playing_blind"):
            req = float(getattr(st, "required_score", 0) or 0)
            cur = float(getattr(st, "current_score", 0) or 0)
            last_blind = {
                "ante": st.ante, "blind": str(getattr(st, "blind", "")),
                "required": req, "current": cur,
                "ratio": (cur / req) if req > 0 else 0.0,
                "hands_left": int(getattr(st, "hands_remaining", getattr(st, "hands", 0)) or 0),
                "discards_left": int(getattr(st, "discards_remaining", getattr(st, "discards", 0)) or 0),
                "jokers": len(getattr(st, "jokers", ()) or ()),
            }
        a = pol.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    s = sim.state
    won = bool(getattr(s, "won", False)) or s.ante >= 9
    return {"seed": seed, "ante": s.ante, "won": won, "death": last_blind}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, seeds))

    wins = sum(r["won"] for r in rows)
    ante_mean = statistics.mean(r["ante"] for r in rows)
    print(f"=== {n} seeds: wins={wins}/{n} ({100*wins/n:.1f}%)  ante mean={ante_mean:.2f} ===")
    losers = [r for r in rows if not r["won"] and r["death"]]
    death_ante = Counter(r["death"]["ante"] for r in losers)
    death_blind = Counter(r["death"]["blind"] for r in losers)
    ratios = [r["death"]["ratio"] for r in losers]
    # bucket the ratios
    blowout = sum(1 for x in ratios if x < 0.5)
    midgap = sum(1 for x in ratios if 0.5 <= x < 0.8)
    close = sum(1 for x in ratios if 0.8 <= x < 1.0)
    print(f"=== death analysis, {n} seeds ({len(losers)} losses) ===")
    print(f"  death ante distribution: {dict(sorted(death_ante.items()))}")
    print(f"  death blind distribution: {dict(death_blind.most_common())}")
    if ratios:
        print(f"  score/target ratio at death: mean={statistics.mean(ratios):.2f} "
              f"median={statistics.median(ratios):.2f}")
        print(f"  BLOWOUT (<0.5): {blowout}   MID (0.5-0.8): {midgap}   CLOSE (0.8-1.0): {close}")
    # worst and closest deaths
    losers_sorted = sorted(losers, key=lambda r: r["death"]["ratio"])
    print("  closest deaths (nearly cleared):")
    for r in sorted(losers, key=lambda r: -r["death"]["ratio"])[:6]:
        d = r["death"]
        print(f"    seed {r['seed']} a{d['ante']} {d['blind']}: {d['current']:.0f}/{d['required']:.0f} "
              f"(ratio {d['ratio']:.2f}) hands_left={d['hands_left']} jokers={d['jokers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
