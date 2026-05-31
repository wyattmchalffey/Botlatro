"""Diagnose why the data-gen SolverPolicy dies at ante 1 (~24% of games).

Ante 1 is trivially survivable (the basic bot never fails it), so an ante-1
death is a catastrophic decision. This finds ante-1-death seeds, then traces
EVERY decision of one game: phase, blind, score vs required, hands/discards,
money, jokers, hand, and the action chosen — so we can see exactly which
decision loses the run (a play that fails to score, a bad shop/sell, etc.).

    PYTHONPATH=src python scripts/solver_ante1_diagnose.py [n_scan]
"""

from __future__ import annotations

import sys


def _cards(hand) -> str:
    out = []
    for c in (hand or ()):
        r = c.get("rank") if isinstance(c, dict) else getattr(c, "rank", "?")
        s = c.get("suit") if isinstance(c, dict) else getattr(c, "suit", "?")
        out.append(f"{r}{s[0] if s else '?'}")
    return " ".join(out)


def run(seed: str, trace: bool):
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0)
    for step in range(4000):
        st = sim.state
        if st.run_over:
            break
        a = pol.choose_action(st)
        if trace:
            jk = ",".join(j.name for j in st.jokers) or "-"
            phase = str(st.phase.value)
            extra = ""
            if phase in ("selecting_hand", "playing_blind"):
                extra = f" score={st.current_score}/{st.required_score} h={st.hands_remaining} d={st.discards_remaining} hand=[{_cards(st.hand)}]"
            elif phase == "shop":
                extra = f" money={st.money} shop_items={[getattr(i,'name',i) for i in (getattr(st,'shop_items',None) or [])][:6]}"
            act = f"{a.action_type.value}{tuple(getattr(a,'card_indices',()) or ())}"
            print(f"  a{st.ante} {phase:14s} ${st.money:<3} jk[{jk}]{extra}\n      -> {act}")
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    s = sim.state
    return s.ante, int(s.current_score), bool(getattr(s, "won", False))


def main() -> int:
    n_scan = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f"scanning seeds 1..{n_scan} for ante-1 deaths...")
    ante1 = []
    for i in range(1, n_scan + 1):
        seed = f"{i:07d}"
        ante, score, won = run(seed, trace=False)
        if ante <= 1 and not won:
            ante1.append((seed, score))
    print(f"ante-1 deaths: {len(ante1)}/{n_scan} -> {[s for s,_ in ante1][:10]}")
    if not ante1:
        print("no ante-1 deaths in this scan")
        return 0
    for seed, score in ante1[:2]:
        print(f"\n===== TRACE seed={seed} (died ante 1, final score={score}) =====")
        run(seed, trace=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
