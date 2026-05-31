"""Full-trajectory audit of ANTE-1 DEATHS — where did each run go wrong?

Scans the numeric seed set, and for every game that DIES at ante 1, prints a
chronological play-by-play: each blind (target, hands/discards, jokers), each
play/discard (cards + score gained + running total/target), and each shop action
(money, buys/sells/uses). Games that survive ante 1 are stopped early (ante>=2),
so the scan is fast (it only fully plays the ante-1 deaths).

    PYTHONPATH=src python scripts/solver_ante1_audit.py [n_seeds] [jobs] [max_to_print]
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor


def _card(c) -> str:
    r = getattr(c, "rank", "?")
    s = getattr(c, "suit", "?")
    enh = getattr(c, "enhancement", None)
    tag = "*" if enh else ""
    return f"{r}{s}{tag}"


def run_task(seed: str) -> dict:
    import balatro_ai.search.hand_search as hs
    from balatro_ai.api.actions import ActionType
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0)

    events: list[str] = []
    prev_blind = None
    for _ in range(4000):
        st = sim.state
        if st.run_over:
            break
        if st.ante >= 2:  # survived ante 1 — stop, we only audit ante-1 deaths
            return {"seed": seed, "died_ante1": False, "events": []}
        phase = str(st.phase.value)

        if phase == "selecting_hand":
            bk = (st.ante, str(st.blind))
            if bk != prev_blind:
                jk = ", ".join(j.name for j in st.jokers) or "(none)"
                lv = {k: v for k, v in (st.hand_levels or {}).items() if v > 1}
                events.append(
                    f"BLIND ante{st.ante} {st.blind}  target={st.required_score} "
                    f"hands={st.hands_remaining} discards={st.discards_remaining} "
                    f"money=${st.money} jokers=[{jk}] levels={lv or '{}'}")
                prev_blind = bk
            hand = list(getattr(st, "hand", ()) or ())
            score_before = float(st.current_score or 0)
            req = float(st.required_score or 0)
            a = pol.choose_action(st)
            if a.action_type.value == "no_op":
                break
            played = [_card(hand[i]) for i in (a.card_indices or ()) if 0 <= i < len(hand)]
            sim.step(a)
            ns = sim.state
            after = float(ns.current_score or 0)
            if a.action_type == ActionType.PLAY_HAND:
                if ns.run_over and ns.ante <= 1:
                    gained = after - score_before  # score retained at death
                    events.append(f"   PLAY [{' '.join(played)}] +{gained:.0f} -> DIED {after:.0f}/{req:.0f}")
                elif str(ns.phase.value) != "selecting_hand":
                    events.append(f"   PLAY [{' '.join(played)}] (was {score_before:.0f}/{req:.0f}) -> CLEARED")
                else:
                    events.append(f"   PLAY [{' '.join(played)}] +{after - score_before:.0f} -> {after:.0f}/{req:.0f}")
            elif a.action_type == ActionType.DISCARD:
                events.append(f"   DISCARD [{' '.join(played)}]")
            continue

        if phase == "shop":
            a = pol.choose_action(st)
            if a.action_type.value == "no_op":
                break
            events.append("   SHOP " + _shop_label(a, st))
            sim.step(a)
            continue

        # other phases (blind_select, round_eval, booster, ...)
        a = pol.choose_action(st)
        if a.action_type.value == "no_op":
            break
        if a.action_type == ActionType.USE_CONSUMABLE:
            events.append("   USE " + _shop_label(a, st))
        sim.step(a)

    s = sim.state
    died_ante1 = bool(s.run_over) and s.ante <= 1
    if died_ante1:
        events.append(f"DIED ante{s.ante} at {int(s.current_score)}/{int(s.required_score or 0)}")
    return {"seed": seed, "died_ante1": died_ante1, "events": events if died_ante1 else []}


def _shop_label(a, st) -> str:
    t = a.action_type.value
    meta = a.metadata or {}
    kind = meta.get("kind", "")
    idx = meta.get("index", None)
    if t == "buy" and kind == "card" and idx is not None:
        shop = getattr(st, "shop", None) or ()
        nm = str(shop[idx]) if 0 <= idx < len(shop) else "?"
        return f"BUY {nm}"
    if t == "sell" and idx is not None:
        jk = getattr(st, "jokers", ()) or ()
        nm = getattr(jk[idx], "name", "?") if 0 <= idx < len(jk) else "?"
        return f"SELL {nm}"
    if t == "use_consumable":
        cons = getattr(st, "consumables", ()) or ()
        nm = str(cons[idx]) if (idx is not None and 0 <= idx < len(cons)) else ""
        return f"USE {nm}"
    if t == "open_pack":
        return "OPEN_PACK"
    if t == "reroll":
        return "REROLL"
    if t == "end_shop":
        return "END_SHOP"
    return t


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    max_print = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, seeds))

    deaths = [r for r in rows if r["died_ante1"]]
    print(f"=== ante-1 deaths: {len(deaths)}/{n} seeds ===")
    print("seeds:", [r["seed"] for r in deaths])
    for r in deaths[:max_print]:
        print(f"\n########## seed {r['seed']} ##########")
        for e in r["events"]:
            print(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
