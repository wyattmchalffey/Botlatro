"""process_time cost breakdown of the DEPLOYED winrate bot
(solver_shop_basic_play_bot = solver SHOP beam + basic greedy PLAY).

phase_timing.py profiles the data-gen SolverPolicy (play-dominated). The
deployed winrate bot is SHOP-dominated (greedy play is cheap), a different
profile that drives every winrate A/B. This wraps the shop cost centers with
process_time (contention-immune under the pool) and aggregates across N seeds.

    PYTHONPATH=src python scripts/deployed_bot_timing.py [n_seeds] [jobs] [faithful]
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


def run_task(args):
    seed, faithful = args
    import time as _t

    import balatro_ai.search.shop_search as ss
    import balatro_ai.search.shop_sampler as smp
    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from dataclasses import replace

    t = defaultdict(float)
    c = defaultdict(int)

    # --- shop internals ---
    _orig_best_shop = ss.best_shop_action
    def timed_best_shop(state, **kw):
        t0 = _t.process_time()
        r = _orig_best_shop(state, **kw)
        t["shop_best_action"] += _t.process_time() - t0
        c["shop_best_action_calls"] += 1
        return r
    ss.best_shop_action = timed_best_shop

    _orig_leaf = ss.shop_leaf_terms
    def timed_leaf(state, **kw):
        t0 = _t.process_time()
        r = _orig_leaf(state, **kw)
        t["shop_leaf_terms"] += _t.process_time() - t0
        c["shop_leaf_terms_calls"] += 1
        return r
    ss.shop_leaf_terms = timed_leaf

    _orig_build = ss._shop_build_score
    def timed_build(state):
        t0 = _t.process_time()
        r = _orig_build(state)
        t["shop_build_score"] += _t.process_time() - t0
        c["shop_build_score_calls"] += 1
        return r
    ss._shop_build_score = timed_build

    # reroll_ev is a method on ShopSampler
    _orig_ev = smp.ShopSampler.reroll_ev
    def timed_ev(self, state, **kw):
        t0 = _t.process_time()
        r = _orig_ev(self, state, **kw)
        t["shop_reroll_ev"] += _t.process_time() - t0
        c["shop_reroll_ev_calls"] += 1
        return r
    smp.ShopSampler.reroll_ev = timed_ev

    kw = {"seed": _stable_seed_int(seed), "stake": "white"}
    if faithful:
        kw["balatro_seed"] = seed
    sim = LocalBalatroSimulator(**kw)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("solver_shop_basic_play_bot", seed=0)
    run_t0 = _t.process_time()
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(4000):
            st = sim.state
            if st.run_over:
                break
            phase = str(st.phase.value)
            ca0 = _t.process_time()
            a = bot.choose_action(st)
            dt = _t.process_time() - ca0
            if phase in ("selecting_hand", "playing_blind"):
                t["ca_play"] += dt
            elif phase == "shop":
                t["ca_shop"] += dt
            else:
                t["ca_other"] += dt
            if a.action_type.value == "no_op":
                break
            s0 = _t.process_time()
            sim.step(a)
            t["sim_step"] += _t.process_time() - s0
    t["total"] = _t.process_time() - run_t0
    c["won"] = int(bool(sim.state.won))
    c["ante"] = sim.state.ante
    return dict(t), dict(c)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    faithful = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    wall0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(run_task, [(s, faithful) for s in seeds]))
    wall = time.perf_counter() - wall0

    t = defaultdict(float)
    c = defaultdict(int)
    antes = []
    for tt, cc in results:
        for k, v in tt.items():
            t[k] += v
        for k, v in cc.items():
            if k == "ante":
                antes.append(v)
            else:
                c[k] += v
    total = t["total"]
    pct = lambda v: f"{100*v/total:4.1f}%"
    print(f"n={n} jobs={jobs} faithful={faithful}  wall={wall:.1f}s  total CPU={total:.0f}s ({total/n:.2f}s/run)")
    print(f"wins {c['won']}/{n}  mean ante {sum(antes)/len(antes):.2f}")
    print("TOP-LEVEL (by call site):")
    print(f"  shop choose_action     {t['ca_shop']:7.1f}s  ({pct(t['ca_shop'])})")
    print(f"  play choose_action     {t['ca_play']:7.1f}s  ({pct(t['ca_play'])})")
    print(f"  other choose_action    {t['ca_other']:7.1f}s  ({pct(t['ca_other'])})")
    print(f"  sim.step               {t['sim_step']:7.1f}s  ({pct(t['sim_step'])})")
    print("SHOP SUB-BREAKDOWN (subset of shop choose_action):")
    print(f"  best_shop_action total {t['shop_best_action']:7.1f}s  ({pct(t['shop_best_action'])})  calls={c['shop_best_action_calls']}")
    print(f"  reroll_ev (sampling)   {t['shop_reroll_ev']:7.1f}s  ({pct(t['shop_reroll_ev'])})  calls={c['shop_reroll_ev_calls']}")
    print(f"  shop_leaf_terms        {t['shop_leaf_terms']:7.1f}s  ({pct(t['shop_leaf_terms'])})  calls={c['shop_leaf_terms_calls']}")
    print(f"  _shop_build_score      {t['shop_build_score']:7.1f}s  ({pct(t['shop_build_score'])})  calls={c['shop_build_score_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
