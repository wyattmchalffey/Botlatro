"""Cost breakdown of the DEPLOYED bot (solver_shop_basic_play_bot), aggregated over seeds.

phase_timing.py profiles SolverPolicy(seed=0) (legacy play beam); our actual runs use
solver_shop_basic_play_bot = solver shop beam + BasicStrategy play. This profiles THAT, splitting
play / shop / other / sim, plus the two big shop sub-costs (shop_leaf_terms eval, reroll EV
sampling). process_time = CPU time, contention-immune.

    PYTHONPATH=src py -3.12 scripts/profile_deployed.py [n_seeds] [jobs]
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


def _worker(seed):
    import time as _t
    from collections import defaultdict as dd
    from dataclasses import replace
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    import balatro_ai.search.shop_search as ss
    from balatro_ai.search.shop_sampler import ShopSampler

    t = dd(float)
    c = dd(int)

    _olt = ss.shop_leaf_terms
    def timed_leaf(*a, **k):
        t0 = _t.process_time(); r = _olt(*a, **k); t["shop_leaf"] += _t.process_time() - t0; c["shop_leaf_calls"] += 1; return r
    ss.shop_leaf_terms = timed_leaf

    _ore = ShopSampler.reroll_ev
    def timed_reroll(self, *a, **k):
        t0 = _t.process_time(); r = _ore(self, *a, **k); t["reroll_ev"] += _t.process_time() - t0; c["reroll_ev_calls"] += 1; return r
    ShopSampler.reroll_ev = timed_reroll

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    bot = create_bot("solver_shop_basic_play_bot", seed=0)
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        run0 = _t.process_time()
        for _ in range(6000):
            st = sim.state
            if st.run_over:
                break
            ph = str(st.phase.value)
            ca0 = _t.process_time()
            a = bot.choose_action(st)
            dt = _t.process_time() - ca0
            if ph in ("selecting_hand", "playing_blind"):
                t["ca_play"] += dt; c["play_dec"] += 1
            elif ph == "shop":
                t["ca_shop"] += dt; c["shop_dec"] += 1
            else:
                t["ca_other"] += dt
            if a.action_type.value == "no_op":
                break
            s0 = _t.process_time()
            sim.step(a)
            t["sim_step"] += _t.process_time() - s0
        t["total"] = _t.process_time() - run0
    return dict(t), dict(c)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    wall0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(_worker, seeds))
    wall = time.perf_counter() - wall0
    t = defaultdict(float); c = defaultdict(int)
    for tt, cc in results:
        for k, v in tt.items():
            t[k] += v
        for k, v in cc.items():
            c[k] += v
    total = t["total"] or 1.0
    pct = lambda v: f"{100*v/total:4.1f}%"
    print(f"DEPLOYED bot  n={n} jobs={jobs}  wall={wall:.1f}s  total CPU={total:.0f}s ({total/n:.1f}s/run)")
    print("TOP-LEVEL:")
    print(f"  play choose_action   {t['ca_play']:7.0f}s  ({pct(t['ca_play'])})  decisions={c['play_dec']}")
    print(f"  shop choose_action   {t['ca_shop']:7.0f}s  ({pct(t['ca_shop'])})  decisions={c['shop_dec']}")
    print(f"  other choose_action  {t['ca_other']:7.0f}s  ({pct(t['ca_other'])})")
    print(f"  sim.step             {t['sim_step']:7.0f}s  ({pct(t['sim_step'])})")
    print("SHOP SUB-COSTS:")
    print(f"  shop_leaf_terms      {t['shop_leaf']:7.0f}s  ({pct(t['shop_leaf'])})  calls={c['shop_leaf_calls']}")
    print(f"  reroll_ev sampling   {t['reroll_ev']:7.0f}s  ({pct(t['reroll_ev'])})  calls={c['reroll_ev_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
