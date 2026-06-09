"""Profile WITHIN the play solver (_solve_blind) for the deployed bot.

Play is 56% of deployed-bot CPU; it's one _solve_blind per play decision, which enumerates
discard actions x draw-odds evaluations. This times _solve_blind total, the draw-evaluations
(count+time), and the clear-line work, to pinpoint the play hotspot.

    PYTHONPATH=src py -3.12 scripts/profile_play.py [n_seeds] [jobs]
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
    import balatro_ai.bots.basic_strategy.blind_solver as bs

    t = dd(float); c = dd(int)

    _osolve = bs._solve_blind_uncached
    def timed_solve(*a, **k):
        t0 = _t.process_time(); r = _osolve(*a, **k); t["solve_blind"] += _t.process_time() - t0; c["solve_calls"] += 1; return r
    bs._solve_blind_uncached = timed_solve

    for fn in ("_straight_draw_evaluation", "_preferred_target_draw_evaluation"):
        orig = getattr(bs, fn, None)
        if orig is None:
            continue
        def make(o, name):
            def timed(*a, **k):
                t0 = _t.process_time(); r = o(*a, **k); t["draw_eval"] += _t.process_time() - t0; c["draw_eval_calls"] += 1; return r
            return timed
        setattr(bs, fn, make(orig, fn))

    _obcl = bs._best_clear_line
    def timed_bcl(*a, **k):
        t0 = _t.process_time(); r = _obcl(*a, **k); t["best_clear_line"] += _t.process_time() - t0; c["bcl_calls"] += 1; return r
    bs._best_clear_line = timed_bcl

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    bot = create_bot("solver_shop_basic_play_bot", seed=0)
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        run0 = _t.process_time()
        for _ in range(6000):
            st = sim.state
            if st.run_over:
                break
            a = bot.choose_action(st)
            if a.action_type.value == "no_op":
                break
            sim.step(a)
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
    print(f"PLAY profile  n={n} jobs={jobs}  wall={wall:.1f}s  total CPU={total:.0f}s ({total/n:.1f}s/run)")
    print(f"  _solve_blind (play)    {t['solve_blind']:7.0f}s  ({pct(t['solve_blind'])})  calls={c['solve_calls']}")
    print(f"    _best_clear_line     {t['best_clear_line']:7.0f}s  ({pct(t['best_clear_line'])})  calls={c['bcl_calls']}")
    print(f"    draw evaluations     {t['draw_eval']:7.0f}s  ({pct(t['draw_eval'])})  calls={c['draw_eval_calls']}")
    print(f"    (per draw-eval: {1000*t['draw_eval']/max(1,c['draw_eval_calls']):.2f} ms; "
          f"per solve: {c['draw_eval_calls']/max(1,c['solve_calls']):.0f} draw-evals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
