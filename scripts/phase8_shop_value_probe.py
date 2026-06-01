"""Diagnostic: heuristic vs neural-value shop leaf on IDENTICAL shop states.

Isolates the decision (no trajectory confound): for each real shop root state,
ask `best_shop_action` with the heuristic leaf and with the calibrated neural
leaf, and tally the chosen action types. If the neural leaf skews toward
END_SHOP / away from BUY, the near-constant value head is causing under-buying
(hoarding) -> weak build -> early death.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    from balatro_ai.api.state import GamePhase
    from balatro_ai.ml.shop_value import NeuralShopLeaf, calibrate_scale
    from balatro_ai.ml.train import load_checkpoint
    from balatro_ai.search.shop_search import ShopSearchContext, best_shop_action
    from balatro_ai.solver.policy import SolverPolicy, _has_shop_action

    sp = SolverPolicy(play_backend="v2", play_depth=3, play_width=1, seed=0)
    config, sampler = sp.shop_config, sp._sampler

    # Collect shop root states from a couple of playthroughs.
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    states = []
    for seed in ("0000005", "0000006"):
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        for _ in range(1500):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            if st.phase == GamePhase.SHOP and _has_shop_action(st):
                states.append(st)
            sim.step(sp.choose_action(st))
            if len(states) >= 70:
                break
        if len(states) >= 70:
            break

    calib_states, probe_states = states[:40], states[40:70]
    model = load_checkpoint(".data/phase8_value_v0.pt")
    calib = calibrate_scale(model, calib_states)
    leaf = NeuralShopLeaf(model, calib)
    print(f"calib n={calib['n']} | heuristic mean={calib['mean_h']:.1f} std={calib['std_h']:.1f}"
          f" | neural mean={calib['mean_n']:.3f} std={calib['std_n']:.3f}")
    print(f"probe states: {len(probe_states)}\n")

    h_cnt, n_cnt, agree = Counter(), Counter(), 0
    for s in probe_states:
        ha = best_shop_action(s, config=config, sampler=sampler,
                              shop_context=ShopSearchContext(), leaf_value_fn=None)
        na = best_shop_action(s, config=config, sampler=sampler,
                              shop_context=ShopSearchContext(), leaf_value_fn=leaf(s))
        hk = ha.action_type.value if ha else "none"
        nk = na.action_type.value if na else "none"
        h_cnt[hk] += 1
        n_cnt[nk] += 1
        agree += int(hk == nk and (ha is None or na is None or ha.card_indices == na.card_indices
                                   or getattr(ha, "target_id", None) == getattr(na, "target_id", None)))

    print(f"heuristic action mix: {dict(h_cnt)}")
    print(f"neural    action mix: {dict(n_cnt)}")
    print(f"same-action agreement: {agree}/{len(probe_states)} = {agree/max(1,len(probe_states)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
