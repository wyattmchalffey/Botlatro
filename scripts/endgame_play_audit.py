"""Endgame build-ceiling vs play-limited audit.

Hypothesis under test: the solver loses late because its BUILD can't clear the blind
threshold, not because it misplays. For each LOSS in the cached solver captures, replay
to the start of the failing blind, fork the sim, and run a STRONG deep play search
(v2 beam, much deeper/wider than the bot's default). If even strong play can't reach
the required score -> BUILD-CEILING. If strong play clears it -> PLAY-limited.

Reuses cached captures (no slow bot re-run). Captures store the full action log, which
replays the exact run.

    PYTHONPATH=src py -3.12 scripts/endgame_play_audit.py \
        --caps .data/onpolicy_solver_caps_384.jsonl --jobs 8 --depth 6 --width 6 \
        --out .data/endgame_play_audit.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_DEPTH = 6
_WIDTH = 6


def _audit_worker(task):
    # depth/width travel WITH the task: Windows spawn workers re-import the
    # module, so parent-side `global _DEPTH` mutations never reach them (the
    # 2026-06-11 depth-curve runs all silently ran the 6/6 defaults).
    cap_dict, depth, width = task
    from dataclasses import replace as dc_replace
    from balatro_ai.api.actions import Action, ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.no_foresight import blind_known_deck
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    if cap_dict.get("won"):
        return None
    active = {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND, GamePhase.ROUND_EVAL}
    left = {GamePhase.SHOP, GamePhase.BLIND_SELECT, GamePhase.RUN_OVER}
    seed = cap_dict["seed"]

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    last_fork = None
    prev_phase = sim.state.phase
    for adict in cap_dict.get("actions", ()):  # replay exact run
        try:
            sim.step(Action.from_mapping(adict))
        except Exception:
            break
        ph = sim.state.phase
        if prev_phase == GamePhase.BLIND_SELECT and ph in active:
            last_fork = (copy.deepcopy(sim), int(sim.state.ante), int(sim.state.required_score),
                         str(sim.state.blind or ""))
        prev_phase = ph
    if last_fork is None:
        return None
    osim, start_ante, required, blind = last_fork
    if required <= 0:
        return None

    driver = SolverPolicy(play_backend="v2", play_depth=depth, play_width=width, seed=0)
    peak = int(osim.state.current_score)
    with bot_config_scope(dc_replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(300):
            st = osim.state
            peak = max(peak, int(st.current_score))
            if st.run_over or st.phase in left or int(st.ante) > start_ante:
                break
            if st.phase not in active:
                break
            try:
                # Honest-mode support: SolverPolicy is not a registry wrapper,
                # so blind the observation explicitly. Identity no-op when
                # BALATRO_NO_FORESIGHT is unset (the original clairvoyant
                # oracle); under shuffle/hide this measures the HONEST
                # deep-play headroom instead.
                a = driver.choose_action(blind_known_deck(st))
            except Exception:
                break
            if a is None or a.action_type == ActionType.NO_OP:
                break
            try:
                osim.step(a)
            except Exception:
                break
    peak = max(peak, int(osim.state.current_score))
    return {
        "seed": seed, "ante": start_ante, "blind": blind, "required": required,
        "bot_score": int(cap_dict.get("final_score", 0)),
        "oracle_peak": int(peak),
        "cleared": bool(peak >= required),
        "bot_ratio": round(cap_dict.get("final_score", 0) / required, 3),
        "oracle_ratio": round(peak / required, 3),
    }


def _grp(rows):
    if not rows:
        return {}
    return {
        "n": len(rows),
        "oracle_clear_rate": round(sum(r["cleared"] for r in rows) / len(rows), 3),
        "median_bot_ratio": round(statistics.median(r["bot_ratio"] for r in rows), 3),
        "median_oracle_ratio": round(statistics.median(r["oracle_ratio"] for r in rows), 3),
        "mean_oracle_ratio": round(statistics.mean(r["oracle_ratio"] for r in rows), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global _DEPTH, _WIDTH
    _DEPTH, _WIDTH = args.depth, args.width

    caps = [json.loads(l) for l in open(args.caps, encoding="utf-8") if l.strip()]
    losses = [c for c in caps if not c.get("won")]
    print(f"[audit] {len(caps)} runs, {len(losses)} losses; strong play v2 depth={_DEPTH} width={_WIDTH}", flush=True)

    tasks = [(c, args.depth, args.width) for c in losses]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = [r for r in ex.map(_audit_worker, tasks) if r is not None]
    else:
        rows = [r for r in (_audit_worker(t) for t in tasks) if r is not None]

    by_ante = defaultdict(list)
    for r in rows:
        by_ante[r["ante"]].append(r)
    out = {
        "depth": _DEPTH, "width": _WIDTH, "n_losses_audited": len(rows),
        "overall": _grp(rows),
        "ante5plus": _grp([r for r in rows if r["ante"] >= 5]),
        "ante8": _grp([r for r in rows if r["ante"] == 8]),
        "by_ante": {str(a): _grp(by_ante[a]) for a in sorted(by_ante)},
        "death_blind_clear": {},
    }
    # clear-rate by blind name (where does even strong play fail?)
    by_blind = defaultdict(list)
    for r in rows:
        by_blind[r["blind"]].append(r)
    out["death_blind_clear"] = {
        b: {"n": len(v), "oracle_clear_rate": round(sum(x["cleared"] for x in v) / len(v), 2),
            "median_oracle_ratio": round(statistics.median(x["oracle_ratio"] for x in v), 2)}
        for b, v in sorted(by_blind.items(), key=lambda kv: -len(kv[1]))[:12]
    }
    print(json.dumps(out, indent=2), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
