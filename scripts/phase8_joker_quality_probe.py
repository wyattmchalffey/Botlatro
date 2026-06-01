"""Count-controlled quality probe: does the averaged rollout value distinguish
GOOD jokers from JUNK jokers, or only "more jokers > fewer"?

The earlier diagnostic showed removing ANY joker lowers value (+0.38 avg) — but
that's dominated by the trivial COUNT effect. This isolates QUALITY:

For each state, run K paired rollouts per joker (value with vs without it), split
the K into two halves, and DEMEAN within the state (subtract the state's mean Δ —
this removes the count/difficulty effect, leaving only how much MORE/LESS each
joker matters than its st-mates). Then correlate half-A vs half-B demeaned Δ across
all (state, joker):
  - demeaned split-half corr >> 0 => the rollout RELIABLY ranks jokers within a
    state => QUALITY signal exists => relabel+retrain could beat the heuristic.
  - demeaned split-half corr ~ 0 => per-joker differences are noise => averaged
    labels buy only COUNT (parity), not a win.
Also reports the raw (count-inclusive) split-half corr for contrast.

    PYTHONPATH=src python scripts/phase8_joker_quality_probe.py \
        --states 14 --samples 10 --max-antes 3 --jobs 8 --metrics .data/phase8_joker_quality.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _frac(state) -> float:
    if state.required_score and state.required_score > 0:
        return min(1.0, max(0.0, state.current_score / state.required_score))
    return 0.0


def _rollout_value(state, *, seed, rollout_bot, max_antes, max_steps=300):
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    sim = LocalBalatroSimulator(seed=seed, stake="white")
    sim.state = state
    bot = create_bot(rollout_bot, seed=seed)
    start_ante = state.ante
    for _ in range(max_steps):
        s = sim.state
        if s.won:
            return float(start_ante + max_antes) + 1.0
        if s.run_over or s.phase == GamePhase.RUN_OVER:
            break
        if s.ante - start_ante >= max_antes:
            return float(s.ante) + _frac(s)
        a = bot.choose_action(s)
        if a is None or a.action_type == ActionType.NO_OP:
            break
        try:
            sim.step(a)
        except (ValueError, IndexError, KeyError, TypeError, AttributeError):
            break
    f = sim.state
    return float(f.ante) + _frac(f)


def _collect_states(seeds, min_jokers, cap, min_ante, per_seed=2):
    """<= per_seed states per seed (spread across seeds) so builds are DIVERSE,
    not many near-identical states from one run."""
    from balatro_ai.api.state import GamePhase
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    drv = SolverPolicy(play_backend="v2", play_depth=2, play_width=1, seed=0)
    out = []
    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        got = 0
        for _ in range(1500):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            if (st.ante >= min_ante and len(st.jokers) >= min_jokers
                    and st.phase in (GamePhase.SELECTING_HAND, GamePhase.SHOP)):
                out.append(st)
                got += 1
                if len(out) >= cap:
                    return out
                if got >= per_seed:
                    break
            sim.step(drv.choose_action(st))
    return out


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def _state_job(arg):
    """Per-joker split-half Δ for one state (count-controlled via demeaning)."""
    st, si, samples, rollout_bot, max_antes = arg
    from balatro_ai.api.state import with_derived_legal_actions

    st_d = with_derived_legal_actions(st)
    njok = len(st.jokers)
    half = samples // 2
    a_sums = [0.0] * njok
    b_sums = [0.0] * njok
    for k in range(samples):
        seed = 7919 * (k + 1) + 31 * si
        v_with = _rollout_value(st_d, seed=seed, rollout_bot=rollout_bot, max_antes=max_antes)
        for j in range(njok):
            without = with_derived_legal_actions(dataclasses.replace(
                st, jokers=tuple(x for q, x in enumerate(st.jokers) if q != j)))
            d = v_with - _rollout_value(without, seed=seed, rollout_bot=rollout_bot, max_antes=max_antes)
            if k < half:
                a_sums[j] += d
            else:
                b_sums[j] += d
    meanA = [s / half for s in a_sums]
    meanB = [s / (samples - half) for s in b_sums]
    mA, mB = statistics.mean(meanA), statistics.mean(meanB)
    names = [getattr(j, "name", "?") for j in st.jokers]
    full = [(a + b) / 2 for a, b in zip(meanA, meanB)]
    return {
        "ante": st.ante, "names": names,
        "demA": [x - mA for x in meanA], "demB": [x - mB for x in meanB],
        "rawA": meanA, "rawB": meanB, "full_demeaned": [x - (mA + mB) / 2 for x in full],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=14)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--max-antes", type=int, default=3)
    ap.add_argument("--min-ante", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rollout-bot", default="basic_strategy_bot")
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    seeds = [f"{600000 + i:07d}" for i in range(1, 61)]
    states = _collect_states(seeds, min_jokers=3, cap=args.states, min_ante=args.min_ante, per_seed=1)
    print(f"[quality] {len(states)} states (ante>={args.min_ante}, jokers>=3), "
          f"{sum(len(s.jokers) for s in states)} jokers; K={args.samples} +{args.max_antes} antes",
          flush=True)
    if len(states) < 3:
        print("[quality] not enough states")
        return 1

    jobs = [(st, si, args.samples, args.rollout_bot, args.max_antes)
            for si, st in enumerate(states)]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_state_job, jobs))
    else:
        results = [_state_job(j) for j in jobs]

    demA = [x for r in results for x in r["demA"]]
    demB = [x for r in results for x in r["demB"]]
    rawA = [x for r in results for x in r["rawA"]]
    rawB = [x for r in results for x in r["rawB"]]
    within_spread = statistics.mean(
        statistics.pstdev(r["full_demeaned"]) for r in results if len(r["full_demeaned"]) > 1)

    out = {
        "states": len(states), "samples": args.samples, "max_antes": args.max_antes,
        "n_jokers": len(demA),
        "demeaned_split_half_corr": round(_pearson(demA, demB), 4),
        "raw_split_half_corr": round(_pearson(rawA, rawB), 4),
        "mean_within_state_delta_spread": round(within_spread, 4),
    }
    print("[quality] RESULT:", json.dumps(out, indent=2), flush=True)
    # qualitative: show a few states' per-joker demeaned Δ (which jokers rate high/low)
    for r in results[:4]:
        ranked = sorted(zip(r["names"], r["full_demeaned"]), key=lambda t: t[1], reverse=True)
        pretty = ", ".join(f"{n}:{d:+.2f}" for n, d in ranked)
        print(f"  ante{r['ante']}: {pretty}", flush=True)

    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump({**out, "examples": [
            {"ante": r["ante"], "names": r["names"], "demeaned": r["full_demeaned"]}
            for r in results]}, fh, indent=2)
    open(f".data/_JOKERQUAL_dem{out['demeaned_split_half_corr']:.2f}"
         f"_raw{out['raw_split_half_corr']:.2f}", "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
