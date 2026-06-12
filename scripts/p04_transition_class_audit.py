"""P0.4: per-transition-class sim->bridge validation (the never-audited classes).

The historical divergence audit validated play/discard/sell/reroll/end_shop to
99.9% but SKIPPED: BUY (~2186 transitions), OPEN_PACK / CHOOSE_PACK_CARD
(~1061), SELECT_BLIND / SKIP_BLIND, stochastic-joker plays (~1046), and boss
blinds with stochastic effects (~73), and bridge validation stopped near
ante 5. Whole-run pass/fail (sim_bridge_replay_check.py) can't attribute a
mismatch to a class; this tool can.

Two modes:

--scan N [--offset K]   (sim-only; no bridge needed)
    Run N seed-faithful clairvoyant trajectories (the replayable kind) and
    rank seeds by how richly they cover the under-audited classes at antes
    6+. Emits a worklist JSON so bridge time gets spent on the most
    informative seeds.

--replay SEED [SEED ...]   (bridge must be up; see reference_bridge_launch)
    Drive each seed's recorded trajectory on the live bridge, comparing
    pre-action state per step (same comparison as sim_bridge_replay_check),
    but AGGREGATE match/mismatch per transition class and report per-class
    totals across all seeds, plus each seed's first divergence.

    PYTHONPATH=src python scripts/p04_transition_class_audit.py --scan 64 \
        --out .data/p04_scan_worklist.json
    PYTHONPATH=src python scripts/p04_transition_class_audit.py \
        --replay 0000012 0000031 ... --out .data/p04_class_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["BALATRO_SEED_FAITHFUL"] = "1"

# Jokers whose effects roll RNG during play/round resolution: a PLAY_HAND
# while one is owned exercises per-play RNG surfaces the audit never covered.
STOCHASTIC_JOKERS = frozenset(
    {
        "Bloodstone", "Space Joker", "8 Ball", "Misprint", "Business Card",
        "Reserved Parking", "Gros Michel", "Cavendish", "Glass Joker",
        "Hallucination", "Madness", "Seance", "Sixth Sense", "Vagabond",
    }
)
SHOWDOWN_BOSSES = frozenset(
    {"Cerulean Bell", "Verdant Leaf", "Violet Vessel", "Amber Acorn", "Crimson Heart"}
)


def _classify(action, state) -> list[str]:
    """Transition classes a (state, action) step exercises (possibly several)."""
    kind = action.action_type.value
    classes: list[str] = []
    if kind == "buy":
        classes.append("buy")
    elif kind == "open_pack":
        classes.append("open_pack")
    elif kind == "choose_pack_card":
        classes.append("choose_pack_card")
    elif kind == "select_blind":
        classes.append("select_blind")
    elif kind == "skip_blind":
        classes.append("skip_blind")
    elif kind == "play_hand":
        joker_names = {j.name for j in state.jokers}
        if joker_names & STOCHASTIC_JOKERS:
            classes.append("play_stochastic_joker")
        blind = str(state.blind or "")
        if blind in SHOWDOWN_BOSSES:
            classes.append("play_showdown_boss")
        elif blind.startswith("The "):
            classes.append("play_boss")
    if state.ante >= 6:
        classes = [f"{c}@a6+" for c in classes] + classes
    return classes


def _trajectory(seed: str, max_steps: int = 3000):
    from sim_bridge_replay_check import run_sim  # reuse the validated plumbing

    return run_sim(seed, max_steps=max_steps)


def _scan(n: int, offset: int, out: str) -> int:
    rows = []
    for i in range(1, n + 1):
        seed = f"{offset + i:07d}"
        final, records = _trajectory(seed)
        counts: Counter[str] = Counter()
        # records hold (action, pre_summary); re-simulate classes from the sim
        # pass directly: run_sim discards the live state, so re-tag by replay.
        from balatro_ai.api.state import with_derived_legal_actions
        from balatro_ai.sim.local_runner import LocalBalatroSimulator
        from balatro_ai.solver.seed_game import SeedGame
        from balatro_ai.solver.trajectory import _stable_seed_int

        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
        sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
        for action, _ in records:
            for cls in _classify(action, sim.state):
                counts[cls] += 1
            try:
                sim.step(action)
            except Exception:  # noqa: BLE001
                break
        # coverage score: late-ante under-audited classes weighted highest
        score = (
            5 * counts.get("buy@a6+", 0)
            + 5 * counts.get("open_pack@a6+", 0)
            + 5 * counts.get("choose_pack_card@a6+", 0)
            + 4 * counts.get("play_showdown_boss@a6+", 0)
            + 3 * counts.get("play_stochastic_joker@a6+", 0)
            + 2 * counts.get("skip_blind", 0)
            + 1 * counts.get("buy", 0)
            + 1 * counts.get("open_pack", 0)
        )
        rows.append(
            {"seed": seed, "final_ante": int(final.ante), "won": bool(final.won),
             "steps": len(records), "coverage_score": score, "classes": dict(counts)}
        )
        print(f"  [{i}/{n}] {seed}: ante {final.ante}, score {score}", flush=True)
    rows.sort(key=lambda r: -r["coverage_score"])
    totals: Counter[str] = Counter()
    for r in rows:
        totals.update(r["classes"])
    result = {"totals": dict(totals), "ranked_seeds": rows}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(dict(totals), indent=1))
    print(f"[scan] top-8 worklist: {[r['seed'] for r in rows[:8]]}")
    print(f"[scan] written to {out}")
    return 0


def _replay(seeds: list[str], out: str) -> int:
    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from sim_bridge_replay_check import replay_on_bridge

    per_class: dict[str, Counter] = defaultdict(Counter)
    details = []
    for seed in seeds:
        final, records = _trajectory(seed)
        # tag every step
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
        sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
        step_classes = []
        for action, _ in records:
            step_classes.append(_classify(action, sim.state))
            try:
                sim.step(action)
            except Exception:  # noqa: BLE001
                step_classes[-1] = step_classes[-1] or ["untagged"]
                break
        bridge_final, first_diff, n = replay_on_bridge(seed, records)
        diverged_at = first_diff[0] if first_diff else None
        for i, classes in enumerate(step_classes[:n]):
            ok = diverged_at is None or i < diverged_at
            for cls in classes:
                per_class[cls]["match" if ok else "after_divergence"] += 1
        if first_diff is not None:
            i, key, sim_pre, bridge_pre, atype, idx = first_diff
            blame = step_classes[i] if i < len(step_classes) else []
            for cls in blame:
                per_class[cls]["first_divergence"] += 1
            details.append(
                {"seed": seed, "diverged_at": i, "field": key, "action": [atype, list(idx)],
                 "classes_at_divergence": blame, "sim_pre": sim_pre, "bridge_pre": bridge_pre}
            )
            print(f"  {seed}: DIVERGED step {i} ({atype}) classes={blame}", flush=True)
        else:
            details.append({"seed": seed, "diverged_at": None,
                            "outcome": {"sim_ante": int(final.ante), "bridge_ante": int(bridge_final.ante)}})
            print(f"  {seed}: full match ({n} steps)", flush=True)
    result = {"per_class": {k: dict(v) for k, v in sorted(per_class.items())}, "seeds": details}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result["per_class"], indent=1))
    print(f"[replay] written to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--replay", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(__file__))  # for sim_bridge_replay_check import
    if args.scan:
        return _scan(args.scan, args.offset, args.out)
    if args.replay:
        return _replay(args.replay, args.out)
    ap.error("one of --scan N or --replay SEED... is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
