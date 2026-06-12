"""Flip diagnostic for the deep-play delegation gate.

Reads the gate's per-pair rows JSONL (env_paired_ab 6th arg), finds the seeds
whose win outcome FLIPPED between control and treatment, and reruns each
flipped seed in treatment mode with delegation tracing: where the deep beam
took over, what it chose vs what basic play would have chosen, whether the
delegated blind cleared, and where the run ultimately diverged/died.

Categorizes each flip:
  - rescued:        a delegated blind the control run died at was cleared
  - butterfly:      first delegation happened at/before a blind BOTH cleared,
                    and the outcome change traces to downstream trajectory
                    divergence (different shops/draws), not the blind itself
  - beam_failed:    a delegated must-clear blind was NOT cleared
  - never_delegated: outcome differed with zero delegations (pure env noise -
                    should not happen; flags a harness bug)

    PYTHONPATH=src python scripts/deep_play_flip_diag.py \
        --rows .data/deep_play_gate_rows.jsonl --jobs 8 \
        --out .data/deep_play_flip_diag.json

Run with the SAME env the gate used (BALATRO_NO_FORESIGHT=shuffle,
BALATRO_DEEP_PLAY_DEPTH/WIDTH/BUDGET); the script sets BALATRO_DEEP_PLAY_ANTE
itself per arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _trace_worker(task) -> dict:
    seed, direction = task
    from dataclasses import replace

    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    os.environ["BALATRO_DEEP_PLAY_ANTE"] = "1"
    bot = create_bot("solver_shop_basic_play_bot", seed=0)
    events: list[dict] = []
    orig = type(bot)._deep_play_action

    def traced(self, state):
        action = orig(self, state)
        if action is not None:
            basic = self._fallback.choose_action(state)
            events.append(
                {
                    "ante": int(state.ante),
                    "blind": str(state.blind or ""),
                    "hands": int(state.hands_remaining),
                    "discards": int(state.discards_remaining),
                    "score": int(state.current_score),
                    "required": int(state.required_score),
                    "beam": [str(action.action_type.value), list(action.card_indices)],
                    "basic_would": [str(basic.action_type.value), list(basic.card_indices)],
                    "agree": (
                        action.action_type == basic.action_type
                        and tuple(action.card_indices) == tuple(basic.card_indices)
                    ),
                }
            )
        return action

    type(bot)._deep_play_action = traced
    blinds_seen: list[tuple[int, str, bool]] = []  # (ante, blind, cleared)
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    prev_blind: tuple[int, str] | None = None
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(6000):
            st = sim.state
            if st.run_over:
                break
            if st.blind and st.phase.value in ("selecting_hand", "playing_blind"):
                prev_blind = (int(st.ante), str(st.blind))
            elif prev_blind is not None and st.phase.value in ("shop", "round_eval", "blind_select"):
                blinds_seen.append((*prev_blind, True))  # left the blind alive => cleared
                prev_blind = None
            action = bot.choose_action(st)
            if action is None or action.action_type.value == "no_op":
                break
            sim.step(action)
    final = sim.state
    if prev_blind is not None and not final.won:
        blinds_seen.append((*prev_blind, False))  # died inside this blind

    delegated_blinds = {(e["ante"], e["blind"]) for e in events}
    death = next(((a, b) for a, b, cleared in blinds_seen if not cleared), None)
    delegated_death = death is not None and death in delegated_blinds
    delegated_cleared = sum(
        1 for a, b, cleared in blinds_seen if cleared and (a, b) in delegated_blinds
    )

    if not events:
        category = "never_delegated"
    elif direction == "lost" and delegated_death:
        category = "beam_failed"
    elif direction == "gained" and delegated_cleared > 0:
        category = "rescued_or_helped"
    else:
        category = "butterfly"

    return {
        "seed": seed,
        "direction": direction,
        "category": category,
        "won": bool(final.won),
        "final_ante": int(final.ante),
        "n_delegations": len(events),
        "n_disagreements": sum(1 for e in events if not e["agree"]),
        "delegated_blinds": sorted(delegated_blinds),
        "delegated_cleared": delegated_cleared,
        "death_blind": death,
        "death_was_delegated": delegated_death,
        "first_delegation": events[0] if events else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8") if l.strip()]
    gained = [r["seed"] for r in rows if r["treat"]["won"] and not r["base"]["won"]]
    lost = [r["seed"] for r in rows if r["base"]["won"] and not r["treat"]["won"]]
    print(f"[flip-diag] {len(rows)} pairs: {len(gained)} gained, {len(lost)} lost", flush=True)
    tasks = [(s, "gained") for s in gained] + [(s, "lost") for s in lost]
    if not tasks:
        print("[flip-diag] no flips to diagnose")
        return 0

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_trace_worker, tasks))
    else:
        results = [_trace_worker(t) for t in tasks]

    by_direction: dict[str, Counter] = {"gained": Counter(), "lost": Counter()}
    for r in results:
        by_direction[r["direction"]][r["category"]] += 1
    summary = {
        "n_gained": len(gained),
        "n_lost": len(lost),
        "categories": {d: dict(c) for d, c in by_direction.items()},
        "mean_delegations_gained": (
            round(sum(r["n_delegations"] for r in results if r["direction"] == "gained") / max(1, len(gained)), 1)
        ),
        "mean_delegations_lost": (
            round(sum(r["n_delegations"] for r in results if r["direction"] == "lost") / max(1, len(lost)), 1)
        ),
        "flips": results,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "flips"}, indent=1))
    print(f"[flip-diag] detail written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
