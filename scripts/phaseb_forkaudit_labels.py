"""Dense fork-audit play labels (Fable-5 plan, day-1 search-improved targets).

The plan's designated fix for play-surface credit assignment (which the
iteration-1 null exposed): at the death blind of LOST runs, a deep d6w6 beam
clears 29.7% of them. Those clearing LINES are search-improved play targets —
the deep search found a better play than the bot took. We extract each
(state, deep-search action) pair on a CLEARING line as a high-weight training
example: clone the action the search found, not the one the bot played. This
is the AlphaZero pattern (search produces better targets than the policy) at
day-1 cost, on exactly the surface where credit assignment is tightest.

Emits a pickle of TrainingExamples (chosen_index = the deep-search action;
value.won=True since the line clears the must-clear blind) to MERGE into the
iteration-0/1 training set with elevated weight.

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python scripts/phaseb_forkaudit_labels.py \
        --caps .data/phaseb_mix_1000.jsonl --max-losses 300 --depth 6 --width 6 \
        --jobs 12 --out .data/phaseb_forkaudit_labels.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _clearing_line(task):
    """Fork a lost run at its death blind, drive the deep beam, and if it
    CLEARS, return the (encoded_state, candidates, chosen_index, value) tuples
    along the clearing line. Returns [] if it doesn't clear."""
    seed, stake, actions, depth, width = task
    import copy
    from dataclasses import replace as dcr

    from balatro_ai.api.actions import Action, ActionType
    from balatro_ai.api.state import GamePhase, with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.no_foresight import blind_known_deck
    from balatro_ai.ml.dataset import _candidate_tokens, TrainingExample, ValueTarget
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    active = {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND, GamePhase.ROUND_EVAL}
    left = {GamePhase.SHOP, GamePhase.BLIND_SELECT, GamePhase.RUN_OVER}

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake=stake)
    sim.state = SeedGame(seed, stake=stake).initial_state()
    last_fork = None
    prev_phase = sim.state.phase
    for adict in actions:
        try:
            sim.step(Action.from_mapping(adict))
        except Exception:  # noqa: BLE001
            break
        ph = sim.state.phase
        if prev_phase == GamePhase.BLIND_SELECT and ph in active:
            last_fork = (copy.deepcopy(sim), int(sim.state.required_score))
        prev_phase = ph
    if last_fork is None:
        return []
    osim, required = last_fork
    if required <= 0:
        return []
    start_ante = int(osim.state.ante)

    driver = SolverPolicy(play_backend="v2", play_depth=depth, play_width=width, seed=0)
    line: list[tuple] = []  # (encoded_state, candidates, chosen_index)
    with bot_config_scope(dcr(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(300):
            st = osim.state
            if st.run_over or st.phase in left or int(st.ante) > start_ante:
                break
            if st.phase not in active:
                break
            try:
                st_obs = blind_known_deck(st)
                a = driver.choose_action(st_obs)
            except Exception:  # noqa: BLE001
                break
            if a is None or a.action_type == ActionType.NO_OP:
                break
            # Record the SELECTING_HAND decisions (play/discard) — the play
            # label surface — with the deep-search action as the target.
            if st.phase == GamePhase.SELECTING_HAND:
                st_legal = with_derived_legal_actions(st_obs)
                candidates, chosen = _candidate_tokens(st_legal, a)
                if candidates and chosen >= 0:
                    line.append((encode_state(st_legal), candidates, chosen))
            try:
                osim.step(a)
            except Exception:  # noqa: BLE001
                break
            if int(osim.state.current_score) >= required:
                # Cleared the must-clear blind -> the line is search-verified.
                won = ValueTarget(won=True, final_ante=start_ante, final_score=int(osim.state.current_score))
                return [
                    TrainingExample(step=i, phase="selecting_hand", encoded_state=es,
                                    action={}, value=won, steps_to_end=len(line) - i,
                                    candidates=cands, chosen_index=ch)
                    for i, (es, cands, ch) in enumerate(line)
                ]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--max-losses", type=int, default=300)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default=".data/phaseb_forkaudit_labels.pkl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.caps, encoding="utf-8") if l.strip()]
    losses = [r for r in rows if not r.get("won")][: args.max_losses]
    print(f"[forkaudit] {len(losses)} losses -> deep d{args.depth}w{args.width} beam at death blinds",
          flush=True)
    tasks = [(r["seed"], r.get("stake", "white"), r["actions"], args.depth, args.width) for r in losses]
    examples = []
    n_cleared = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, line in enumerate(ex.map(_clearing_line, tasks), start=1):
            if line:
                n_cleared += 1
                examples.extend(line)
            if i % 50 == 0:
                print(f"[forkaudit] {i}/{len(losses)} losses, {n_cleared} cleared, "
                      f"{len(examples)} play-label examples", flush=True)
    with open(args.out, "wb") as fh:
        pickle.dump(examples, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[forkaudit] {n_cleared}/{len(losses)} clearable ({n_cleared/max(1,len(losses)):.1%}); "
          f"{len(examples)} search-improved play labels -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
