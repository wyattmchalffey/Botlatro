"""Value-in-search 1-ply test (the operator PROGRESS.md 2026-06-14 pre-registered).

Arm A (v1ply): B0 proposes; on the play surface (SELECTING_HAND) the top-K
policy candidates are expanded one ply with forward_sim (draws sampled from the
BLINDED DeckModel belief — honest, and shared across candidates within a
decision for common-random-numbers variance reduction) and re-ranked by V0's
state-only value head. Every other phase is B0 argmax, byte-identical to Arm B.

Arm B (b0): pure B0 argmax (neural_policy_bot), the cloud-gate baseline.

The paired delta therefore isolates exactly one question: does the learned
value, used as 1-ply search on the play surface, beat the policy head's own
ranking? Play is where the measured recoverable headroom lives (fork-audit:
~25-30% of losses have a clearing line) and where afterstates are cleanly
simulable.

    PYTHONPATH=src BALATRO_NO_FORESIGHT=shuffle python scripts/phaseb_gate_value1ply.py \
        --b0 .data/cloud_b0.pt --v0 .data/cloud_v0.pt \
        --seeds 1024 --eval-offset 5300000 --jobs 10 --topk 8 \
        --result .data/value1ply_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _state_win_prob(net, state) -> float:
    """V(state) from the value head (state-only; candidates don't feed it)."""
    import torch

    from balatro_ai.ml.dataset import CandidateToken, TrainingExample, ValueTarget
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.policy_net import collate_candidates

    dummy = CandidateToken(
        action_type_index=0, n_cards=0.0, amount=0.0, has_target=0.0,
        play_score=0.0, has_play_score=0.0, heuristic_choice=0.0,
    )
    ex = TrainingExample(
        step=0, phase="", encoded_state=encode_state(state), action={},
        value=ValueTarget(False, 0, 0), steps_to_end=0,
        candidates=(dummy,), chosen_index=0,
    )
    with torch.no_grad():
        _, win_logit = net.candidate_logits(collate_candidates([ex]))
    return float(torch.sigmoid(win_logit[0]))


class Value1PlyBot:
    """B0 argmax everywhere except SELECTING_HAND, where top-K policy
    candidates are 1-ply-expanded and re-ranked by V0."""

    name = "value1ply_bot"

    def __init__(self, b0_ckpt: str, v0_ckpt: str, *, topk: int = 8,
                 min_margin: float = 0.0, seed: int = 0):
        from balatro_ai.bots.neural_policy import NeuralPolicyBot
        from balatro_ai.ml.policy_net import load_policy

        self._inner = NeuralPolicyBot(seed=seed, ckpt=b0_ckpt)
        self._b0 = load_policy(b0_ckpt)
        self._b0.eval()
        self._v0 = load_policy(v0_ckpt)
        self._v0.eval()
        self.topk = topk
        self.min_margin = min_margin  # override only if V beats base by this much
        self._decision_i = 0
        # instrumentation: how often the value overrode the policy argmax
        self.n_play_decisions = 0
        self.n_overrides = 0

    def _afterstate(self, state, action, rng_seed: int):
        """Honest one-ply afterstate: forward_sim with belief-sampled draws."""
        from random import Random

        from balatro_ai.api.actions import ActionType
        from balatro_ai.search.deck_model import DeckModel
        from balatro_ai.search.forward_sim import simulate_discard, simulate_play

        drawn = ()
        n_draw = len(action.card_indices or ())
        if n_draw > 0:
            model = DeckModel.from_state(state)
            n_draw = min(n_draw, model.total_cards)
            if n_draw > 0:
                # One rng_seed per DECISION (not per action): all candidates
                # see the same replacement draws — CRN across the comparison.
                drawn = model.sample_draws(n_draw, Random(rng_seed))
        if action.action_type == ActionType.PLAY_HAND:
            return simulate_play(state, action, drawn_cards=drawn)
        return simulate_discard(state, action, drawn_cards=drawn)

    def _value(self, after) -> float:
        from balatro_ai.api.state import GamePhase
        from balatro_ai.search.state_value import _cash_out_leaf_state

        if after.won:
            return 1.0
        if after.run_over:
            return 0.0
        cleared = after.phase == GamePhase.ROUND_EVAL or (
            after.required_score > 0 and after.current_score >= after.required_score
        )
        if cleared:
            return _state_win_prob(self._v0, _cash_out_leaf_state(after))
        if after.phase == GamePhase.SELECTING_HAND and after.hands_remaining <= 0:
            return 0.0  # out of hands, blind not cleared -> dead
        return _state_win_prob(self._v0, after)

    def choose_action(self, state):
        import torch

        from balatro_ai.api.actions import ActionType
        from balatro_ai.api.state import GamePhase
        from balatro_ai.bots.no_foresight import blind_known_deck
        from balatro_ai.ml.dataset import candidate_tokens_for_state
        from balatro_ai.ml.encoding import encode_state
        from balatro_ai.ml.policy_net import candidate_logit_vector

        self._decision_i += 1
        base = self._inner.choose_action(state)  # B0 argmax w/ its own fallbacks
        st = blind_known_deck(state)
        if st.phase != GamePhase.SELECTING_HAND:
            return base
        # Pure RERANKING of the base action's peers: if B0 chose something other
        # than a play/discard (use_consumable = planets/tarots, sell), pass it
        # through untouched — forcing a play over a planet is a behavior change,
        # not a value comparison, and it wrecks hand-leveling.
        if base.action_type not in (ActionType.PLAY_HAND, ActionType.DISCARD):
            return base
        legals = st.legal_actions
        candidates = candidate_tokens_for_state(st)
        if not candidates or len(candidates) != len(legals):
            return base
        try:
            logits = candidate_logit_vector(self._b0, encode_state(st), candidates)
            order = torch.argsort(logits, descending=True).tolist()
        except Exception:  # noqa: BLE001 — experiment must never crash a run
            return base
        top = [
            i for i in order
            if legals[i] is not None
            and legals[i].action_type in (ActionType.PLAY_HAND, ActionType.DISCARD)
        ][: self.topk]
        if not top:
            return base
        self.n_play_decisions += 1
        rng_seed = self._decision_i  # deterministic, shared across candidates
        best_i, best_v, base_v = None, float("-inf"), None
        for i in top:
            try:
                v = self._value(self._afterstate(st, legals[i], rng_seed))
            except Exception:  # noqa: BLE001 — infeasible candidate, skip
                continue
            if (base_v is None
                    and legals[i].action_type == base.action_type
                    and legals[i].card_indices == base.card_indices):
                base_v = v
            if v > best_v:
                best_i, best_v = i, v
        if best_i is None:
            return base
        # Confidence gate: only override when V's preference clears the margin
        # over the base action's own afterstate value (0.0 = pre-registered
        # pure argmax-V; >0 tests whether gating on V's noise floor rescues it).
        if self.min_margin > 0.0 and base_v is not None and best_v < base_v + self.min_margin:
            return base
        chosen = legals[best_i]
        if chosen.action_type != base.action_type or chosen.card_indices != base.card_indices:
            self.n_overrides += 1
        return chosen


def _eval_seed(task) -> tuple[bool, int, int, int]:
    """One seed, one arm. Determinism discipline mirrors phaseb_iter1_full."""
    seed, arm, b0, v0, topk, min_margin = task
    os.environ["BALATRO_DEVICE"] = "cpu"
    import random
    from dataclasses import replace

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    if arm == "v1ply":
        bot = Value1PlyBot(b0, v0, topk=topk, min_margin=min_margin)
    else:
        os.environ["BALATRO_POLICY_CKPT"] = b0
        bot = create_bot("neural_policy_bot", seed=0)
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(4000):
            st = sim.state
            if st.run_over:
                break
            a = bot.choose_action(st)
            if a.action_type.value == "no_op":
                break
            try:
                sim.step(a)
            except Exception:  # noqa: BLE001
                break
    n_play = getattr(bot, "n_play_decisions", 0)
    n_over = getattr(bot, "n_overrides", 0)
    return bool(sim.state.won), int(sim.state.ante), n_play, n_over


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", default=".data/cloud_b0.pt")
    ap.add_argument("--v0", default=".data/cloud_v0.pt")
    ap.add_argument("--seeds", type=int, default=1024)
    ap.add_argument("--eval-offset", type=int, default=5300000)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="override only when V(best) > V(base) + margin (0 = pure argmax-V)")
    ap.add_argument("--result", default=".data/value1ply_result.json")
    args = ap.parse_args()

    from balatro_ai.bench_stats import mcnemar_exact_p, paired_delta_ci, paired_mean_diff_ci

    seeds = [f"{args.eval_offset + i:07d}" for i in range(1, args.seeds + 1)]
    print(f"[v1ply] paired gate: {len(seeds)} seeds, topk={args.topk}, "
          f"b0={args.b0} v0={args.v0}", flush=True)
    results = {}
    for arm in ("v1ply", "b0"):
        tasks = [(s, arm, args.b0, args.v0, args.topk, args.min_margin) for s in seeds]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results[arm] = list(ex.map(_eval_seed, tasks))
        wins = sum(1 for w, _, _, _ in results[arm] if w)
        print(f"[v1ply] arm {arm}: {wins}/{len(seeds)} ({wins/len(seeds):.2%})", flush=True)

    n = len(seeds)
    a, b = results["v1ply"], results["b0"]
    a_w = sum(1 for w, _, _, _ in a if w)
    b_w = sum(1 for w, _, _, _ in b if w)
    gained = sum(1 for (x, *_), (y, *_) in zip(a, b) if x and not y)
    lost = sum(1 for (x, *_), (y, *_) in zip(a, b) if y and not x)
    p = mcnemar_exact_p(gained, lost)
    lo, hi = paired_delta_ci(gained, lost, n)
    ante_diffs = [ia - ib for (_, ia, *_), (_, ib, *_) in zip(a, b)]
    amean, alo, ahi = paired_mean_diff_ci(ante_diffs)
    tot_play = sum(np for _, _, np, _ in a)
    tot_over = sum(no for _, _, _, no in a)

    result = {
        "n_seeds": n, "eval_offset": args.eval_offset, "topk": args.topk,
        "min_margin": args.min_margin,
        "v1ply_wins": a_w, "v1ply_winrate": round(a_w / n, 4),
        "b0_wins": b_w, "b0_winrate": round(b_w / n, 4),
        "gained": gained, "lost": lost,
        "d_winrate": round((a_w - b_w) / n, 4),
        "d_winrate_ci": [round(lo, 4), round(hi, 4)],
        "mcnemar_p": round(p, 4),
        "mean_ante_delta": round(amean, 4),
        "mean_ante_delta_ci": [round(alo, 4), round(ahi, 4)],
        "play_decisions": tot_play, "value_overrides": tot_over,
        "override_rate": round(tot_over / max(1, tot_play), 4),
    }
    print(f"[v1ply] v1ply {a_w}/{n} ({a_w/n:.2%}) vs b0 {b_w}/{n} ({b_w/n:.2%}); "
          f"d={100*(a_w-b_w)/n:+.2f}pp (CI {100*lo:+.2f}..{100*hi:+.2f}), McNemar p={p:.3f}", flush=True)
    print(f"[v1ply] mean-ante {amean:+.3f} (CI {alo:+.3f}..{ahi:+.3f}); "
          f"value overrode policy on {tot_over}/{tot_play} play decisions "
          f"({tot_over/max(1,tot_play):.1%})", flush=True)
    with open(args.result, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[v1ply] result -> {args.result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
