"""Diagnostic: does PolicyCandidateProvider rank plays sensibly, or near-random?

Steps a sim a few decisions in, grabs SELECTING_HAND states with many legal
plays, and compares the policy's ranking to the immediate-score heuristic's:
- agreement@1 : do they pick the same top play?
- spearman-ish: rank-correlation of the two orderings over the candidates.
- shows the top-3 of each for eyeballing.

Near-zero agreement/correlation => wiring/alignment bug. Positive-but-imperfect
=> the policy learned but is a weaker ranker than immediate-score (a real finding).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _immediate_order(state, plays):
    """Order plays (as indices into `plays`) by the heuristic immediate ranker."""
    from balatro_ai.solver.search_v2.play import _rank_plays
    ranked = _rank_plays(state, tuple(plays), limit=len(plays))
    key_to_idx = {tuple(p.card_indices): i for i, p in enumerate(plays)}
    return [key_to_idx[tuple(a.card_indices)] for a in ranked]


def _policy_order(provider, state, plays):
    import torch

    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.model import collate_states
    from balatro_ai.ml.policy import _classify_hand

    enc = encode_state(state)
    hand = enc.hand
    batch = collate_states([enc])
    h = batch.card_mask.shape[1]
    c = len(plays)
    masks = torch.zeros(1, c, h)
    htypes = torch.zeros(1, c, dtype=torch.long)
    sizes = torch.zeros(1, c)
    for k, a in enumerate(plays):
        idxs = [j for j in a.card_indices if 0 <= j < len(hand)]
        for j in idxs:
            masks[0, k, j] = 1.0
        sizes[0, k] = len(idxs)
        if idxs:
            htypes[0, k] = _classify_hand(
                [hand[j].rank_index for j in idxs], [hand[j].suit_index for j in idxs])
    with torch.no_grad():
        scores = provider.model.play_candidate_scores(batch, masks, htypes, sizes)[0].tolist()
    order = sorted(range(c), key=lambda k: scores[k], reverse=True)
    return order, scores


def _rank_corr(order_a, order_b):
    """Spearman rank correlation between two orderings of the same items."""
    n = len(order_a)
    rank_a = {item: i for i, item in enumerate(order_a)}
    rank_b = {item: i for i, item in enumerate(order_b)}
    d2 = sum((rank_a[i] - rank_b[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else 1.0


def main() -> int:
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.ml.neural_search import PolicyCandidateProvider
    from balatro_ai.ml.train import load_checkpoint
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    provider = PolicyCandidateProvider(load_checkpoint(".data/phase8_playpolicy_v0.pt"))
    driver = SolverPolicy(play_backend="v2", play_depth=1, play_width=1, seed=0)

    samples, agree1, agree3, corrs = 0, 0, 0, []
    for seed in ("0000001", "0000002", "0000003"):
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        for _ in range(120):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            if st.phase == GamePhase.SELECTING_HAND:
                plays = [a for a in st.legal_actions
                         if a.action_type == ActionType.PLAY_HAND and a.card_indices]
                if len(plays) >= 6:
                    io = _immediate_order(st, plays)
                    po, _ = _policy_order(provider, st, plays)
                    samples += 1
                    agree1 += int(io[0] == po[0])
                    agree3 += int(io[0] in po[:3])
                    corrs.append(_rank_corr(io, po))
                    if samples <= 4:
                        print(f"  ante{st.ante} cands={len(plays)} "
                              f"imm_top={list(plays[io[0]].card_indices)} "
                              f"pol_top={list(plays[po[0]].card_indices)} "
                              f"corr={corrs[-1]:.2f}")
            sim.step(driver.choose_action(st))

    n = max(1, samples)
    print(f"\nsamples={samples}  agree@1={agree1/n:.2f}  "
          f"imm_top_in_policy_top3={agree3/n:.2f}  "
          f"mean_rank_corr={sum(corrs)/len(corrs):.3f}" if corrs else "no samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
