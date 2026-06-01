"""Wire the neural components into the v2 beam (Stage 2.3).

The v2 play search (`solver_beam_play_action` / `SearchV2PlayPolicy`) already
takes an injectable `candidate_provider` (proposes/ranks candidate plays) and
`leaf_evaluator` (scores leaves). So a neural-guided beam needs no new search
code — just:

- `PolicyCandidateProvider`: ranks the legal play candidates with the trained
  play-policy (`ValueNet.play_candidate_scores`) and returns the top few, so the
  beam expands only the policy's promising candidates (the AlphaZero prior).
- the distilled value leaf (`ValueNetLeaf(head="clear")`) for leaf evaluation.

`neural_play_policy()` assembles both into a `SearchV2PlayPolicy` drop-in.
Discards keep the heuristic ranking (the play policy was trained on plays).
"""

from __future__ import annotations

import torch

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.ml.encoding import encode_state
from balatro_ai.ml.model import ValueNet, collate_states
from balatro_ai.ml.policy import _classify_hand


class PolicyCandidateProvider:
    """v2 `CandidateProvider`: rank play candidates by the learned play policy."""

    def __init__(self, model: ValueNet, *, play_oversample: int = 2, discard_share: int = 1) -> None:
        self.model = model
        self.model.eval()
        self.play_oversample = play_oversample
        self.discard_share = discard_share

    def __call__(self, state: GameState, *, width: int) -> tuple[Action, ...]:
        plays = tuple(
            a for a in state.legal_actions
            if a.action_type == ActionType.PLAY_HAND and a.card_indices)
        discards = tuple(
            a for a in state.legal_actions
            if a.action_type == ActionType.DISCARD and a.card_indices)
        ranked_plays = self._rank_plays(state, plays, max(1, width * self.play_oversample))
        # Discards: reuse the heuristic ranker (play policy doesn't score discards).
        from balatro_ai.solver.search_v2.play import _rank_discards
        ranked_discards = _rank_discards(state, discards, limit=max(0, width * self.discard_share))
        return (*ranked_plays, *ranked_discards)

    @torch.no_grad()
    def _rank_plays(self, state: GameState, plays: tuple[Action, ...], limit: int) -> tuple[Action, ...]:
        if not plays or limit <= 0:
            return ()
        if len(plays) <= limit:
            return plays
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
                    [hand[j].rank_index for j in idxs],
                    [hand[j].suit_index for j in idxs])
        scores = self.model.play_candidate_scores(batch, masks, htypes, sizes)[0].tolist()
        order = sorted(range(c), key=lambda k: scores[k], reverse=True)
        return tuple(plays[k] for k in order[:limit])


def neural_play_policy(
    value_ckpt: str,
    policy_ckpt: str,
    *,
    depth: int = 3,
    width: int = 2,
    leaf_head: str = "clear",
    seed: int = 0,
):
    """A `SearchV2PlayPolicy` driven by the learned play policy + value leaf."""
    from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
    from balatro_ai.ml.leaf import ValueNetLeaf
    from balatro_ai.ml.train import load_checkpoint
    from balatro_ai.solver.search_v2.play import SearchV2PlayPolicy

    leaf = ValueNetLeaf(value_ckpt, head=leaf_head)
    provider = PolicyCandidateProvider(load_checkpoint(policy_ckpt))
    return SearchV2PlayPolicy(
        depth=depth, width=width, leaf_evaluator=leaf,
        candidate_provider=provider, seed=seed, fallback=BasicStrategyBot(seed=seed))
