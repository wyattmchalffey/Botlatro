"""Value-guided shop leaf (Stage 2.4).

The shop beam (`best_shop_action`) scores leaf states with a `leaf_value_fn`.
By default that's the heuristic `shop_leaf_terms(...).total`. Here we swap in the
learned value head (expected-ante) so the *economy* decision — where heuristics
are crudest and the bot's lost winrate lives (dies ~ante 5-6 = weak late build) —
is guided by the net.

Scale matters: the beam computes `score = action_score + leaf_score * leaf_weight`
(`leaf_weight=0.35`), and `action_score` (per-action heuristic value) stays active.
The heuristic leaf `.total` is in heuristic units (tens-hundreds); the value head
is [0,1]. So we **z-score calibrate** the net leaf to the heuristic leaf's empirical
mean/std, making it a scale-neutral drop-in — the A/B then isolates *ranking
quality*, not a scale artifact.

Inject via `SolverPolicy(shop_leaf_value_fn=NeuralShopLeaf(model, calib))`.
"""

from __future__ import annotations

import statistics

import torch

from balatro_ai.api.state import GameState
from balatro_ai.ml.encoding import encode_state
from balatro_ai.ml.model import ValueNet, collate_states


def heuristic_total(state: GameState) -> float:
    """The heuristic shop leaf score for `state`, treated as its own root."""
    from balatro_ai.search.shop_search import _shop_build_score, shop_leaf_terms

    return shop_leaf_terms(
        state, root_state=state, root_build_score=_shop_build_score(state)).total


@torch.no_grad()
def neural_value(model: ValueNet, state: GameState, head: str = "ante") -> float:
    """Raw value-head estimate for `state` (no won/run_over special-casing)."""
    batch = collate_states([encode_state(state)])
    if head == "ante":
        return float(model.ante_value(batch)[0])
    if head == "win":
        return float(model.win_prob(batch)[0])
    if head == "clear":
        return float(model.clear_value(batch)[0])
    raise ValueError(f"unknown head: {head!r}")


def calibrate_scale(model: ValueNet, states, head: str = "ante") -> dict[str, float]:
    """Affine (z-score) params matching the net leaf to the heuristic leaf scale."""
    hs = [heuristic_total(s) for s in states]
    ns = [neural_value(model, s, head) for s in states]
    return {
        "mean_h": statistics.mean(hs), "std_h": statistics.pstdev(hs) or 1.0,
        "mean_n": statistics.mean(ns), "std_n": statistics.pstdev(ns) or 1.0,
        "n": len(states),
    }


class NeuralShopLeaf:
    """Shop `leaf_value_fn` factory backed by the value head, calibrated to the
    heuristic leaf scale. Call signature mirrors the archetype path:
    `(root_state) -> (leaf_state -> float)`.
    """

    def __init__(self, model: ValueNet, calib: dict[str, float], *, head: str = "ante") -> None:
        self.model = model
        self.model.eval()
        self.calib = calib
        self.head = head

    def __call__(self, root_state: GameState):
        c = self.calib

        def _value(leaf_state: GameState) -> float:
            z = (neural_value(self.model, leaf_state, self.head) - c["mean_n"]) / c["std_n"]
            return c["mean_h"] + z * c["std_h"]

        return _value
