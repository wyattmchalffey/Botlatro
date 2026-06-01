"""Rollout distillation (Stage 1.3, Option A).

Trains a net to reproduce the play beam's `clear_probability` leaf — the exact
within-blind signal it needs — at the net's speed, on the **beam's own state
distribution**. This attacks both reasons the ante-head leaf was worse:

- **Granularity:** labels are the rollout leaf's own output (clear-prob /
  cleared-bonus), so the net learns to discriminate exactly the within-blind
  choices the beam faces — not a coarse whole-run value.
- **Distribution shift:** `CollectingClearLeaf` records the states the *beam*
  actually evaluates (its hypothetical play lines), so the net is trained on the
  distribution it will be queried on.

Ceiling note: distillation can only *match* the rollout, never beat it — this
buys speed (a cheap leaf), not strength. Strength comes later (policy head +
self-play). A fast accurate leaf is the prerequisite that makes both affordable.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
from torch import nn

from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.ml.encoding import EncodedState, encode_state
from balatro_ai.ml.model import ValueNet, collate_states
from balatro_ai.solver.search_v2.leaf_value import ClearProbabilityLeaf

Pair = tuple[EncodedState, float]


class CollectingClearLeaf:
    """A `clear_probability` leaf that records `(encoded_state, value)` per call.

    Terminal states (won / run-over) are skipped — the deployed `ValueNetLeaf`
    special-cases those (2.0 / 0.0); the net only needs to learn the in-between.
    """

    def __init__(self, sink: list[Pair], *, samples: int = 4, seed: int = 0) -> None:
        self.base = ClearProbabilityLeaf(samples=samples, seed=seed)
        self.sink = sink

    def evaluate(self, state: GameState) -> float:
        value = self.base.evaluate(state)
        if not (state.won or state.run_over or state.phase == GamePhase.RUN_OVER):
            self.sink.append((encode_state(state), float(value)))
        return value


def collect_distill_pairs(
    seeds: Sequence[str],
    *,
    depth: int = 3,
    width: int = 2,
    stake: str = "white",
    max_states: int | None = None,
) -> list[Pair]:
    """Run the v2 play beam over `seeds`, recording every leaf state it scores."""
    from balatro_ai.api.actions import ActionType
    from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.search_v2.play import SearchV2PlayPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    sink: list[Pair] = []
    leaf = CollectingClearLeaf(sink)
    play_policy = SearchV2PlayPolicy(
        depth=depth, width=width, leaf_evaluator=leaf, seed=0,
        fallback=BasicStrategyBot(seed=0))
    solver = SolverPolicy(play_policy=play_policy, play_backend="v2",
                          play_depth=depth, play_width=width, seed=0)
    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake=stake)
        sim.state = SeedGame(seed, stake=stake).initial_state()
        for _ in range(2000):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            action = solver.choose_action(st)
            if action.action_type == ActionType.NO_OP:
                break
            sim.step(action)
        if max_states is not None and len(sink) >= max_states:
            break
    return sink


def train_distill(
    pairs: Sequence[Pair],
    *,
    epochs: int = 15,
    lr: float = 1e-2,
    batch_size: int = 512,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    seed: int = 0,
) -> ValueNet:
    """Train `ValueNet.clear_head` to regress the rollout leaf values."""
    if not pairs:
        raise ValueError("train_distill needs at least one pair")
    torch.manual_seed(seed)
    states = [p[0] for p in pairs]
    labels = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    model = ValueNet(dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    n = len(states)
    bs = batch_size if batch_size and batch_size > 0 else n
    rng = random.Random(seed)
    model.train()
    for _ in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            batch = collate_states([states[i] for i in idx])
            opt.zero_grad()
            loss = mse(model.clear_value(batch), labels[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def eval_distill(model: ValueNet, pairs: Sequence[Pair]) -> dict:
    """MSE + Pearson correlation between the net and the rollout labels."""
    model.eval()
    states = [p[0] for p in pairs]
    labels = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    pred = model.clear_value(collate_states(states))
    mse = float(nn.MSELoss()(pred, labels))
    a, b = pred.tolist(), labels.tolist()
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    corr = num / (da * db) if da and db else float("nan")
    return {"n": n, "mse": mse, "corr": corr,
            "label_mean": round(mb, 4), "pred_mean": round(ma, 4)}
