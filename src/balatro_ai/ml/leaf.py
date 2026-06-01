"""Learned leaf evaluator for the solver beam (Stage 1.3).

`ValueNetLeaf` implements the `solver.search_v2.leaf_value.LeafEvaluator`
protocol (duck-typed: `evaluate(state) -> float`) using a trained `ValueNet`
checkpoint. It is the NNUE-analog drop-in for the `clear_probability` rollout
leaf — a single cheap forward pass instead of multiple greedy playouts.

Kept in the `ml/` layer (not `search_v2/leaf_value.py`) so the solver's normal
code path stays torch-free; this module is only imported when a learned leaf is
actually used (A/B experiments, eventual production wiring).

Leaf-value convention (matches the rollout leaves): won -> 2.0, run-over -> 0.0,
otherwise the net's value head in [0, 1]. The default head is `"ante"` (expected
final ante / 8) — a denser, better-generalizing signal than the rare binary win.
"""

from __future__ import annotations

import torch

from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.ml.encoding import encode_state
from balatro_ai.ml.model import ValueNet, collate_states
from balatro_ai.ml.train import load_checkpoint


class ValueNetLeaf:
    """Beam leaf evaluator backed by a trained `ValueNet` checkpoint."""

    def __init__(
        self,
        checkpoint: str | ValueNet,
        *,
        head: str = "ante",
    ) -> None:
        if isinstance(checkpoint, ValueNet):
            self.model = checkpoint
        else:
            self.model = load_checkpoint(checkpoint)
        self.model.eval()
        if head not in ("ante", "win"):
            raise ValueError(f"head must be 'ante' or 'win', got {head!r}")
        self.head = head
        # Single-thread the per-leaf forward pass: batch-size-1 inference has
        # no parallelism to exploit and thread dispatch is pure overhead.
        torch.set_num_threads(1)

    def evaluate(self, state: GameState) -> float:
        if state.won:
            return 2.0
        if state.run_over or state.phase == GamePhase.RUN_OVER:
            return 0.0
        batch = collate_states([encode_state(state)])
        with torch.no_grad():
            value = (
                self.model.ante_value(batch) if self.head == "ante"
                else self.model.win_prob(batch)
            )
        return float(value[0])
