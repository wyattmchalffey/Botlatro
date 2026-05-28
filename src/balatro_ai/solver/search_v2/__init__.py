"""Solver-specific search infrastructure (Tier 1 of SOLVER_OPTIMIZATION_PLAN.md).

The `search_v2` package is the rewrite of the play/shop search that the
M1-M6 solver wrapped from `balatro_ai.search.hand_search`. The wrapped
version was correct but inherited live-bot preprocessing that cost
~6-8s per play decision; this package replaces it with a leaner beam
that has no preprocessing shortcut and lets the leaf evaluator be the
only "expensive" part of each ply.

Modules:
- `play`       - `solver_beam_play_action`, the custom whole-blind beam.
- `leaf_value` - leaf evaluators (`LeafEvaluator`, `FastHeuristicLeaf`,
                 `ClearProbabilityLeaf`, `FutureBlindSurvivalLeaf`,
                 `ArchetypeAwareLeaf`).

See `SOLVER_OPTIMIZATION_PLAN.md` §3 for the full design rationale.
"""

from balatro_ai.solver.search_v2.leaf_value import (
    ArchetypeAwareLeaf,
    ClearProbabilityLeaf,
    FastHeuristicLeaf,
    FutureBlindSurvivalLeaf,
    LeafEvaluator,
    PlanningValueLeaf,
)
from balatro_ai.solver.search_v2.play import (
    CandidateProvider,
    SearchV2PlayPolicy,
    TopKByImmediateScore,
    solver_beam_play_action,
)

__all__ = [
    "ArchetypeAwareLeaf",
    "CandidateProvider",
    "ClearProbabilityLeaf",
    "FastHeuristicLeaf",
    "FutureBlindSurvivalLeaf",
    "LeafEvaluator",
    "PlanningValueLeaf",
    "SearchV2PlayPolicy",
    "TopKByImmediateScore",
    "solver_beam_play_action",
]
