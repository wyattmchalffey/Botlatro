# cython: language_level=3, boundscheck=False, wraparound=False
"""Native (Cython) port of `hand_evaluator` hot path (Tier 2 #4).

This module is a scaffolding placeholder for now — it exposes a single
`is_native_available()` smoke-test function so the build infrastructure
can be validated before any real logic is ported.

The plan, per SOLVER_OPTIMIZATION_PLAN.md §4 #4:
- `_identify_hand_type` → `identify_hand_type_native` (Tier 2 #4b)
- `_scoring_indices` → `scoring_indices_native` (Tier 2 #4c)
- `_card_chip_value` → `card_chip_value_native` (Tier 2 #4d)
- `_effect_adjustments` → `effect_adjustments_native` (Tier 2 #4d)

Each port lives behind a try/except import in `hand_evaluator.py`
so the project still runs in a Python-only mode when Cython isn't
compiled (or the build fails). A parity test in
`tests/test_hand_evaluator_native_parity.py` runs both
implementations on every audit transition and asserts identical
output — the correctness gate that prevents silent regressions
from a porting bug.
"""


def is_native_available() -> bool:
    """Smoke-test entry point. Returns True if the module imported cleanly.

    Used by `hand_evaluator.py` to decide whether to dispatch to the
    native or Python implementation at runtime.
    """

    return True
