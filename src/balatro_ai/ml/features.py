"""State -> feature vector for the Phase 8 learned value function.

Features are deliberately built from the bot's own build-strength signals
(role scores, sample build capacity, shop pressure) plus economy/progress/
resource context. The value model learns the calibrated, nonlinear map from
these to actual win probability — recalibrating where the heuristic pressure
formula is wrong, and capturing interactions it misses.

Pure-numpy friendly: returns a flat float tuple. Never raises — missing
signals fall back to neutral defaults so data generation can't crash.
"""

from __future__ import annotations

import math

from balatro_ai.api.state import GameState

FEATURE_NAMES: tuple[str, ...] = (
    "ante",
    "ante_frac",          # ante / 8
    "log_money",          # log1p(money)
    "money",
    "hands_remaining",
    "discards_remaining",
    "n_jokers",
    "open_joker_slots",
    "n_consumables",
    "n_vouchers",
    "chip_score",
    "mult_score",
    "xmult_score",
    "scaling_score",
    "economy_score",
    "has_chips",
    "has_mult",
    "has_xmult",
    "has_scaling",
    "has_economy",
    "n_missing_roles",
    "max_hand_level",
    "total_leveling",
    "log_build_capacity",  # log1p(sample build score)
    "pressure_ratio",      # target_score / build_capacity (capped)
    "pressure_danger",     # max(0, ratio-1) capped
)
N_FEATURES = len(FEATURE_NAMES)


def features_from_state(state: GameState) -> list[float]:
    """Rich value-function features for ``state``. Never raises."""

    ante = float(getattr(state, "ante", 1) or 1)
    money = float(getattr(state, "money", 0) or 0)
    hands = float(getattr(state, "hands_remaining", 0) or 0)
    discards = float(getattr(state, "discards_remaining", 0) or 0)
    jokers = getattr(state, "jokers", ()) or ()
    consumables = getattr(state, "consumables", ()) or ()
    vouchers = getattr(state, "vouchers", ()) or ()

    levels = [int(v) for v in (getattr(state, "hand_levels", {}) or {}).values()]
    max_level = float(max(levels)) if levels else 1.0
    total_leveling = float(sum(max(0, v - 1) for v in levels))

    # Build-strength signals from the bot's own profile / capacity / pressure.
    chip = mult = xmult = scaling = economy = 0.0
    has = {"chips": 0.0, "mult": 0.0, "xmult": 0.0, "scaling": 0.0, "economy": 0.0}
    n_missing = 0.0
    open_slots = max(0.0, 5.0 - len(jokers))
    try:
        from balatro_ai.bots.basic_strategy.build_profile import _build_profile
        p = _build_profile(state)
        chip, mult, xmult = p.chip_score, p.mult_score, p.xmult_score
        scaling, economy = p.scaling_score, p.economy_score
        has = {
            "chips": float(p.has_chips), "mult": float(p.has_mult),
            "xmult": float(p.has_xmult), "scaling": float(p.has_scaling),
            "economy": float(p.has_economy),
        }
        n_missing = float(len(p.missing_roles))
        open_slots = float(p.open_joker_slots)
    except Exception:  # noqa: BLE001
        pass

    build_capacity = 0.0
    try:
        from balatro_ai.bots.basic_strategy.build_scoring import _sample_build_score
        build_capacity = float(_sample_build_score(state, tuple(jokers)))
    except Exception:  # noqa: BLE001
        pass

    pressure_ratio = 1.0
    try:
        from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure
        pr = _shop_pressure(state)
        pressure_ratio = float(min(5.0, max(0.0, pr.ratio)))
    except Exception:  # noqa: BLE001
        pass

    return [
        ante,
        ante / 8.0,
        math.log1p(max(0.0, money)),
        money,
        hands,
        discards,
        float(len(jokers)),
        open_slots,
        float(len(consumables)),
        float(len(vouchers)),
        chip,
        mult,
        xmult,
        scaling,
        economy,
        has["chips"],
        has["mult"],
        has["xmult"],
        has["scaling"],
        has["economy"],
        n_missing,
        max_level,
        total_leveling,
        math.log1p(max(0.0, build_capacity)),
        pressure_ratio,
        max(0.0, min(4.0, pressure_ratio - 1.0)),
    ]
