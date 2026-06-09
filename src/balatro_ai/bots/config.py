"""Tunable parameters for BasicStrategyBot.

Defaults reproduce the behavior of the rule-coded constants and inline magic
numbers that previously lived inside `basic_strategy_bot.py`. Field defaults
must match the historical values so existing benchmarks remain reproducible
when no config is supplied.

Override a field by passing a `BotConfig` to `BasicStrategyBot(config=...)` or
by entering `bot_config_scope(config)` directly. Threading uses `threading.local`
to mirror the existing `decision_cache_scope` pattern.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import local
from typing import Iterator


@dataclass(frozen=True, slots=True)
class BotConfig:
    # Shop buy / reroll decision tolerance.
    shop_value_tolerance: float = 0.25

    # When True (default), shop decisions attach a full audit payload (re-runs the
    # planner for every option — ~2x the late-shop cost) for the offline eval tools
    # (decision_flip_audit / loss_audit / replay_analyzer). Set False in rollout
    # pilots / high-throughput label generation, which never read it.
    shop_audit_enabled: bool = True

    # Calibrated shop/build planner. The default keeps the legacy shop value as
    # the anchor and adds bounded, auditable adjustments around it.
    calibrated_shop_planner_enabled: bool = True
    calibrated_shop_legacy_weight: float = 1.0
    calibrated_shop_leaf_delta_weight: float = 0.35
    calibrated_shop_pressure_delta_weight: float = 0.65
    calibrated_shop_role_fill_weight: float = 1.0
    calibrated_shop_late_conversion_weight: float = 1.0
    calibrated_shop_reserve_risk_weight: float = 0.0
    calibrated_shop_slot_risk_weight: float = 1.0
    calibrated_shop_boss_risk_weight: float = 1.0

    # Pace-vs-target safety multipliers.
    # shop_target_safety_base raised 1.15 -> 1.30 (2026-05-29 causal config
    # sweep): the value model's calibration showed the bot was over-optimistic
    # at antes 1-2 and under-built early; demanding more score headroom before
    # settling lifted winrate ~12.5% -> 17.0% across 200 seeds (stable 17/17 on
    # both seed halves, never below control) and pushed +6 runs to ante 8. The
    # benefit is early-only: pushing to 1.40 or raising shop_safety_cap (which
    # extends it past ante 2) both regressed back toward baseline.
    shop_target_safety_base: float = 1.30
    hand_pace_safety_base: float = 1.05

    # _shop_target_safety_multiplier ante bonuses and slot bonus.
    shop_safety_ante3_bonus: float = 0.10
    shop_safety_ante4_bonus: float = 0.10
    shop_safety_ante5_bonus: float = 0.05
    shop_safety_full_slots_bonus: float = 0.05
    shop_safety_cap: float = 1.45

    # _shop_capacity_safety_factor penalties (applied at ante >= 4 when no money-scaling joker).
    capacity_no_xmult_penalty: float = 0.10
    capacity_no_planet_penalty: float = 0.08
    capacity_full_slots_penalty: float = 0.06
    capacity_rare_hand_penalty: float = 0.05
    capacity_ante5_penalty: float = 0.04
    capacity_floor: float = 0.72

    # Joker sample-score coefficient: how much weight to give simulated
    # score delta vs hand-tuned heuristics. Used in _joker_card_value,
    # _candidate_joker_value_for_replacement, _owned_joker_value.
    joker_sample_coefficient: float = 0.08
    # _panic_discard_ratio: how far below pace before bot bails to a discard.
    panic_discard_base_ratio: float = 0.45
    panic_discard_ante4_bonus: float = 0.10
    panic_discard_boss_bonus: float = 0.05
    panic_discard_low_hands_bonus: float = 0.10
    panic_discard_desperate_penalty: float = 0.25
    panic_discard_floor: float = 0.20
    panic_discard_cap: float = 0.70


DEFAULT_CONFIG: BotConfig = BotConfig()

_BOT_CONFIG_LOCAL = local()


def active_config() -> BotConfig:
    """Return the active BotConfig, or DEFAULT_CONFIG if no scope is open."""

    return getattr(_BOT_CONFIG_LOCAL, "config", DEFAULT_CONFIG)


@contextmanager
def bot_config_scope(config: BotConfig | None) -> Iterator[BotConfig]:
    """Install `config` as the active config for the duration of the with-block.

    Passing None preserves any previously-active scope. Nested scopes restore
    the previous config on exit.
    """

    if config is None:
        yield active_config()
        return

    previous = getattr(_BOT_CONFIG_LOCAL, "config", None)
    _BOT_CONFIG_LOCAL.config = config
    try:
        yield config
    finally:
        if previous is None:
            try:
                del _BOT_CONFIG_LOCAL.config
            except AttributeError:
                pass
        else:
            _BOT_CONFIG_LOCAL.config = previous
