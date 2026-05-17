"""Basic rule bot with simple play/discard and shop discipline."""

from __future__ import annotations

from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.basic_strategy.actions import (
    _action_index_for_strategy,
    _annotated_action,
    _blind_memory_key,
    _blind_select_action,
    _first_action_of_type,
    _shop_memory_key,
    _with_target_indices,
)
from balatro_ai.bots.basic_strategy.ante_one_hunt import (
    ANTE_ONE_UPGRADE_MIN_GAIN,
    ANTE_ONE_UPGRADE_NEAR_CLEAR_RATIO,
    ANTE_ONE_UPGRADE_TARGET_RATIO,
    _ante_one_card_upgrade_key,
    _ante_one_flush_upgrade_core_indices,
    _ante_one_near_clear_upgrade_discard_action,
    _ante_one_straight_upgrade_core_indices,
    _ante_one_upgrade_core_score,
    _ante_one_upgrade_discard_indices,
    _ante_one_upgrade_keep_candidates,
    _ante_one_upgrade_projection_is_good,
    _first_blind_one_hand_hunt_action,
    _matching_discard_action,
    _straight_core_high_end,
    _unique_index_tuples,
)
from balatro_ai.bots.basic_strategy.banner_policy import (
    BANNER_DISCARD_FUTURE_TAX_WEIGHT,
    _banner_ev_reason,
    _banner_preserve_play_reason,
    _banner_vetoed_play_action,
)
from balatro_ai.bots.basic_strategy.blind_state import _is_boss_blind
from balatro_ai.bots.basic_strategy.blind_setup import (
    _mystic_summit_setup_discard_action,
    _opening_joker_setup_play_action,
    _opening_setup_is_safe,
    _strategic_discard_is_safe,
    _strategic_joker_discard_action,
)
from balatro_ai.bots.basic_strategy.blind_solver import (
    CLEAR_LINE_HAND_TYPES,
    _BlindSolution,
    _ClearLine,
    _best_clear_line,
    _best_discard_for_solution,
    _blind_capacity_from_score,
    _solve_blind,
)
from balatro_ai.bots.basic_strategy.blind_tactics import _tactical_blind_action
from balatro_ai.bots.basic_strategy.blind_reasons import (
    _ante_one_upgrade_discard_reason,
    _discard_reason,
    _first_blind_discard_reason,
    _joker_discard_reason,
    _joker_discard_triggers,
    _last_hand_hunt_discard_reason,
    _mystic_summit_setup_discard_reason,
    _panic_discard_reason,
    _play_reason,
    _preferred_hand_hunt_discard_reason,
    _preferred_hand_hunt_redraw_play_reason,
    _safety_discard_reason,
    _winning_economy_hunt_discard_reason,
)
from balatro_ai.bots.basic_strategy.build_profile import (
    _build_archetype,
    _build_profile,
    _build_profile_uncached,
    _build_role_scores,
    _durable_joker_role_scores,
    _joker_late_durability_factor,
    _joker_role_scores,
    _owned_role_value,
    _reroll_role_hunt_bonus,
    _urgent_late_role_hunt,
)
from balatro_ai.bots.basic_strategy.build_scoring import (
    _archetype_score_samples,
    _buy_would_overfill_joker_slots,
    _card_sharp_repeat_projection_weight,
    _jokers_after_buy_for_scoring,
    _jokers_after_sell_for_scoring,
    _normal_slot_joker_card,
    _sample_build_score,
    _sample_build_score_uncached,
    _sample_hand_build_score,
    _sample_score_delta_for_joker,
    _sample_score_gain_for_joker,
    _score_samples_for_state,
    _should_project_card_sharp_repeat_value,
    _visible_hand_sample_score,
)
from balatro_ai.bots.basic_strategy.cache import (
    _DECISION_CACHE_LOCAL,
    _card_cache_key,
    _current_decision_cache,
    _decision_cached,
    _decision_scoped_cache,
    _freeze_for_cache,
    _identity_cached_value,
    _joker_cache_key,
    _sample_build_score_cache_key,
    _state_scoped_cache,
    decision_cache_scope,
)
from balatro_ai.bots.basic_strategy.discard_state import (
    _conditional_discard_money_delta_for_economy_hunt,
    _discard_draw_count,
    _discard_scoring_state,
    _effective_hand_size,
    _hand_levels_after_discard_for_economy_hunt,
    _has_explicit_hand_size_modifier,
    _jokers_after_discard_for_scoring,
    _known_draw_for_discard,
    _modifiers_after_discard_for_economy_hunt,
    _round_discard_used_count,
    _round_discard_used_count_from_modifiers,
    _serpent_draws_three_for_strategy,
    _state_after_discard_for_projection,
    _state_after_known_discard_for_economy_hunt,
    _trading_card_dollars,
    _yorick_after_discard,
)
from balatro_ai.bots.basic_strategy.discard_policy import (
    DISCARD_DETAIL_LIMIT,
    LATE_DISCARD_DETAIL_LIMIT,
    _best_discard_action,
    _discard_action_playstyle_bonus,
    _discard_can_reduce_hands_needed,
    _discard_detail_limit,
    _discard_penalty_jokers_active,
    _estimated_hands_needed,
    _is_first_discard_window,
    _pace_safety_multiplier,
    _panic_discard_ratio,
    _prefilter_discard_actions,
    _score_is_on_pace,
    _should_chase_discard,
    _should_last_hand_hunt_discard,
    _should_panic_discard,
    _should_safety_discard,
    _unknown_discard_projection_is_trustworthy,
)
from balatro_ai.bots.basic_strategy.cards import (
    _basic_consumable_open_slots,
    _card_cost,
    _card_key,
    _card_label,
    _card_modifier,
    _card_rank,
    _card_set,
    _card_suit,
    _card_value,
    _consumable_card_for_name,
    _consumable_open_slots_after_storage_use,
    _consumable_slot_limit,
    _early_power_bonus,
    _edition_bonus,
    _edition_chips_value,
    _edition_mult_value,
    _edition_xmult_value,
    _has_consumable_room,
    _is_black_hole_card,
    _is_blue_seal,
    _is_face_card_for_state,
    _is_gold_enhancement,
    _is_gold_seal,
    _is_joker_card,
    _is_planet_card,
    _is_playing_card,
    _is_spectral_card,
    _is_tarot_card,
    _joker_from_shop_card,
    _joker_would_overfill_slots,
    _mail_in_rebate_rank_from_text,
    _normalize_rank,
    _normalize_suit,
    _normal_joker_open_slots,
    _normal_joker_slot_limit,
    _normal_joker_slots_used,
    _normalize_card_attr,
    _pack_card_requires_targets,
    _rank_from_text,
    _rank_matches,
    _uses_normal_joker_slot,
)
from balatro_ai.bots.basic_strategy.data import (
    ANTE_SMALL_BLIND_SCORES,
    BURNT_JOKER_DISCARD_HAND_VALUES,
    CARD_SHARP_REPEATABILITY_WEIGHTS,
    DANGEROUS_BOSS_BLINDS,
    DECAYING_SCORE_JOKERS,
    DEDICATED_TWO_PAIR_BUILD_JOKERS,
    DISCARD_PENALTY_JOKERS,
    EARLY_POWER_JOKERS,
    FINAL_BOSS_BLINDS,
    FINAL_BOSS_FRAGILE_JOKERS,
    FINITE_SCORE_JOKERS,
    FLEX_SCALING_JOKERS,
    FLUSH_ARCHETYPE_HANDS,
    FLUSH_DECK_MANIPULATION_TAROTS,
    FOUR_KIND_CONTAINS_HANDS,
    GLASS_CANNON_JOKERS,
    IMPOSSIBLE_HAND_MANIPULATION_GAP,
    IMPOSSIBLE_STARTER_DECK_HANDS,
    JOKER_BASE_VALUES,
    JOKER_ECONOMY_VALUES,
    JOKER_HAND_SYNERGY,
    JOKER_ORDER_CHIPS,
    JOKER_ORDER_MULT,
    JOKER_ORDER_XMULT,
    JOKER_PRIMARY_HAND,
    JOKER_SCALING_VALUES,
    LOW_PRIORITY_JOKERS,
    MONEY_SCALING_RESERVE_TARGETS,
    NARROW_EARLY_JOKERS,
    ORDER_SENSITIVE_JOKERS,
    PAIR_CONTAINS_HANDS,
    PLANET_TO_HAND,
    PREFERRED_HAND_HUNT_TYPES,
    RANK_DECK_MANIPULATION_TAROTS,
    RARE_FLUSH_HAND_TYPES,
    RARE_HAND_JOKER_TARGETS,
    RARE_HAND_SUPPORT_TAROTS,
    RARE_HAND_TYPES,
    RARE_RANK_HAND_TYPES,
    ROLE_MISSING_BONUSES,
    ROLE_UNIQUE_VALUES,
    ROUND_RESET_SCORE_JOKERS,
    SCALING_JOKERS,
    SPECTRAL_CARD_NAMES,
    SPECTRAL_SEAL_VALUES,
    SUIT_TAROT_TARGET_SUITS,
    TARGET_REQUIRED_TAROTS,
    TAROT_VALUES,
    TEMPORARY_SCORE_JOKERS,
    THREE_KIND_CONTAINS_HANDS,
    TWO_PAIR_SUPPORT_JOKERS,
    TWO_PAIR_CONTAINS_HANDS,
    VOUCHER_BUY_DENYLIST,
    VOUCHER_INTEREST_CAP_MONEY,
    VOUCHER_IMMEDIATE_SCORE_NAMES,
    VOUCHER_PRESSURE_ALLOWED_NAMES,
    VOUCHER_VALUES,
    WHITE_STAKE_SAMPLE_HANDS,
)
from balatro_ai.bots.basic_strategy.decision_context import _DecisionContext
from balatro_ai.bots.basic_strategy.hand_models import (
    _BlindContext,
    _PlayCandidate,
    _SampleHand,
    _StraightDrawEvaluation,
    _TargetDrawEvaluation,
)
from balatro_ai.bots.basic_strategy.draw_math import (
    _rank_matches_straight_value,
    _straight_rank_for_value,
    _straight_values_for_card,
    _straight_values_present,
    _straight_window_duplicate_penalty,
    _straight_window_is_open_ended,
    _strategy_rank_order,
)
from balatro_ai.bots.basic_strategy.economy_hunt import (
    BLUE_SEAL_ROUND_END_VALUE,
    _card_is_economy_hunt_target,
    _delayed_gratification_dollars,
    _discard_sensitive_cash_out_value,
    _drawn_economy_hunt_value,
    _economy_hunt_card_value,
    _held_round_end_economy_value,
)
from balatro_ai.bots.basic_strategy.draw_evaluation import (
    _flush_completion_cards_for_suit,
    _flush_completion_probability,
    _flush_suit_out_count,
    _flush_target_draw_evaluations,
    _full_house_completion_cards_for_ranks,
    _full_house_target_draw_evaluations,
    _hand_matches_preferred_family,
    _preferred_hand_family,
    _preferred_target_draw_evaluation,
    _rank_completion_cards_for_rank,
    _rank_completion_probability,
    _rank_fill_cards,
    _rank_group_completion_probability,
    _rank_out_count,
    _rank_target_draw_evaluations,
    _straight_completion_cards_for_values,
    _straight_completion_probability,
    _straight_completion_score,
    _straight_draw_evaluation,
    _straight_draw_quality,
    _straight_draw_reason_detail,
    _straight_known_draw_completes,
    _straight_missing_value_out_count,
    _straight_out_values_for_window,
    _straight_top_draw_out_count,
    _target_draw_quality,
    _target_draw_reason_detail,
)
from balatro_ai.bots.basic_strategy.hand_preferences import (
    _advanced_hand_level_is_playable,
    _dominant_suit,
    _flexible_hand_types,
    _hand_archetype_support_count,
    _hand_level_vote,
    _has_dedicated_pair_plan,
    _has_dedicated_two_pair_plan,
    _preferred_hand_type,
    _preferred_hand_type_uncached,
    _primary_hand_vote_weight,
    _single_narrow_chip_signal_is_noise,
)
from balatro_ai.bots.basic_strategy.hand_value import (
    _card_keep_scores,
    _card_long_term_value,
    _cheap_kept_hand_potential,
    _kept_hand_potential,
    _straight_draw_potential,
)
from balatro_ai.bots.basic_strategy.held_consumables import (
    _held_consumable_action,
    _held_consumable_value,
)
from balatro_ai.bots.basic_strategy.jokers import (
    _active_joker_names,
    _castle_target_suit,
    _current_plus_for_joker,
    _format_xmult,
    _joker_current_plus_value,
    _joker_current_xmult_value,
    _joker_effect_text,
    _joker_has_sticker,
    _joker_is_disabled_for_build,
    _joker_metadata_int_value,
    _joker_metadata_numeric_value,
    _joker_metadata_sources,
    _joker_metadata_value,
    _joker_names,
    _joker_remaining_count,
    _joker_roles,
    _joker_sell_total,
    _joker_with_added_current_plus,
    _joker_with_added_current_xmult,
    _joker_with_current_xmult,
    _mail_in_rebate_rank,
    _static_chip_role_score,
    _static_mult_role_score,
    _static_xmult_role_score,
    _truthy_modifier,
)
from balatro_ai.bots.basic_strategy.pack_choice import (
    _pack_choice_action,
    _pack_skip_action,
    _pack_skip_value,
    _red_card_pack_skip_value,
    _stale_empty_pack_action,
)
from balatro_ai.bots.basic_strategy.joker_ordering import (
    JOKER_REARRANGE_EXHAUSTIVE_COUNT,
    JOKER_REARRANGE_MIN_GAIN,
    MAX_JOKER_REARRANGE_COUNT,
    _best_play_score_for_joker_order,
    _can_use_direct_best_play_score,
    _full_play_action_count,
    _joker_order_can_matter,
    _joker_order_role,
    _joker_order_sort_key,
    _joker_rearrange_action,
    _joker_rearrange_candidate_orders,
    _order_with_copy_before_target,
    _order_with_target_first,
)
from balatro_ai.bots.basic_strategy.pack_targets import (
    _pack_card_is_pickable,
    _pack_card_target_indices,
    _suit_tarot_target_indices,
    _target_required_tarot_is_supported,
)
from balatro_ai.bots.basic_strategy.play_scoring import (
    _action_index_sum,
    _best_play_action,
    _boss_adjusted_score,
    _evaluate_play_action,
    _play_candidate,
    _play_candidates,
    _played_hand_counts,
    _played_hand_types_this_round,
    _score_play_action,
)
from balatro_ai.bots.basic_strategy.preferred_hunt import (
    _preferred_hand_hunt_allowed,
    _preferred_hand_hunt_discard_action,
    _preferred_hand_hunt_redraw_play_action,
    _preferred_hunt_blind_blocks_hunt,
    _preferred_hunt_blind_blocks_redraw_play,
    _preferred_hunt_discard_detail_limit,
    _preferred_hunt_min_draw_strength,
    _preferred_hunt_play_ceiling,
    _preferred_hunt_projection_is_safe,
    _preferred_hunt_protected_indices,
    _should_play_now,
    _straight_hunt_core_indices,
    _straight_hunt_projection_is_safe,
)
from balatro_ai.bots.basic_strategy.score_projection import (
    _best_score_from_cards,
    _discard_realism_factor,
    _dominant_suit_from_cards,
    _fill_with_high_cards,
    _flush_completion_cards,
    _hand_multiset_cache_key,
    _jokers_cache_key,
    _optimistic_completion_score,
    _projected_discard_cache_key,
    _projected_score_after_discard,
    _rank_completion_cards,
    _scoring_state_cache_key,
    _straight_completion_cards,
    _strong_draw_size,
)
from balatro_ai.bots.basic_strategy.profile import (
    CRITICAL_BUILD_ROLES,
    _BuildProfile,
    _ShopContext,
    _ShopPressure,
    _build_profile_payload,
    _pressure_payload,
    _role_requirement,
)
from balatro_ai.bots.basic_strategy.rare_hands import (
    _has_impossible_hand_manipulation_path,
    _rare_hand_deck_manipulation_need,
    _rare_hand_investment_penalty,
    _rare_hand_plan,
    _rare_hand_support_gap,
    _rare_hand_support_score,
    _rare_hand_support_threshold,
    _rare_rank_target,
    _tarot_supports_rare_hand,
    _unsupported_rare_joker_extra_penalty,
    _unsupported_two_pair_joker_penalty,
    _visible_rank_count,
)
from balatro_ai.bots.basic_strategy.shop_items import (
    _action_payload,
    _has_shop_decision_surface,
    _indexed_shop_item,
    _item_payload_for_action,
    _shop_item_for_action,
    _shop_item_payload,
    _shop_option_payload,
)
from balatro_ai.bots.basic_strategy.shop_flow import (
    _owned_joker_value_payloads,
    _pressure_forced_shop_action,
    _replacement_sell_action,
    _has_shop_policy_action,
    _shop_action,
    _shop_decision_audit,
    _shop_information_first_action,
)
from balatro_ai.bots.basic_strategy.shop_jokers import (
    _build_conflict_penalty,
    _build_synergy_value,
    _candidate_joker_value_for_replacement,
    _capped_red_card_candidate_value,
    _future_headroom_required_score,
    _future_score_headroom_joker_bonus,
    _hallucination_utility_bonus,
    _joker_card_value,
    _joker_heuristic_value,
    _joker_role_bonus,
    _joker_sample_reliability,
    _joker_stencil_would_fill_slots,
    _narrow_rank_inconsistency_penalty_for_joker,
    _owned_joker_value,
    _pressure_joker_role_bonus,
    _rare_hand_inconsistency_penalty_for_joker,
    _red_card_has_visible_skip_plan,
    _red_card_heuristic_value,
    _replacement_cost_weight,
    _replacement_role_upgrade_bonus,
    _replacement_upgrade_threshold,
    _shop_build_capacity_for_jokers,
    _temporary_score_late_penalty,
)
from balatro_ai.bots.basic_strategy.shop_cards import (
    _black_hole_card_value,
    _hand_joker_alignment_score,
    _planet_capacity_gain,
    _planet_card_value,
    _playing_card_shop_value,
    _rare_hand_deck_manipulation_bonus,
    _tarot_card_value,
    _weak_hand_plan_penalty,
)
from balatro_ai.bots.basic_strategy.shop_money import (
    BASE_INTEREST_CAP_MONEY,
    _cost_penalty,
    _desired_money_reserve,
    _economy_joker_interest_penalty_scale,
    _interest_cap_money,
    _late_closing_money_reserve_cap,
    _late_pressure_interest_floor,
    _money_after_spend_penalty,
    _money_gain_value,
    _money_plan_payload,
    _spendable_money,
)
from balatro_ai.bots.basic_strategy.shop_packs import (
    _celestial_candidate_hand_types,
    _celestial_pack_capacity_gain,
    _is_buffoon_pack,
    _late_pack_is_worth_opening,
    _late_pack_limit,
    _minimum_late_pack_capacity_gain,
    _pack_capacity_gain,
    _pack_value,
    _rare_hand_pack_bonus,
    _rare_hand_pack_capacity_bonus,
    _score_loss_after_spending,
)
from balatro_ai.bots.basic_strategy.shop_forecast import (
    _blind_name_from_mapping,
    _blind_score_from_mapping,
    _boss_capacity_factor,
    _boss_preview_weight,
    _boss_score_target_multiplier,
    _boss_target_safety_bonus,
    _effective_boss_target_multiplier,
    _estimated_final_win_required_score,
    _estimated_next_required_score,
    _estimated_shop_planning_required_score,
    _extrapolated_small_blind_score,
    _final_boss_fragility_factor,
    _final_win_target_weight,
    _has_planet_investment,
    _shop_cleared_blind_kind,
    _shop_has_followup_big_blind_shop,
    _shop_pressure_boss_capacity_factor,
    _shop_pressure_boss_discards_remaining,
    _shop_pressure_boss_hands_remaining,
    _shop_pressure_effective_hands,
    _shop_pressure_score_state,
    _shop_pressure_uses_boss_score_state,
    _shop_pressure_uses_exact_needle_hand,
    _suit_boss_capacity_factor,
    _upcoming_boss_blind_name,
    _upcoming_boss_score,
    _weighted_boss_capacity_factor,
    _weighted_final_boss_fragility_factor,
)
from balatro_ai.bots.basic_strategy.shop_pressure import (
    _early_build_pressure_floor,
    _shop_capacity_safety_factor,
    _shop_hand_realism_factor,
    _shop_pressure,
    _shop_pressure_uncached,
    _shop_target_safety_multiplier,
)
from balatro_ai.bots.basic_strategy.shop_reroll import (
    _best_visible_non_reroll_shop_value,
    _early_reroll_is_allowed,
    _late_bank_conversion_mode,
    _late_bank_conversion_reserve_cap,
    _late_extra_pressure_reroll_allowance,
    _late_extra_pressure_reroll_is_worth_it,
    _late_pressure_closer_mode,
    _late_pressure_closer_reroll_limit,
    _late_reroll_is_worth_it,
    _late_reroll_limit,
    _minimum_reroll_bank,
    _missing_critical_roles,
    _pressure_spend_mode,
    _pressure_spend_reserve_slack,
    _reroll_cost_escalation_penalty,
    _rich_late_role_hunt,
    _visible_early_power_path,
)
from balatro_ai.bots.basic_strategy.shop_vouchers import (
    _antimatter_voucher_adjustment,
    _ante_step_voucher_adjustment,
    _discount_voucher_adjustment,
    _discard_voucher_adjustment,
    _edition_voucher_adjustment,
    _generator_voucher_adjustment,
    _hand_size_voucher_adjustment,
    _hand_voucher_adjustment,
    _interest_cap_voucher_adjustment,
    _late_pressure_blocks_voucher,
    _observatory_voucher_adjustment,
    _omen_globe_voucher_adjustment,
    _planet_for_hand,
    _reroll_voucher_adjustment,
    _retcon_voucher_adjustment,
    _shop_slot_voucher_adjustment,
    _voucher_buy_is_blocked,
    _voucher_does_not_solve_current_boss_shop,
    _voucher_dynamic_adjustment,
    _voucher_name_key,
    _voucher_value,
)
from balatro_ai.bots.basic_strategy.shop_values import (
    _pack_card_value,
    _pressure_pack_bonus,
    _scaling_commitment_pack_bonus,
    _scaling_commitment_shop_bonus,
    _shop_action_cost,
    _shop_action_reveals_information_before_joker_buy,
    _shop_action_value,
    _shop_buy_threshold,
    _shop_card_value,
    _shop_information_action_can_take_joker_slot,
    _shop_pack_can_trigger_hidden_target_error,
    _shop_reroll_cost,
    _shop_safety_pack_bonus,
    _spectral_card_value,
    _visible_safety_pack_before_reroll,
)
from balatro_ai.bots.basic_strategy.shop_safety import (
    _early_hand_type_is_supported,
    _early_shop_safety_adjustment,
    _hand_has_natural_support,
    _has_money_scaling_joker,
    _has_real_scoring_joker,
)
from balatro_ai.bots.basic_strategy.utils import _int_or_default
from balatro_ai.bots.basic_strategy.winning_economy import (
    WINNING_ECONOMY_HUNT_MIN_GAIN,
    _best_clear_economy_value,
    _clear_economy_value_for_evaluated_play,
    _clear_economy_value_for_play,
    _winning_economy_hunt_discard_action,
)
from balatro_ai.bots.config import DEFAULT_CONFIG as _DEFAULT_CONFIG
from balatro_ai.bots.config import BotConfig, active_config, bot_config_scope
from balatro_ai.bots.random_bot import RandomBot
from balatro_ai.rules.hand_evaluator import (
    HandType,
    RANK_VALUES,
    STRAIGHT_VALUES,
    _prepare_joker_evaluation_context,
    best_play_from_hand,
    debuffed_suits_for_blind,
    evaluate_played_cards,
)
from balatro_ai.search.hand_viability import (
    ADVANCED_HAND_TYPES,
    hand_type_is_viable,
    hand_type_viability_multiplier,
)


# Compatibility aliases: prefer `active_config().X` in new code. Kept as
# module-level constants so existing imports (tests, external tooling) work.
SHOP_VALUE_TOLERANCE = _DEFAULT_CONFIG.shop_value_tolerance
SHOP_TARGET_SAFETY_BASE = _DEFAULT_CONFIG.shop_target_safety_base
HAND_PACE_SAFETY_BASE = _DEFAULT_CONFIG.hand_pace_safety_base


@dataclass(slots=True)
class BasicStrategyBot:
    """A small step above immediate-score greed.

    The bot avoids random shop spending, plays hands that are good enough for
    the current blind, and uses discards when the best immediate hand is behind
    the score pace.
    """

    seed: int | None = None
    name: str = "basic_strategy_bot"
    config: BotConfig | None = None
    _fallback: RandomBot = field(init=False, repr=False)
    _shop_key: tuple[int | None, int, str, int] | None = field(default=None, init=False, repr=False)
    _rerolls_in_shop: int = field(default=0, init=False, repr=False)
    _packs_opened_in_shop: int = field(default=0, init=False, repr=False)
    _filled_last_joker_slot_in_shop: bool = field(default=False, init=False, repr=False)
    _blind_key: tuple[int | None, int, str, int] | None = field(default=None, init=False, repr=False)
    _played_hand_types_in_blind: tuple[HandType, ...] = field(default=(), init=False, repr=False)
    _discards_in_blind: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback = RandomBot(seed=self.seed)

    def choose_action(self, state: GameState) -> Action:
        with bot_config_scope(self.config), decision_cache_scope():
            return self._choose_action_uncached(state)

    def _choose_action_uncached(self, state: GameState) -> Action:
        self._sync_shop_memory(state)
        self._sync_blind_memory(state)

        blind_select = _blind_select_action(state)
        if blind_select is not None:
            return blind_select

        cash_out = _first_action_of_type(state, ActionType.CASH_OUT)
        if cash_out is not None:
            return cash_out

        context = self._decision_context(state)

        shop_action = (
            _shop_action(
                state,
                context.shop,
                pressure=context.shop_pressure,
                profile=context.build_profile,
            )
            if _has_shop_policy_action(state)
            else None
        )
        if shop_action is not None:
            if shop_action.action_type == ActionType.END_SHOP:
                held_consumable = _held_consumable_action(state)
                if held_consumable is not None:
                    return held_consumable
                return shop_action
            else:
                self._record_shop_action(state, shop_action)
                return shop_action

        end_shop = _first_action_of_type(state, ActionType.END_SHOP)
        if end_shop is not None:
            held_consumable = _held_consumable_action(state)
            if held_consumable is not None:
                return held_consumable
            return end_shop

        stale_empty_pack = _stale_empty_pack_action(state)
        if stale_empty_pack is not None:
            return stale_empty_pack

        pack_choice = _pack_choice_action(state, context.shop)
        if pack_choice is not None:
            self._record_shop_action(state, pack_choice)
            return pack_choice

        held_consumable = _held_consumable_action(state)
        if held_consumable is not None:
            return held_consumable

        blind_context = context.blind
        joker_rearrange = _joker_rearrange_action(state, blind_context)
        if joker_rearrange is not None:
            return joker_rearrange

        blind_solution = context.blind_solution
        best_play = blind_solution.best_play
        if best_play is None:
            return self._fallback.choose_action(state)

        action = _tactical_blind_action(state, best_play, blind_context, solution=blind_solution)
        self._record_blind_action(state, action, blind_context)
        return action

    def _shop_context(self) -> _ShopContext:
        return _ShopContext(
            rerolls_in_shop=self._rerolls_in_shop,
            packs_opened_in_shop=self._packs_opened_in_shop,
            filled_last_joker_slot=self._filled_last_joker_slot_in_shop,
        )

    def _decision_context(self, state: GameState) -> _DecisionContext:
        return _DecisionContext(
            state=state,
            shop=self._shop_context(),
            blind=self._blind_context(state),
        )

    def _blind_context(self, state: GameState) -> _BlindContext:
        played_from_state = _played_hand_types_this_round(state)
        if state.blind == "The Mouth" and self._played_hand_types_in_blind:
            return _BlindContext(played_hand_types=self._played_hand_types_in_blind, discards_taken=self._discards_in_blind)
        if played_from_state:
            return _BlindContext(played_hand_types=played_from_state, discards_taken=self._discards_in_blind)
        return _BlindContext(played_hand_types=self._played_hand_types_in_blind, discards_taken=self._discards_in_blind)

    def _sync_shop_memory(self, state: GameState) -> None:
        if not (_has_shop_decision_surface(state) or state.modifiers.get("pack_cards")):
            return

        key = _shop_memory_key(state)
        if key != self._shop_key:
            self._shop_key = key
            self._rerolls_in_shop = 0
            self._packs_opened_in_shop = 0
            self._filled_last_joker_slot_in_shop = False

    def _record_shop_action(self, state: GameState, action: Action) -> None:
        if action.action_type == ActionType.REROLL:
            self._rerolls_in_shop += 1
        elif action.action_type == ActionType.OPEN_PACK:
            self._packs_opened_in_shop += 1
        elif action.action_type == ActionType.SELL:
            self._filled_last_joker_slot_in_shop = False
        elif action.action_type == ActionType.BUY:
            item = _shop_item_for_action(state, action)
            if (
                _normal_slot_joker_card(item)
                and _normal_joker_open_slots(state) <= 1
            ):
                self._filled_last_joker_slot_in_shop = True
        elif action.action_type == ActionType.CHOOSE_PACK_CARD:
            item = _shop_item_for_action(state, action)
            if (
                _normal_slot_joker_card(item)
                and _normal_joker_open_slots(state) <= 1
            ):
                self._filled_last_joker_slot_in_shop = True

    def _sync_blind_memory(self, state: GameState) -> None:
        if not (state.hand or state.current_score or state.hands_remaining):
            return

        key = _blind_memory_key(state)
        if key != self._blind_key:
            self._blind_key = key
            self._played_hand_types_in_blind = ()
            self._discards_in_blind = 0

    def _record_blind_action(self, state: GameState, action: Action, context: _BlindContext) -> None:
        if action.action_type == ActionType.DISCARD:
            self._discards_in_blind = context.discards_taken + 1
            return
        if action.action_type != ActionType.PLAY_HAND or not action.card_indices:
            return
        hand_type = _evaluate_play_action(state, action, context).hand_type
        self._played_hand_types_in_blind = (*context.played_hand_types, hand_type)


