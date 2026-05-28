//! End-to-end no-joker evaluation (Phase 2 wire-up).
//!
//! Composes `identify_hand_type`, `scoring_indices`, `card_chip_value`,
//! and BASE_HAND_VALUES into a single function: given played cards
//! (+ hand levels + debuffed suits), returns (chips, mult, score, hand_type).
//!
//! Coverage: the no-joker, no-blind-effect fast path. The Python
//! caller passes `jokers=()` and a blind name that has no chip/mult
//! adjustments (typically white-stake regular blinds). For hands
//! with jokers OR boss-blind effects, the Rust function returns
//! `None` and the Python `evaluate_played_cards` takes over.
//!
//! Empirically this fast path covers the majority of evaluations
//! during the first blind of a run (no jokers acquired yet), which
//! is where solver rollouts spend a lot of time.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::hand_eval::chips::card_chip_value;
use crate::hand_eval::effects::{
    ability_joker_effect, enhancement_effect, held_card_effect,
    held_pass_possible, held_retrigger_count, is_face_with_pareidolia,
    is_supported_joker, per_card_joker_effect, raised_fist_target,
    scored_card_trigger_count, HandContext, JokerMetadata,
};
use crate::hand_eval::hand_type::{
    identify_hand_type_simple, identify_hand_type_with_jokers, HandType, IdJokers,
};
use crate::hand_eval::scoring::{scoring_indices_simple, scoring_indices_with_jokers};
use crate::state::card::{Card, Rank, Seal, Suit};

/// Base (level-1) chips + mult per hand type, matching
/// `BASE_HAND_VALUES` at `hand_evaluator.py:74`.
#[inline]
fn base_hand_values(ht: HandType) -> (u32, u32) {
    match ht {
        HandType::HighCard => (5, 1),
        HandType::Pair => (10, 2),
        HandType::TwoPair => (20, 2),
        HandType::ThreeOfAKind => (30, 3),
        HandType::Straight => (30, 4),
        HandType::Flush => (35, 4),
        HandType::FullHouse => (40, 4),
        HandType::FourOfAKind => (60, 7),
        HandType::StraightFlush => (100, 8),
        HandType::FiveOfAKind => (120, 12),
        HandType::FlushHouse => (140, 14),
        HandType::FlushFive => (160, 16),
    }
}

/// Per-level chip + mult increment, matching `LEVEL_INCREMENTS` at
/// `hand_evaluator.py:89`.
#[inline]
fn level_increment(ht: HandType) -> (u32, u32) {
    match ht {
        HandType::HighCard => (10, 1),
        HandType::Pair => (15, 1),
        HandType::TwoPair => (20, 1),
        HandType::ThreeOfAKind => (20, 2),
        HandType::Straight => (30, 3),
        HandType::Flush => (15, 2),
        HandType::FullHouse => (25, 2),
        HandType::FourOfAKind => (30, 3),
        HandType::StraightFlush => (40, 4),
        HandType::FiveOfAKind => (35, 3),
        HandType::FlushHouse => (40, 4),
        HandType::FlushFive => (50, 3),
    }
}

/// Native end-to-end evaluation. Returns
/// `(chips, mult, score, hand_type_str)` for the simple case, or
/// `None` if anything in the path bailed.
///
/// `jokers` is the list of joker NAMES present in the run. Any joker
/// name NOT in the Rust-supported set causes the function to bail
/// (returns None) — the caller falls back to Python `evaluate_played_cards`
/// rather than risk producing a wrong score by skipping an
/// unsupported joker's effect.
pub fn evaluate_simple(
    cards: &[Card],
    hand_level: u32,
    debuffed_suits: &[Suit],
    joker_names: &[String],
    joker_editions: &[crate::state::card::Edition],
    joker_metadata: &[JokerMetadata],
    held_cards: &[Card],
    played_hand_types: &[String],
    ctx: HandContext,
) -> Option<(u64, u64, u64, &'static str)> {
    if cards.is_empty() {
        return None;
    }
    // Joker editions are parallel to joker_names. Empty list is treated
    // as "all None" — keeps backward-compatible call sites simple.
    let editions_for_jokers: Vec<crate::state::card::Edition> = if joker_editions.len() == joker_names.len() {
        joker_editions.to_vec()
    } else {
        vec![crate::state::card::Edition::None; joker_names.len()]
    };
    // Same pattern for joker metadata — default to "no scaling" entries.
    let metadata_for_jokers: Vec<JokerMetadata> = if joker_metadata.len() == joker_names.len() {
        joker_metadata.to_vec()
    } else {
        vec![JokerMetadata::default(); joker_names.len()]
    };

    // Identification joker flags. When any of these are set OR a
    // Wild card is in the played hand, take the joker-aware path;
    // otherwise the simple path (faster, no combinations enumeration).
    let id_jokers = IdJokers {
        smeared: joker_names.iter().any(|n| n == "Smeared Joker"),
        four_fingers: joker_names.iter().any(|n| n == "Four Fingers"),
        shortcut: joker_names.iter().any(|n| n == "Shortcut"),
    };
    let has_wild = cards.iter().any(|c| matches!(c.enhancement, crate::state::card::Enhancement::Wild))
        || held_cards.iter().any(|c| matches!(c.enhancement, crate::state::card::Enhancement::Wild));
    let take_joker_path = id_jokers.smeared || id_jokers.four_fingers
        || id_jokers.shortcut || has_wild;
    let ht = if take_joker_path {
        identify_hand_type_with_jokers(cards, id_jokers)?
    } else {
        identify_hand_type_simple(cards)?
    };
    let has_splash = joker_names.iter().any(|n| n == "Splash");
    let scoring = if take_joker_path {
        scoring_indices_with_jokers(cards, ht, has_splash, id_jokers)?
    } else {
        crate::hand_eval::scoring::scoring_indices_simple_with_splash(
            cards, ht, has_splash,
        )?
    };

    // Compute Card Sharp / Supernova / Obelisk ctx fields from
    // played_hand_types. These override the same fields in `ctx`
    // (we expect callers to leave them defaulted when passing
    // played_hand_types).
    let ht_str = ht.to_str();
    let played_count_this_hand_type = played_hand_types
        .iter()
        .filter(|s| s.as_str() == ht_str)
        .count() as u32;
    let hand_type_played_before = played_count_this_hand_type > 0;
    // Highest count among OTHER hand_types this round — Obelisk gate.
    // Linear scan with a tiny inline histogram; played_hand_types is
    // small (≤10 entries typically) so the O(n²) here is fine.
    let played_count_max_other_hand_type = {
        let mut max_other: u32 = 0;
        for s in played_hand_types {
            if s.as_str() == ht_str { continue; }
            let count = played_hand_types
                .iter()
                .filter(|x| x.as_str() == s.as_str())
                .count() as u32;
            if count > max_other { max_other = count; }
        }
        max_other.max(ctx.played_count_max_other_hand_type)
    };
    let pareidolia_active = joker_names.iter().any(|n| n == "Pareidolia");
    let ctx = HandContext {
        played_count_this_hand_type: played_count_this_hand_type
            .max(ctx.played_count_this_hand_type),
        hand_type_played_before: hand_type_played_before || ctx.hand_type_played_before,
        played_count_max_other_hand_type,
        pareidolia_active: pareidolia_active || ctx.pareidolia_active,
        ..ctx
    };

    let (base_chips, base_mult) = base_hand_values(ht);
    let (chip_inc, mult_inc) = level_increment(ht);
    let level_delta = hand_level.saturating_sub(1);
    let hand_chips = base_chips + chip_inc * level_delta;
    let hand_mult = base_mult + mult_inc * level_delta;

    // Pre-check: every supplied joker must be supported by EITHER
    // per-card or ability pass. If any joker is unknown, bail to
    // Python entirely. This is more conservative than partial
    // evaluation but prevents silent miscalculation.
    for name in joker_names {
        if !is_supported_joker(name) {
            return None;
        }
    }

    // Running totals — mirror Python's `ordered_chips` / `ordered_mult`
    // that get add_chips / add_mult / multiply_mult applied in order.
    let mut chips: i64 = hand_chips as i64;
    let mut mult: f64 = hand_mult as f64;

    // First scored face index (for Photograph). Computed once;
    // re-used to gate Photograph's x2 mult to ONLY the first face.
    // Pareidolia makes every card count as a face → first scored
    // card is the "first face".
    let first_face_index_in_scoring: Option<usize> = scoring.iter().position(|&i| {
        is_face_with_pareidolia(cards[i], ctx.pareidolia_active)
    });

    // === Per-card pass ===
    // For each scoring card, the Python order is:
    //   1. add_chips(card_chip_value)        — base + bonus enhancement
    //   2. add_chips(edition_chips)          — Foil = +50
    //   3. add_mult(enhancement_mult)        — Mult enhancement = +4
    //   4. add_mult(edition_mult)            — Holographic = +10
    //   5. multiply_mult(enhancement_xmult)  — Glass = x2 (we bail on Glass)
    //   6. multiply_mult(edition_xmult)      — Polychrome = x1.5
    //   7. apply_scored_card_jokers           — Triboulet etc fire here
    // First scored card index (for Hanging Chad retrigger).
    let first_scoring_card_index = scoring.first().copied();
    for (pos, &i) in scoring.iter().enumerate() {
        let c = cards[i];
        // Retrigger count: fire all the per-card effects N times.
        // Computed once per card; the inner loop just applies the
        // base effects per trigger. Python matches: card_chip_value
        // contributes per trigger, as do edition/enhancement/per-card
        // joker effects. Defaults to 1 trigger if no retrigger jokers
        // are present (the common case).
        let is_first_scored = first_scoring_card_index == Some(i);
        let triggers = scored_card_trigger_count(
            c, is_first_scored, joker_names, ctx.hands_remaining,
            ctx.pareidolia_active,
        );
        let is_first_face = first_face_index_in_scoring == Some(pos);

        for _ in 0..triggers {
            let v = card_chip_value(c, debuffed_suits)?;
            chips += v as i64;
            chips += c.edition.chips();
            let enh = enhancement_effect(c.enhancement);
            chips += enh.chip_delta;
            mult += enh.mult_delta as f64;
            mult += c.edition.mult() as f64;
            mult *= enh.xmult;
            mult *= c.edition.xmult();

            // Per-card jokers (Triboulet x2 applies HERE before
            // ability jokers add more mult; Hanging Chad's
            // retrigger count is already baked into `triggers`).
            // Pass per-joker metadata so Ancient Joker / The Idol
            // can read target_suit / target_rank.
            for (j_idx, name) in joker_names.iter().enumerate() {
                let j_meta = metadata_for_jokers[j_idx];
                if let Some(e) = per_card_joker_effect(
                    name, c, is_first_face, j_meta, ctx.pareidolia_active,
                ) {
                    chips += e.chip_delta;
                    mult += e.mult_delta as f64;
                    mult *= e.xmult;
                }
            }
        }
    }

    // === Ability pass ===
    // Per ability joker (in held order):
    //   1. add chips from joker edition (Foil = +50)
    //   2. add mult from joker edition (Holographic = +10)
    //   3. apply joker effect (add chips/mult/xmult as applicable)
    //   3b. apply Baseball Card x1.5 mult per OTHER baseball card
    //       (only when this joker is uncommon rarity)
    //   4. multiply mult by joker edition xmult (Polychrome = x1.5)
    // Matches Python's `_effect_adjustments` ability-pass ordering at
    // lines 1297-1436.
    let scoring_cards: Vec<Card> = scoring.iter().map(|&i| cards[i]).collect();
    let baseball_count: u32 = joker_names
        .iter()
        .filter(|n| n.as_str() == "Baseball Card")
        .count() as u32;
    // Wee Joker: precompute total per-trigger count over scored 2s.
    // Wee Joker's chip contribution = current_plus_chips + 8 * total.
    // Skip the loop entirely when no Wee Joker is held (the common
    // case).
    let total_triggers_on_2s: u32 = if joker_names.iter().any(|n| n == "Wee Joker") {
        scoring.iter().filter_map(|&i| {
            if cards[i].rank == Rank::Two {
                let is_first_scored = first_scoring_card_index == Some(i);
                Some(scored_card_trigger_count(
                    cards[i], is_first_scored, joker_names,
                    ctx.hands_remaining, ctx.pareidolia_active,
                ))
            } else { None }
        }).sum()
    } else {
        0
    };
    for (idx, name) in joker_names.iter().enumerate() {
        let ed = editions_for_jokers[idx];

        // Steps 1-2: edition pre-effects
        chips += ed.chips();
        mult += ed.mult() as f64;

        // Step 3: joker effect. Note: returns Some(default) when the
        // joker is known but its condition isn't met (no contribution).
        // The is_supported_joker pre-check at the top guarantees we
        // never reach here with an unknown joker.
        //
        // Wee Joker is special-cased here because its chip
        // contribution needs the precomputed `total_triggers_on_2s`,
        // which isn't visible to ability_joker_effect.
        let meta = metadata_for_jokers[idx];
        if name == "Wee Joker" {
            chips += meta.current_plus_chips as i64;
            chips += 8 * (total_triggers_on_2s as i64);
        } else if let Some(e) = ability_joker_effect(name, ht, cards.len(), &scoring_cards, held_cards, ctx, meta) {
            chips += e.chip_delta;
            mult += e.mult_delta as f64;
            mult *= e.xmult;
        }

        // Step 3b: Baseball Card x1.5 mult per other Baseball Card,
        // gated on the current joker's rarity being "uncommon" (1).
        // Matches Python: joker_rarity(physical_joker) == "uncommon".
        if baseball_count > 0 && meta.rarity == 1 {
            let other_baseball = if name.as_str() == "Baseball Card" {
                baseball_count - 1
            } else {
                baseball_count
            };
            for _ in 0..other_baseball {
                mult *= 1.5;
            }
        }

        // Step 4: edition xmult post-effect
        mult *= ed.xmult();
    }

    // === Held-card pass ===
    // After ability jokers, iterate held cards. Each held card may
    // contribute via Steel enhancement (x1.5 mult) or via per-held-
    // card joker effects (Shoot the Moon, Baron, Raised Fist). The
    // effect retriggers (1 + mime_count) times, +1 more if the
    // card has a Red Seal. Matches Python's loop at
    // `_effect_adjustments` lines 1277-1294.
    //
    // Fast-skip: most evaluations don't have Steel cards or held-
    // card jokers. `held_pass_possible` is a cheap O(held) check
    // that avoids the full nested loop in those cases.
    if held_pass_possible(held_cards, joker_names) {
        let raised_fist_idx = raised_fist_target(held_cards);
        for (i, c) in held_cards.iter().enumerate() {
            let is_target = raised_fist_idx == Some(i);
            let red_seal = c.seal == Seal::Red;
            let count = held_retrigger_count(joker_names, red_seal);
            for _ in 0..count {
                let e = held_card_effect(*c, joker_names, debuffed_suits, is_target);
                chips += e.chip_delta;
                mult += e.mult_delta as f64;
                mult *= e.xmult;
            }
        }
    }

    // Final score. Match Python `_score_floor(ordered_chips * ordered_mult)`.
    let total_chips = chips.max(0) as u64;
    let final_mult = mult.max(1.0);
    let score = ((total_chips as f64) * final_mult).floor() as u64;
    // For the returned `mult` field, Python reports the ADDITIVE-only
    // mult (excluding xmult). We compute the integer-floor of the final
    // mult to match the score's effective mult.
    let total_mult = final_mult.floor() as u64;
    Some((total_chips, total_mult.max(1), score, ht.to_str()))
}

/// PyO3 wrapper. Returns a tuple `(chips, mult, score, hand_type)`
/// or None to signal Python fallback.
///
/// Python caller pattern:
///     result = balatro_core.evaluate_simple(
///         rust_cards, hand_level=hand_levels.get(...), debuffed_suits=["H"]
///     )
///     if result is None:
///         # Fall back to Python evaluate_played_cards for the slow path
///         result = evaluate_played_cards(cards, ..., jokers=jokers)
///     chips, mult, score, hand_type = result
#[pyfunction]
#[pyo3(name = "evaluate_simple")]
#[pyo3(signature = (cards, hand_level, debuffed_suits, joker_names=None,
                    joker_editions=None, held_cards=None,
                    joker_current_plus_mult=None,
                    joker_current_plus_chips=None,
                    joker_current_xmult=None,
                    joker_loyalty_ready=None,
                    joker_drivers_active=None,
                    joker_leading_plus_mult=None,
                    joker_leading_plus_chips=None,
                    joker_sell_value=None,
                    joker_rarity=None,
                    joker_target_suit=None,
                    joker_target_rank=None,
                    joker_obelisk_gain=None,
                    money=0, joker_slot_limit=5, discards_remaining=0,
                    hands_remaining=0,
                    played_hand_types=None,
                    played_count_this_hand_type=0,
                    hand_type_played_before=false,
                    deck_size=0))]
#[allow(clippy::too_many_arguments)]
pub fn py_evaluate_simple(
    cards: Vec<Card>,
    hand_level: u32,
    debuffed_suits: Vec<String>,
    joker_names: Option<Vec<String>>,
    joker_editions: Option<Vec<Option<String>>>,
    held_cards: Option<Vec<Card>>,
    joker_current_plus_mult: Option<Vec<i32>>,
    joker_current_plus_chips: Option<Vec<i32>>,
    joker_current_xmult: Option<Vec<f64>>,
    joker_loyalty_ready: Option<Vec<bool>>,
    joker_drivers_active: Option<Vec<bool>>,
    joker_leading_plus_mult: Option<Vec<i32>>,
    joker_leading_plus_chips: Option<Vec<i32>>,
    joker_sell_value: Option<Vec<i32>>,
    joker_rarity: Option<Vec<u8>>,
    joker_target_suit: Option<Vec<Option<String>>>,
    joker_target_rank: Option<Vec<Option<String>>>,
    joker_obelisk_gain: Option<Vec<f64>>,
    money: i32,
    joker_slot_limit: u32,
    discards_remaining: u32,
    hands_remaining: u32,
    played_hand_types: Option<Vec<String>>,
    played_count_this_hand_type: u32,
    hand_type_played_before: bool,
    deck_size: u32,
) -> Option<(u64, u64, u64, &'static str)> {
    let suits: Vec<Suit> = debuffed_suits
        .iter()
        .filter_map(|s| Suit::from_str(s))
        .collect();
    let jokers = joker_names.unwrap_or_default();
    let editions: Vec<crate::state::card::Edition> = joker_editions
        .map(|opts| opts.iter().map(|o| crate::state::card::Edition::from_option_str(o.as_deref())).collect())
        .unwrap_or_default();
    let held = held_cards.unwrap_or_default();
    let metadata = build_metadata(
        jokers.len(),
        joker_current_plus_mult.as_deref(),
        joker_current_plus_chips.as_deref(),
        joker_current_xmult.as_deref(),
        joker_loyalty_ready.as_deref(),
        joker_drivers_active.as_deref(),
        joker_leading_plus_mult.as_deref(),
        joker_leading_plus_chips.as_deref(),
        joker_sell_value.as_deref(),
        joker_rarity.as_deref(),
        joker_target_suit.as_deref(),
        joker_target_rank.as_deref(),
        joker_obelisk_gain.as_deref(),
    );
    let ctx = HandContext {
        money,
        joker_count: jokers.len() as u32,
        joker_slot_limit,
        discards_remaining,
        hands_remaining,
        played_count_this_hand_type,
        hand_type_played_before,
        deck_size,
        played_count_max_other_hand_type: 0,
        pareidolia_active: false,
    };
    let played_types_vec = played_hand_types.unwrap_or_default();
    evaluate_simple(&cards, hand_level, &suits, &jokers, &editions, &metadata, &held, &played_types_vec, ctx)
}

/// Convenience: same as `evaluate_simple` but takes the hand_levels
/// dict (Python style — `dict[hand_type_str, int]`) and looks up
/// the right level after identifying the hand type. Saves a Python
/// dict-lookup roundtrip.
#[pyfunction]
#[pyo3(name = "evaluate_simple_with_levels")]
#[pyo3(signature = (cards, hand_levels, debuffed_suits, joker_names=None,
                    joker_editions=None, held_cards=None,
                    joker_current_plus_mult=None,
                    joker_current_plus_chips=None,
                    joker_current_xmult=None,
                    joker_loyalty_ready=None,
                    joker_drivers_active=None,
                    joker_leading_plus_mult=None,
                    joker_leading_plus_chips=None,
                    joker_sell_value=None,
                    joker_rarity=None,
                    joker_target_suit=None,
                    joker_target_rank=None,
                    joker_obelisk_gain=None,
                    money=0, joker_slot_limit=5, discards_remaining=0,
                    hands_remaining=0,
                    played_hand_types=None,
                    played_count_this_hand_type=0,
                    hand_type_played_before=false,
                    deck_size=0))]
#[allow(clippy::too_many_arguments)]
pub fn py_evaluate_simple_with_levels(
    cards: Vec<Card>,
    hand_levels: &Bound<'_, PyDict>,
    debuffed_suits: Vec<String>,
    joker_names: Option<Vec<String>>,
    joker_editions: Option<Vec<Option<String>>>,
    held_cards: Option<Vec<Card>>,
    joker_current_plus_mult: Option<Vec<i32>>,
    joker_current_plus_chips: Option<Vec<i32>>,
    joker_current_xmult: Option<Vec<f64>>,
    joker_loyalty_ready: Option<Vec<bool>>,
    joker_drivers_active: Option<Vec<bool>>,
    joker_leading_plus_mult: Option<Vec<i32>>,
    joker_leading_plus_chips: Option<Vec<i32>>,
    joker_sell_value: Option<Vec<i32>>,
    joker_rarity: Option<Vec<u8>>,
    joker_target_suit: Option<Vec<Option<String>>>,
    joker_target_rank: Option<Vec<Option<String>>>,
    joker_obelisk_gain: Option<Vec<f64>>,
    money: i32,
    joker_slot_limit: u32,
    discards_remaining: u32,
    hands_remaining: u32,
    played_hand_types: Option<Vec<String>>,
    played_count_this_hand_type: u32,
    hand_type_played_before: bool,
    deck_size: u32,
) -> PyResult<Option<(u64, u64, u64, &'static str)>> {
    let ht_str = match identify_hand_type_simple(&cards) {
        Some(ht) => ht.to_str(),
        None => return Ok(None),
    };
    let level: u32 = hand_levels
        .get_item(ht_str)?
        .map(|v| v.extract().unwrap_or(1u32))
        .unwrap_or(1);
    let suits: Vec<Suit> = debuffed_suits
        .iter()
        .filter_map(|s| Suit::from_str(s))
        .collect();
    let jokers = joker_names.unwrap_or_default();
    let editions: Vec<crate::state::card::Edition> = joker_editions
        .map(|opts| opts.iter().map(|o| crate::state::card::Edition::from_option_str(o.as_deref())).collect())
        .unwrap_or_default();
    let held = held_cards.unwrap_or_default();
    let metadata = build_metadata(
        jokers.len(),
        joker_current_plus_mult.as_deref(),
        joker_current_plus_chips.as_deref(),
        joker_current_xmult.as_deref(),
        joker_loyalty_ready.as_deref(),
        joker_drivers_active.as_deref(),
        joker_leading_plus_mult.as_deref(),
        joker_leading_plus_chips.as_deref(),
        joker_sell_value.as_deref(),
        joker_rarity.as_deref(),
        joker_target_suit.as_deref(),
        joker_target_rank.as_deref(),
        joker_obelisk_gain.as_deref(),
    );
    let ctx = HandContext {
        money,
        joker_count: jokers.len() as u32,
        joker_slot_limit,
        discards_remaining,
        hands_remaining,
        played_count_this_hand_type,
        hand_type_played_before,
        deck_size,
        played_count_max_other_hand_type: 0,
        pareidolia_active: false,
    };
    let played_types_vec = played_hand_types.unwrap_or_default();
    Ok(evaluate_simple(&cards, level, &suits, &jokers, &editions, &metadata, &held, &played_types_vec, ctx))
}

/// Compose per-joker metadata from up to thirteen parallel optional
/// lists. When a list is None or wrong-length, defaults to
/// zeros/1.0/false/none.
#[allow(clippy::too_many_arguments)]
fn build_metadata(
    n: usize,
    plus_mult: Option<&[i32]>,
    plus_chips: Option<&[i32]>,
    xmult: Option<&[f64]>,
    loyalty_ready: Option<&[bool]>,
    drivers_active: Option<&[bool]>,
    leading_plus_mult: Option<&[i32]>,
    leading_plus_chips: Option<&[i32]>,
    sell_value: Option<&[i32]>,
    rarity: Option<&[u8]>,
    target_suit: Option<&[Option<String>]>,
    target_rank: Option<&[Option<String>]>,
    obelisk_gain: Option<&[f64]>,
) -> Vec<JokerMetadata> {
    let mut out = vec![JokerMetadata::default(); n];
    if let Some(v) = plus_mult {
        for (i, m) in v.iter().take(n).enumerate() {
            out[i].current_plus_mult = *m;
        }
    }
    if let Some(v) = plus_chips {
        for (i, c) in v.iter().take(n).enumerate() {
            out[i].current_plus_chips = *c;
        }
    }
    if let Some(v) = xmult {
        for (i, x) in v.iter().take(n).enumerate() {
            out[i].current_xmult = *x;
        }
    }
    if let Some(v) = loyalty_ready {
        for (i, b) in v.iter().take(n).enumerate() {
            out[i].loyalty_ready = *b;
        }
    }
    if let Some(v) = drivers_active {
        for (i, b) in v.iter().take(n).enumerate() {
            out[i].drivers_active = *b;
        }
    }
    if let Some(v) = leading_plus_mult {
        for (i, m) in v.iter().take(n).enumerate() {
            out[i].leading_plus_mult = *m;
        }
    }
    if let Some(v) = leading_plus_chips {
        for (i, c) in v.iter().take(n).enumerate() {
            out[i].leading_plus_chips = *c;
        }
    }
    if let Some(v) = sell_value {
        for (i, s) in v.iter().take(n).enumerate() {
            out[i].sell_value = *s;
        }
    }
    if let Some(v) = rarity {
        for (i, r) in v.iter().take(n).enumerate() {
            out[i].rarity = *r;
        }
    }
    if let Some(v) = target_suit {
        for (i, s) in v.iter().take(n).enumerate() {
            if let Some(s_str) = s.as_deref() {
                if let Some(suit) = Suit::from_str(s_str) {
                    out[i].has_target_suit = true;
                    out[i].target_suit = suit as u8;
                }
            }
        }
    }
    if let Some(v) = target_rank {
        for (i, r) in v.iter().take(n).enumerate() {
            if let Some(r_str) = r.as_deref() {
                if let Some(rank) = crate::state::card::Rank::from_str(r_str) {
                    out[i].has_target_rank = true;
                    out[i].target_rank = rank as u8;
                }
            }
        }
    }
    if let Some(v) = obelisk_gain {
        for (i, g) in v.iter().take(n).enumerate() {
            out[i].obelisk_gain = *g;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::card::{Edition, Enhancement, Rank, Seal};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card {
            rank,
            suit,
            enhancement: Enhancement::None,
            edition: Edition::None,
            seal: Seal::None,
            debuffed: false,
        }
    }

    #[test]
    fn straight_at_level_1_scores_correctly() {
        // A-2-3-4-5 wheel straight, level 1.
        // base = (30 chips, 4 mult)
        // card chips = 11 (Ace) + 2 + 3 + 4 + 5 = 25
        // total = (30 + 25) * 4 = 220
        let cards = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Two, Suit::Diamonds),
            card(Rank::Three, Suit::Clubs),
            card(Rank::Four, Suit::Spades),
            card(Rank::Five, Suit::Hearts),
        ];
        let (chips, mult, score, ht) = evaluate_simple(&cards, 1, &[], &[], &[], &[], &[], &[], HandContext::default()).unwrap();
        assert_eq!(chips, 55);
        assert_eq!(mult, 4);
        assert_eq!(score, 220);
        assert_eq!(ht, "Straight");
    }

    #[test]
    fn pair_of_aces_at_level_1() {
        // Pair: base (10 chips, 2 mult), card chips = 11 + 11 = 22
        // total = (10 + 22) * 2 = 64
        let cards = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Spades),
        ];
        let (chips, mult, score, ht) = evaluate_simple(&cards, 1, &[], &[], &[], &[], &[], &[], HandContext::default()).unwrap();
        assert_eq!(chips, 32);
        assert_eq!(mult, 2);
        assert_eq!(score, 64);
        assert_eq!(ht, "Pair");
    }

    #[test]
    fn pair_at_level_3() {
        // Pair level 3 → base (10, 2) + 2 * (15, 1) = (40, 4)
        // + card chips 22 = 62 chips * 4 mult = 248
        let cards = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Spades),
        ];
        let (chips, mult, score, _) = evaluate_simple(&cards, 3, &[], &[], &[], &[], &[], &[], HandContext::default()).unwrap();
        assert_eq!(chips, 62);
        assert_eq!(mult, 4);
        assert_eq!(score, 248);
    }

    #[test]
    fn pair_of_aces_with_plain_joker() {
        // Pair: (10 + 22) chips × (2 + 4) mult = 32 × 6 = 192
        let cards = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Spades),
        ];
        let jokers = vec!["Joker".to_string()];
        let (chips, mult, score, _) = evaluate_simple(&cards, 1, &[], &jokers, &[], &[], &[], &[], HandContext::default()).unwrap();
        assert_eq!(chips, 32);
        assert_eq!(mult, 6);
        assert_eq!(score, 192);
    }

    #[test]
    fn unknown_joker_bails_to_none() {
        // "Future Joker" not in supported set → None.
        let cards = [card(Rank::Ace, Suit::Hearts), card(Rank::Ace, Suit::Spades)];
        let jokers = vec!["Future Joker".to_string()];
        assert!(evaluate_simple(&cards, 1, &[], &jokers, &[], &[], &[], &[], HandContext::default()).is_none());
    }

    #[test]
    fn empty_hand_returns_none() {
        assert_eq!(evaluate_simple(&[], 1, &[], &[], &[], &[], &[], &[], HandContext::default()), None);
    }
}
