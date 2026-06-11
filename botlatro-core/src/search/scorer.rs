//! Batched play-action scorer (Phase 4a of RUST_PORT_PLAN.md).
//!
//! Today Python's `_rank_plays` iterates ~218 candidate actions per
//! state and calls `_score_play_action` for each. Each call goes
//! through the Rust evaluate_simple wire-in but pays a Python-side
//! per-call cost (building the wrapper inputs, dispatching to
//! evaluate_simple_with_levels, unpacking the result).
//!
//! This function takes the shared inputs ONCE and a list of
//! per-action card-index sets, then scores all actions internally —
//! one FFI call per state instead of ~218.
//!
//! Returns the list of integer scores (matching Python's
//! `_score_play_action` return type — the score field of the
//! evaluation, NOT the boss-adjusted version; boss adjustments are
//! applied Python-side by the existing `_boss_adjusted_score`).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::hand_eval::effects::{HandContext, JokerMetadata};
use crate::hand_eval::evaluate::evaluate_simple;
use crate::hand_eval::hand_type::identify_hand_type_simple;
use crate::state::card::{Card, Edition, Rank, Suit};

/// Port of Python's `_discard_candidate_score` from
/// `search/discard_search.py:125`. Used to rank discard actions
/// by how good the resulting hand looks. Heuristic, not exact.
#[pyfunction]
#[pyo3(name = "discard_candidate_score")]
pub fn py_discard_candidate_score(
    hand: Vec<Card>,
    discard_indices: Vec<usize>,
) -> f64 {
    if discard_indices.is_empty() {
        return f64::NEG_INFINITY;
    }
    let n = hand.len();
    let mut in_sel = vec![false; n.max(1)];
    for &i in &discard_indices {
        if i < n { in_sel[i] = true; }
    }
    let selected: Vec<Card> = discard_indices.iter()
        .filter(|&&i| i < n)
        .map(|&i| hand[i])
        .collect();
    if selected.is_empty() {
        return f64::NEG_INFINITY;
    }
    let held: Vec<Card> = hand.iter().enumerate()
        .filter(|(i, _)| !in_sel.get(*i).copied().unwrap_or(false))
        .map(|(_, c)| *c)
        .collect();
    let mut score = (selected.len() as f64) * 0.5;
    for c in &selected {
        let rv = scorer_rank_value(c.rank) as f64;
        score += (15.0 - rv) * 0.25;
    }
    score += scorer_duplicate_support(&held) * 2.0;
    score += scorer_suit_support(&held) * 1.2;
    score += scorer_straight_support(&held) * 0.9;
    score -= scorer_duplicate_support(&selected) * 1.5;
    score -= scorer_suit_support(&selected) * 0.6;
    score
}

/// Batched version: rank all discard candidates for a hand in one
/// FFI call.
#[pyfunction]
#[pyo3(name = "discard_candidate_scores_batch")]
pub fn py_discard_candidate_scores_batch(
    hand: Vec<Card>,
    action_indices_list: Vec<Vec<usize>>,
) -> Vec<f64> {
    action_indices_list.into_iter()
        .map(|indices| py_discard_candidate_score(hand.clone(), indices))
        .collect()
}

#[inline]
fn scorer_rank_value(r: Rank) -> u8 {
    match r {
        Rank::Two => 2, Rank::Three => 3, Rank::Four => 4, Rank::Five => 5,
        Rank::Six => 6, Rank::Seven => 7, Rank::Eight => 8, Rank::Nine => 9,
        Rank::Ten => 10, Rank::Jack => 11, Rank::Queen => 12, Rank::King => 13,
        Rank::Ace => 14,
    }
}

fn scorer_duplicate_support(cards: &[Card]) -> f64 {
    let mut counts = [0u32; 16];
    for c in cards { counts[c.rank as usize] += 1; }
    counts.iter().map(|&n| if n >= 2 { (n * n) as f64 } else { 0.0 }).sum()
}

fn scorer_suit_support(cards: &[Card]) -> f64 {
    let mut counts = [0u32; 4];
    for c in cards { counts[c.suit as usize] += 1; }
    *counts.iter().max().unwrap_or(&0) as f64
}

fn scorer_straight_support(cards: &[Card]) -> f64 {
    let mut values: Vec<u8> = cards.iter().map(|c| scorer_rank_value(c.rank)).filter(|&v| v > 0).collect();
    values.sort_unstable();
    values.dedup();
    if values.contains(&14) {
        values.insert(0, 1);
        values.sort_unstable();
        values.dedup();
    }
    let mut best = 0u32;
    for start in 0..values.len() {
        let mut run = 1u32;
        for i in start+1..values.len() {
            if values[i] == values[i-1] + 1 {
                run += 1;
                if run > best { best = run; }
            } else if values[i] > values[i-1] + 1 {
                break;
            }
        }
    }
    best as f64
}

/// PyO3 wrapper. Takes:
/// - `hand`: the full hand as RustCards (converted once)
/// - `action_indices_list`: per-action card-index lists
/// - shared joker + ctx data (matching `evaluate_simple_with_levels`'s
///   signature)
/// Returns a Vec<i64> of scores (one per action), or — when
/// `with_hand_types` is set — a Vec of (score, hand_type_str) pairs so
/// callers that need the hand type (e.g. building play candidates) can
/// batch instead of paying one FFI call per action. The hand type was
/// always computed internally; this just stops discarding it.
///
/// Bails to Python (returns None) for any action whose hand type
/// can't be identified by the simple path (Wild cards, etc.).
#[pyfunction]
#[pyo3(name = "score_play_actions_batch")]
#[pyo3(signature = (hand, action_indices_list, hand_levels, debuffed_suits,
                    joker_names=None, joker_editions=None,
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
                    deck_size=0,
                    with_hand_types=false))]
#[allow(clippy::too_many_arguments)]
pub fn py_score_play_actions_batch(
    py: Python<'_>,
    hand: Vec<Card>,
    action_indices_list: Vec<Vec<usize>>,
    hand_levels: &Bound<'_, PyDict>,
    debuffed_suits: Vec<String>,
    joker_names: Option<Vec<String>>,
    joker_editions: Option<Vec<Option<String>>>,
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
    with_hand_types: bool,
) -> PyResult<PyObject> {
    let suits: Vec<Suit> = debuffed_suits.iter()
        .filter_map(|s| Suit::from_str(s))
        .collect();
    let jokers = joker_names.unwrap_or_default();
    let editions: Vec<Edition> = joker_editions
        .map(|opts| opts.iter().map(|o| Edition::from_option_str(o.as_deref())).collect())
        .unwrap_or_default();

    // Build per-joker metadata array once (shared across all actions).
    let n = jokers.len();
    let mut metadata = vec![JokerMetadata::default(); n];
    if let Some(v) = joker_current_plus_mult.as_deref() {
        for (i, m) in v.iter().take(n).enumerate() { metadata[i].current_plus_mult = *m; }
    }
    if let Some(v) = joker_current_plus_chips.as_deref() {
        for (i, c) in v.iter().take(n).enumerate() { metadata[i].current_plus_chips = *c; }
    }
    if let Some(v) = joker_current_xmult.as_deref() {
        for (i, x) in v.iter().take(n).enumerate() { metadata[i].current_xmult = *x; }
    }
    if let Some(v) = joker_loyalty_ready.as_deref() {
        for (i, b) in v.iter().take(n).enumerate() { metadata[i].loyalty_ready = *b; }
    }
    if let Some(v) = joker_drivers_active.as_deref() {
        for (i, b) in v.iter().take(n).enumerate() { metadata[i].drivers_active = *b; }
    }
    if let Some(v) = joker_leading_plus_mult.as_deref() {
        for (i, m) in v.iter().take(n).enumerate() { metadata[i].leading_plus_mult = *m; }
    }
    if let Some(v) = joker_leading_plus_chips.as_deref() {
        for (i, c) in v.iter().take(n).enumerate() { metadata[i].leading_plus_chips = *c; }
    }
    if let Some(v) = joker_sell_value.as_deref() {
        for (i, s) in v.iter().take(n).enumerate() { metadata[i].sell_value = *s; }
    }
    if let Some(v) = joker_rarity.as_deref() {
        for (i, r) in v.iter().take(n).enumerate() { metadata[i].rarity = *r; }
    }
    if let Some(v) = joker_target_suit.as_deref() {
        for (i, s) in v.iter().take(n).enumerate() {
            if let Some(s_str) = s.as_deref() {
                if let Some(suit) = Suit::from_str(s_str) {
                    metadata[i].has_target_suit = true;
                    metadata[i].target_suit = suit as u8;
                }
            }
        }
    }
    if let Some(v) = joker_target_rank.as_deref() {
        for (i, r) in v.iter().take(n).enumerate() {
            if let Some(r_str) = r.as_deref() {
                if let Some(rank) = crate::state::card::Rank::from_str(r_str) {
                    metadata[i].has_target_rank = true;
                    metadata[i].target_rank = rank as u8;
                }
            }
        }
    }
    if let Some(v) = joker_obelisk_gain.as_deref() {
        for (i, g) in v.iter().take(n).enumerate() { metadata[i].obelisk_gain = *g; }
    }

    let played_types_vec = played_hand_types.unwrap_or_default();
    let ctx = HandContext {
        money,
        joker_count: n as u32,
        joker_slot_limit,
        discards_remaining,
        hands_remaining,
        played_count_this_hand_type,
        hand_type_played_before,
        deck_size,
        played_count_max_other_hand_type: 0,  // computed inside evaluate_simple
        pareidolia_active: false,             // computed inside evaluate_simple
    };

    // Iterate actions, building (played, held) slices per action.
    let mut scores: Vec<Option<(u64, &'static str)>> = Vec::with_capacity(action_indices_list.len());
    for indices in &action_indices_list {
        // Build played + held card lists.
        let played: Vec<Card> = indices.iter().filter_map(|&i| hand.get(i).copied()).collect();
        if played.len() != indices.len() {
            scores.push(None);
            continue;
        }
        // Membership mask instead of a per-action HashSet alloc+hash. Hands are
        // <=16 cards and the guard above guarantees every i in indices indexes a
        // real hand slot (i < hand.len() <= 16), so this reproduces
        // !played_set.contains(i) exactly with no heap allocation.
        let mut in_played = [false; 16];
        for &i in indices.iter() {
            if i < 16 {
                in_played[i] = true;
            }
        }
        let held: Vec<Card> = hand.iter().enumerate()
            .filter(|(i, _)| !(*i < 16 && in_played[*i]))
            .map(|(_, c)| *c)
            .collect();

        // Identify hand type to look up level.
        let level = match identify_hand_type_simple(&played) {
            Some(ht) => {
                let ht_str = ht.to_str();
                hand_levels.get_item(ht_str)?
                    .map(|v| v.extract().unwrap_or(1u32))
                    .unwrap_or(1)
            }
            None => {
                // Wild card / can't identify simple → bail this action only.
                scores.push(None);
                continue;
            }
        };

        let r = evaluate_simple(
            &played, level, &suits,
            &jokers, &editions, &metadata,
            &held, &played_types_vec, ctx,
        );
        scores.push(r.map(|(_chips, _mult, score, ht)| (score, ht)));
    }
    if with_hand_types {
        Ok(scores.into_py(py))
    } else {
        let plain: Vec<Option<u64>> = scores.into_iter().map(|s| s.map(|(score, _)| score)).collect();
        Ok(plain.into_py(py))
    }
}

/// Best play action across all legal 1..=5 card combinations of the
/// hand. Returns `(best_indices, best_score)` or None to bail (Wild
/// cards in hand, unsupported jokers, etc.). Saves the Python action-
/// enumeration loop on top of the batched scorer (Phase 4c).
///
/// Caller still has the bail predicates (blind safety, etc.); this
/// function trusts that hand_levels + jokers are fully Rust-supported.
#[pyfunction]
#[pyo3(name = "best_play_action")]
#[pyo3(signature = (hand, hand_levels, debuffed_suits,
                    joker_names=None, joker_editions=None,
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
pub fn py_best_play_action(
    hand: Vec<Card>,
    hand_levels: &Bound<'_, PyDict>,
    debuffed_suits: Vec<String>,
    joker_names: Option<Vec<String>>,
    joker_editions: Option<Vec<Option<String>>>,
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
) -> PyResult<Option<(Vec<usize>, u64)>> {
    // Build shared metadata array (same as batched scorer).
    let suits: Vec<Suit> = debuffed_suits.iter()
        .filter_map(|s| Suit::from_str(s))
        .collect();
    let jokers = joker_names.unwrap_or_default();
    let editions: Vec<Edition> = joker_editions
        .map(|opts| opts.iter().map(|o| Edition::from_option_str(o.as_deref())).collect())
        .unwrap_or_default();
    let n_jokers = jokers.len();
    let mut metadata = vec![JokerMetadata::default(); n_jokers];
    if let Some(v) = joker_current_plus_mult.as_deref() {
        for (i, m) in v.iter().take(n_jokers).enumerate() { metadata[i].current_plus_mult = *m; }
    }
    if let Some(v) = joker_current_plus_chips.as_deref() {
        for (i, c) in v.iter().take(n_jokers).enumerate() { metadata[i].current_plus_chips = *c; }
    }
    if let Some(v) = joker_current_xmult.as_deref() {
        for (i, x) in v.iter().take(n_jokers).enumerate() { metadata[i].current_xmult = *x; }
    }
    if let Some(v) = joker_loyalty_ready.as_deref() {
        for (i, b) in v.iter().take(n_jokers).enumerate() { metadata[i].loyalty_ready = *b; }
    }
    if let Some(v) = joker_drivers_active.as_deref() {
        for (i, b) in v.iter().take(n_jokers).enumerate() { metadata[i].drivers_active = *b; }
    }
    if let Some(v) = joker_leading_plus_mult.as_deref() {
        for (i, m) in v.iter().take(n_jokers).enumerate() { metadata[i].leading_plus_mult = *m; }
    }
    if let Some(v) = joker_leading_plus_chips.as_deref() {
        for (i, c) in v.iter().take(n_jokers).enumerate() { metadata[i].leading_plus_chips = *c; }
    }
    if let Some(v) = joker_sell_value.as_deref() {
        for (i, s) in v.iter().take(n_jokers).enumerate() { metadata[i].sell_value = *s; }
    }
    if let Some(v) = joker_rarity.as_deref() {
        for (i, r) in v.iter().take(n_jokers).enumerate() { metadata[i].rarity = *r; }
    }
    if let Some(v) = joker_target_suit.as_deref() {
        for (i, s) in v.iter().take(n_jokers).enumerate() {
            if let Some(s_str) = s.as_deref() {
                if let Some(suit) = Suit::from_str(s_str) {
                    metadata[i].has_target_suit = true;
                    metadata[i].target_suit = suit as u8;
                }
            }
        }
    }
    if let Some(v) = joker_target_rank.as_deref() {
        for (i, r) in v.iter().take(n_jokers).enumerate() {
            if let Some(r_str) = r.as_deref() {
                if let Some(rank) = crate::state::card::Rank::from_str(r_str) {
                    metadata[i].has_target_rank = true;
                    metadata[i].target_rank = rank as u8;
                }
            }
        }
    }
    if let Some(v) = joker_obelisk_gain.as_deref() {
        for (i, g) in v.iter().take(n_jokers).enumerate() { metadata[i].obelisk_gain = *g; }
    }

    let played_types_vec = played_hand_types.unwrap_or_default();
    let ctx = HandContext {
        money,
        joker_count: n_jokers as u32,
        joker_slot_limit,
        discards_remaining,
        hands_remaining,
        played_count_this_hand_type,
        hand_type_played_before,
        deck_size,
        played_count_max_other_hand_type: 0,
        pareidolia_active: false,
    };

    // Enumerate all combinations of size 1..=min(5, hand_size) inline.
    // Score each, track argmax. No Python loop, no Python tuple
    // allocations per action.
    let n = hand.len();
    let max_size = n.min(5);
    let mut best_score: u64 = 0;
    let mut best_indices: Option<Vec<usize>> = None;
    let mut first = true; // ensure first valid result wins over default 0
    let mut idx = vec![0usize; max_size];
    for size in 1..=max_size {
        // Initialize first combination of this size.
        for i in 0..size { idx[i] = i; }
        loop {
            // Build played + held card lists.
            let played: Vec<Card> = idx[..size].iter().map(|&i| hand[i]).collect();
            let mut in_played = [false; 16];
            for &i in &idx[..size] { in_played[i] = true; }
            let held: Vec<Card> = hand.iter().enumerate()
                .filter(|(i, _)| !in_played[*i])
                .map(|(_, c)| *c)
                .collect();

            // Identify hand type → level.
            let level = match identify_hand_type_simple(&played) {
                Some(ht) => {
                    let ht_str = ht.to_str();
                    hand_levels.get_item(ht_str)?
                        .map(|v| v.extract().unwrap_or(1u32))
                        .unwrap_or(1)
                }
                None => 1, // shouldn't matter; result will likely be None
            };

            if let Some((_chips, _mult, score, _ht)) = evaluate_simple(
                &played, level, &suits,
                &jokers, &editions, &metadata,
                &held, &played_types_vec, ctx,
            ) {
                if first || score > best_score {
                    first = false;
                    best_score = score;
                    best_indices = Some(idx[..size].to_vec());
                }
            }

            // Advance combination idx[..size] (lex next).
            let mut k: Option<usize> = None;
            for i in (0..size).rev() {
                if idx[i] < n - size + i {
                    k = Some(i);
                    break;
                }
            }
            let Some(k) = k else { break };
            idx[k] += 1;
            for i in k+1..size {
                idx[i] = idx[i-1] + 1;
            }
        }
    }

    match best_indices {
        Some(v) => Ok(Some((v, best_score))),
        None => Ok(None),
    }
}
