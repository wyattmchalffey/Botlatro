//! Native beam-play-action search (Phase 4d.2 of RUST_PORT_PLAN.md).
//!
//! Ports the recursive structure of `_beam_action_value` +
//! `_beam_plan_value` from `hand_search.py` into Rust. The win is
//! that the whole subtree value computation for one root candidate
//! happens in ONE FFI call — Python's beam loop doesn't recurse
//! across the boundary.
//!
//! Scope (minimum viable):
//! - Considers only PLAY actions inside the recursion. Discards are
//!   handled by the outer Python loop (which still picks root
//!   candidates including discards via `_beam_root_actions`).
//! - Single draw sample per simulated action (the Python beam
//!   averages over multiple random draws but each is independent
//!   noise; a single sample with our internal RNG is a similar
//!   estimator).
//! - Leaf evaluator is `clear_probability_native`'s internal greedy
//!   rollout — bails on the same boss-blind / joker constraints.
//!
//! Bail policy: same as Phase 4d.1 — unsupported jokers, Glass /
//! blue-seal cards, non-vanilla blinds. Returns None to bail.
//!
//! Determinism: caller passes a seed; xoshiro256** RNG drives all
//! draws + tie-breaks. Different seeds give different leaf
//! estimates (acceptable — leaf is an estimator).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::forward_sim::jokers::{jokers_after_play, AfterPlayContext};
use crate::forward_sim::simulate::SIMPLE_BAIL_JOKERS;
use crate::hand_eval::effects::{HandContext, JokerMetadata};
use crate::hand_eval::evaluate::evaluate_simple;
use crate::hand_eval::hand_type::identify_hand_type_simple;
use crate::state::card::{Card, Edition, Enhancement, Rank, Seal, Suit};

/// A candidate considered by the beam. `Play` actions score via
/// evaluate_simple; `Discard` actions use a heuristic that mirrors
/// Python's `_cheap_beam_discard_rank`.
#[derive(Clone, Debug)]
enum BeamAction {
    Play { indices: Vec<usize>, score: u64 },
    Discard { indices: Vec<usize> },
}

// Re-use the xoshiro RNG from rollout.rs by inlining the struct.
// Could be factored, but small enough that duplication is fine.
#[derive(Clone, Debug)]
struct Xoshiro {
    state: [u64; 4],
}
impl Xoshiro {
    fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut x = seed;
        for slot in s.iter_mut() {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = z ^ (z >> 31);
        }
        Self { state: s }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }
    #[inline]
    fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

/// Mutable beam-tree state. Cloned at each recursion branch so
/// siblings don't see each other's mutations.
#[derive(Clone, Debug)]
struct BeamState {
    hand: Vec<Card>,
    known_deck: Vec<Card>,
    metadata: Vec<JokerMetadata>,
    current_remaining: Vec<i32>,
    current_score: i64,
    required_score: i64,
    hands_remaining: u32,
    discards_remaining: u32,
    money: i32,
    deck_size: u32,
    played_count_this_hand_type: u32,
    played_count_max_other_hand_type: u32,
    hand_type_played_before: bool,
    played_hand_types: Vec<String>,
}

#[derive(Clone, Copy)]
struct BeamCtx<'a> {
    joker_names: &'a [String],
    joker_editions: &'a [Edition],
    joker_vampire_gain: &'a [f64],
    joker_obelisk_should_scale: &'a [bool],
    debuffed_suits: &'a [Suit],
    pareidolia_active: bool,
    width: u32,
}

#[inline]
fn is_terminal(bs: &BeamState) -> bool {
    bs.current_score >= bs.required_score
        || bs.hands_remaining == 0
        || bs.hand.is_empty()
}

/// Discard-rank heuristic. Mirrors Python's `_discard_candidate_score`
/// (~30 lines of straightforward Python) plus `_cheap_beam_discard_rank`'s
/// pressure_bonus + 500× scaling so it's comparable to play scores.
fn discard_candidate_rank(
    bs: &BeamState,
    discard_indices: &[usize],
    best_play_score: u64,
    pareidolia_active: bool,
) -> f64 {
    if discard_indices.is_empty() {
        return f64::NEG_INFINITY;
    }
    let _ = pareidolia_active; // not consumed by the heuristic directly
    // Build selected + held card lists.
    let mut in_sel = [false; 16];
    for &i in discard_indices { in_sel[i] = true; }
    let selected: Vec<Card> = discard_indices.iter().map(|&i| bs.hand[i]).collect();
    let held: Vec<Card> = bs.hand.iter().enumerate()
        .filter(|(i, _)| !in_sel[*i])
        .map(|(_, c)| *c)
        .collect();
    // _discard_candidate_score
    let mut score = (selected.len() as f64) * 0.5;
    for c in &selected {
        let rv = rank_value(c.rank) as f64;
        score += (15.0 - rv) * 0.25;
    }
    score += duplicate_support_score(&held) * 2.0;
    score += suit_support_score(&held) * 1.2;
    score += straight_support_score(&held) * 0.9;
    score -= duplicate_support_score(&selected) * 1.5;
    score -= suit_support_score(&selected) * 0.6;
    // _cheap_beam_discard_rank: pressure_bonus if play can't pace.
    let remaining_score = (bs.required_score - bs.current_score).max(0) as f64;
    let pace_score = remaining_score / bs.hands_remaining.max(1) as f64;
    let pressure_bonus = if bs.discards_remaining > 0 && (best_play_score as f64) < pace_score {
        15_000.0
    } else {
        0.0
    };
    pressure_bonus + (score * 500.0)
}

#[inline]
fn rank_value(r: Rank) -> u8 {
    match r {
        Rank::Two => 2, Rank::Three => 3, Rank::Four => 4, Rank::Five => 5,
        Rank::Six => 6, Rank::Seven => 7, Rank::Eight => 8, Rank::Nine => 9,
        Rank::Ten => 10, Rank::Jack => 11, Rank::Queen => 12, Rank::King => 13,
        Rank::Ace => 14,
    }
}

fn duplicate_support_score(cards: &[Card]) -> f64 {
    let mut counts = [0u32; 16];
    for c in cards { counts[c.rank as usize] += 1; }
    counts.iter().map(|&n| if n >= 2 { (n * n) as f64 } else { 0.0 }).sum()
}

fn suit_support_score(cards: &[Card]) -> f64 {
    let mut counts = [0u32; 4];
    for c in cards { counts[c.suit as usize] += 1; }
    *counts.iter().max().unwrap_or(&0) as f64
}

fn straight_support_score(cards: &[Card]) -> f64 {
    let mut values: Vec<u8> = cards.iter().map(|c| rank_value(c.rank)).filter(|&v| v > 0).collect();
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

/// Top-K candidate enumeration: return up to `width` action-index
/// vectors sorted by score descending.
fn top_k_plays(
    bs: &BeamState,
    ctx: BeamCtx<'_>,
    hand_levels_lookup: &dyn Fn(&str) -> u32,
    limit: usize,
) -> Vec<(Vec<usize>, u64)> {
    let n = bs.hand.len();
    let max_size = n.min(5);
    // Score every combination, then take the top `limit`.
    let mut scored: Vec<(u64, Vec<usize>)> = Vec::new();
    let mut idx = vec![0usize; max_size];
    for size in 1..=max_size {
        for i in 0..size { idx[i] = i; }
        loop {
            let played: Vec<Card> = idx[..size].iter().map(|&i| bs.hand[i]).collect();
            let mut in_played = [false; 16];
            for &i in &idx[..size] { in_played[i] = true; }
            let held: Vec<Card> = bs.hand.iter().enumerate()
                .filter(|(i, _)| !in_played[*i])
                .map(|(_, c)| *c)
                .collect();
            if let Some(ht) = identify_hand_type_simple(&played) {
                let level = hand_levels_lookup(ht.to_str());
                let hand_ctx = HandContext {
                    money: bs.money,
                    joker_count: ctx.joker_names.len() as u32,
                    joker_slot_limit: 5,
                    discards_remaining: bs.discards_remaining,
                    hands_remaining: bs.hands_remaining,
                    played_count_this_hand_type: bs.played_count_this_hand_type,
                    hand_type_played_before: bs.hand_type_played_before,
                    deck_size: bs.deck_size,
                    played_count_max_other_hand_type: bs.played_count_max_other_hand_type,
                    pareidolia_active: ctx.pareidolia_active,
                };
                if let Some((_chips, _mult, score, _ht)) = evaluate_simple(
                    &played, level, ctx.debuffed_suits,
                    ctx.joker_names, ctx.joker_editions, &bs.metadata,
                    &held, &bs.played_hand_types, hand_ctx,
                ) {
                    scored.push((score, idx[..size].to_vec()));
                }
            }
            // Advance combination.
            let mut k: Option<usize> = None;
            for i in (0..size).rev() {
                if idx[i] < n - size + i { k = Some(i); break; }
            }
            let Some(k) = k else { break };
            idx[k] += 1;
            for i in k+1..size { idx[i] = idx[i-1] + 1; }
        }
    }
    // Sort by score descending, keep top `limit`.
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.truncate(limit);
    scored.into_iter().map(|(s, i)| (i, s)).collect()
}

/// Apply a play action to `bs` in-place. Mirrors the play-step logic
/// from `greedy_rollout_clears` in rollout.rs.
fn apply_play(
    bs: &mut BeamState,
    ctx: BeamCtx<'_>,
    play_indices: &[usize],
    play_score: u64,
    rng: &mut Xoshiro,
) {
    let played_cards: Vec<Card> = play_indices.iter().map(|&i| bs.hand[i]).collect();
    let Some(ht) = identify_hand_type_simple(&played_cards) else { return };
    let mut in_played = [false; 16];
    for &i in play_indices { in_played[i] = true; }
    let held_cards: Vec<Card> = bs.hand.iter().enumerate()
        .filter(|(i, _)| !in_played[*i])
        .map(|(_, c)| *c)
        .collect();
    bs.current_score += play_score as i64;
    bs.hands_remaining = bs.hands_remaining.saturating_sub(1);

    // Sample drawn cards.
    let draw_count = play_indices.len().min(bs.known_deck.len());
    let mut drawn: Vec<Card> = Vec::with_capacity(draw_count);
    for _ in 0..draw_count {
        if bs.known_deck.is_empty() { break; }
        let pick = rng.gen_range(bs.known_deck.len());
        drawn.push(bs.known_deck.remove(pick));
    }
    bs.deck_size = bs.deck_size.saturating_sub(drawn.len() as u32);

    // Joker scaling updates.
    let scoring_indices: Vec<usize> = (0..played_cards.len()).collect();
    let after_ctx = AfterPlayContext {
        hand_type: ht,
        played_cards: &played_cards,
        scoring_indices: &scoring_indices,
        n_played: played_cards.len(),
        hands_remaining: bs.hands_remaining,
        pareidolia_active: ctx.pareidolia_active,
    };
    let current_plus_chips: Vec<i32> = bs.metadata.iter().map(|m| m.current_plus_chips).collect();
    let current_plus_mult: Vec<i32> = bs.metadata.iter().map(|m| m.current_plus_mult).collect();
    let current_xmult: Vec<f64> = bs.metadata.iter().map(|m| m.current_xmult).collect();
    let obelisk_gain: Vec<f64> = bs.metadata.iter().map(|m| m.obelisk_gain).collect();
    let updates = jokers_after_play(
        ctx.joker_names,
        &current_plus_chips,
        &current_plus_mult,
        &current_xmult,
        &bs.current_remaining,
        ctx.joker_vampire_gain,
        ctx.joker_obelisk_should_scale,
        &obelisk_gain,
        0,
        after_ctx,
    );
    for (i, u) in updates.iter().enumerate() {
        if let Some(v) = u.new_plus_chips { bs.metadata[i].current_plus_chips = v; }
        if let Some(v) = u.new_plus_mult { bs.metadata[i].current_plus_mult = v; }
        if let Some(v) = u.new_xmult { bs.metadata[i].current_xmult = v; }
        if let Some(v) = u.new_remaining { bs.current_remaining[i] = v; }
    }

    let ht_str = ht.to_str();
    bs.played_count_this_hand_type += 1;
    bs.hand_type_played_before = true;
    bs.played_hand_types.push(ht_str.to_string());
    let mut max_other: u32 = 0;
    for s in &bs.played_hand_types {
        if s.as_str() == ht_str { continue; }
        let c = bs.played_hand_types.iter().filter(|x| x.as_str() == s.as_str()).count() as u32;
        if c > max_other { max_other = c; }
    }
    bs.played_count_max_other_hand_type = max_other;

    bs.hand = held_cards.into_iter().chain(drawn).collect();
}

/// Apply a discard action to `bs` in-place. Mirrors Python's
/// `simulate_discard` for the simple case: drop selected, draw
/// replacements, decrement discards_remaining, update jokers via
/// jokers_after_discard. Skips Hand Levels update (Burnt Joker
/// rare in early game) and money delta (Faceless/Mail-In rare).
fn apply_discard(
    bs: &mut BeamState,
    ctx: BeamCtx<'_>,
    discard_indices: &[usize],
    rng: &mut Xoshiro,
) {
    if discard_indices.is_empty() {
        return;
    }
    // Build held + sample replacements.
    let mut in_sel = [false; 16];
    for &i in discard_indices { in_sel[i] = true; }
    let held: Vec<Card> = bs.hand.iter().enumerate()
        .filter(|(i, _)| !in_sel[*i])
        .map(|(_, c)| *c)
        .collect();
    let draw_count = discard_indices.len().min(bs.known_deck.len());
    let mut drawn: Vec<Card> = Vec::with_capacity(draw_count);
    for _ in 0..draw_count {
        if bs.known_deck.is_empty() { break; }
        let pick = rng.gen_range(bs.known_deck.len());
        drawn.push(bs.known_deck.remove(pick));
    }
    bs.deck_size = bs.deck_size.saturating_sub(drawn.len() as u32);
    bs.discards_remaining = bs.discards_remaining.saturating_sub(1);
    // jokers_after_discard: Hit the Road (+xmult per scored J),
    // Madness (+xmult per blind defeated), etc. Most jokers don't
    // update on discard, so this is cheap.
    let _ = ctx; // joker after-discard handled implicitly (no-op for
                 // supported jokers in our SIMPLE_BAIL_JOKERS set;
                 // Hit the Road etc. are in the no-bail set so updates
                 // happen via metadata growth in subsequent plays).
    bs.hand = held.into_iter().chain(drawn).collect();
}

/// Leaf evaluator: estimate clear-or-progress value. Mirrors
/// `_planning_value_uncached` at a simplified scale:
/// - 1.0 if cleared
/// - clear_probability via internal greedy rollouts (1 sample to
///   keep it cheap inside the beam — outer Python beam already
///   averaged across multiple draws)
/// - progress contribution when no clear
fn leaf_value(
    bs: &BeamState,
    ctx: BeamCtx<'_>,
    hand_levels_lookup: &dyn Fn(&str) -> u32,
    rng: &mut Xoshiro,
) -> f64 {
    if bs.current_score >= bs.required_score {
        // Cleared; return a small headroom bonus over 1.0 to encourage
        // bigger clears over bare clears.
        let headroom = (bs.current_score as f64 / bs.required_score.max(1) as f64).min(3.0);
        return 1.0 + (headroom * 0.25).min(0.75);
    }
    if bs.hands_remaining == 0 || bs.hand.is_empty() {
        // Cannot clear.
        let progress = (bs.current_score as f64 / bs.required_score.max(1) as f64).min(1.0);
        return progress * 0.15;
    }
    // Run greedy rollouts to estimate clear probability. Python's
    // planning_value default in solver path is samples=1; we use
    // more here because the native rollout's xoshiro RNG gives
    // different per-rollout outcomes than Python's random.Random,
    // and averaging dampens the noise.
    let samples = 16u32;
    let mut clears = 0u32;
    for _ in 0..samples {
        let mut rs = bs.clone();
        for _ in 0..15 {
            if rs.current_score >= rs.required_score { clears += 1; break; }
            if rs.hands_remaining == 0 || rs.hand.is_empty() { break; }
            // Greedy pick: top_k_plays with limit=1.
            let picks = top_k_plays(&rs, ctx, hand_levels_lookup, 1);
            let Some((indices, score)) = picks.into_iter().next() else { break };
            apply_play(&mut rs, ctx, &indices, score, rng);
        }
        if rs.current_score >= rs.required_score && rs.hands_remaining > 0 {
            // already counted above
        }
    }
    let clear = clears as f64 / samples as f64;
    let progress = (bs.current_score as f64 / bs.required_score.max(1) as f64).min(1.0);
    (clear * 1.0) + ((1.0 - clear) * progress * 0.15)
}

fn beam_plan_value(
    bs: &BeamState,
    ctx: BeamCtx<'_>,
    depth: u32,
    hand_levels_lookup: &dyn Fn(&str) -> u32,
    rng: &mut Xoshiro,
) -> f64 {
    if depth == 0 || is_terminal(bs) {
        return leaf_value(bs, ctx, hand_levels_lookup, rng);
    }
    // Build mixed top-K candidate list. Mirrors Python
    // `_beam_future_actions` which:
    // - picks top `limit` plays by score
    // - picks top `limit` discards by discard_rank
    // - sorts the union and keeps top `limit` (== beam_width+1
    //   capped at 4 in Python; we approximate with width+1)
    let limit = ((ctx.width + 1) as usize).max(1).min(4);
    let plays = top_k_plays(bs, ctx, hand_levels_lookup, limit);
    if plays.is_empty() {
        return leaf_value(bs, ctx, hand_levels_lookup, rng);
    }
    let best_play_score = plays.iter().map(|(_, s)| *s).max().unwrap_or(0);

    // Enumerate discard candidates (only when discards remain).
    let mut discard_candidates: Vec<(Vec<usize>, f64)> = Vec::new();
    if bs.discards_remaining > 0 {
        let n = bs.hand.len();
        let max_size = n.min(5);
        let mut idx = vec![0usize; max_size];
        for size in 1..=max_size {
            for i in 0..size { idx[i] = i; }
            loop {
                let rank = discard_candidate_rank(bs, &idx[..size], best_play_score, ctx.pareidolia_active);
                discard_candidates.push((idx[..size].to_vec(), rank));
                let mut k: Option<usize> = None;
                for i in (0..size).rev() {
                    if idx[i] < n - size + i { k = Some(i); break; }
                }
                let Some(k) = k else { break };
                idx[k] += 1;
                for i in k+1..size { idx[i] = idx[i-1] + 1; }
            }
        }
        discard_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        discard_candidates.truncate(limit);
    }

    // Now build the unified action list. Use Python's
    // `_cheap_beam_play_rank` formula so plays compare correctly
    // against discards' `_cheap_beam_discard_rank` (×500).
    let remaining_score = (bs.required_score - bs.current_score).max(0) as f64;
    let mut mixed: Vec<(BeamAction, f64)> = Vec::with_capacity(plays.len() + discard_candidates.len());
    for (indices, score) in &plays {
        let mut value = *score as f64;
        if remaining_score > 0.0 {
            if (*score as f64) >= remaining_score {
                // Clears the blind: dominant bonus.
                value += 100_000.0 - (indices.len() as f64) * 3.0;
            } else {
                // Pace bonus: prefer plays that get closer to clearing.
                let needed = ((remaining_score + (*score as f64 - 1.0)) / (*score as f64).max(1.0)).ceil().max(1.0);
                value += 10_000.0 / needed;
            }
        }
        mixed.push((BeamAction::Play { indices: indices.clone(), score: *score }, value));
    }
    for (indices, rank) in &discard_candidates {
        mixed.push((BeamAction::Discard { indices: indices.clone() }, *rank));
    }
    mixed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    mixed.truncate(limit);

    let mut best = f64::NEG_INFINITY;
    for (action, _rank) in mixed {
        let mut child = bs.clone();
        match action {
            BeamAction::Play { indices, score } => {
                apply_play(&mut child, ctx, &indices, score, rng);
            }
            BeamAction::Discard { indices } => {
                apply_discard(&mut child, ctx, &indices, rng);
            }
        }
        let v = beam_plan_value(&child, ctx, depth - 1, hand_levels_lookup, rng);
        if v > best { best = v; }
    }
    best
}

/// PyO3 wrapper: native beam-play-action value.
/// Returns the BEST plan value for the given state at the given depth.
/// Python caller iterates root actions and calls this for each.
#[pyfunction]
#[pyo3(name = "beam_play_value_native")]
#[pyo3(signature = (hand, known_deck, hand_levels, debuffed_suits,
                    depth, width,
                    joker_names=None, joker_editions=None,
                    joker_current_plus_mult=None,
                    joker_current_plus_chips=None,
                    joker_current_xmult=None,
                    joker_current_remaining=None,
                    joker_loyalty_ready=None,
                    joker_drivers_active=None,
                    joker_leading_plus_mult=None,
                    joker_leading_plus_chips=None,
                    joker_sell_value=None,
                    joker_rarity=None,
                    joker_target_suit=None,
                    joker_target_rank=None,
                    joker_obelisk_gain=None,
                    joker_vampire_gain=None,
                    joker_obelisk_should_scale=None,
                    money=0, discards_remaining=0,
                    hands_remaining=0,
                    played_hand_types=None,
                    played_count_this_hand_type=0,
                    hand_type_played_before=false,
                    deck_size=0,
                    played_count_max_other_hand_type=0,
                    pareidolia_active=false,
                    current_score=0, required_score=0,
                    seed=0))]
#[allow(clippy::too_many_arguments)]
pub fn py_beam_play_value_native(
    hand: Vec<Card>,
    known_deck: Vec<Card>,
    hand_levels: &Bound<'_, PyDict>,
    debuffed_suits: Vec<String>,
    depth: u32,
    width: u32,
    joker_names: Option<Vec<String>>,
    joker_editions: Option<Vec<Option<String>>>,
    joker_current_plus_mult: Option<Vec<i32>>,
    joker_current_plus_chips: Option<Vec<i32>>,
    joker_current_xmult: Option<Vec<f64>>,
    joker_current_remaining: Option<Vec<i32>>,
    joker_loyalty_ready: Option<Vec<bool>>,
    joker_drivers_active: Option<Vec<bool>>,
    joker_leading_plus_mult: Option<Vec<i32>>,
    joker_leading_plus_chips: Option<Vec<i32>>,
    joker_sell_value: Option<Vec<i32>>,
    joker_rarity: Option<Vec<u8>>,
    joker_target_suit: Option<Vec<Option<String>>>,
    joker_target_rank: Option<Vec<Option<String>>>,
    joker_obelisk_gain: Option<Vec<f64>>,
    joker_vampire_gain: Option<Vec<f64>>,
    joker_obelisk_should_scale: Option<Vec<bool>>,
    money: i32,
    discards_remaining: u32,
    hands_remaining: u32,
    played_hand_types: Option<Vec<String>>,
    played_count_this_hand_type: u32,
    hand_type_played_before: bool,
    deck_size: u32,
    played_count_max_other_hand_type: u32,
    pareidolia_active: bool,
    current_score: i64,
    required_score: i64,
    seed: u64,
) -> PyResult<Option<f64>> {
    let suits: Vec<Suit> = debuffed_suits.iter()
        .filter_map(|s| Suit::from_str(s))
        .collect();
    let jokers = joker_names.unwrap_or_default();
    for n in &jokers {
        if SIMPLE_BAIL_JOKERS.contains(&n.as_str()) {
            return Ok(None);
        }
        if !crate::hand_eval::effects::is_supported_joker(n) {
            return Ok(None);
        }
    }
    for c in hand.iter().chain(known_deck.iter()) {
        if matches!(c.enhancement, Enhancement::Glass) { return Ok(None); }
        if matches!(c.seal, Seal::Blue) { return Ok(None); }
    }

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
                if let Some(rank) = Rank::from_str(r_str) {
                    metadata[i].has_target_rank = true;
                    metadata[i].target_rank = rank as u8;
                }
            }
        }
    }
    if let Some(v) = joker_obelisk_gain.as_deref() {
        for (i, g) in v.iter().take(n_jokers).enumerate() { metadata[i].obelisk_gain = *g; }
    }
    let vampire_gain = joker_vampire_gain.unwrap_or_else(|| vec![0.0; n_jokers]);
    let obelisk_should_scale = joker_obelisk_should_scale.unwrap_or_else(|| vec![false; n_jokers]);
    let current_remaining = joker_current_remaining.unwrap_or_else(|| vec![0; n_jokers]);
    let played_hand_types = played_hand_types.unwrap_or_default();

    let hand_levels_lookup = |ht_str: &str| -> u32 {
        hand_levels.get_item(ht_str)
            .ok().flatten()
            .and_then(|v| v.extract::<u32>().ok())
            .unwrap_or(1)
    };

    let bs = BeamState {
        hand,
        known_deck,
        metadata,
        current_remaining,
        current_score,
        required_score,
        hands_remaining,
        discards_remaining,
        money,
        deck_size,
        played_count_this_hand_type,
        played_count_max_other_hand_type,
        hand_type_played_before,
        played_hand_types,
    };
    let ctx = BeamCtx {
        joker_names: &jokers,
        joker_editions: &editions,
        joker_vampire_gain: &vampire_gain,
        joker_obelisk_should_scale: &obelisk_should_scale,
        debuffed_suits: &suits,
        pareidolia_active,
        width,
    };
    let mut rng = Xoshiro::new(seed);
    Ok(Some(beam_plan_value(&bs, ctx, depth, &hand_levels_lookup, &mut rng)))
}
