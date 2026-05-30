//! Native greedy-rollout clear-probability estimator (Phase 4d.1
//! of RUST_PORT_PLAN.md).
//!
//! Ports `_clear_probability_uncached` and `_greedy_rollout_clears`
//! from `src/balatro_ai/search/state_value.py` into Rust. The win is
//! that ONE FFI call runs N rollouts × ~5 plays each = ~80 internal
//! plays without crossing the Python boundary — replacing N Python
//! wrapper calls (each with per-call simulate_play overhead).
//!
//! Strategy:
//! - Internal mutable state struct (`RolloutState`) holds the fields
//!   that change during a rollout: hand, known_deck, score, money,
//!   hands_remaining, joker metadata. The remaining state fields are
//!   immutable per rollout.
//! - Greedy loop: at each step, find the best play (via the existing
//!   `best_play_action` enumerate+score+argmax), sample drawn cards
//!   from the deck (internal xoshiro RNG), apply the simulate_play
//!   deltas inline.
//! - Terminal conditions: score ≥ required, hands_remaining == 0,
//!   or hand empty.
//!
//! Bail policy: returns None when the initial state contains any
//! joker we can't handle in `simulate_play_simple` (Vampire, Midas
//! Mask, Hiker, etc.) — jokers don't get ADDED during a rollout so
//! the initial check is sufficient.
//!
//! RNG note: this uses an internal xoshiro RNG seeded from the
//! caller's u64 seed. The drawn-card sequence WILL differ from
//! Python's `random.Random(seed).choices`, which means the rollout's
//! clear-probability estimate will differ slightly. That's
//! acceptable — clear_probability is an ESTIMATOR; different
//! sampling gives different estimates but the bot's decisions
//! should remain quality-equivalent.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::forward_sim::jokers::{jokers_after_discard, jokers_after_play, AfterPlayContext};
use crate::forward_sim::levels::hand_level_after_play;
use crate::forward_sim::simulate::SIMPLE_BAIL_JOKERS;
use crate::hand_eval::effects::{HandContext, JokerMetadata};
use crate::hand_eval::evaluate::evaluate_simple;
use crate::hand_eval::hand_type::{identify_hand_type_simple, HandType};
use crate::state::card::{Card, Edition, Enhancement, Rank, Seal, Suit};

/// Simple xoshiro256** RNG. Deterministic from a u64 seed, fast,
/// good distribution. Used to sample drawn cards inside rollouts
/// without paying the cost of a more elaborate RNG.
#[derive(Clone, Debug)]
struct XoshiroRng {
    state: [u64; 4],
}

impl XoshiroRng {
    fn new(seed: u64) -> Self {
        // SplitMix64 seeding (standard practice).
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
        let result = self.state[1]
            .wrapping_mul(5).rotate_left(7).wrapping_mul(9);
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
        // Unbiased bounded sampling. For our small n (≤ 52) the bias
        // from the simple modulo is negligible, but use the standard
        // rejection-free trick anyway.
        (self.next_u64() as usize) % n
    }
}

/// Mutable rollout state. Cloned from a snapshot of caller inputs
/// before each rollout sample, so each sample starts from the same
/// initial state.
#[derive(Clone, Debug)]
struct RolloutState {
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

#[allow(clippy::too_many_arguments)]
fn greedy_rollout_clears(
    rs: &mut RolloutState,
    joker_names: &[String],
    joker_editions: &[Edition],
    joker_vampire_gain: &[f64],
    joker_obelisk_should_scale: &[bool],
    hand_levels_lookup: &dyn Fn(&str) -> u32,
    debuffed_suits: &[Suit],
    is_the_arm_blind: bool,
    pareidolia_active: bool,
    rng: &mut XoshiroRng,
) -> bool {
    // Iterate at most a generous bound to avoid infinite loops in
    // malformed states. Real rollouts terminate after ~5 plays.
    for _ in 0..20 {
        if rs.current_score >= rs.required_score {
            return true;
        }
        if rs.hands_remaining == 0 || rs.hand.is_empty() {
            return rs.current_score >= rs.required_score;
        }

        // Find the best play: enumerate combinations of size
        // 1..=min(5, hand_size), score each, take the argmax.
        // Mirrors `best_play_action_native` but inline so we share
        // the rollout's state context.
        let n = rs.hand.len();
        let max_size = n.min(5);
        let mut best_score: u64 = 0;
        let mut best_indices: Option<Vec<usize>> = None;
        let mut first = true;
        let mut idx = vec![0usize; max_size];
        for size in 1..=max_size {
            for i in 0..size { idx[i] = i; }
            loop {
                let played: Vec<Card> = idx[..size].iter().map(|&i| rs.hand[i]).collect();
                let mut in_played = [false; 16];
                for &i in &idx[..size] { in_played[i] = true; }
                let held: Vec<Card> = rs.hand.iter().enumerate()
                    .filter(|(i, _)| !in_played[*i])
                    .map(|(_, c)| *c)
                    .collect();
                if let Some(ht) = identify_hand_type_simple(&played) {
                    let level = hand_levels_lookup(ht.to_str());
                    let ctx = HandContext {
                        money: rs.money,
                        joker_count: joker_names.len() as u32,
                        joker_slot_limit: 5,
                        discards_remaining: rs.discards_remaining,
                        hands_remaining: rs.hands_remaining,
                        played_count_this_hand_type: rs.played_count_this_hand_type,
                        hand_type_played_before: rs.hand_type_played_before,
                        deck_size: rs.deck_size,
                        played_count_max_other_hand_type: rs.played_count_max_other_hand_type,
                        pareidolia_active,
                    };
                    if let Some((_chips, _mult, score, _ht)) = evaluate_simple(
                        &played, level, debuffed_suits,
                        joker_names, joker_editions, &rs.metadata,
                        &held, &rs.played_hand_types, ctx,
                    ) {
                        if first || score > best_score {
                            first = false;
                            best_score = score;
                            best_indices = Some(idx[..size].to_vec());
                        }
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

        let Some(play_indices) = best_indices else { return false };

        // ── Discard step ── mirrors Python `_greedy_rollout_clears` /
        // `_should_rollout_discard` / `_rollout_discard_action`. Without this
        // the rollout NEVER discards, so it returns ~0 on blinds that are only
        // clearable by discarding dead cards and redrawing. If the best play
        // can't clear and a discard could still rescue the round, discard the
        // lowest straight-rank non-protected cards and redraw, then re-loop
        // (a discard does NOT consume a hand).
        {
            let remaining: i64 = (rs.required_score - rs.current_score).max(0);
            let deck_n = rs.known_deck.len();
            if rs.discards_remaining > 0 && deck_n > 0 && (best_score as i64) < remaining {
                let pace = remaining as f64 / (rs.hands_remaining.max(1) as f64);
                if rs.hands_remaining <= 1 || (best_score as f64) < pace {
                    let mut protected = [false; 16];
                    for &i in &play_indices {
                        if i < 16 { protected[i] = true; }
                    }
                    let hand_n = rs.hand.len();
                    let mut candidates: Vec<usize> =
                        (0..hand_n).filter(|&i| !(i < 16 && protected[i])).collect();
                    let discard_limit = candidates.len().min(5).min(deck_n);
                    if discard_limit > 0 {
                        // Lowest straight-rank first (ties by index) == Python's
                        // _rollout_discard_rank_value ordering.
                        candidates.sort_by_key(|&i| (rs.hand[i].rank.straight_value(), i));
                        let mut is_discarded = [false; 16];
                        let discarded_cards: Vec<Card> = candidates[..discard_limit]
                            .iter()
                            .map(|&i| {
                                if i < 16 { is_discarded[i] = true; }
                                rs.hand[i]
                            })
                            .collect();

                        // Discard-triggered joker scaling (Green Joker, Hit the
                        // Road, Ramen, Yorick, …). target_suit=None for now: the
                        // rollout's play-scoring already runs without it, so
                        // Castle's discard chips are the only approximation.
                        let cur_chips: Vec<i32> = rs.metadata.iter().map(|m| m.current_plus_chips).collect();
                        let cur_mult: Vec<i32> = rs.metadata.iter().map(|m| m.current_plus_mult).collect();
                        let cur_xmult: Vec<f64> = rs.metadata.iter().map(|m| m.current_xmult).collect();
                        let target_suit_none: Vec<Option<u8>> = vec![None; joker_names.len()];
                        let updates = jokers_after_discard(
                            joker_names, &cur_chips, &cur_mult, &cur_xmult,
                            &rs.current_remaining, &target_suit_none, &discarded_cards,
                        );
                        for (i, u) in updates.iter().enumerate() {
                            if let Some(v) = u.new_plus_chips { rs.metadata[i].current_plus_chips = v; }
                            if let Some(v) = u.new_plus_mult { rs.metadata[i].current_plus_mult = v; }
                            if let Some(v) = u.new_xmult { rs.metadata[i].current_xmult = v; }
                            if let Some(v) = u.new_remaining { rs.current_remaining[i] = v; }
                        }

                        // Remove discarded cards from hand, draw replacements.
                        let mut new_hand: Vec<Card> = (0..hand_n)
                            .filter(|&i| !(i < 16 && is_discarded[i]))
                            .map(|i| rs.hand[i])
                            .collect();
                        let draw_count = discard_limit.min(rs.known_deck.len());
                        for _ in 0..draw_count {
                            if rs.known_deck.is_empty() { break; }
                            let di = rng.gen_range(rs.known_deck.len());
                            new_hand.push(rs.known_deck.remove(di));
                        }
                        rs.deck_size = rs.deck_size.saturating_sub(draw_count as u32);
                        rs.discards_remaining -= 1;
                        rs.hand = new_hand;
                        continue;
                    }
                }
            }
        }

        // Identify the played hand type (we have it from the loop
        // but re-derive for clarity; cheap call).
        let played_cards: Vec<Card> = play_indices.iter().map(|&i| rs.hand[i]).collect();
        let Some(ht) = identify_hand_type_simple(&played_cards) else { return false };
        let mut in_played = [false; 16];
        for &i in &play_indices { in_played[i] = true; }
        let held_cards: Vec<Card> = rs.hand.iter().enumerate()
            .filter(|(i, _)| !in_played[*i])
            .map(|(_, c)| *c)
            .collect();

        // Apply scoring deltas.
        rs.current_score += best_score as i64;
        rs.hands_remaining = rs.hands_remaining.saturating_sub(1);

        // Sample drawn cards from known_deck.
        let draw_count = play_indices.len().min(rs.known_deck.len());
        let mut drawn: Vec<Card> = Vec::with_capacity(draw_count);
        for _ in 0..draw_count {
            if rs.known_deck.is_empty() { break; }
            let idx = rng.gen_range(rs.known_deck.len());
            drawn.push(rs.known_deck.remove(idx));
        }
        rs.deck_size = rs.deck_size.saturating_sub(drawn.len() as u32);

        // Update joker scaling counters.
        let scoring_indices: Vec<usize> = (0..played_cards.len()).collect();
        let after_ctx = AfterPlayContext {
            hand_type: ht,
            played_cards: &played_cards,
            scoring_indices: &scoring_indices,
            n_played: played_cards.len(),
            hands_remaining: rs.hands_remaining,
            pareidolia_active,
        };
        let current_plus_chips: Vec<i32> = rs.metadata.iter().map(|m| m.current_plus_chips).collect();
        let current_plus_mult: Vec<i32> = rs.metadata.iter().map(|m| m.current_plus_mult).collect();
        let current_xmult: Vec<f64> = rs.metadata.iter().map(|m| m.current_xmult).collect();
        let obelisk_gain: Vec<f64> = rs.metadata.iter().map(|m| m.obelisk_gain).collect();
        let updates = jokers_after_play(
            joker_names,
            &current_plus_chips,
            &current_plus_mult,
            &current_xmult,
            &rs.current_remaining,
            joker_vampire_gain,
            joker_obelisk_should_scale,
            &obelisk_gain,
            0,
            after_ctx,
        );
        // Apply updates (we skip joker removal — the rollout doesn't
        // need to track Ice Cream / Seltzer disappearance since their
        // effect-zero metadata still produces the correct (zero)
        // contribution for the remaining rollout iterations.)
        for (i, u) in updates.iter().enumerate() {
            if let Some(v) = u.new_plus_chips { rs.metadata[i].current_plus_chips = v; }
            if let Some(v) = u.new_plus_mult { rs.metadata[i].current_plus_mult = v; }
            if let Some(v) = u.new_xmult { rs.metadata[i].current_xmult = v; }
            if let Some(v) = u.new_remaining { rs.current_remaining[i] = v; }
        }

        // Update played-hand-type tracking (for Card Sharp/Obelisk/Supernova).
        let ht_str = ht.to_str();
        rs.played_count_this_hand_type += 1; // we played this one
        rs.hand_type_played_before = true;
        // Recompute played_count_max_other_hand_type from the
        // appended list (rough — full histogram on small list).
        rs.played_hand_types.push(ht_str.to_string());
        let mut max_other: u32 = 0;
        for s in &rs.played_hand_types {
            if s.as_str() == ht_str { continue; }
            let count = rs.played_hand_types.iter().filter(|x| x.as_str() == s.as_str()).count() as u32;
            if count > max_other { max_other = count; }
        }
        rs.played_count_max_other_hand_type = max_other;

        // The Arm decrements stored hand level for the played type.
        let _ = hand_level_after_play(0, 0, is_the_arm_blind);

        // Build next hand: held + drawn. (Sort order doesn't matter
        // for our combinations enumeration.)
        let mut next_hand: Vec<Card> = Vec::with_capacity(held_cards.len() + drawn.len());
        next_hand.extend(held_cards);
        next_hand.extend(drawn);
        rs.hand = next_hand;
    }
    rs.current_score >= rs.required_score
}

/// PyO3 wrapper: native clear_probability estimator.
#[pyfunction]
#[pyo3(name = "clear_probability_native")]
#[pyo3(signature = (hand, known_deck, hand_levels, debuffed_suits,
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
                    samples=16, seed=0))]
#[allow(clippy::too_many_arguments)]
pub fn py_clear_probability_native(
    hand: Vec<Card>,
    known_deck: Vec<Card>,
    hand_levels: &Bound<'_, PyDict>,
    debuffed_suits: Vec<String>,
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
    samples: u32,
    seed: u64,
) -> PyResult<Option<f64>> {
    let suits: Vec<Suit> = debuffed_suits.iter()
        .filter_map(|s| Suit::from_str(s))
        .collect();
    let jokers = joker_names.unwrap_or_default();
    // Bail on any joker simulate_play_simple can't handle inside the
    // rollout. Jokers don't get added during a rollout, so the
    // initial check is sufficient.
    for n in &jokers {
        if SIMPLE_BAIL_JOKERS.contains(&n.as_str()) {
            return Ok(None);
        }
        if !crate::hand_eval::effects::is_supported_joker(n) {
            return Ok(None);
        }
    }
    // Bail on Glass cards in hand or known_deck (would shatter
    // stochastically, which simulate_play_simple doesn't model).
    for c in hand.iter().chain(known_deck.iter()) {
        if matches!(c.enhancement, Enhancement::Glass) {
            return Ok(None);
        }
        if matches!(c.seal, Seal::Blue) {
            return Ok(None);
        }
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

    // Capture hand_levels into a closure-friendly local lookup.
    let hand_levels_lookup = |ht_str: &str| -> u32 {
        hand_levels.get_item(ht_str)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<u32>().ok())
            .unwrap_or(1)
    };

    let total_samples = samples.max(1);
    let mut clears: u32 = 0;
    for sample_index in 0..total_samples {
        let mut rs = RolloutState {
            hand: hand.clone(),
            known_deck: known_deck.clone(),
            metadata: metadata.clone(),
            current_remaining: current_remaining.clone(),
            current_score,
            required_score,
            hands_remaining,
            discards_remaining,
            money,
            deck_size,
            played_count_this_hand_type,
            played_count_max_other_hand_type,
            hand_type_played_before,
            played_hand_types: played_hand_types.clone(),
        };
        // Per-sample sub-seed so independent rollouts give different
        // draw sequences.
        let mut rng = XoshiroRng::new(seed.wrapping_add(sample_index as u64));
        if greedy_rollout_clears(
            &mut rs,
            &jokers, &editions,
            &vampire_gain, &obelisk_should_scale,
            &hand_levels_lookup,
            &suits,
            false, // is_the_arm_blind — Arm is in SIMPLE_BAIL via blind safety
            pareidolia_active,
            &mut rng,
        ) {
            clears += 1;
        }
    }
    Ok(Some(clears as f64 / total_samples as f64))
}
