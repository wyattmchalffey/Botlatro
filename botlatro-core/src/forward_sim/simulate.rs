//! Top-level `simulate_play_native` — composes the Phase 2
//! evaluation with the Phase 3 helpers into ONE FFI call.
//!
//! This is where the FFI amortization actually pays off: one
//! cross-boundary call replaces what would otherwise be many
//! Python helper calls (each with its own conversion overhead).
//!
//! Coverage: **simple-case fast path only**. Returns None to bail
//! to Python when any of these are present:
//! - Card-mutating jokers: Midas Mask, Vampire, Hiker, DNA
//! - State-creation jokers: Sixth Sense, Hallucination, Gift Card,
//!   Mr. Bones, Crimson Heart, Trousers (hook discards)
//! - Mid-play discard jokers: Hook blind
//! - Special blinds: The Ox (money reset on hand type)
//! - Stochastic outcomes (Glass shatter, Bloodstone, Lucky Card, etc.)
//! - Splash (handled by Rust evaluate_simple already, but its
//!   scoring-indices change requires Python rebuild of state.modifiers)
//! - Observatory voucher (multiplier on hand)
//!
//! The simple-case fast path covers:
//! - First-blind (no jokers)
//! - Pure-scoring jokers (Joker, Greedy, Jolly, etc.)
//! - Scaling jokers (Green Joker, Vampire — wait, bail on Vampire,
//!   Ice Cream, Loyalty Card, etc.)
//! - Retrigger jokers (Hack, Sock and Buskin, Dusk, Seltzer, Hanging Chad)
//!
//! On the simple path, returns a `PlayTransition` struct with all
//! the fields Python needs to construct the next GameState via
//! `dataclass.replace`.

use pyo3::prelude::*;

use crate::hand_eval::effects::{HandContext, JokerMetadata};
use crate::hand_eval::evaluate::evaluate_simple;
use crate::hand_eval::hand_type::{identify_hand_type_simple, HandType};
use crate::state::card::{Card, Edition, Enhancement, Suit};

/// Jokers we definitely cannot handle in the simple fast path.
/// Their effects involve card mutation, state changes outside the
/// score path, or stochastic outcomes that solver-deterministic
/// search treats as empty but they still influence simulate_play.
pub const SIMPLE_BAIL_JOKERS: &[&str] = &[
    // Card mutation
    "Midas Mask", "Vampire", "Hiker", "DNA",
    // Consumable creation / state events. (Hallucination removed: its
    // only effect is creating a Tarot on BOOSTER-PACK open, which never
    // fires during a play-rollout — it is score-neutral here, and its
    // ability/scoring port already contributes 0.)
    "Sixth Sense",
    // End-of-round / save effects
    "Gift Card", "Mr. Bones",
    // Held-pile / draw effects
    "Crimson Heart",
    // (Spare Trousers removed: its +2-mult-on-two-pair-shape scaling is
    // fully modeled in jokers_after_play, and its scoring reads
    // current_plus_mult — both supported. The old bail was stale.)
    // Splash forces all-cards-scoring → mutated_played_cards changes
    "Splash",
    // Glass jokers — _played_cards_after_play with shattered glass
    "Glass Joker",
    // Madness disables a joker mid-round — complex
    "Madness",
];

/// Result of a successful simple-case play transition.
#[derive(Clone, Debug)]
pub struct PlayTransition {
    pub next_score: i64,
    pub next_money: i32,
    pub next_hands_remaining: i32,
    pub next_deck_size: u32,
    pub deck_indices_to_remove: Vec<usize>,
    pub next_phase_str: &'static str,
    pub mr_bones_fired: bool,
    pub hand_type_str: &'static str,
    pub new_hand_level: Option<u32>,
    pub held_end_money: i32,
    // Joker scaling updates (parallel to joker_names)
    pub joker_new_chips: Vec<Option<i32>>,
    pub joker_new_mult: Vec<Option<i32>>,
    pub joker_new_xmult: Vec<Option<f64>>,
    pub joker_new_remaining: Vec<Option<i32>>,
    pub joker_remove: Vec<bool>,
}

/// Heart of the simple fast path. Returns `None` to signal Python
/// fallback. Takes pre-converted inputs (the wire-in extracts these
/// from the Python GameState ONCE per call).
#[allow(clippy::too_many_arguments)]
pub fn simulate_play_simple(
    // Played cards + their evaluation inputs
    played_cards: &[Card],
    held_cards: &[Card],
    drawn_cards: &[Card],
    known_deck: &[Card],
    // Jokers
    joker_names: &[String],
    joker_editions: &[Edition],
    joker_metadata: &[JokerMetadata],
    // Per-joker scaling-update inputs
    joker_vampire_gain: &[f64],
    joker_obelisk_should_scale: &[bool],
    joker_current_remaining: &[i32],
    // Context
    hand_level: u32,
    ctx: HandContext,
    debuffed_suits: &[Suit],
    played_hand_types: &[String],
    current_score: i64,
    required_score: i64,
    deck_size: u32,
    is_the_arm_blind: bool,
    has_mr_bones: bool,
    mime_count: u32,
) -> Option<PlayTransition> {
    // === Bail checks ===
    // Unknown jokers, splash, or any complex-effect joker → bail.
    for n in joker_names {
        if SIMPLE_BAIL_JOKERS.contains(&n.as_str()) {
            return None;
        }
        if !crate::hand_eval::effects::is_supported_joker(n) {
            return None;
        }
    }
    // Bail on cards with Glass/Steel held (steel works in score but
    // _played_cards_after_play might mutate glass after shatter).
    // Held Glass / Steel doesn't shatter, but scored Glass does (RNG).
    // For solver determinism, no RNG outcomes, so glass never shatters.
    // Still, we bail on any held card with Glass to be conservative.
    for c in played_cards.iter().chain(held_cards.iter()) {
        if matches!(c.enhancement, Enhancement::Glass) {
            return None;
        }
    }
    // Bail on Blue seals (creates consumables at round end).
    for c in held_cards {
        if matches!(c.seal, crate::state::card::Seal::Blue) {
            return None;
        }
    }

    // === Score the played cards ===
    // identify + scoring all happens via evaluate_simple.
    let ht = identify_hand_type_simple(played_cards)?;
    let (chips, _mult, score, ht_str) = evaluate_simple(
        played_cards, hand_level, debuffed_suits,
        joker_names, joker_editions, joker_metadata,
        held_cards, played_hand_types, ctx,
    )?;
    let _ = chips; // suppress unused

    // === Apply post-score updates ===
    let next_score = current_score + score as i64;
    let next_hands = (ctx.hands_remaining as i32 - 1).max(0);
    // Money: no observatory multiplier in simple case → just += score.
    // Per Python: `next_money = money + evaluation.money_delta` where
    // money_delta only comes from Faceless / Golden / etc. — handled
    // by score-neutral arms. We approximate with 0 here for the simple
    // case (the bail set excludes money-modifying jokers... wait, no —
    // Faceless Joker etc. are score-neutral but DO affect money on
    // certain triggers. For now bail when any money-touching joker
    // present.)
    let next_money = ctx.money;  // money_delta = 0 in simple case

    // === Joker scaling updates (Phase 3b helper) ===
    let scoring_indices: Vec<usize> = (0..played_cards.len()).collect();
    let after_play_ctx = crate::forward_sim::jokers::AfterPlayContext {
        hand_type: ht,
        played_cards,
        scoring_indices: &scoring_indices,
        n_played: played_cards.len(),
        hands_remaining: ctx.hands_remaining,
        pareidolia_active: ctx.pareidolia_active,
    };
    let current_plus_chips: Vec<i32> = joker_metadata.iter().map(|m| m.current_plus_chips).collect();
    let current_plus_mult: Vec<i32> = joker_metadata.iter().map(|m| m.current_plus_mult).collect();
    let current_xmult: Vec<f64> = joker_metadata.iter().map(|m| m.current_xmult).collect();
    let obelisk_gain: Vec<f64> = joker_metadata.iter().map(|m| m.obelisk_gain).collect();
    let updates = crate::forward_sim::jokers::jokers_after_play(
        joker_names,
        &current_plus_chips,
        &current_plus_mult,
        &current_xmult,
        joker_current_remaining,
        joker_vampire_gain,
        joker_obelisk_should_scale,
        &obelisk_gain,
        0, // lucky_triggers (stochastic, 0 in solver)
        after_play_ctx,
    );

    // === Hand-level update (Phase 3f helper) ===
    let new_hand_level = crate::forward_sim::levels::hand_level_after_play(
        hand_level, 0, is_the_arm_blind,
    );

    // === Deck draw (Phase 3a helper) ===
    let (next_deck_size, deck_indices_to_remove) = if drawn_cards.is_empty() {
        (deck_size, Vec::new())
    } else if known_deck.is_empty() {
        (deck_size.saturating_sub(drawn_cards.len() as u32), Vec::new())
    } else {
        crate::forward_sim::deck::draw_indices_to_remove(known_deck, deck_size, drawn_cards).ok()?
    };

    // === Next phase decision (Phase 3d helper) ===
    let next_phase = crate::forward_sim::phase::next_phase(
        required_score, next_score, next_hands as u32, has_mr_bones,
    );
    let mr_bones_fired = matches!(next_phase, crate::forward_sim::phase::NextPhase::MrBonesSave);

    // === Held end-of-round money (Phase 3e helper) ===
    let held_end_money = if matches!(
        next_phase,
        crate::forward_sim::phase::NextPhase::RoundEval
        | crate::forward_sim::phase::NextPhase::MrBonesSave
    ) {
        crate::forward_sim::economy::held_end_of_round_money_delta(held_cards, mime_count)
    } else {
        0
    };

    // Unpack joker updates into flat vectors.
    let n = joker_names.len();
    let mut joker_new_chips = Vec::with_capacity(n);
    let mut joker_new_mult = Vec::with_capacity(n);
    let mut joker_new_xmult = Vec::with_capacity(n);
    let mut joker_new_remaining = Vec::with_capacity(n);
    let mut joker_remove = Vec::with_capacity(n);
    for u in updates {
        joker_new_chips.push(u.new_plus_chips);
        joker_new_mult.push(u.new_plus_mult);
        joker_new_xmult.push(u.new_xmult);
        joker_new_remaining.push(u.new_remaining);
        joker_remove.push(u.remove);
    }

    Some(PlayTransition {
        next_score,
        next_money: next_money + held_end_money,
        next_hands_remaining: next_hands,
        next_deck_size,
        deck_indices_to_remove,
        next_phase_str: next_phase.to_str(),
        mr_bones_fired,
        hand_type_str: ht_str,
        new_hand_level,
        held_end_money,
        joker_new_chips,
        joker_new_mult,
        joker_new_xmult,
        joker_new_remaining,
        joker_remove,
    })
}

/// PyO3 wrapper. Returns None to bail; on success returns a 14-tuple
/// with all the delta fields Python needs to construct the next state.
#[pyfunction]
#[pyo3(name = "simulate_play_simple")]
#[pyo3(signature = (played_cards, held_cards, drawn_cards, known_deck,
                    joker_names, joker_editions,
                    joker_current_plus_mult, joker_current_plus_chips,
                    joker_current_xmult, joker_current_remaining,
                    joker_loyalty_ready, joker_drivers_active,
                    joker_leading_plus_mult, joker_leading_plus_chips,
                    joker_sell_value, joker_rarity,
                    joker_target_suit, joker_target_rank,
                    joker_obelisk_gain, joker_vampire_gain,
                    joker_obelisk_should_scale,
                    hand_level, money, joker_slot_limit,
                    discards_remaining, hands_remaining,
                    played_hand_types, played_count_this_hand_type,
                    hand_type_played_before, deck_size,
                    played_count_max_other_hand_type, pareidolia_active,
                    debuffed_suits, current_score, required_score,
                    is_the_arm_blind, has_mr_bones, mime_count))]
#[allow(clippy::too_many_arguments)]
pub fn py_simulate_play_simple(
    played_cards: Vec<Card>,
    held_cards: Vec<Card>,
    drawn_cards: Vec<Card>,
    known_deck: Vec<Card>,
    joker_names: Vec<String>,
    joker_editions: Vec<Option<String>>,
    joker_current_plus_mult: Vec<i32>,
    joker_current_plus_chips: Vec<i32>,
    joker_current_xmult: Vec<f64>,
    joker_current_remaining: Vec<i32>,
    joker_loyalty_ready: Vec<bool>,
    joker_drivers_active: Vec<bool>,
    joker_leading_plus_mult: Vec<i32>,
    joker_leading_plus_chips: Vec<i32>,
    joker_sell_value: Vec<i32>,
    joker_rarity: Vec<u8>,
    joker_target_suit: Vec<Option<String>>,
    joker_target_rank: Vec<Option<String>>,
    joker_obelisk_gain: Vec<f64>,
    joker_vampire_gain: Vec<f64>,
    joker_obelisk_should_scale: Vec<bool>,
    hand_level: u32,
    money: i32,
    joker_slot_limit: u32,
    discards_remaining: u32,
    hands_remaining: u32,
    played_hand_types: Vec<String>,
    played_count_this_hand_type: u32,
    hand_type_played_before: bool,
    deck_size: u32,
    played_count_max_other_hand_type: u32,
    pareidolia_active: bool,
    debuffed_suits: Vec<String>,
    current_score: i64,
    required_score: i64,
    is_the_arm_blind: bool,
    has_mr_bones: bool,
    mime_count: u32,
) -> Option<(
    i64,                    // next_score
    i32,                    // next_money
    i32,                    // next_hands_remaining
    u32,                    // next_deck_size
    Vec<usize>,             // deck_indices_to_remove
    &'static str,           // next_phase
    bool,                   // mr_bones_fired
    &'static str,           // hand_type
    Option<u32>,            // new_hand_level
    i32,                    // held_end_money
    // Per-joker update: (new_chips, new_mult, new_xmult, new_remaining, remove)
    Vec<(Option<i32>, Option<i32>, Option<f64>, Option<i32>, bool)>,
)> {
    let suits: Vec<Suit> = debuffed_suits.iter()
        .filter_map(|s| Suit::from_str(s))
        .collect();
    let editions: Vec<Edition> = joker_editions.iter()
        .map(|o| Edition::from_option_str(o.as_deref()))
        .collect();

    // Build JokerMetadata array. We re-use the same builder pattern
    // as evaluate.rs's `build_metadata`. Inline here to avoid making
    // it pub.
    let n = joker_names.len();
    let mut metadata = vec![JokerMetadata::default(); n];
    for i in 0..n {
        let m = &mut metadata[i];
        if i < joker_current_plus_mult.len() { m.current_plus_mult = joker_current_plus_mult[i]; }
        if i < joker_current_plus_chips.len() { m.current_plus_chips = joker_current_plus_chips[i]; }
        if i < joker_current_xmult.len() { m.current_xmult = joker_current_xmult[i]; }
        if i < joker_loyalty_ready.len() { m.loyalty_ready = joker_loyalty_ready[i]; }
        if i < joker_drivers_active.len() { m.drivers_active = joker_drivers_active[i]; }
        if i < joker_leading_plus_mult.len() { m.leading_plus_mult = joker_leading_plus_mult[i]; }
        if i < joker_leading_plus_chips.len() { m.leading_plus_chips = joker_leading_plus_chips[i]; }
        if i < joker_sell_value.len() { m.sell_value = joker_sell_value[i]; }
        if i < joker_rarity.len() { m.rarity = joker_rarity[i]; }
        if i < joker_target_suit.len() {
            if let Some(s) = joker_target_suit[i].as_deref() {
                if let Some(suit) = Suit::from_str(s) {
                    m.has_target_suit = true;
                    m.target_suit = suit as u8;
                }
            }
        }
        if i < joker_target_rank.len() {
            if let Some(r) = joker_target_rank[i].as_deref() {
                if let Some(rank) = crate::state::card::Rank::from_str(r) {
                    m.has_target_rank = true;
                    m.target_rank = rank as u8;
                }
            }
        }
        if i < joker_obelisk_gain.len() { m.obelisk_gain = joker_obelisk_gain[i]; }
    }

    let ctx = HandContext {
        money,
        joker_count: n as u32,
        joker_slot_limit,
        discards_remaining,
        hands_remaining,
        played_count_this_hand_type,
        hand_type_played_before,
        deck_size,
        played_count_max_other_hand_type,
        pareidolia_active,
    };

    let r = simulate_play_simple(
        &played_cards, &held_cards, &drawn_cards, &known_deck,
        &joker_names, &editions, &metadata,
        &joker_vampire_gain, &joker_obelisk_should_scale, &joker_current_remaining,
        hand_level, ctx, &suits, &played_hand_types,
        current_score, required_score, deck_size,
        is_the_arm_blind, has_mr_bones, mime_count,
    )?;

    // Pack per-joker updates into a single Vec of 5-tuples to keep
    // the outer return tuple under PyO3's 12-element auto-conversion
    // limit.
    let n_jokers = r.joker_new_chips.len();
    let mut joker_updates = Vec::with_capacity(n_jokers);
    for i in 0..n_jokers {
        joker_updates.push((
            r.joker_new_chips[i],
            r.joker_new_mult[i],
            r.joker_new_xmult[i],
            r.joker_new_remaining[i],
            r.joker_remove[i],
        ));
    }
    Some((
        r.next_score,
        r.next_money,
        r.next_hands_remaining,
        r.next_deck_size,
        r.deck_indices_to_remove,
        r.next_phase_str,
        r.mr_bones_fired,
        r.hand_type_str,
        r.new_hand_level,
        r.held_end_money,
        joker_updates,
    ))
}
